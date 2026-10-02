"""TrajImpute 直接缺失预测评估器（无未来泄漏）。

评估口径（任务书 §十）：
    minADE_K / minFDE_K  K 条预测同一组取 min（与 src/metrics 同语义）
    ADE@1 / FDE@1        最高概率模式的单模态误差（sort by pi, top-1）
    MR                   全部 K 条预测终点误差均 > threshold 才算 miss
                         （与 src/metrics.MR 的 all(-1) 语义一致）

层级：scene / difficulty / split / missing_count / variant / seed。
整体聚合 = 按样本 micro-average（每组报告样本数，整体均值不隐藏分组差异）。
官方插补后预测 ADE/FDE 只能作 external_reference，禁止混入本文 direct 指标。
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def _ade_fde(pred: torch.Tensor, target: torch.Tensor):
    """pred [B, K, T, 2], target [B, T, 2] -> ade [B, K], fde [B, K]"""
    d = torch.norm(pred - target.unsqueeze(1), p=2, dim=-1)  # [B, K, T]
    return d.mean(-1), d[..., -1]


def evaluate_predictions(pred: torch.Tensor, prob: torch.Tensor, target: torch.Tensor,
                         miss_threshold: float = 2.0):
    """一组预测的 direct 指标（纯函数，无状态，便于测试）。

    Args:
        pred:   [B, K, T, 2]
        prob:   [B, K] 模态概率（未归一化 logit 或概率均可，仅用于排序）
        target: [B, T, 2]
    Returns:
        dict: minADE_K/minFDE_K/ADE@1/FDE@1/MR（每样本一个值的均值前的张量）
    """
    B, K, T, _ = pred.shape
    ade, fde = _ade_fde(pred, target)
    # top-1 = 最高概率模式（按 pi 排序取第一）
    order = torch.argsort(prob, dim=-1, descending=True)
    top1 = order[:, 0]
    ade1 = ade[torch.arange(B), top1]
    fde1 = fde[torch.arange(B), top1]
    return {
        "minADE_K": ade.min(-1).values,
        "minFDE_K": fde.min(-1).values,
        "ADE@1": ade1,
        "FDE@1": fde1,
        "MR": (fde > miss_threshold).all(-1).float(),
        "_K": K,
    }


class DirectEvaluator:
    """按 missing_count 分组累积指标；compute() 输出层级化结果。"""

    def __init__(self, miss_threshold: float = 2.0):
        self.miss_threshold = miss_threshold
        # (scene, difficulty, split, missing_count) -> {metric: [values]}
        self.groups = defaultdict(lambda: defaultdict(list))
        # dimension -> value -> {metric: [values]}
        self.dimension_groups = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

    def update(self, pred, prob, target, scene, difficulty, split, missing_count,
               valid_count=None, anchor_lag=None, forecast_gap=None):
        res = evaluate_predictions(pred, prob, target, self.miss_threshold)
        key = (scene, difficulty, split, int(missing_count))
        for name, vals in res.items():
            if name.startswith("_"):
                continue
            self.groups[key][name].extend(vals.tolist())
        dimensions = {"missing_count": int(missing_count)}
        if valid_count is not None:
            dimensions["valid_count"] = int(valid_count)
        if anchor_lag is not None:
            dimensions["anchor_lag"] = int(anchor_lag)
            dimensions["terminal_missing"] = bool(int(anchor_lag) > 0)
        if forecast_gap is not None:
            dimensions["forecast_gap"] = int(forecast_gap)
        for dimension, value in dimensions.items():
            value_key = str(value).lower() if isinstance(value, bool) else str(value)
            for name, vals in res.items():
                if name.startswith("_"):
                    continue
                self.dimension_groups[dimension][value_key][name].extend(vals.tolist())

    def compute(self):
        """-> dict[layered results]；每级均含 n、micro 整体与分 missing_count 明细。"""
        out: dict = {"by_group": {}, "by_dimension": {}, "overall": {}}
        for (scene, diff, split, mc), metrics in sorted(self.groups.items()):
            entry: dict = {"n": len(next(iter(metrics.values())))}
            for name, vals in metrics.items():
                entry[name] = sum(vals) / len(vals)
            out["by_group"][f"{scene}/{diff}/{split}/missing={mc}"] = entry
        for dimension, values in sorted(self.dimension_groups.items()):
            out["by_dimension"][dimension] = {}
            for value, metrics in sorted(values.items(), key=lambda item: item[0]):
                entry = {"n": len(next(iter(metrics.values())))}
                for name, vals in metrics.items():
                    entry[name] = sum(vals) / len(vals)
                out["by_dimension"][dimension][value] = entry
        # micro-average：所有样本平权
        all_vals = defaultdict(list)
        for metrics in self.groups.values():
            for name, vals in metrics.items():
                all_vals[name].extend(vals)
        for name, vals in all_vals.items():
            out["overall"][name] = sum(vals) / len(vals)
        out["overall"]["n"] = len(all_vals.get("minFDE_K", []))
        out["aggregation"] = "micro-average over all focal samples"  # type: ignore[assignment]
        return out


def save_results(results: dict, meta: dict, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return out_path
