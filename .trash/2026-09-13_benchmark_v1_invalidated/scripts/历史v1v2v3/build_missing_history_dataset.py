"""第一轮缺失历史数据集制作脚本（missing_history_v1）。

按 docs/research/ETHUCY与SDD第一轮缺失历史数据集制作说明.md 生成：
  - data/ETHUCY_missing_v1/{complete,random_single,random_block2}/fold_*/{train,val,test}/<scene>/*.pt
  - data/SDD_missing_v1/{complete,random_single,random_block2}/{train,val,test}/*.pt

要点：
  1. 只对历史 8 帧施加缺失，未来 12 帧保持完整；
  2. 帧编号 0-7，第 6、7 帧强制可见，缺失只从第 0-5 帧中抽；
  3. 掩码种子由 dataset|fold|split|scene_id|source_index|focal_id|condition|mask_seed
     经 SHA256 派生，完全确定、可复现，与遍历顺序/进程数无关；
  4. complete 条件不调用随机数生成器；
  5. SDD train/val 划分严格复刻 moflow_sdd_dataset.py（default_rng(2024) 置换）；
  6. 缺失历史坐标写 0.0，valid_mask[:, :8] = history_mask，valid_mask[:, 8:] 全 True。

用法：v1（历史，掩码生成路径与现在完全兼容）：
  python3 scripts/数据集构建/build_missing_history_dataset.py --dataset ethucy \
      --source-root data/ETHUCY_benchmark_v1 --output-root data/ETHUCY_missing_v1 \
      --conditions complete random_single random_block2 --mask-seed 42
  python3 scripts/数据集构建/build_missing_history_dataset.py --dataset sdd \
      --source-root data/sdd --output-root data/SDD_missing_v1 \
      --conditions complete random_single random_block2 --split-seed 2024 --mask-seed 42

v2 高缺失（missing_history_v2_high，docs/research/ETHUCY与SDD第二轮高缺失历史实验方案.md）：
  python3 scripts/数据集构建/build_missing_history_dataset.py --dataset ethucy \
      --source-root data/ETHUCY_benchmark_v1 --output-root data/ETHUCY_missing_v2_high \
      --conditions random_fixed3 random_fixed4 random_block3 random_block4 random_block6 \
      --mask-seed 42 --version missing_history_v2_high
  python3 scripts/数据集构建/build_missing_history_dataset.py --dataset sdd \
      --source-root data/sdd --output-root data/SDD_missing_v2_high \
      --conditions random_fixed3 random_fixed4 random_block3 random_block4 random_block6 \
      --split-seed 2024 --mask-seed 42 --version missing_history_v2_high
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

OBS_LEN = 8
PRED_LEN = 12
SEQ_LEN = 20
DT = 0.4
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
SPLITS = ["train", "val", "test"]
# 可缺失帧范围：第 0-5 帧（第 6、7 帧强制可见）
MISSABLE = list(range(6))
CONDITIONS = [
    "complete", "random_single", "random_block2",                    # v1
    "random_fixed3", "random_fixed4", "random_fixed5",               # v2_high 随机缺失
    "random_block3", "random_block4", "random_block6",               # v2_high 连续缺失
    "random_fixed3_ng", "random_fixed4_ng",                          # v3_noguard 随机缺失
    "random_block3_ng", "random_block4_ng", "random_block6_ng",      # v3_noguard 连续缺失
    "uniform_hard_ng",                                                # v3_noguard TrajImpute hard 对齐
]
# v3_noguard：无保护条件（候选帧 0..7，每 actor ≥1 可见帧；见第二轮方案 §5.1）
V3_CONDITIONS = {
    "random_fixed3_ng", "random_fixed4_ng",
    "random_block3_ng", "random_block4_ng", "random_block6_ng",
    "uniform_hard_ng",
}
# 每个条件的名义缺失帧数 / 缺失率 / 掩码规则（写入 manifest，方案 §2.2 / §7.3）
CONDITION_SPECS = {
    "complete":      {"missing_frames": 0, "nominal_rate": 0.0,   "rule": "all_visible"},
    "random_single": {"missing_frames": 1, "nominal_rate": 0.125, "rule": "random_1_distinct_of_0-5"},
    "random_block2": {"missing_frames": 2, "nominal_rate": 0.25,  "rule": "contiguous_block2_start_0-4"},
    "random_fixed3": {"missing_frames": 3, "nominal_rate": 0.375, "rule": "random_3_distinct_of_0-5"},
    "random_fixed4": {"missing_frames": 4, "nominal_rate": 0.5,   "rule": "random_4_distinct_of_0-5"},
    "random_fixed5": {"missing_frames": 5, "nominal_rate": 0.625, "rule": "random_5_distinct_of_0-5"},
    "random_block3": {"missing_frames": 3, "nominal_rate": 0.375, "rule": "contiguous_block3_start_0-3"},
    "random_block4": {"missing_frames": 4, "nominal_rate": 0.5,   "rule": "contiguous_block4_start_0-2"},
    "random_block6": {"missing_frames": 6, "nominal_rate": 0.75,  "rule": "fixed_block_0-5"},
    "random_fixed3_ng": {"missing_frames": 3, "nominal_rate": 0.375, "rule": "random_3_distinct_of_0-7_noguard"},
    "random_fixed4_ng": {"missing_frames": 4, "nominal_rate": 0.5,   "rule": "random_4_distinct_of_0-7_noguard"},
    "random_block3_ng": {"missing_frames": 3, "nominal_rate": 0.375, "rule": "contiguous_block3_start_0-5_noguard"},
    "random_block4_ng": {"missing_frames": 4, "nominal_rate": 0.5,   "rule": "contiguous_block4_start_0-4_noguard"},
    "random_block6_ng": {"missing_frames": 6, "nominal_rate": 0.75,  "rule": "contiguous_block6_start_0-2_noguard"},
    "uniform_hard_ng":  {"missing_frames": "4-7", "nominal_rate": 0.6875, "rule": "m~Uniform{4,5,6,7}_then_m_distinct_of_0-7_noguard"},
}


def mask_seed_from(key: str) -> int:
    """由样本身份字符串派生确定性掩码种子。"""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def derive_history_mask(
    dataset: str,
    fold,
    split: str,
    scene_id: str,
    source_index: int,
    source_file: str,
    focal_id: int,
    condition: str,
    mask_seed: int,
    num_agents: int,
) -> torch.Tensor:
    """按行人生成历史掩码 [A, 8]，True=可见。

    身份键含 source_file：同一 (fold,split,scene) 的不同源 .txt 样本即使
    sample_index 相同也获得独立掩码（防跨源掩码碰撞）。

    v1 条件（complete/random_single/random_block2）的采样代码路径保持逐字节
    不变（相同的 rng 调用序列），v1 数据集重跑可精确复现。
    """
    if condition == "complete":
        return torch.ones(num_agents, OBS_LEN, dtype=torch.bool)

    key = "|".join(
        str(x) for x in [dataset, fold, split, scene_id, source_index, source_file, focal_id, condition, mask_seed]
    )
    rng = np.random.default_rng(mask_seed_from(key))
    mask = torch.ones(num_agents, OBS_LEN, dtype=torch.bool)

    if condition in V3_CONDITIONS:
        # ---- v3_noguard：候选帧 0..7，无帧 6/7 保护，每 actor ≥1 可见帧 ----
        # 若某 actor 掩码全 False，以 key+"|retry{k}" 重派生 rng 重抽（确定性，方案 §5.1.4）
        for a in range(num_agents):
            if condition == "uniform_hard_ng":
                # m ~ Uniform{4,5,6,7}，再无放回抽 m 帧（TrajImpute hard 档对齐）
                need_retry = True
                for k in range(0, 64):
                    rng_a = (np.random.default_rng(mask_seed_from(key + f"|a{a}"))
                             if k == 0 else
                             np.random.default_rng(mask_seed_from(key + f"|a{a}|retry{k}")))
                    m = 4 + int(rng_a.integers(0, 4))
                    if m <= OBS_LEN - 1:  # m<=7 保证 ≥1 可见帧
                        ts = rng_a.choice(OBS_LEN, size=m, replace=False)
                        mask[a, ts.long() if hasattr(ts, "long") else torch.as_tensor(ts, dtype=torch.long)] = False
                        need_retry = False
                        break
                if need_retry:
                    raise RuntimeError(f"uniform_hard_ng mask retry exhausted: {key}")
            elif condition in ("random_fixed3_ng", "random_fixed4_ng"):
                k_n = int(condition[len("random_fixed")])
                need_retry = True
                for k in range(0, 64):
                    rng_a = (np.random.default_rng(mask_seed_from(key + f"|a{a}"))
                             if k == 0 else
                             np.random.default_rng(mask_seed_from(key + f"|a{a}|retry{k}")))
                    ts = rng_a.choice(OBS_LEN, size=k_n, replace=False)
                    if k_n <= OBS_LEN - 1:
                        mask[a, torch.as_tensor(ts, dtype=torch.long)] = False
                        need_retry = False
                        break
                if need_retry:
                    raise RuntimeError(f"{condition} mask retry exhausted: {key}")
            else:
                # 连续块：block3 s∈0..5 / block4 s∈0..4 / block6 s∈0..2
                m = int(condition[len("random_block")])
                s_hi = OBS_LEN - m  # 保证块尾 <=7
                need_retry = True
                for k in range(0, 64):
                    rng_a = (np.random.default_rng(mask_seed_from(key + f"|a{a}"))
                             if k == 0 else
                             np.random.default_rng(mask_seed_from(key + f"|a{a}|retry{k}")))
                    s = int(rng_a.integers(0, s_hi + 1))
                    if s + m <= OBS_LEN:  # 连续块必满足 m<=7，恒有可见帧
                        mask[a, s:s + m] = False
                        need_retry = False
                        break
                if need_retry:
                    raise RuntimeError(f"{condition} mask retry exhausted: {key}")
        # v3 无帧 6/7 强制可见；每 actor 至少 1 可见帧已由构造保证（m<=7 / 连续块 m<=7）
        return mask

    if condition == "random_single":
        # 每个行人独立随机缺失 1 帧（从 0-5 中抽）
        for a in range(num_agents):
            t = int(rng.integers(0, len(MISSABLE)))
            mask[a, t] = False
    elif condition == "random_block2":
        # 每个行人独立随机缺失连续 2 帧（起点从 0-4 中抽，块为 {s, s+1}）
        for a in range(num_agents):
            s = int(rng.integers(0, len(MISSABLE) - 1))
            mask[a, s] = False
            mask[a, s + 1] = False
    elif condition in ("random_fixed3", "random_fixed4", "random_fixed5"):
        # v2：从候选帧 0-5 中无放回抽 k 个不同帧（方案 §3.1）
        k = int(condition[-1])
        for a in range(num_agents):
            ts = rng.choice(len(MISSABLE), size=k, replace=False)
            for t in ts:
                mask[a, int(t)] = False
    elif condition in ("random_block3", "random_block4"):
        # v2：连续 m 帧，起点 s ∈ [0, 6-m]（block3: s∈0-3, block4: s∈0-2）
        m = int(condition[-1])
        for a in range(num_agents):
            s = int(rng.integers(0, len(MISSABLE) - m + 1))
            mask[a, s:s + m] = False
    elif condition == "random_block6":
        # v2：唯一合法块 {0,...,5}，无需随机（方案 §3.1）；condition/seed 仍保留在元数据
        mask[:, :6] = False
    else:
        raise ValueError(f"unknown condition: {condition}")
    # 强制第 6、7 帧可见（双保险）
    mask[:, 6] = True
    mask[:, 7] = True
    return mask


def apply_mask_to_sample(sample: dict, history_mask: torch.Tensor) -> dict:
    """将历史掩码应用到 canonical 样本：缺失处置 0.0，重写 valid_mask。

    v3_noguard：额外写入逐 actor 的 last_valid_idx / anchor_lag_steps / forecast_gap_steps
    （方案 §5.1.2/5.1.4）。v1/v2 条件下三字段同样写入（anchor_lag 恒 0、
    forecast_gap 恒 1），字段集统一，读取端无需分版本分支。
    """
    positions = sample["positions"].clone()
    valid_mask = sample["valid_mask"].clone()
    positions[:, :OBS_LEN] = positions[:, :OBS_LEN] * history_mask.unsqueeze(-1).float()
    valid_mask[:, :OBS_LEN] = history_mask
    valid_mask[:, OBS_LEN:] = True
    out = dict(sample)
    out["positions"] = positions
    out["valid_mask"] = valid_mask
    out["history_mask"] = history_mask.clone()
    # 逐 actor 时间间隔字段：仅 v3_noguard 写入（保持 v1/v2 输出文件逐字节不变；
    # v1/v2 下 anchor_lag 恒 0 / forecast_gap 恒 1，读取端按缺省处理）
    if getattr(apply_mask_to_sample, "_v3_mode", False):
        last_valid_idx = torch.full((history_mask.size(0),), -1, dtype=torch.long)
        for a in range(history_mask.size(0)):
            idxs = torch.nonzero(history_mask[a]).flatten()
            if len(idxs):
                last_valid_idx[a] = int(idxs[-1].item())
        out["last_valid_idx"] = last_valid_idx
        out["anchor_lag_steps"] = (7 - last_valid_idx).clamp(min=0)
        out["forecast_gap_steps"] = (8 - last_valid_idx).clamp(min=1)
    return out


# ---------------------------------------------------------------------------
# ETH/UCY
# ---------------------------------------------------------------------------

def list_ethucy_units(source_root: Path, scenes=None):
    """枚举 (fold, split, scene, files) 工作单元，顺序固定。

    scenes 非 None 时只保留这些 scene 名（冒烟测试用；修复历史 no-op：
    之前 --scenes 声明了但从未传入此处）。
    """
    units = []
    for fold in FOLDS:
        for split in SPLITS:
            split_dir = source_root / f"fold_{fold}" / split
            if not split_dir.exists():
                continue
            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                if scenes is not None and scene_dir.name not in scenes:
                    continue
                files = sorted(scene_dir.glob("*.pt"))
                if files:
                    units.append((fold, split, scene_dir.name, [str(f) for f in files]))
    return units


def build_ethucy_unit(args):
    """处理一个 (condition, fold, split, scene) 单元。在子进程中运行。"""
    condition, fold, split, scene, files, source_root, output_root, mask_seed = args
    apply_mask_to_sample._v3_mode = condition in V3_CONDITIONS
    out_dir = Path(output_root) / condition / f"fold_{fold}" / split / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "samples": 0, "agents": 0, "missing_frames": 0,
        "rejected": 0, "reject_reasons": {},
        "source_index_min": None, "source_index_max": None,
        "sha256": [],  # (filename, digest)
        "m_hist": {},
    }
    for f in files:
        try:
            src = torch.load(f, weights_only=False)
            A, T, _ = src["positions"].shape
            if T != SEQ_LEN or A < 1:
                raise ValueError(f"bad shape {src['positions'].shape}")
            if not bool(src["valid_mask"].all()):
                raise ValueError("source valid_mask not all True")
        except Exception as e:  # noqa: BLE001
            stats["rejected"] += 1
            stats["reject_reasons"][str(e)[:80]] = stats["reject_reasons"].get(str(e)[:80], 0) + 1
            continue

        source_index = int(src["sample_index"])
        hmask = derive_history_mask(
            "ethucy", fold, split, src["scene_id"], source_index,
            str(src["source_file"]), int(src["focal_id"]), condition, mask_seed, A,
        )
        out = apply_mask_to_sample(src, hmask)
        out["condition"] = condition
        out["source_file"] = src["source_file"]
        out["source_index"] = source_index
        out["mask_seed"] = mask_seed
        out["obs_len"] = OBS_LEN
        out["pred_len"] = PRED_LEN
        out["dt"] = DT
        out["unit"] = "meter"

        name = Path(f).name
        out_path = out_dir / name
        torch.save(out, out_path)

        stats["samples"] += 1
        stats["agents"] += A
        stats["missing_frames"] += int((~hmask).sum())
        if condition == "uniform_hard_ng":
            for m, c in zip(*np.unique((~hmask).sum(dim=1).tolist(), return_counts=True)):
                mk = str(int(m)); cv = int(c)
                stats.setdefault("m_hist", {})[mk] = stats["m_hist"].get(mk, 0) + cv
        si = source_index
        stats["source_index_min"] = si if stats["source_index_min"] is None else min(stats["source_index_min"], si)
        stats["source_index_max"] = si if stats["source_index_max"] is None else max(stats["source_index_max"], si)
        h = hashlib.sha256()
        with open(out_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        stats["sha256"].append((name, h.hexdigest()))
    return (condition, fold, split, scene), stats


def build_ethucy(source_root: Path, output_root: Path, conditions, mask_seed, workers, version="missing_history_v1", scenes=None):
    units = list_ethucy_units(source_root, scenes=scenes)
    total_src = sum(len(u[3]) for u in units)
    print(f"[ethucy] source units={len(units)} files={total_src}", flush=True)

    tasks = [(c, u[0], u[1], u[2], u[3], str(source_root), str(output_root), mask_seed)
             for c in conditions for u in units]
    # sample_index：按固定遍历顺序赋值（fold,split,scene 文件名序），与条件无关
    manifest_stats = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, stats in ex.map(build_ethucy_unit, tasks, chunksize=1):
            condition, fold, split, scene = key
            md = manifest_stats.setdefault(
                (condition, fold, split),
                {"samples": 0, "agents": 0, "missing_frames": 0, "history_frames": 0,
                 "rejected": 0, "reject_reasons": {}, "scenes": {},
                 "source_index_min": None, "source_index_max": None},
            )
            md["samples"] += stats["samples"]
            md["agents"] += stats["agents"]
            md["missing_frames"] += stats["missing_frames"]
            md["history_frames"] += stats["agents"] * OBS_LEN
            md["rejected"] += stats["rejected"]
            for k, v in stats["reject_reasons"].items():
                md["reject_reasons"][k] = md["reject_reasons"].get(k, 0) + v
            md["scenes"][scene] = {
                "samples": stats["samples"],
                "file_digests": dict(stats["sha256"]),
            }
            if stats.get("m_hist"):
                md["m_hist"] = {k: md.get("m_hist", {}).get(k, 0) + v for k, v in stats["m_hist"].items()}
            md["source_index_min"] = stats["source_index_min"] if md["source_index_min"] is None else min(md["source_index_min"], stats["source_index_min"])
            md["source_index_max"] = stats["source_index_max"] if md["source_index_max"] is None else max(md["source_index_max"], stats["source_index_max"])
            done += 1
            if done % 30 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                print(f"[ethucy] {done}/{len(tasks)} units, {rate:.1f} units/s", flush=True)

    manifest = {
        "version": version,
        "dataset": "ETHUCY",
        "obs_len": OBS_LEN,
        "pred_len": PRED_LEN,
        "dt": DT,
        "unit": "meter",
        "mask_seed": mask_seed,
        "split_seed": None,
        "conditions": conditions,
        "condition_specs": {c: CONDITION_SPECS[c] for c in conditions},
        "history_visibility_rule": ("no_guard_min_1_visible" if version == "missing_history_v3_noguard"
                                    else "frames_6_and_7_visible"),
        "future_policy": "complete",
        "coordinate_policy": "raw_scene_coordinates",
        "feature_policy": "recompute_from_visible_history",
        "source_root": str(source_root),
        "source_version": "ETHUCY_benchmark_v1",
        "mask_rng_key_format": "dataset|fold|split|scene_id|source_index|source_file|focal_id|condition|mask_seed -> sha256[:8] -> np.random.default_rng",
        "splits": {},
    }
    for (condition, fold, split), md in sorted(manifest_stats.items()):
        cd = manifest["splits"].setdefault(condition, {}).setdefault(f"fold_{fold}", {})
        cd[split] = {
            "samples": md["samples"],
            "agents_total": md["agents"],
            "missing_history_frames": md["missing_frames"],
            "history_frames_total": md["history_frames"],
            "actual_missing_rate": round(md["missing_frames"] / md["history_frames"], 6) if md["history_frames"] else 0.0,
            "source_index_range": [md["source_index_min"], md["source_index_max"]],
            "rejected_samples": md["rejected"],
            "reject_reasons": md["reject_reasons"],
            "scenes": {s: v["samples"] for s, v in md["scenes"].items()},
            "per_file_sha256": {s: v["file_digests"] for s, v in md["scenes"].items()},
            "m_hist": md.get("m_hist", {}),
        }
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[ethucy] manifest -> {output_root / 'manifest.json'}", flush=True)


# ---------------------------------------------------------------------------
# SDD
# ---------------------------------------------------------------------------

def sdd_split_indices(n: int, split_seed: int, val_ratio: float = 0.1):
    """严格复刻 moflow_sdd_dataset.py 的确定性 90/10 划分。

    返回 (train_indices, val_indices)，均为 sdd_train.pkl 列表中的原始索引，
    顺序与 MoFlowSddDataset 中的 scenes 顺序一致。
    """
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_val = int(n * val_ratio)
    return perm[n_val:], perm[:n_val]


def build_sdd_unit(args):
    """处理一个 (condition, split) 单元。在子进程中运行。"""
    condition, split, records, output_root, mask_seed = args
    apply_mask_to_sample._v3_mode = condition in V3_CONDITIONS
    out_dir = Path(output_root) / condition / split
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "samples": 0, "agents": 0, "missing_frames": 0,
        "rejected": 0, "reject_reasons": {},
        "source_index_min": None, "source_index_max": None,
        "sha256": {},
        "m_hist": {},
    }
    for sample_index, (orig_idx, past, fut) in enumerate(records):
        try:
            past = np.asarray(past, dtype=np.float32)
            fut = np.asarray(fut, dtype=np.float32)
            if past.shape != (OBS_LEN, 2) or fut.shape != (PRED_LEN, 2):
                raise ValueError(f"bad shapes {past.shape} {fut.shape}")
            if not (np.isfinite(past).all() and np.isfinite(fut).all()):
                raise ValueError("non-finite coordinates")
        except Exception as e:  # noqa: BLE001
            stats["rejected"] += 1
            stats["reject_reasons"][str(e)[:80]] = stats["reject_reasons"].get(str(e)[:80], 0) + 1
            continue

        positions = torch.from_numpy(np.concatenate([past, fut], axis=0)).unsqueeze(0)  # [1,20,2]
        src_file = "sdd_train.pkl" if split in ("train", "val") else "sdd_test.pkl"

        hmask = derive_history_mask(
            "sdd", None, split, "SDD", int(orig_idx), src_file, 0, condition, mask_seed, 1,
        )
        sample = {
            "positions": positions,
            "valid_mask": torch.ones(1, SEQ_LEN, dtype=torch.bool),
            "agent_ids": torch.zeros(1, dtype=torch.long),
            "focal_id": 0,
            "scene_id": "SDD",
            "fold": None,
            "split": split,
        }
        out = apply_mask_to_sample(sample, hmask)
        out["condition"] = condition
        out["source_file"] = src_file
        out["source_index"] = int(orig_idx)
        out["sample_index"] = sample_index
        out["mask_seed"] = mask_seed
        out["obs_len"] = OBS_LEN
        out["pred_len"] = PRED_LEN
        out["dt"] = DT
        out["unit"] = "pixel"

        out_path = out_dir / f"{sample_index:06d}.pt"
        torch.save(out, out_path)

        stats["samples"] += 1
        stats["agents"] += 1
        stats["missing_frames"] += int((~hmask).sum())
        if condition == "uniform_hard_ng":
            for m, c in zip(*np.unique((~hmask).sum(dim=1).tolist(), return_counts=True)):
                mk = str(int(m)); cv = int(c)
                stats.setdefault("m_hist", {})[mk] = stats["m_hist"].get(mk, 0) + cv
        stats["source_index_min"] = int(orig_idx) if stats["source_index_min"] is None else min(stats["source_index_min"], int(orig_idx))
        stats["source_index_max"] = int(orig_idx) if stats["source_index_max"] is None else max(stats["source_index_max"], int(orig_idx))
        h = hashlib.sha256()
        with open(out_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 16), b""):
                h.update(chunk)
        stats["sha256"][out_path.name] = h.hexdigest()
    return (condition, split), stats


def build_sdd(source_root: Path, output_root: Path, conditions, split_seed, mask_seed, workers, version="missing_history_v1"):
    with open(source_root / "original" / "sdd_train.pkl", "rb") as f:
        train_all = pickle.load(f)
    with open(source_root / "original" / "sdd_test.pkl", "rb") as f:
        test_all = pickle.load(f)
    print(f"[sdd] train.pkl={len(train_all)} test.pkl={len(test_all)}", flush=True)

    train_idx, val_idx = sdd_split_indices(len(train_all), split_seed)
    print(f"[sdd] split: train={len(train_idx)} val={len(val_idx)} (split_seed={split_seed})", flush=True)

    split_records = {
        "train": [(int(i), train_all[i][0], train_all[i][1]) for i in train_idx],
        "val": [(int(i), train_all[i][0], train_all[i][1]) for i in val_idx],
        "test": [(int(i), t[0], t[1]) for i, t in enumerate(test_all)],
    }
    tasks = [(c, s, split_records[s], str(output_root), mask_seed)
             for c in conditions for s in SPLITS]

    manifest_stats = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
        for key, stats in ex.map(build_sdd_unit, tasks, chunksize=1):
            condition, split = key
            md = manifest_stats.setdefault(
                (condition, split),
                {"samples": 0, "agents": 0, "missing_frames": 0, "history_frames": 0,
                 "rejected": 0, "reject_reasons": {},
                 "source_index_min": None, "source_index_max": None, "sha256": {}},
            )
            md["samples"] += stats["samples"]
            md["agents"] += stats["agents"]
            md["missing_frames"] += stats["missing_frames"]
            md["history_frames"] += stats["agents"] * OBS_LEN
            md["rejected"] += stats["rejected"]
            for k, v in stats["reject_reasons"].items():
                md["reject_reasons"][k] = md["reject_reasons"].get(k, 0) + v
            md["source_index_min"] = stats["source_index_min"] if md["source_index_min"] is None else min(md["source_index_min"], stats["source_index_min"])
            md["source_index_max"] = stats["source_index_max"] if md["source_index_max"] is None else max(md["source_index_max"], stats["source_index_max"])
            md["sha256"].update(stats["sha256"])
            if stats.get("m_hist"):
                md["m_hist"] = {k: md.get("m_hist", {}).get(k, 0) + v for k, v in stats["m_hist"].items()}

    manifest = {
        "version": version,
        "dataset": "SDD",
        "obs_len": OBS_LEN,
        "pred_len": PRED_LEN,
        "dt": DT,
        "unit": "pixel",
        "mask_seed": mask_seed,
        "split_seed": split_seed,
        "val_ratio": 0.1,
        "conditions": conditions,
        "condition_specs": {c: CONDITION_SPECS[c] for c in conditions},
        "history_visibility_rule": ("no_guard_min_1_visible" if version == "missing_history_v3_noguard"
                                    else "frames_6_and_7_visible"),
        "future_policy": "complete",
        "coordinate_policy": "raw_scene_coordinates",
        "feature_policy": "recompute_from_visible_history",
        "source_root": str(source_root / "original"),
        "source_version": "sdd_train.pkl / sdd_test.pkl (split_seed=2024)",
        "mask_rng_key_format": "dataset|fold|split|scene_id|source_index|source_file|focal_id|condition|mask_seed -> sha256[:8] -> np.random.default_rng",
        "splits": {},
    }
    for (condition, split), md in sorted(manifest_stats.items()):
        manifest["splits"].setdefault(condition, {})[split] = {
            "samples": md["samples"],
            "agents_total": md["agents"],
            "missing_history_frames": md["missing_frames"],
            "history_frames_total": md["history_frames"],
            "actual_missing_rate": round(md["missing_frames"] / md["history_frames"], 6) if md["history_frames"] else 0.0,
            "source_index_range": [md["source_index_min"], md["source_index_max"]],
            "rejected_samples": md["rejected"],
            "reject_reasons": md["reject_reasons"],
            "per_file_sha256": md["sha256"],
            "m_hist": md.get("m_hist", {}),
        }
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[sdd] manifest -> {output_root / 'manifest.json'} ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build missing-history datasets (v1 / v2_high)")
    ap.add_argument("--dataset", choices=["ethucy", "sdd"], required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--mask-seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=2024)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--folds", nargs="+", default=FOLDS, help="ethucy only: subset of folds")
    ap.add_argument("--scenes", nargs="+", default=None, help="ethucy only: only these scene names (smoke test)")
    ap.add_argument("--version", default="missing_history_v1",
                    help="manifest version 标签：missing_history_v1 | missing_history_v2_high | missing_history_v3_noguard")
    args = ap.parse_args()

    if args.folds != FOLDS:
        # 通过模块级约定传给子进程可见的枚举逻辑：直接改写本模块全局
        globals()["FOLDS"] = args.folds

    if args.version == "missing_history_v3_noguard":
        bad = [c for c in args.conditions if c not in V3_CONDITIONS]
        assert not bad, f"v3_noguard 版本只接受 *_ng 条件，得到: {bad}"
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "ethucy":
        build_ethucy(source_root, output_root, args.conditions, args.mask_seed, args.workers,
                     version=args.version, scenes=args.scenes)
    else:
        build_sdd(source_root, output_root, args.conditions, args.split_seed, args.mask_seed, args.workers, version=args.version)


if __name__ == "__main__":
    main()
