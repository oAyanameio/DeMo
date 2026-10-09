"""TrajGap-Bench Mixed 直接缺失预测评估入口。

正式协议固定为 `--difficulty Mixed`：
  PYTHONPATH=. python scripts/结果分析/evaluate_trajimpute_direct.py \
      --checkpoint outputs/.../checkpoints/xxx.ckpt \
      --scene ETH-M --difficulty Mixed --split test --K 20 --variant M0

输出：outputs/trajgap_bench/<scene>_Mixed_<split>_<variant>_seed<seed>/results.json
层级：scene/split/missing_count/variant/seed；K 显式写入 meta；
外部参考数字（TrajImpute 原论文）只能以 external_reference 单独存放，不混入。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.datamodule.trajimpute_dataset import (  # noqa: E402
    MIXED_DIFFICULTY, TrajGapDataset, trajimpute_collate_fn,
    inspect_clean_source, classify_evidence_bin,
)
from src.evaluation.trajimpute_direct import DirectEvaluator, save_results  # noqa: E402

VARIANTS = {
    "M0": {},
    # M1：自然缺失证据条件化输出校准（方案 §6）。校准头进 checkpoint，
    # 评估时必须以相同开关重建模型，否则校准权重被 strict=False 静默丢弃。
    "M1": {"calibration": True},
}


def get_git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def build_model(variant: str, num_modes: int, bimamba: bool = False):
    from src.model.trajectory_forecaster import TrajectoryForecaster
    return TrajectoryForecaster(
        embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
        qkv_bias=False, drop_path=0.2, num_actor_types=1,
        num_modes=num_modes, bimamba=bimamba, dt=0.4, obs_len=8,
        **VARIANTS[variant],
    )


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_config = ckpt.get("hyper_parameters", {}).get("model", {})
    checkpoint_modes = model_config.get("num_modes") if isinstance(model_config, dict) else None
    expected_modes = model.time_decoder.num_modes
    if checkpoint_modes is not None and int(checkpoint_modes) != expected_modes:
        raise ValueError(
            f"checkpoint num_modes={checkpoint_modes} 与评估 K={expected_modes} 不一致"
        )
    state = ckpt.get("state_dict", ckpt)
    cleaned = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
    # 强校验：calibration 开关与 checkpoint 参数必须匹配，防止 M0/M1 静默错配
    cal_keys = {k for k in cleaned if k.startswith("calibration_head.")}
    expects_cal = hasattr(model, "calibration_head")
    if expects_cal and not cal_keys:
        raise ValueError(
            f"variant 要求 calibration=true，但 checkpoint 无 calibration_head 参数: "
            f"{ckpt_path}（这是 M0 checkpoint，不能标成 M1 评估）"
        )
    if not expects_cal and cal_keys:
        raise ValueError(
            f"variant=M0 要求原始模型，但 checkpoint 含 calibration_head 参数（{len(cal_keys)} 个）: "
            f"{ckpt_path}（M1 checkpoint 必须用 --variant M1 评估）"
        )
    if expects_cal:
        # 完整性：部分加载会让缺失参数留在零初始化，静默退化为 raw 语义。
        # cleaned 键已剥 net. 前缀（如 calibration_head.tau_branch.0.weight），
        # state_dict 键相对模块（tau_branch.0.weight），比较前补齐前缀。
        expected_keys = {
            f"calibration_head.{k}"
            for k in model.calibration_head.state_dict().keys()
        }
        if cal_keys != expected_keys:
            raise ValueError(
                f"checkpoint 校准参数不完整: 缺失 {sorted(expected_keys - cal_keys)}，"
                f"多余 {sorted(cal_keys - expected_keys)}（{ckpt_path}）"
            )
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


@torch.no_grad()
def run_evaluation(model, dataset, K, scene, difficulty, split, variant, seed,
                   device="cuda:0", batch_size=64, num_workers=2, max_batches=None):
    """返回 (results_cal, results_raw)；非 M1 校准模型两者相同。"""
    from torch.utils.data import DataLoader
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=trajimpute_collate_fn)
    evaluator = DirectEvaluator()
    evaluator_raw = DirectEvaluator()
    n_batches = 0
    for batch in loader:
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        out = model(batch_dev)
        # 与 trainer.test_step 一致：最终输出 new_y_hat/new_pi 优先
        pred = out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]
        prob_raw = out["new_pi"] if out.get("new_pi") is not None else out["pi"]
        prob_cal = out.get("new_pi_cal", prob_raw)
        pred = pred[..., :2]
        if pred.shape[1] != K:
            raise RuntimeError(
                f"模型实际输出 K={pred.shape[1]}，但评估协议要求 K={K}"
            )
        target = batch_dev["target"][:, 0]  # focal [B, T, 2]
        prob_raw = prob_raw.float()
        prob_cal = prob_cal.float()
        focal_valid = batch["x_valid_mask"][:, 0]
        valid_count = focal_valid.sum(-1)
        anchor_lag = batch["x_anchor_lag_steps"][:, 0]
        forecast_gap = batch["x_forecast_gap_steps"][:, 0]
        # M1 机制诊断维度（方案 §6.5）：evidence bin / max_missing_run /
        # mode entropy / calibrated scale —— 只由历史 mask 与模型输出计算
        if "max_missing_run" in batch:
            max_run = batch["max_missing_run"]
        else:
            from src.model.layers.missingness_calibration import max_missing_run_batch
            max_run = max_missing_run_batch(focal_valid)
        has_cal = out.get("new_pi_cal") is not None
        ent_raw = -(torch.softmax(prob_raw, dim=-1)
                    * torch.log_softmax(prob_raw, dim=-1)).sum(-1)  # [B]
        ent_cal = (-(torch.softmax(prob_cal, dim=-1)
                     * torch.log_softmax(prob_cal, dim=-1)).sum(-1)
                   if has_cal else ent_raw)
        scale_cal_mean = (out["scal_cal"].float().mean(dim=(1, 2, 3))
                          if has_cal else None)
        scale_raw_mean = (out["scal_new"].float().mean(dim=(1, 2, 3))
                          if out.get("scal_new") is not None else None)
        # 逐样本分组（batch 内证据条件可能不同）
        for i in range(pred.shape[0]):
            common = dict(
                scene=scene, difficulty=difficulty, split=split,
                missing_count=int(batch["missing_count"][i]),
                valid_count=int(valid_count[i]),
                anchor_lag=int(anchor_lag[i]),
                forecast_gap=int(forecast_gap[i]),
                max_missing_run=int(max_run[i]),
                evidence_bin=classify_evidence_bin(focal_valid[i].cpu()),
            )
            evaluator.update(
                pred[i:i + 1].float().cpu(), prob_cal[i:i + 1].cpu(),
                target[i:i + 1].cpu(),
                mode_entropy=float(ent_cal[i]),
                scale_mean=(float(scale_cal_mean[i])
                            if scale_cal_mean is not None else None),
                **common,
            )
            # raw 概率的 entropy / raw scale 作为对照诊断
            evaluator_raw.update(
                pred[i:i + 1].float().cpu(), prob_raw[i:i + 1].cpu(),
                target[i:i + 1].cpu(),
                mode_entropy=float(ent_raw[i]),
                scale_mean=(float(scale_raw_mean[i])
                            if scale_raw_mean is not None else None),
                **common,
            )
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return evaluator.compute(), evaluator_raw.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--difficulty", default="Mixed", choices=["Mixed"],
                    help="TrajGap-Bench 正式主协议固定为 Mixed")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--variant", default="M0", choices=list(VARIANTS))
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--untrained", action="store_true",
                    help="未训练模型冒烟（结果无意义，仅验证管线）")
    ap.add_argument("--zero-missing-only", action="store_true",
                    help="已废弃：TrajGap Mixed 主协议不允许子集筛选")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--output-root", default="outputs/trajgap_bench")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bimamba", action=argparse.BooleanOptionalAction, default=False,
                    help="主链固定单向(2026-09-12裁定)；须与训练时一致")
    args = ap.parse_args()

    if args.K != 20:
        raise SystemExit("TrajGap-Bench Mixed direct 正式评估固定 K=20")

    torch.manual_seed(args.seed)
    if args.difficulty != MIXED_DIFFICULTY:
        raise SystemExit("TrajGap-Bench 正式评估只允许 Mixed")
    if args.zero_missing_only:
        raise SystemExit("Mixed 协议不支持 zero_missing_only 诊断筛选")
    dataset = TrajGapDataset(args.data_root, args.scene, args.split)
    # Mixed 主协议不做子集过滤；保留总行数用于完整性审计。
    n_total_rows = int(dataset.missing_counts.shape[0])
    n_kept_focal = len(dataset)
    n_filtered_out = n_total_rows - n_kept_focal if args.zero_missing_only else 0
    model = build_model(
        args.variant,
        args.K,
        bimamba=args.bimamba,
    )

    ckpt_info = None
    if args.checkpoint:
        missing, unexpected = load_checkpoint(model, args.checkpoint)
        ckpt_info = {
            "path": str(args.checkpoint),
            "missing_keys": [k for k in missing if "gap_embed" not in k],
            "unexpected_keys": list(unexpected)[:20],
            "calibration_enabled": bool(VARIANTS[args.variant].get("calibration")),
            "calibration_params_loaded": (
                not any(k.startswith("calibration_head.") for k in missing)
                if VARIANTS[args.variant].get("calibration") else None
            ),
        }
    elif not args.untrained:
        raise SystemExit("需要 --checkpoint 或 --untrained（冒烟）之一")

    results, results_raw = run_evaluation(
        model, dataset, args.K, args.scene, args.difficulty, args.split,
        args.variant, args.seed, device=args.device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_batches=args.max_batches,
    )

    meta = {
        "data_root": args.data_root,
        "scene": args.scene,
        "difficulty": args.difficulty,
        "split": args.split,
        "variant": args.variant,
        "seed": args.seed,
        "K": args.K,
        "evaluator": "src/evaluation/trajimpute_direct.py (TrajGap Mixed direct, no imputation)",
        "git_revision": get_git_revision(),
        "untrained_smoke": bool(args.untrained),
        "checkpoint": ckpt_info,
        "metric_semantics": {
            "minADE_K": f"min over K={args.K} predicted modes, same set for minFDE_K",
            "ADE@1": "top-1 by predicted probability (highest-pi mode)",
            "FDE@1": "top-1 by predicted probability (highest-pi mode)",
            "MR": f"miss if ALL K modes' final displacement > {2.0}",
        },
        "external_reference": None,
        "release_subset": inspect_clean_source(args.data_root),
        "protocol": "mixed-direct",
        "zero_missing_only": bool(args.zero_missing_only),
        "n_samples_kept": n_kept_focal,
        "n_samples_total_rows": n_total_rows,
        "n_filtered_out": n_filtered_out,
    }
    tag = "untrained" if args.untrained else "ckpt"
    out_dir = Path(args.output_root) / (
        f"{args.scene}_{args.difficulty}_{args.split}_{args.variant}_seed{args.seed}_{tag}")
    if out_dir.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖: {out_dir}")
    # M1：raw 与 calibrated 双套结果并存（方案 §6.5/§9.2）；
    # calibrated 写主 results.json，raw 单独落盘，禁止只保留校准结果。
    payload = {
        "meta": meta,
        "results": results,
    }
    if args.variant in VARIANTS and VARIANTS[args.variant].get("calibration"):
        payload["results_raw"] = results_raw
    out_path = save_results(results, meta, out_dir / "results.json")
    if "results_raw" in payload:
        save_results(results_raw, meta, out_dir / "results_raw.json")
    print(f"\n[results] {out_path}")
    ov = results["overall"]
    print(f"overall n={ov['n']} minADE{args.K}={ov['minADE_K']:.4f} "
          f"minFDE{args.K}={ov['minFDE_K']:.4f} MR={ov['MR']:.4f} "
          f"ADE@1={ov['ADE@1']:.4f} FDE@1={ov['FDE@1']:.4f}")
    print("[groups]")
    for g, entry in sorted(results["by_group"].items()):
        print(f"  {g}: n={entry['n']} minFDE_K={entry['minFDE_K']:.4f}")


if __name__ == "__main__":
    main()
