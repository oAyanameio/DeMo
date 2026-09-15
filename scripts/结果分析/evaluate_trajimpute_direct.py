"""TrajImpute 直接缺失预测评估脚本。

用法（两类）：
  1) checkpoint 评估：
     PYTHONPATH=. python scripts/结果分析/evaluate_trajimpute_direct.py \
         --checkpoint outputs/.../checkpoints/xxx.ckpt \
         --scene ETH-M --difficulty Easy --split test --K 20 --variant M0_base
  2) 未训练模型冒烟（验证数据管线与评估器本身，结果无意义）：
     PYTHONPATH=. python scripts/结果分析/evaluate_trajimpute_direct.py \
         --scene ETH-M --difficulty Easy --split test --K 20 --variant M0_base --untrained

输出：outputs/trajimpute_direct/<scene>_<difficulty>_<split>_<variant>_seed<seed>/results.json
层级：scene/difficulty/split/missing_count/variant/seed；K 显式写入 meta；
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
    TrajImputeDataset, trajimpute_collate_fn, inspect_clean_source,
)
from src.evaluation.trajimpute_direct import DirectEvaluator, save_results  # noqa: E402

VARIANTS = {
    "M0_base": {"use_observation_features": False, "use_missing_summary": False,
                "use_motion_features": False},
    "M0-current": {"use_observation_features": False, "use_missing_summary": False,
                   "use_motion_features": False},
    "B0": {"use_observation_features": False, "use_missing_summary": False,
           "use_motion_features": False},
    "B1": {"use_observation_features": False, "use_missing_summary": False,
           "use_motion_features": True},
    "M1-evidence": {"use_observation_features": False, "use_missing_summary": False,
                    "use_motion_features": True, "use_evidence_clock": True},
    "M2-social": {"use_observation_features": False, "use_missing_summary": False,
                  "use_motion_features": False, "use_evidence_clock": False,
                  "use_social_evidence": True},
    "M1_obs": {"use_observation_features": True, "use_missing_summary": False,
               "use_motion_features": False},
    "M2_history": {"use_observation_features": True, "use_missing_summary": True,
                   "use_motion_features": False},
}


def get_git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def build_model(variant: str, num_modes: int, bimamba: bool = True):
    from src.model.model_forecast import ModelForecast
    switches = VARIANTS[variant]
    return ModelForecast(
        embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
        qkv_bias=False, drop_path=0.2, num_actor_types=1,
        num_modes=num_modes, bimamba=bimamba, dt=0.4, obs_len=8,
        **switches,
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
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    return missing, unexpected


@torch.no_grad()
def run_evaluation(model, dataset, K, scene, difficulty, split, variant, seed,
                   device="cuda:0", batch_size=64, num_workers=2, max_batches=None):
    from torch.utils.data import DataLoader
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=trajimpute_collate_fn)
    evaluator = DirectEvaluator()
    n_batches = 0
    for batch in loader:
        batch_dev = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        out = model(batch_dev)
        # 选点/测试口径统一（方案 A，2026-09-15）：训练选点用 val_minFDE20
        # （基础 y_hat 分支），测试同用基础 y_hat/pi，禁止 new_y_hat 混用
        pred = out["y_hat"]
        prob = out["pi"]
        pred = pred[..., :2]
        if pred.shape[1] != K:
            raise RuntimeError(
                f"模型实际输出 K={pred.shape[1]}，但评估协议要求 K={K}"
            )
        target = batch_dev["target"][:, 0]  # focal [B, T, 2]
        prob = prob.float()
        # 逐样本分组（batch 内 missing_count 可能不同）
        for i in range(pred.shape[0]):
            evaluator.update(
                pred[i:i + 1].float().cpu(), prob[i:i + 1].cpu(),
                target[i:i + 1].cpu(),
                scene=scene, difficulty=difficulty, split=split,
                missing_count=int(batch["missing_count"][i]),
            )
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return evaluator.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/home/lbh/TrajImpute/dataset/TrajImpute")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--difficulty", default="Easy", choices=["Easy", "Hard"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--variant", default="M0-current", choices=list(VARIANTS))
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--untrained", action="store_true",
                    help="未训练模型冒烟（结果无意义，仅验证管线）")
    ap.add_argument("--zero-missing-only", action="store_true",
                    help="诊断用途：只评估 Easy 中 focal missing_count==0 的样本；不是 Clean-direct")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--output-root", default="outputs/trajimpute_retrain")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bimamba", action=argparse.BooleanOptionalAction, default=False,
                    help="主链固定单向(2026-09-12裁定)；须与训练时一致")
    args = ap.parse_args()

    if args.K != 20:
        raise SystemExit("TrajImpute direct 正式评估固定 K=20")

    torch.manual_seed(args.seed)
    dataset = TrajImputeDataset(
        args.data_root, args.scene, args.difficulty, args.split,
        zero_missing_only=args.zero_missing_only,
    )
    # 被 zero_missing_only 过滤掉的样本数（Clean-direct 边界记录）
    n_total_rows = int(dataset.missing_counts.shape[0])
    n_kept_focal = len(dataset)
    n_filtered_out = n_total_rows - n_kept_focal if args.zero_missing_only else 0
    model = build_model(args.variant, args.K, bimamba=args.bimamba)

    ckpt_info = None
    if args.checkpoint:
        missing, unexpected = load_checkpoint(model, args.checkpoint)
        ckpt_info = {
            "path": str(args.checkpoint),
            "missing_keys": [k for k in missing if "gap_embed" not in k],
            "unexpected_keys": list(unexpected)[:20],
        }
    elif not args.untrained:
        raise SystemExit("需要 --checkpoint 或 --untrained（冒烟）之一")

    results = run_evaluation(
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
        "evaluator": "src/evaluation/trajimpute_direct.py (direct, no imputation)",
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
        "release_subset": (
            {
                "type": "easy_zero_missing_diagnostic",
                "test": {"source": f"Easy/data_test.pkl",
                         "filter": "focal missing_count == 0"},
                "is_clean_direct": False,
            } if args.zero_missing_only else inspect_clean_source(args.data_root)
        ),
        "protocol": (
            "easy-zero-missing-diagnostic"
            if args.zero_missing_only else f"{args.difficulty.lower()}-direct"
        ),
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
    out_path = save_results(results, meta, out_dir / "results.json")
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
