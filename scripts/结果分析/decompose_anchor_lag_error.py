"""anchor_lag 误差分解：把 test 集误差按「锚点回退步数」分层统计。

背景：v3_noguard 协议下锚点=最后可见帧，anchor_lag = 7 - last_valid_idx。
假设：lag>=1 的行人承担了"不可恢复的锚点几何损失"，是 M0 与 v2 有保护协议
(~0.10 minFDE6) 差距的主要来源。

口径与 trainer_forecast 的 val_metrics 完全一致：
  - minFDE6: per-agent, min over K=6 modes of ||y_hat[...,-1,:2] - target[...,-1,:2]||
  - minADE6: 同上，对 12 帧取 mean
  - MR: 末端误差 > 2m
分层仅统计 x_key_valid_mask=True 的真实 agent（padding agent 的 lag 填充值为 0，
不过滤会把 padding 全部塞进 lag=0 层）。

用法：
  ~/.conda/envs/DeMo/bin/python scripts/结果分析/decompose_anchor_lag_error.py \
      --condition random_fixed4_ng --seed 2024 \
      --variants M0_base M1_obs M2_history --gpu 1 --batch-size 64
输出：
  outputs/missing_aware/analysis/<condition>_seed<seed>/anchor_lag_decomposition.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)  # hydra config_path 必须相对 cwd


def load_trainer(variant, fold, condition, seed):
    """hydra compose 后手动实例化 Trainer，返回已加载 ckpt 的 trainer。"""
    import hydra
    from hydra.utils import instantiate

    arm_root = REPO / "outputs/missing_aware/ethucy/train_adapt" / variant / condition / f"seed_{seed}"
    results = json.load(open(arm_root / "results.json"))
    row = [r for r in results["rows"] if r["fold"] == fold][0]
    ckpt_path = row["checkpoint_path"]
    assert os.path.exists(ckpt_path), ckpt_path

    flags = results.get("model_flags", {})
    overrides = [
        f"fold={fold}",
        f"data_root={REPO}/data/ETHUCY_missing_v3_noguard/{condition}",
        f"seed={seed}",
        "test=true",
        f"model.target.model.use_observation_features={str(flags.get('use_observation_features', False)).lower()}",
        f"model.target.model.use_missing_summary={str(flags.get('use_missing_summary', False)).lower()}",
    ]
    with hydra.initialize(config_path="../../conf", version_base=None, job_name="lag_decomp"):
        cfg = hydra.compose(config_name="config_missing_aware_ethucy", overrides=overrides)

    model_cfg = OmegaConf.to_container(cfg.model.target, resolve=True)["model"]
    from src.model.trainer_forecast import Trainer

    trainer = Trainer(model=dict(model_cfg))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"]
    if any(k.startswith("net.") for k in state):
        state = {k[len("net."):]: v for k, v in state.items()}
    missing, unexpected = trainer.net.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    trainer.net = trainer.net.cuda().eval()
    return trainer


def build_test_loader(fold, condition, seed, batch_size):
    import hydra
    from hydra.utils import instantiate

    with hydra.initialize(config_path="../../conf", version_base=None, job_name="lag_decomp_dm"):
        cfg = hydra.compose(
            config_name="config_missing_aware_ethucy",
            overrides=[
                f"fold={fold}",
                f"data_root={REPO}/data/ETHUCY_missing_v3_noguard/{condition}",
                f"seed={seed}",
                "test=true",
            ],
        )
    dm = instantiate(cfg.datamodule.target)
    dm.setup("test")
    return dm.test_dataloader()


@torch.no_grad()
def run_arm(variant, folds, condition, seed, batch_size, gpu):
    # 设备选择由调用方通过 CUDA_VISIBLE_DEVICES 控制；此处仅兜底
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        torch.cuda.set_device(gpu)
    records = []  # per-agent dicts
    for fold in folds:
        trainer = load_trainer(variant, fold, condition, seed)
        loader = build_test_loader(fold, condition, seed, batch_size)
        for data in loader:
            data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            out = trainer.net(data)
            y_hat = out["new_y_hat"] if out.get("new_y_hat") is not None else out["y_hat"]
            # 评估口径：focal agent only（与 trainer_forecast.test_step 一致，target[:, 0]）
            target = data["target"][:, 0]  # [B, T, 2]
            lag = data["x_anchor_lag_steps"][:, 0].long()  # [B] focal 的锚点回退步数

            err = torch.norm(y_hat[..., :2] - target.unsqueeze(1), dim=-1)  # [B, K, T]
            fde6 = err[..., -1].min(dim=1).values  # [B]
            ade6 = err.mean(dim=-1).min(dim=1).values  # [B]
            miss = (fde6 > 2.0).float()

            for b in range(err.shape[0]):
                records.append({
                    "fold": fold,
                    "anchor_lag": int(lag[b].item()),
                    "minFDE6": float(fde6[b].item()),
                    "minADE6": float(ade6[b].item()),
                    "miss": int(miss[b].item()),
                })
        print(f"  {variant}/{fold}: {len(records)} agents accumulated")
    return records


def stratify(records):
    import statistics

    groups = {0: [], 1: [], 2: [], 3: []}
    for r in records:
        groups[min(r["anchor_lag"], 3)].append(r)
    out = {}
    for lag, recs in sorted(groups.items()):
        if not recs:
            continue
        n = len(recs)
        out[str(lag) if lag < 3 else "3+"] = {
            "n_agents": n,
            "pct": round(100.0 * n / len(records), 1),
            "minFDE6_mean": round(statistics.mean(r["minFDE6"] for r in recs), 4),
            "minFDE6_median": round(statistics.median(r["minFDE6"] for r in recs), 4),
            "minADE6_mean": round(statistics.mean(r["minADE6"] for r in recs), 4),
            "MR": round(statistics.mean(r["miss"] for r in recs), 4),
        }
    # lag=0 vs >=1 汇总
    lag0 = [r for r in records if r["anchor_lag"] == 0]
    lagp = [r for r in records if r["anchor_lag"] >= 1]
    out["_summary"] = {
        "total_agents": len(records),
        "lag0": {
            "n": len(lag0), "pct": round(100.0 * len(lag0) / len(records), 1),
            "minFDE6": round(statistics.mean(r["minFDE6"] for r in lag0), 4),
            "minADE6": round(statistics.mean(r["minADE6"] for r in lag0), 4),
            "MR": round(statistics.mean(r["miss"] for r in lag0), 4),
        },
        "lag>=1": {
            "n": len(lagp), "pct": round(100.0 * len(lagp) / len(records), 1),
            "minFDE6": round(statistics.mean(r["minFDE6"] for r in lagp), 4),
            "minADE6": round(statistics.mean(r["minADE6"] for r in lagp), 4),
            "MR": round(statistics.mean(r["miss"] for r in lagp), 4),
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="random_fixed4_ng")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--variants", nargs="+", default=["M0_base", "M1_obs", "M2_history"])
    ap.add_argument("--folds", nargs="+", default=["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"])
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_result = {"meta": vars(args), "variants": {}}
    for variant in args.variants:
        print(f"== {variant} ==")
        records = run_arm(variant, args.folds, args.condition, args.seed,
                          args.batch_size, args.gpu)
        all_result["variants"][variant] = stratify(records)

    out_path = args.out or str(
        REPO / "outputs/missing_aware/analysis"
        / f"{args.condition}_seed{args.seed}" / "anchor_lag_decomposition.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_result, f, ensure_ascii=False, indent=2)
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
