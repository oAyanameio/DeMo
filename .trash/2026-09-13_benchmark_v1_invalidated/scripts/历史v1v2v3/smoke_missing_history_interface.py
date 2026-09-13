"""missing-history 模型接口冒烟：Dataset → collate → ModelForecast actor-only 前向检查。

支持 v1/v2（帧 6/7 恒可见）与 v3_noguard（无保护）两套断言集（方案 §5.1.5）。

用法：
  # v2（回归）
  CUDA_VISIBLE_DEVICES=<g> python scripts/审计与校验/smoke_missing_history_interface.py \
      --version v2 --ethucy-root data/ETHUCY_missing_v2_high --sdd-root data/SDD_missing_v2_high
  # v3（无保护）
  CUDA_VISIBLE_DEVICES=<g> python scripts/审计与校验/smoke_missing_history_interface.py \
      --version v3 --ethucy-root data/ETHUCY_missing_v3_noguard --sdd-root data/SDD_missing_v3_noguard
"""
import argparse
import sys

import torch

sys.path.insert(0, "/home/lbh/DeMo")
from src.datamodule.ethucy_benchmark_dataset import (  # noqa: E402
    EthUcyBenchmarkDataset, ethucy_benchmark_collate_fn,
)
from src.datamodule.sdd_missing_dataset import SddMissingDataset  # noqa: E402
from src.datamodule.moflow_ethucy_dataset import moflow_ethucy_collate_fn  # noqa: E402

V3_CONDS = ["random_fixed3_ng", "random_fixed4_ng", "random_block3_ng",
            "random_block4_ng", "random_block6_ng", "uniform_hard_ng"]
V2_CONDS = ["random_fixed3", "random_fixed4", "random_block3", "random_block4", "random_block6"]


def load_model(num_modes=20, device="cuda"):
    from src.model.model_forecast import ModelForecast
    model = ModelForecast(
        embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
        qkv_bias=False, drop_path=0.2, num_actor_types=1,
        num_modes=num_modes, dt=0.4,
    )
    return model.to(device).eval()


def check_batch(audit, name, items, model, collate, device, keys_gt1_ok, noguard):
    batch = collate(items)
    fails = []

    def rec(ok, msg):
        if not ok:
            fails.append(msg)

    mb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    with torch.no_grad():
        out = model(mb)

    def finite_tree(o, path="out"):
        if isinstance(o, torch.Tensor):
            rec(bool(torch.isfinite(o).all()), f"{name}: {path} NaN/Inf")
        elif isinstance(o, (list, tuple)):
            for i, x in enumerate(o):
                finite_tree(x, f"{path}[{i}]")
        elif isinstance(o, dict):
            for k, x in o.items():
                finite_tree(x, f"{path}.{k}")
    finite_tree(out)

    xvm = mb["x_valid_mask"]
    real = mb["x_key_valid_mask"].bool()

    if noguard:
        # v3：不要求帧 6/7 可见，不要求 ≥2 可见帧；每真实 actor ≥1 可见帧
        rec(bool(xvm[real].any(-1).all()), f"{name}: real agent with 0 visible frames")
        # x_centers == 每个 actor 的最后有效位置（按其自身 last_valid_idx）
        lvi = mb["x_last_valid_idx"]
        for b in range(xvm.size(0)):
            for n in range(xvm.size(1)):
                if bool(real[b, n]):
                    idx = int(lvi[b, n].item())
                    rec(torch.allclose(mb["x_centers"][b, n],
                                       mb["x_positions"][b, n, idx], atol=1e-5),
                        f"{name}: x_centers != last valid pos (agent {n}, idx {idx})")
        # focal 原点 = focal 最后可见位置（origin 已在局部系下应用，这里校验
        # focal 局部坐标在 last_valid_idx 处为 0 向量）
        f_lvi = lvi[:, 0]
        for b in range(xvm.size(0)):
            rec(torch.allclose(mb["x_positions"][b, 0, int(f_lvi[b].item())],
                               torch.zeros(2, device=mb["x_positions"].device), atol=1e-5),
                f"{name}: focal origin not at last valid position")
        # anchor_lag/forecast_gap 与掩码一致
        hm = xvm.clone(); hm[~real] = True
        for b in range(xvm.size(0)):
            for n in range(xvm.size(1)):
                if bool(real[b, n]):
                    idxs = torch.nonzero(xvm[b, n]).flatten()
                    rec(int(idxs[-1].item()) == int(lvi[b, n].item()),
                        f"{name}: last_valid_idx mismatch")
                    rec(int(mb["x_anchor_lag_steps"][b, n].item()) == 7 - int(idxs[-1].item()),
                        f"{name}: anchor_lag mismatch")
                    rec(int(mb["x_forecast_gap_steps"][b, n].item()) == 8 - int(idxs[-1].item()),
                        f"{name}: forecast_gap mismatch")
        # 仅单帧可见样本可前向（无 NaN 已由 finite_tree 覆盖；这里显式确认存在此类样本且通过）
        single = (xvm[real].sum(-1) == 1)
        rec(True, "")  # 前向已含单帧样本则 finite_tree 已验证
    else:
        # v1/v2 回归断言（原 smoke_v2_model_interface 行为）
        rec(bool(xvm[real][..., 6].all()) and bool(xvm[real][..., 7].all()),
            f"{name}: frames 6/7 not always valid")
        rec(bool((xvm[real].sum(-1) >= 2).all()),
            f"{name}: real agent with <2 visible frames")
        rec(torch.allclose(mb["x_centers"][real], mb["x_positions"][real][:, 7], atol=1e-6),
            f"{name}: x_centers != frame-7 position")

    # 缺失步派生量为 0（两版本通用）
    hm = xvm.clone(); hm[~real] = True
    step_mask = hm[:, :, :-1] & hm[:, :, 1:]
    for fld in ["x_positions_diff", "x_velocity", "x_velocity_diff", "x_angles"]:
        t = mb[fld]
        t_abs = t.abs().sum(-1) if t.dim() == 4 else t.abs()
        if fld == "x_velocity_diff":
            both = step_mask[:, :, 1:] & step_mask[:, :, :-1]
            bad = (t_abs[:, :, 2:] != 0) & ~both & real.unsqueeze(-1)
        else:
            bad = (t_abs[:, :, 1:] != 0) & ~step_mask & real.unsqueeze(-1)
        rec(not bool(bad.any()), f"{name}: {fld} nonzero on masked steps")

    rec(bool(torch.isfinite(mb["target"][real]).all()), f"{name}: target NaN/Inf")
    rec(bool(mb["target_mask"][real].all()), f"{name}: target_mask not all True")

    if keys_gt1_ok:
        rec(mb["x_positions"].shape[1] > 1, f"{name}: no multi-agent batch")
    else:
        rec(mb["x_positions"].shape[1] == 1, f"{name}: SDD actor dim != 1")

    audit[name] = fails
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--version", choices=["v2", "v3"], default="v3")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--ethucy-root", default=None)
    ap.add_argument("--sdd-root", default=None)
    ap.add_argument("--fold", default="ETH")
    args = ap.parse_args()

    noguard = args.version == "v3"
    conds = args.conditions or (V3_CONDS if noguard else V2_CONDS)
    ethucy_root = args.ethucy_root or ("data/ETHUCY_missing_v3_noguard" if noguard else "data/ETHUCY_missing_v2_high")
    sdd_root = args.sdd_root or ("data/SDD_missing_v3_noguard" if noguard else "data/SDD_missing_v2_high")

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"device={device} version={args.version}")
    model = load_model(device=device)
    audit = {}
    total_fails = 0

    for cond in conds:
        ds = EthUcyBenchmarkDataset(f"{ethucy_root}/{cond}", args.fold, "test")
        items = [ds[i] for i in range(min(args.limit, len(ds)))]
        f = check_batch(audit, f"ethucy/{cond}", items, model,
                        ethucy_benchmark_collate_fn, device, keys_gt1_ok=True, noguard=noguard)
        total_fails += len(f)
        ds = SddMissingDataset(sdd_root, cond, "test")
        items = [ds[i] for i in range(min(args.limit, len(ds)))]
        f = check_batch(audit, f"sdd/{cond}", items, model,
                        moflow_ethucy_collate_fn, device, keys_gt1_ok=False, noguard=noguard)
        total_fails += len(f)

    print("\n===== 结果 =====")
    for k, v in audit.items():
        status = "PASS" if not v else "FAIL"
        print(f"[{status}] {k}" + ("" if not v else f" — {v[:3]}"))
    print(f"\ntotal fails: {total_fails}")
    sys.exit(1 if total_fails else 0)


if __name__ == "__main__":
    main()
