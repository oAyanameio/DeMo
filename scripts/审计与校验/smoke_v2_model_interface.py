"""v2 模型接口冒烟：Dataset → collate → ModelForecast(use_map=False) 前向检查（方案 §7.2）。

对每个新条件（缺省 random_fixed3/4, random_block3/4/6）：
  ETH/UCY：EthUcyBenchmarkDataset(data/ETHUCY_missing_v2_high/<cond>, fold=ETH, split=test)
  SDD：    SddMissingDataset(data/SDD_missing_v2_high, condition, split=test)

检查项：
  1. collate 后字段形状 / dtype；
  2. 前向输出无 NaN / Inf；
  3. focal 帧 6/7 有效（x_valid_mask[..., 6/7] 全 True）；
  4. 缺失步的 diff/velocity/velocity_diff/x_angles 为 0；
  5. x_centers == 最后一个有效历史位置（帧 7）；
  6. ETH/UCY 邻居 actor token 保留（x_positions 第二维 > 1 的 batch 存在）；
     SDD actor 维度恒为 1；
  7. target / target_mask 未被掩码污染（target 全有限、target_mask 全 True）；
  8. block6 下 focal 至少 2 个有效历史位置。

用法：
  CUDA_VISIBLE_DEVICES=<gpu> python scripts/审计与校验/smoke_v2_model_interface.py [--gpu] [--limit N]
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


def load_model(num_modes=20, device="cuda"):
    """与 conf/model/ethucy_benchmark_model_forecast.yaml 相同的构造参数。"""
    from src.model.model_forecast import ModelForecast
    model = ModelForecast(
        embed_dim=128,
        future_steps=12,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        use_map=False,
        num_actor_types=1,
        num_modes=num_modes,
        dt=0.4,
    )
    return model.to(device).eval()


def check_batch_common(audit, name, items, model, collate, device, keys_gt1_ok):
    batch = collate(items)
    fails = []

    def rec(ok, msg):
        if not ok:
            fails.append(msg)

    # 移动到 device（只取模型需要的字段）
    mb = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            mb[k] = v.to(device)
        else:
            mb[k] = v

    with torch.no_grad():
        out = model(mb)

    # 输出有限性
    def finite_tree(o, path="out"):
        if isinstance(o, torch.Tensor):
            rec(bool(torch.isfinite(o).all()), f"{name}: {path} has NaN/Inf")
        elif isinstance(o, (list, tuple)):
            for i, x in enumerate(o):
                finite_tree(x, f"{path}[{i}]")
        elif isinstance(o, dict):
            for k, x in o.items():
                finite_tree(x, f"{path}.{k}")
    finite_tree(out)

    xvm = mb["x_valid_mask"]      # [B, N, 8]
    # collate 对 agent 维 padding（valid/mask 填 False）：所有断言只作用于真实 agent
    real = mb["x_key_valid_mask"].bool()  # [B, N]；padding 槽位为 False（focal 恒 True）
    rec(bool(xvm[real][..., 6].all()) and bool(xvm[real][..., 7].all()),
        f"{name}: frames 6/7 not always valid on real agents")
    rec(bool((xvm[real].sum(-1) >= 2).all()),
        f"{name}: real agent with <2 visible history frames")

    # 缺失步派生量必须为 0（对 t=1..7 的步，涉及缺失帧的步；仅真实 agent）
    hm = xvm.clone()
    hm[~real] = True  # padding 行不参与（其派生量为 0，合法）
    step_mask = hm[:, :, :-1] & hm[:, :, 1:]  # [B,N,7] step t 合法性（t=1..7）
    for fld in ["x_positions_diff", "x_velocity", "x_velocity_diff", "x_angles"]:
        t = mb[fld]
        # 聚合掉坐标维：[B,N,8,2] -> [B,N,8]
        t_abs = t.abs().sum(-1) if t.dim() == 4 else t.abs()
        if fld == "x_velocity_diff":
            # t=2..7 依赖两步位移
            both = step_mask[:, :, 1:] & step_mask[:, :, :-1]
            bad = (t_abs[:, :, 2:] != 0) & ~both & real.unsqueeze(-1)
            rec(not bool(bad.any()), f"{name}: {fld} nonzero on masked steps")
        else:
            bad = (t_abs[:, :, 1:] != 0) & ~step_mask & real.unsqueeze(-1)
            rec(not bool(bad.any()), f"{name}: {fld} nonzero on masked steps")

    # x_centers == 帧 7（最后一个有效历史位置；仅真实 agent）
    rec(torch.allclose(mb["x_centers"][real], mb["x_positions"][real][:, 7], atol=1e-6),
        f"{name}: x_centers != last visible position")

    # target 未被污染（仅真实 agent；padding 槽位 target_mask=False 是设计行为）
    rec(bool(torch.isfinite(mb["target"][real]).all()), f"{name}: target has NaN/Inf")
    rec(bool(mb["target_mask"][real].all()), f"{name}: target_mask not all True on real agents")

    # actor 维度
    if keys_gt1_ok:
        rec(mb["x_positions"].shape[1] > 1, f"{name}: no multi-agent batch (N=={mb['x_positions'].shape[1]})")
    else:
        rec(mb["x_positions"].shape[1] == 1, f"{name}: SDD actor dim != 1")

    audit[name] = fails
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="用 CUDA（缺省 CPU）")
    ap.add_argument("--limit", type=int, default=64, help="每条件取多少样本进 batch")
    ap.add_argument("--conditions", nargs="+", default=[
        "random_fixed3", "random_fixed4", "random_block3", "random_block4", "random_block6"])
    ap.add_argument("--ethucy-root", default="data/ETHUCY_missing_v2_high")
    ap.add_argument("--sdd-root", default="data/SDD_missing_v2_high")
    ap.add_argument("--fold", default="ETH")
    args = ap.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = load_model(device=device)
    audit = {}
    total_fails = 0

    for cond in args.conditions:
        # ETH/UCY
        ds = EthUcyBenchmarkDataset(f"{args.ethucy_root}/{cond}", args.fold, "test")
        items = [ds[i] for i in range(min(args.limit, len(ds)))]
        f = check_batch_common(audit, f"ethucy/{cond}", items, model, ethucy_benchmark_collate_fn, device, keys_gt1_ok=True)
        total_fails += len(f)
        # SDD
        ds = SddMissingDataset(args.sdd_root, cond, "test")
        items = [ds[i] for i in range(min(args.limit, len(ds)))]
        f = check_batch_common(audit, f"sdd/{cond}", items, model, moflow_ethucy_collate_fn, device, keys_gt1_ok=False)
        total_fails += len(f)

    print("\n===== 结果 =====")
    for k, v in audit.items():
        status = "PASS" if not v else "FAIL"
        print(f"[{status}] {k}" + ("" if not v else f" — {v}"))
    print(f"\ntotal fails: {total_fails}")
    sys.exit(1 if total_fails else 0)


if __name__ == "__main__":
    main()
