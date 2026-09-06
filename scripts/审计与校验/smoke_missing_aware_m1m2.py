"""M1/M2 真实数据前向 smoke（任务 2/3 接口验证，临时脚本）。

真实 v2/v3 数据：Dataset → collate → ModelForecast(M1+M2) 前向，
验证输出形状/有限性 + complete 条件中性值 + v1/v2 帧 6/7 gap=0。
"""
import sys

sys.path.insert(0, "/home/lbh/DeMo")
import torch

from src.datamodule.ethucy_benchmark_dataset import (
    EthUcyBenchmarkDataset, ethucy_benchmark_collate_fn)
from src.datamodule.sdd_missing_dataset import SddMissingDataset
from src.datamodule.moflow_ethucy_dataset import moflow_ethucy_collate_fn
from src.model.model_forecast import ModelForecast

device = "cuda"
fails = []


def rec(ok, msg):
    if not ok:
        fails.append(msg)
    print(("PASS " if ok else "FAIL ") + msg)


model = ModelForecast(
    embed_dim=128, future_steps=12, num_heads=8, mlp_ratio=4.0,
    drop_path=0.2, num_actor_types=1, num_modes=6, dt=0.4,
    use_observation_features=True, use_missing_summary=True,
).to(device).eval()

CASES = [
    # (名称, 根, 条件, dataset 构造, collate, num_modes)
    ("v2/block6", "data/ETHUCY_missing_v2_high", "random_block6",
     lambda r, c: EthUcyBenchmarkDataset(f"{r}/{c}", "ETH", "test"),
     ethucy_benchmark_collate_fn),
    ("v2/complete", "data/ETHUCY_missing_v1", "complete",
     lambda r, c: EthUcyBenchmarkDataset(f"{r}/{c}", "ETH", "test"),
     ethucy_benchmark_collate_fn),
    ("v3/block6_ng", "data/ETHUCY_missing_v3_noguard", "random_block6_ng",
     lambda r, c: EthUcyBenchmarkDataset(f"{r}/{c}", "ETH", "test"),
     ethucy_benchmark_collate_fn),
    ("v3/uniform_hard_ng", "data/ETHUCY_missing_v3_noguard", "uniform_hard_ng",
     lambda r, c: EthUcyBenchmarkDataset(f"{r}/{c}", "ETH", "test"),
     ethucy_benchmark_collate_fn),
    ("SDDv2/block6", "data/SDD_missing_v2_high", "random_block6",
     lambda r, c: SddMissingDataset(r, c, "test"),
     moflow_ethucy_collate_fn),
    ("SDDv3/uniform_hard_ng", "data/SDD_missing_v3_noguard", "uniform_hard_ng",
     lambda r, c: SddMissingDataset(r, c, "test"),
     moflow_ethucy_collate_fn),
]

for name, root, cond, mk, collate in CASES:
    ds = mk(root, cond)
    items = [ds[i] for i in range(min(32, len(ds)))]
    batch = collate(items)
    mb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    real = mb["x_key_valid_mask"].bool()

    # v1/v2 协议帧 6/7 恒可见 → gap=0（v3 不适用）
    if not name.startswith("v3") and not name.startswith("SDDv3"):
        rec(bool((mb["x_gap_steps"][real][..., 6:8] == 0).all()),
           f"{name}: gap_steps[6:8]==0 for real actors")

    # 新字段有限
    for k in ("x_gap_steps", "x_prev_valid_gap", "x_motion_valid",
              "x_motion_run", "x_missing_summary"):
        rec(bool(torch.isfinite(mb[k]).all()), f"{name}: {k} finite")

    with torch.no_grad():
        out = model(mb)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            rec(bool(torch.isfinite(v).all()), f"{name}: out.{k} finite")
    rec(out["y_hat"].shape == (mb["x_positions"].size(0), 6, 12, 2),
        f"{name}: y_hat shape {tuple(out['y_hat'].shape)}")

# complete 中性值验证（v1 complete）
ds = EthUcyBenchmarkDataset("data/ETHUCY_missing_v1/complete", "ETH", "test")
item = ds[0]
rec(torch.allclose(item["x_missing_summary"][0],
                   torch.tensor([0., 0., 0., 1., 1., 0.])),
    f"complete: focal summary neutral, got {item['x_missing_summary'][0].tolist()}")

# M0 不带新字段仍可跑（旧 batch 兼容）
model_m0 = ModelForecast(embed_dim=128, future_steps=12, num_actor_types=1).to(device).eval()
batch = ethucy_benchmark_collate_fn([ds[0]])
for k in ("x_gap_steps", "x_prev_valid_gap", "x_motion_valid", "x_motion_run", "x_missing_summary"):
    batch.pop(k)
mb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
with torch.no_grad():
    out = model_m0(mb)
rec(torch.isfinite(out["y_hat"]).all(), "M0 forward without new fields (legacy batch)")

print(f"\ntotal fails: {len(fails)}")
sys.exit(1 if fails else 0)
