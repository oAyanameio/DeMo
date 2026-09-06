"""Trainer 端到端一步训练 smoke + hydra 默认配置组合检查（临时脚本）。"""
import sys

sys.path.insert(0, "/home/lbh/DeMo")

import torch

# (a) Trainer 一步训练（M1+M2）
from src.datamodule.ethucy_benchmark_dataset import (
    EthUcyBenchmarkDataset, ethucy_benchmark_collate_fn)
from src.model.trainer_forecast import Trainer

device = "cuda"
model_cfg = {
    "type": "ModelForecast", "embed_dim": 128, "future_steps": 12,
    "num_heads": 8, "mlp_ratio": 4.0, "qkv_bias": False, "drop_path": 0.2,
    "num_actor_types": 1, "num_modes": 6, "dt": 0.4,
    "use_observation_features": True, "use_missing_summary": True,
}
trainer = Trainer(model=dict(model_cfg), lr=1e-3).to(device)

ds = EthUcyBenchmarkDataset("data/ETHUCY_missing_v2_high/random_block6", "ETH", "train")
items = [ds[i] for i in range(4)]
batch = ethucy_benchmark_collate_fn(items)
mb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

out = trainer(mb)          # forward -> ret_dict
loss, loss_dict = trainer.cal_loss(out, mb)
loss.backward()
grads_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.parameters())
print(f"(a) M1+M2 trainer step: total={float(loss):.4f} grads finite: {grads_finite}")
print(f"    losses={ {k: round(v, 4) for k, v in loss_dict.items()} }")
assert grads_finite and torch.isfinite(loss)

# (b) M0 一步训练（新字段在 batch 中存在但被忽略）
model_cfg_m0 = {k: v for k, v in model_cfg.items()
                if k not in ("use_observation_features", "use_missing_summary")}
trainer_m0 = Trainer(model=dict(model_cfg_m0), lr=1e-3).to(device)
out0 = trainer_m0(mb)
loss0, _ = trainer_m0.cal_loss(out0, mb)
loss0.backward()
print(f"(b) M0 trainer step (new fields ignored): total={float(loss0):.4f}")
assert torch.isfinite(loss0)

# (c) hydra 默认 config 组合
import hydra
with hydra.initialize(config_path="conf", version_base=None):
    cfg = hydra.compose(config_name="config")
dm = hydra.utils.instantiate(cfg.datamodule.target)
trainer_h = hydra.utils.instantiate(cfg.model.target)
n_params = sum(p.numel() for p in trainer_h.net.parameters())
has_new = any("missing_summary" in k for k in trainer_h.net.state_dict())
mlp_in = trainer_h.net.hist_embed_mlp[0].in_features
print(f"(c) hydra compose ok: datamodule={type(dm).__name__}, net params={n_params}, "
      f"hist_mlp in={mlp_in}, missing_summary present: {has_new}")
assert not has_new and mlp_in == 4, "default config must stay M0 (in=4, no new modules)"
print("ALL OK")
