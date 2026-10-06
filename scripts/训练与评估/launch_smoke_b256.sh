#!/usr/bin/env bash
# batch=256 / 5 epoch throughput smoke (GPU0, 独立输出根, 不触碰正式结果)
set -euo pipefail
cd /home/lbh/DeMo
mkdir -p outputs/smoke_b256

setsid env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 \
  /home/lbh/.conda/envs/DeMo/bin/python -u train.py \
  --config-name=config_missing_aware_trajimpute \
  scene=ETH-M difficulty=Mixed \
  train_difficulties=['Mixed'] val_difficulties=['Mixed'] \
  train_sampling_scheme=natural train_sampling_seed=2024 \
  zero_missing_only=false \
  data_root=/home/lbh/TrajImpute/dataset/TrajImpute \
  seed=2024 batch_size=256 num_workers=8 epochs=5 \
  monitor=val_minFDE20 model.target.model.num_modes=20 \
  lr=0.001 weight_decay=0.0001 bimamba=false \
  model_version=0 clean_suffix=_trajgap \
  hydra.run.dir=outputs/smoke_b256/M0_ETH-M_b256_lr1e-3_seed2024/train \
  > outputs/smoke_b256/launch.log 2>&1 < /dev/null &

echo "launched pid=$!"
