#!/bin/bash
# ZARA2-M 单场景拆到 GPU0（与 HOTEL 尾段共卡，HOTEL 收官后独占）
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/uni_encoder_round/parallel_b1_uni.log
ts() { date '+%F %T'; }
echo "[$(ts)] GPU0 arm: ZARA2-M start" >> "$LOG"
setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol easy-direct --variant B1 --scenes ZARA2-M --gpu 0 \
  --output-root outputs/trajimpute_retrain_uni --screening \
  > outputs/uni_encoder_round/runner_B1_zara2_gpu0.log 2>&1 < /dev/null
echo "[$(ts)] GPU0 arm ZARA2-M done rc=$?" >> "$LOG"
