#!/bin/bash
# 任务1：ETH-M B1 补种子复核（seed 2025）——判定 +7.6% 是真退化还是单种子伪象
# M0-uni ETH-M seed2025 对照臂 + B1-uni ETH-M seed2025，GPU3 串行
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/uni_encoder_round/eth_seed_recheck.log
ts(){ date '+%F %T'; }

echo "[$(ts)] arm1: M0-current ETH-M seed2025 start" >> "$LOG"
setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol easy-direct --variant M0-current --scenes ETH-M --gpu 3 --seed 2025 \
  --output-root outputs/trajimpute_retrain_uni --screening \
  > outputs/uni_encoder_round/runner_eth_s2025_m0.log 2>&1 < /dev/null
echo "[$(ts)] arm1 M0 rc=$?" >> "$LOG"

echo "[$(ts)] arm2: B1 ETH-M seed2025 start" >> "$LOG"
setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol easy-direct --variant B1 --scenes ETH-M --gpu 3 --seed 2025 \
  --output-root outputs/trajimpute_retrain_uni --screening \
  > outputs/uni_encoder_round/runner_eth_s2025_b1.log 2>&1 < /dev/null
echo "[$(ts)] arm2 B1 rc=$?" >> "$LOG"
echo "[$(ts)] ALL DONE" >> "$LOG"
