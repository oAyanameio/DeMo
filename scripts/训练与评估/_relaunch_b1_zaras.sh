#!/bin/bash
# B1-uni ZARA1/ZARA2 重拉（15:18 被杀后恢复）
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/uni_encoder_round/parallel_b1_uni.log
ts() { date '+%F %T'; }

(
  echo "[$(ts)] relaunch ZARA1-M GPU3" >> "$LOG"
  setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol easy-direct --variant B1 --scenes ZARA1-M --gpu 3 \
    --output-root outputs/trajimpute_retrain_uni --screening \
    > outputs/uni_encoder_round/runner_B1_zara1_gpu3.log 2>&1 < /dev/null
  echo "[$(ts)] ZARA1 done rc=$?" >> "$LOG"
) &

(
  echo "[$(ts)] relaunch ZARA2-M GPU0" >> "$LOG"
  setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol easy-direct --variant B1 --scenes ZARA2-M --gpu 0 \
    --output-root outputs/trajimpute_retrain_uni --screening \
    > outputs/uni_encoder_round/runner_B1_zara2_gpu0b.log 2>&1 < /dev/null
  echo "[$(ts)] ZARA2 done rc=$?" >> "$LOG"
) &
wait
echo "[$(ts)] ZARA1+ZARA2 relaunch ALL DONE" >> "$LOG"
