#!/bin/bash
# B1-easy-uni 并行拆链：GPU1 跑 UNIV→ZARA2（串行两场景），GPU3 跑 ZARA1
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/uni_encoder_round/parallel_b1_uni.log
ts() { date '+%F %T'; }

(
  echo "[$(ts)] GPU1 arm: UNIV-M + ZARA2-M start" >> "$LOG"
  setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol easy-direct --variant B1 --scenes UNIV-M ZARA2-M --gpu 1 \
    --output-root outputs/trajimpute_retrain_uni --screening \
    > outputs/uni_encoder_round/runner_B1_uni_gpu1.log 2>&1 < /dev/null
  echo "[$(ts)] GPU1 arm done rc=$?" >> "$LOG"
) &

(
  echo "[$(ts)] GPU3 arm: ZARA1-M start" >> "$LOG"
  setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol easy-direct --variant B1 --scenes ZARA1-M --gpu 3 \
    --output-root outputs/trajimpute_retrain_uni --screening \
    > outputs/uni_encoder_round/runner_B1_uni_gpu3.log 2>&1 < /dev/null
  echo "[$(ts)] GPU3 arm done rc=$?" >> "$LOG"
) &
wait
echo "[$(ts)] parallel arms ALL DONE" >> "$LOG"
