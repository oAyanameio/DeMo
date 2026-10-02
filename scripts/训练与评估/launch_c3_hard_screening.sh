#!/usr/bin/env bash
set -euo pipefail
cd /home/lbh/DeMo
export LD_LIBRARY_PATH=/home/lbh/.conda/envs/DeMo/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=.

RUNNER="scripts/训练与评估/run_trajimpute_experiments.py"
ROOT=outputs/trajgap_bench_c3_seed2024
COMMON=(
  --protocol mixed-direct
  --variant C3
  --seed 2024
  --epochs 100
  --batch-size 64
  --num-workers 4
  --K 20
  --max-train-retries 6
  --screening
)

mkdir -p "$ROOT/gpu3" "$ROOT/gpu1"
setsid env CUDA_VISIBLE_DEVICES=3 /home/lbh/.conda/envs/DeMo/bin/python -u "$RUNNER" \
  "${COMMON[@]}" --scenes ETH-M HOTEL-M UNIV-M \
  --gpu 3 --output-root "$ROOT/gpu3" \
  > "$ROOT/gpu3/runner.log" 2>&1 < /dev/null &
PID_GPU3=$!

setsid env CUDA_VISIBLE_DEVICES=1 /home/lbh/.conda/envs/DeMo/bin/python -u "$RUNNER" \
  "${COMMON[@]}" --scenes ZARA1-M ZARA2-M \
  --gpu 1 --output-root "$ROOT/gpu1" \
  > "$ROOT/gpu1/runner.log" 2>&1 < /dev/null &
PID_GPU1=$!

echo "C3 TrajGap stream A PID=$PID_GPU3 GPU=3 scenes=ETH-M,HOTEL-M,UNIV-M"
echo "C3 TrajGap stream B PID=$PID_GPU1 GPU=1 scenes=ZARA1-M,ZARA2-M"
echo "output=$ROOT/{gpu3,gpu1}"
