#!/usr/bin/env bash
set -euo pipefail

cd /home/lbh/DeMo
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate DeMo
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=.

RUNNER="scripts/训练与评估/run_trajimpute_experiments.py"
OUTPUT_ROOT="outputs/c3_hard_screening_seed2024"
COMMON=(
  --protocol hard-direct
  --variant C3
  --seed 2024
  --epochs 100
  --batch-size 64
  --num-workers 4
  --K 20
  --max-train-retries 6
  --screening
)

mkdir -p outputs
setsid nohup python -u "$RUNNER" \
  "${COMMON[@]}" \
  --scenes ETH-M HOTEL-M UNIV-M \
  --gpu 3 \
  --output-root "$OUTPUT_ROOT/gpu3" \
  > "$OUTPUT_ROOT.gpu3.log" 2>&1 </dev/null &
PID_GPU3=$!

setsid nohup python -u "$RUNNER" \
  "${COMMON[@]}" \
  --scenes ZARA1-M ZARA2-M \
  --gpu 1 \
  --output-root "$OUTPUT_ROOT/gpu1" \
  > "$OUTPUT_ROOT.gpu1.log" 2>&1 </dev/null &
PID_GPU1=$!

echo "C3 Hard stream A PID=$PID_GPU3 GPU=3 scenes=ETH-M,HOTEL-M,UNIV-M"
echo "C3 Hard stream B PID=$PID_GPU1 GPU=1 scenes=ZARA1-M,ZARA2-M"
echo "logs=$OUTPUT_ROOT.gpu3.log,$OUTPUT_ROOT.gpu1.log"
echo "output=$OUTPUT_ROOT/{gpu3,gpu1}"
