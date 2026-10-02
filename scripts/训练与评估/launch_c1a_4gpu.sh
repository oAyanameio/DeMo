#!/usr/bin/env bash
set -euo pipefail
cd /home/lbh/DeMo
export LD_LIBRARY_PATH=/home/lbh/.conda/envs/DeMo/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=.

RUNNER="scripts/训练与评估/run_trajimpute_experiments.py"
BASE=outputs/trajgap_bench_c1a_m0_seed2024
COMMON=(
  --protocol mixed-direct
  --seed 2024
  --epochs 100
  --batch-size 64
  --num-workers 4
  --K 20
  --max-train-retries 6
  --screening
)

launch() {
  local variant="$1" gpu="$2" root="$3"
  mkdir -p "$root"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" /home/lbh/.conda/envs/DeMo/bin/python -u "$RUNNER" \
    "${COMMON[@]}" --variant "$variant" --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
    --gpu "$gpu" --output-root "$root" \
    > "$root/runner.log" 2>&1 < /dev/null &
  echo "$variant PID=$! GPU=$gpu root=$root"
}

# TrajGap-Bench Mixed：M0 与 C1-A 各一条独立全场景链。
launch M0 1 "$BASE/m0"
launch C1-A 3 "$BASE/c1a"
