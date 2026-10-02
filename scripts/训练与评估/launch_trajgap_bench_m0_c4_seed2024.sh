#!/usr/bin/env bash
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
BASE=outputs/trajgap_bench_seed2024
mkdir -p "$BASE/m0_gpu1" "$BASE/m0_gpu3" "$BASE/c4_gpu1" "$BASE/c4_gpu3"

launch() {
  local variant="$1" gpu="$2" root="$3" scenes="$4"
  local log="$root/runner.log" pidfile="$root/runner.pid"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 PYTHONPATH=. \
    "$PY" -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol mixed-direct --variant "$variant" --scenes $scenes \
    --seed 2024 --gpu "$gpu" --output-root "$root" \
    --screening --num-workers 4 --max-train-retries 1 \
    > "$log" 2>&1 < /dev/null &
  echo "$!" > "$pidfile"
  echo "${variant}_${gpu}_PID=$! scenes=$scenes root=$root"
}

# GPU1/GPU3 当前空闲；每卡各运行一个 M0 与一个 C4 runner，输出根目录完全隔离。
launch M0 1 "$BASE/m0_gpu1" "ETH-M HOTEL-M UNIV-M"
launch M0 3 "$BASE/m0_gpu3" "ZARA1-M ZARA2-M"
launch C4 1 "$BASE/c4_gpu1" "ETH-M HOTEL-M UNIV-M"
launch C4 3 "$BASE/c4_gpu3" "ZARA1-M ZARA2-M"
