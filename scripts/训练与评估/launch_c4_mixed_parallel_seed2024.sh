#!/usr/bin/env bash
set +e
cd /home/lbh/DeMo || exit 1
export LD_LIBRARY_PATH=/home/lbh/.conda/envs/DeMo/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
PY=/home/lbh/.conda/envs/DeMo/bin/python
ROOT=outputs/mixed_pair_seed2024_c4_parallel
mkdir -p "$ROOT"
launch() {
  scene="$1"
  gpu="$2"
  scene_root="$ROOT/$scene"
  dir="$scene_root/C4_${scene}_mixed-direct_seed2024_uni"
  mkdir -p "$scene_root"
  log="$scene_root/runner.log"
  pidfile="$scene_root/runner.pid"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol mixed-direct --variant C4 --scenes "$scene" \
    --seed 2024 --gpu "$gpu" --output-root "$scene_root" \
    --screening --num-workers 4 --max-train-retries 1 \
    > "$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  printf '%s_PID=%s GPU=%s\n' "$scene" "$!" "$gpu"
}
launch ETH-M 3
launch HOTEL-M 1
launch UNIV-M 2
launch ZARA1-M 3
launch ZARA2-M 1
