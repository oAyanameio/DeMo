#!/usr/bin/env bash
set +e
cd /home/lbh/DeMo || exit 1
export LD_LIBRARY_PATH=/home/lbh/.conda/envs/DeMo/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
PY=/home/lbh/.conda/envs/DeMo/bin/python
mkdir -p outputs/mixed_pair_seed2024_m0 outputs/mixed_pair_seed2024_c4
setsid env CUDA_VISIBLE_DEVICES=3 "$PY" -u scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol mixed-direct --variant M0 \
  --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
  --seed 2024 --gpu 3 \
  --output-root outputs/mixed_pair_seed2024_m0 \
  --screening --num-workers 4 --max-train-retries 1 \
  > outputs/mixed_pair_seed2024_m0/runner.log 2>&1 < /dev/null &
echo $! > outputs/mixed_pair_seed2024_m0/runner.pid
setsid env CUDA_VISIBLE_DEVICES=3 "$PY" -u scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol mixed-direct --variant C4 \
  --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
  --seed 2024 --gpu 3 \
  --output-root outputs/mixed_pair_seed2024_c4 \
  --screening --num-workers 4 --max-train-retries 1 \
  > outputs/mixed_pair_seed2024_c4/runner.log 2>&1 < /dev/null &
echo $! > outputs/mixed_pair_seed2024_c4/runner.pid
printf 'M0_PID=%s\nC4_PID=%s\n' "$(cat outputs/mixed_pair_seed2024_m0/runner.pid)" "$(cat outputs/mixed_pair_seed2024_c4/runner.pid)"
