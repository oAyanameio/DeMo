#!/usr/bin/env bash
set +e
cd /home/lbh/DeMo
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate DeMo
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=.

RUNNER="scripts/训练与评估/run_trajimpute_experiments.py"
COMMON="--scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M --seed 2024 --epochs 100 --batch-size 64 --num-workers 4 --K 20 --max-train-retries 6"

# 每卡一条链：GPU0=M0-Easy, GPU1=C1A-Easy, GPU2=M0-Hard, GPU3=C1A-Hard
setsid nohup env CUDA_VISIBLE_DEVICES=0 python -u "$RUNNER" --protocol easy-direct --variant M0   --gpu 0 --output-root outputs/c1a_m0_par_m0_easy  $COMMON > outputs/c1a_m0_par_m0_easy.log 2>&1 </dev/null &
echo "M0   Easy PID=$! GPU=0"
setsid nohup env CUDA_VISIBLE_DEVICES=1 python -u "$RUNNER" --protocol easy-direct --variant C1-A --gpu 1 --output-root outputs/c1a_m0_par_c1a_easy $COMMON > outputs/c1a_m0_par_c1a_easy.log 2>&1 </dev/null &
echo "C1-A Easy PID=$! GPU=1"
setsid nohup env CUDA_VISIBLE_DEVICES=2 python -u "$RUNNER" --protocol hard-direct --variant M0   --gpu 2 --output-root outputs/c1a_m0_par_m0_hard  $COMMON > outputs/c1a_m0_par_m0_hard.log 2>&1 </dev/null &
echo "M0   Hard PID=$! GPU=2"
setsid nohup env CUDA_VISIBLE_DEVICES=3 python -u "$RUNNER" --protocol hard-direct --variant C1-A --gpu 3 --output-root outputs/c1a_m0_par_c1a_hard $COMMON > outputs/c1a_m0_par_c1a_hard.log 2>&1 </dev/null &
echo "C1-A Hard PID=$! GPU=3"
