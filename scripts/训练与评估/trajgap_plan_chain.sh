#!/bin/bash
# TrajGap plan 正式链：M0 配对复验(5场景) -> M1 校准(5场景)，GPU 3 串行。
# 目录命名按方案 §8：outputs/trajgap_plan/{M0_pair_seed2024, M1_calibration_seed2024}
cd /home/lbh/DeMo
P=/home/lbh/.conda/envs/DeMo/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH=.

$P -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol mixed-direct --variant M0 \
    --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
    --seed 2024 --gpu 3 --output-root outputs/trajgap_plan/M0_pair_seed2024 \
    >> outputs/trajgap_plan/chain_M0.log 2>&1
M0_RC=$?
$P -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol mixed-direct --variant M1 \
    --scenes ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M \
    --seed 2024 --gpu 3 --output-root outputs/trajgap_plan/M1_calibration_seed2024 \
    >> outputs/trajgap_plan/chain_M1.log 2>&1
M1_RC=$?
echo "CHAIN_DONE m0_rc=$M0_RC m1_rc=$M1_RC $(date +%F_%T)" >> outputs/trajgap_plan/chain_M1.log
