#!/bin/bash
# TrajGap M1 直训链（2026-10-10 裁定：M0 产物已删，M1 不等 M0 配对臂）：
#   GPU1 臂：ETH-M + UNIV-M + ZARA1-M
#   GPU2 臂：HOTEL-M + ZARA2-M
# M1 选点 monitor=val_cal_b-minFDE20；产物 outputs/trajgap_plan/M1_calibration_seed2024/
cd /home/lbh/DeMo
P=/home/lbh/.conda/envs/DeMo/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH=.
PLAN=outputs/trajgap_plan
M1_ROOT=$PLAN/M1_calibration_seed2024

ARM=${1:?usage: trajgap_plan_chain.sh <gpu1|gpu2>}

if [ "$ARM" = gpu1 ]; then
    $P -u scripts/训练与评估/run_trajimpute_experiments.py \
        --protocol mixed-direct --variant M1 \
        --scenes ETH-M UNIV-M ZARA1-M \
        --seed 2024 --gpu 1 --output-root "$M1_ROOT" \
        >> "$PLAN/chain_M1_gpu1.log" 2>&1
    echo "ARM_DONE gpu1 rc=$? $(date +%F_%T)" >> "$PLAN/chain_gpu1.log"
elif [ "$ARM" = gpu2 ]; then
    $P -u scripts/训练与评估/run_trajimpute_experiments.py \
        --protocol mixed-direct --variant M1 \
        --scenes HOTEL-M ZARA2-M \
        --seed 2024 --gpu 2 --output-root "$M1_ROOT" \
        >> "$PLAN/chain_M1_gpu2.log" 2>&1
    echo "ARM_DONE gpu2 rc=$? $(date +%F_%T)" >> "$PLAN/chain_gpu2.log"
fi
