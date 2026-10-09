#!/bin/bash
# TrajGap plan 双卡并行链（2026-10-09 拆分提速）：
#   阶段1（M0 配对复验，并行）：GPU3 跑 ETH-M(断点续训)+UNIV-M+ZARA1-M，
#                              GPU2 跑 HOTEL-M+ZARA2-M
#   阶段2（M1 校准，等 M0 全部完成后并行）：GPU3 跑 ETH-M+UNIV-M+ZARA1-M，
#                              GPU2 跑 HOTEL-M+ZARA2-M
# 门禁：M1 的进入条件是 M0 配对复验完成（方案 §4）；双臂用 marker 文件
#   互等，两臂 M0 都完成才启动各自 M1。
# 守卫：runner 对已存在且无 last.ckpt 的目录拒绝复用（防跳过/竞态）；
#   ETH-M 已有 epoch=3 断点，runner 自动断点续训。
cd /home/lbh/DeMo
P=/home/lbh/.conda/envs/DeMo/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH=.
PLAN=outputs/trajgap_plan
M0_ROOT=$PLAN/M0_pair_seed2024
M1_ROOT=$PLAN/M1_calibration_seed2024
MARK3=$PLAN/.m0_done_gpu3
MARK2=$PLAN/.m0_done_gpu2

run() {  # run <gpu> <variant> <root> <scenes...>
    local gpu=$1 variant=$2 root=$3; shift 3
    $P -u scripts/训练与评估/run_trajimpute_experiments.py \
        --protocol mixed-direct --variant "$variant" \
        --scenes "$@" --seed 2024 --gpu "$gpu" \
        --output-root "$root" \
        >> "$PLAN/chain_${variant}_gpu${gpu}.log" 2>&1
}

ARM=${1:?usage: trajgap_plan_chain.sh <gpu3|gpu2>}

if [ "$ARM" = gpu3 ]; then
    run 3 M0 "$M0_ROOT" ETH-M UNIV-M ZARA1-M
    echo "M0_RC=$? $(date +%F_%T)" >> "$MARK3"
    # 等另一臂 M0 完成（其 marker 由 gpu2 臂写入）
    while [ ! -f "$MARK2" ]; do sleep 60; done
    run 3 M1 "$M1_ROOT" ETH-M UNIV-M ZARA1-M
    echo "ARM_DONE gpu3 m1_rc=$? $(date +%F_%T)" >> "$PLAN/chain_gpu3.log"
elif [ "$ARM" = gpu2 ]; then
    run 2 M0 "$M0_ROOT" HOTEL-M ZARA2-M
    echo "M0_RC=$? $(date +%F_%T)" >> "$MARK2"
    while [ ! -f "$MARK3" ]; do sleep 60; done
    run 2 M1 "$M1_ROOT" HOTEL-M ZARA2-M
    echo "ARM_DONE gpu2 m1_rc=$? $(date +%F_%T)" >> "$PLAN/chain_gpu2.log"
fi
