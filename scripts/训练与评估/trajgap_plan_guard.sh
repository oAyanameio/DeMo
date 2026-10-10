#!/bin/bash
# trajgap_plan 双臂守护（2026-10-10 M1 直训版）：某臂进程消失且未写
# ARM_DONE 时重拉该臂；重拉幂等性由 runner last.ckpt 断点续训承担。
GUARD_LOG=/home/lbh/DeMo/outputs/trajgap_plan/guard.log
CHAIN=/home/lbh/DeMo/scripts/训练与评估/trajgap_plan_chain.sh
PLAN=/home/lbh/DeMo/outputs/trajgap_plan

while true; do
    sleep 300
    for arm in gpu1 gpu2; do
        done_marker=$PLAN/chain_${arm}.log
        if grep -q "ARM_DONE ${arm}" "$done_marker" 2>/dev/null; then
            continue  # 该臂已完成
        fi
        if pgrep -f "bash ${CHAIN} ${arm}" > /dev/null 2>&1 \
           || pgrep -f "/bin/bash ${CHAIN} ${arm}" > /dev/null 2>&1; then
            continue  # 该臂活着
        fi
        echo "$(date +%F_%T) guard: arm ${arm} dead, relaunch" >> "$GUARD_LOG"
        setsid -f "$CHAIN" "$arm" < /dev/null > /dev/null 2>&1
    done
    if grep -q "ARM_DONE gpu1" "$PLAN/chain_gpu1.log" 2>/dev/null \
       && grep -q "ARM_DONE gpu2" "$PLAN/chain_gpu2.log" 2>/dev/null; then
        echo "$(date +%F_%T) guard: all arms done, exit" >> "$GUARD_LOG"
        exit 0
    fi
done
