#!/bin/bash
# M1 三卡守护：臂退出且未 ARM_DONE 时按断点续训重拉。
GUARD_LOG=/home/lbh/DeMo/outputs/trajgap_plan/guard.log
CHAIN=/home/lbh/DeMo/scripts/训练与评估/trajgap_plan_chain.sh
PLAN=/home/lbh/DeMo/outputs/trajgap_plan
while true; do
    sleep 300
    for arm in gpu0 gpu1 gpu2; do
        done_marker=$PLAN/chain_${arm}.log
        if grep -q "ARM_DONE ${arm}" "$done_marker" 2>/dev/null; then
            continue
        fi
        if pgrep -f "bash ${CHAIN} ${arm}" >/dev/null 2>&1 \
           || pgrep -f "/bin/bash ${CHAIN} ${arm}" >/dev/null 2>&1; then
            continue
        fi
        echo "$(date +%F_%T) guard: ${arm} dead, relaunch" >> "$GUARD_LOG"
        setsid -f "$CHAIN" "$arm" < /dev/null > /dev/null 2>&1
    done
    if grep -q "ARM_DONE gpu0" "$PLAN/chain_gpu0.log" 2>/dev/null \
       && grep -q "ARM_DONE gpu1" "$PLAN/chain_gpu1.log" 2>/dev/null \
       && grep -q "ARM_DONE gpu2" "$PLAN/chain_gpu2.log" 2>/dev/null; then
        echo "$(date +%F_%T) guard: all arms done, exit" >> "$GUARD_LOG"
        exit 0
    fi
done
