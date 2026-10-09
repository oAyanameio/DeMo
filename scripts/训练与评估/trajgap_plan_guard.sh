#!/bin/bash
# trajgap_plan 链守护臂：链 bash 被外源信号杀掉时自动重拉（断点续训由
# runner 的 last.ckpt 逻辑承担）；见 CHAIN_DONE 标记后自灭。
# pgrep 锚定完整脚本路径（教训：宽松子串会匹配空，误判链死造成双写）。
GUARD_LOG=/home/lbh/DeMo/outputs/trajgap_plan/guard.log
CHAIN=/home/lbh/DeMo/scripts/训练与评估/trajgap_plan_chain.sh
DONE_MARKER=/home/lbh/DeMo/outputs/trajgap_plan/chain_M1.log

while true; do
    sleep 300
    if grep -q "CHAIN_DONE" "$DONE_MARKER" 2>/dev/null; then
        echo "$(date +%F_%T) guard: CHAIN_DONE, exit" >> "$GUARD_LOG"
        exit 0
    fi
    if pgrep -f "bash ${CHAIN}" > /dev/null 2>&1 || pgrep -f "/bin/bash ${CHAIN}" > /dev/null 2>&1; then
        :  # 链活着，不干预
    else
        echo "$(date +%F_%T) guard: chain dead, relaunch" >> "$GUARD_LOG"
        setsid -f "$CHAIN" < /dev/null > /dev/null 2>&1
    fi
done
