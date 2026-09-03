#!/bin/bash
# 监控 bi_B(ZARA2) 训练结束 → 自动触发 round2 全量 eval 补跑
# 用法: bash watch_biB_then_rerun.sh &   (由 nohup 守护)
cd /home/lbh/DeMo
LOG=outputs/missing_v2_round2_logs/watch_rerun.log
echo "[$(date '+%F %T')] watcher start" >> "$LOG"

# 等 bi_B 链 DONE 标记（编排脚本写 CHAIN_bi_B_DONE 到 bi_B.log）
while ! grep -q "CHAIN_bi_B_DONE" outputs/missing_v2_round2_logs/bi_B.log 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%F %T')] bi_B chain done, starting eval rerun" >> "$LOG"

# 触发全量补跑
bash scripts/训练与评估/rerun_round2_evals.sh >> "$LOG" 2>&1
echo "[$(date '+%F %T')] all done" >> "$LOG"
