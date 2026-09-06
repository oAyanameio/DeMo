#!/bin/bash
# ETH/UCY v3_noguard 全量构建 → 全量审计 → v3 零样本（GPU0）
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
OUT=outputs/missing_v3_zeroshot_logs
mkdir -p "$OUT"
LOG="$OUT/v3_chain.log"

echo "[$(date '+%F %T')] ETH/UCY v3 build start" >> "$LOG"
$PY scripts/数据集构建/build_missing_history_dataset.py --dataset ethucy \
  --source-root data/ETHUCY_benchmark_v1 --output-root data/ETHUCY_missing_v3_noguard \
  --conditions random_fixed3_ng random_fixed4_ng random_block3_ng random_block4_ng random_block6_ng uniform_hard_ng \
  --mask-seed 42 --version missing_history_v3_noguard --workers 24 \
  > /tmp/v3_eth_build.log 2>&1
rc=$?
echo "[$(date '+%F %T')] ETH/UCY v3 build rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && exit $rc

echo "[$(date '+%F %T')] ETH/UCY v3 audit start" >> "$LOG"
$PY scripts/审计与校验/audit_missing_history_dataset.py --dataset ethucy \
  --data-root data/ETHUCY_missing_v3_noguard --source-root data/ETHUCY_benchmark_v1 --full \
  > /tmp/v3_eth_audit.log 2>&1
rc=$?
echo "[$(date '+%F %T')] ETH/UCY v3 audit rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && exit $rc

echo "[$(date '+%F %T')] v3 zeroshot GPU0 start" >> "$LOG"
bash scripts/训练与评估/run_zeroshot_v3.sh 0 random_block6_ng uniform_hard_ng random_fixed4_ng \
  >> "$OUT/nohup.log" 2>&1
echo "[$(date '+%F %T')] v3 zeroshot done rc=$?" >> "$LOG"
