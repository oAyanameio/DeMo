#!/bin/bash
# 分析臂：轮询等待 M0/M1/M2 三组 folds_ok 均=5 且 experiment_meta status=complete，
# 然后自动运行 analyze_missing_aware.py 生成 summary.json/csv/md。
set -uo pipefail
cd /home/lbh/DeMo
LOG=/home/lbh/DeMo/outputs/missing_aware
OUT=/home/lbh/DeMo/outputs/missing_aware/ethucy/train_adapt
PY=/home/lbh/.conda/envs/DeMo/bin/python

for i in $(seq 1 2880); do  # 最多等 24h
  ok=1
  for v in M0_base M1_obs M2_history; do
    n=$("$PY" -c "import json;print(json.load(open('$OUT/$v/random_fixed4_ng/seed_2024/results.json'))['folds_ok'])" 2>/dev/null || echo 0)
    [ "$n" = "5" ] || { ok=0; break; }
  done
  [ "$ok" = "1" ] && break
  sleep 30
done

if [ "$ok" != "1" ]; then
  echo "[$(date '+%F %T')] ANALYZE 等待超时（24h），三组未全部 5/5，放弃" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
  exit 3
fi

echo "[$(date '+%F %T')] ANALYZE start（三组均 5/5）" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. "$PY" scripts/结果分析/analyze_missing_aware.py \
  --input-root /home/lbh/DeMo/outputs/missing_aware \
  --dataset ethucy \
  --protocol train_adapt \
  --conditions random_fixed4_ng \
  --variants M0_base M1_obs M2_history \
  --seeds 2024 \
  --output-dir "$LOG/analysis/random_fixed4_ng_seed2024" \
  >> "$LOG/chain_random_fixed4_ng_seed2024.log" 2>&1
echo "[$(date '+%F %T')] ANALYZE rc=$? CHAIN DONE" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
