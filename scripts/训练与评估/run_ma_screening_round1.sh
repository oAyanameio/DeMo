#!/bin/bash
# Missing-Aware 第一组正式筛选：M0 -> M1 -> M2 -> analyze（rc 门控链）。
# 数据：ETH/UCY v3_noguard random_fixed4_ng；seed 2024；五折；epochs 100。
set -uo pipefail
cd /home/lbh/DeMo

PY=/home/lbh/.conda/envs/DeMo/bin/python
RUNNER=scripts/训练与评估/run_missing_aware_ethucy.py
OUT=/home/lbh/DeMo/outputs/missing_aware/ethucy/train_adapt
DATA=data/ETHUCY_missing_v3_noguard/random_fixed4_ng
COND=random_fixed4_ng
LOGDIR=/home/lbh/DeMo/outputs/missing_aware
mkdir -p "$LOGDIR"
CHAIN_LOG="$LOGDIR/chain_random_fixed4_ng_seed2024.log"

run_variant() {
  local variant=$1
  echo "[$(date '+%F %T')] START variant=$variant" >> "$CHAIN_LOG"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. "$PY" "$RUNNER" \
    --variant "$variant" \
    --condition "$COND" \
    --data-root "$DATA" \
    --output-root "$OUT" \
    --seed 2024 \
    --gpu 1 \
    --folds ETH HOTEL UNIV ZARA1 ZARA2 \
    --epochs 100 \
    --batch-size 64 \
    --num-workers 16 \
    --precision bf16 \
    > "$LOGDIR/${variant}_${COND}_seed2024.out" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] variant=$variant rc=$rc" >> "$CHAIN_LOG"
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] CHAIN STOPPED at $variant (rc=$rc)" >> "$CHAIN_LOG"
    exit $rc
  fi
  # 逐 variant 验 5/5 status
  local ok
  ok=$("$PY" - "$OUT/$variant/$COND/seed_2024/results.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("folds_ok", 0))
EOF
)
  echo "[$(date '+%F %T')] variant=$variant folds_ok=$ok" >> "$CHAIN_LOG"
  if [ "$ok" != "5" ]; then
    echo "[$(date '+%F %T')] CHAIN STOPPED: $variant folds_ok=$ok != 5" >> "$CHAIN_LOG"
    exit 2
  fi
}

run_variant M0_base
run_variant M1_obs
run_variant M2_history

echo "[$(date '+%F %T')] ANALYZE start" >> "$CHAIN_LOG"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. "$PY" scripts/结果分析/analyze_missing_aware.py \
  --input-root /home/lbh/DeMo/outputs/missing_aware \
  --dataset ethucy \
  --protocol train_adapt \
  --conditions "$COND" \
  --variants M0_base M1_obs M2_history \
  --seeds 2024 \
  --output-dir "$LOGDIR/analysis/random_fixed4_ng_seed2024" \
  >> "$CHAIN_LOG" 2>&1
arc=$?
echo "[$(date '+%F %T')] ANALYZE rc=$arc" >> "$CHAIN_LOG"
echo "[$(date '+%F %T')] CHAIN DONE" >> "$CHAIN_LOG"
