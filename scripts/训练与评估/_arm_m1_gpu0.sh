#!/bin/bash
# M1_obs 独立五折（GPU0）——提速并行臂；完成条件 folds_ok=5。
set -uo pipefail
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
OUT=/home/lbh/DeMo/outputs/missing_aware/ethucy/train_adapt
LOG=/home/lbh/DeMo/outputs/missing_aware
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "$PY" scripts/训练与评估/run_missing_aware_ethucy.py \
  --variant M1_obs \
  --condition random_fixed4_ng \
  --data-root data/ETHUCY_missing_v3_noguard/random_fixed4_ng \
  --output-root "$OUT" \
  --seed 2024 \
  --gpu 0 \
  --folds ETH HOTEL UNIV ZARA1 ZARA2 \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 16 \
  --precision bf16 \
  > "$LOG/M1_obs_random_fixed4_ng_seed2024.out" 2>&1
rc=$?
ok=$("$PY" -c "import json;print(json.load(open('$OUT/M1_obs/random_fixed4_ng/seed_2024/results.json'))['folds_ok'])" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] M1_obs rc=$rc folds_ok=$ok" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
