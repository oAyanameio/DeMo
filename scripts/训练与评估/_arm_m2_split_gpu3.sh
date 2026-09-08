#!/bin/bash
# M2_history 拆分臂-B（GPU3）：ZARA1+ZARA2 两折，与原臂-A（HOTEL+UNIV+剩余ETH收尾）同卡并行。
# 目录隔离：只写 fold_ZARA1/fold_ZARA2；results.json 由后结束的臂补全五折汇总。
set -uo pipefail
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
OUT=/home/lbh/DeMo/outputs/missing_aware/ethucy/train_adapt
LOG=/home/lbh/DeMo/outputs/missing_aware
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. "$PY" scripts/训练与评估/run_missing_aware_ethucy.py \
  --variant M2_history \
  --condition random_fixed4_ng \
  --data-root data/ETHUCY_missing_v3_noguard/random_fixed4_ng \
  --output-root "$OUT" \
  --seed 2024 \
  --gpu 3 \
  --folds ZARA1 ZARA2 \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 16 \
  --precision bf16 \
  > "$LOG/M2_history_splitB_seed2024.out" 2>&1
rc=$?
ok=$("$PY" -c "import json;print(json.load(open('$OUT/M2_history/random_fixed4_ng/seed_2024/results.json'))['folds_ok'])" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] M2_splitB(ZARA1,ZARA2) rc=$rc folds_ok=$ok" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
