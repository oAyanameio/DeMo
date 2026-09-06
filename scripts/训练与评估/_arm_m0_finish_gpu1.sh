#!/bin/bash
# M0 收尾臂：等 ZARA2 训练进程（1363518）自然结束后，--skip-train 走 runner
# 的 eval+汇总路径，补出 M0 的 results.json（folds_ok=5）。
set -uo pipefail
cd /home/lbh/DeMo
LOG=/home/lbh/DeMo/outputs/missing_aware
OUT=/home/lbh/DeMo/outputs/missing_aware/ethucy/train_adapt
PY=/home/lbh/.conda/envs/DeMo/bin/python

# 等待 ZARA2 训练进程结束（最多等 20 小时）
for i in $(seq 1 2400); do
  ps -p 1363518 >/dev/null 2>&1 || break
  sleep 30
done

echo "[$(date '+%F %T')] M0 ZARA2 train 进程已结束，开始 eval+汇总" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
# ZARA2 的 train run 目录已在，runner skip-train 会重新收集 best ckpt 并 eval
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. "$PY" scripts/训练与评估/run_missing_aware_ethucy.py \
  --variant M0_base \
  --condition random_fixed4_ng \
  --data-root data/ETHUCY_missing_v3_noguard/random_fixed4_ng \
  --output-root "$OUT" \
  --seed 2024 \
  --gpu 1 \
  --folds ETH HOTEL UNIV ZARA1 ZARA2 \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 16 \
  --precision bf16 \
  --skip-train \
  >> "$LOG/M0_base_random_fixed4_ng_seed2024.out" 2>&1
rc=$?
ok=$("$PY" -c "import json;print(json.load(open('$OUT/M0_base/random_fixed4_ng/seed_2024/results.json'))['folds_ok'])" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] M0_base 收尾 rc=$rc folds_ok=$ok" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
