#!/bin/bash
# M2 臂-A 看护：等 ETH fold_result.json 落盘后，把原五折臂（会重复跑 ZARA1/2）
# 替换为只跑 HOTEL+UNIV 的新臂。
set -uo pipefail
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
OUT=/home/lbh/DeMo/outputs/missing_aware/ethucy/train_adapt
LOG=/home/lbh/DeMo/outputs/missing_aware
RES="$OUT/M2_history/random_fixed4_ng/seed_2024/fold_ETH/fold_result.json"

# 等 ETH 完成（最多 2h）
for i in $(seq 1 240); do
  [ -f "$RES" ] && break
  sleep 30
done
if [ ! -f "$RES" ]; then
  echo "[$(date '+%F %T')] M2 臂-A 看护超时: ETH 未完成, 不动原臂" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
  exit 4
fi

# kill 原五折臂 bash + 其 runner 子进程（此刻它在跑 HOTEL 的最初几个 epoch, 损失可忽略）
pkill -f "_arm_m2_gpu3.sh" 2>/dev/null
pkill -f "run_missing_aware_ethucy.py --variant M2_history --condition random_fixed4_ng --data-root data/ETHUCY_missing_v3_noguard/random_fixed4_ng --output-root /home/lbh/DeMo/outputs/missing_aware/ethucy/train_adapt --seed 2024 --gpu 3 --folds ETH" 2>/dev/null
sleep 3
echo "[$(date '+%F %T')] M2 原五折臂已停, 重启臂-A(HOTEL+UNIV)" >> "$LOG/chain_random_fixed4_ng_seed2024.log"

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. "$PY" scripts/训练与评估/run_missing_aware_ethucy.py \
  --variant M2_history \
  --condition random_fixed4_ng \
  --data-root data/ETHUCY_missing_v3_noguard/random_fixed4_ng \
  --output-root "$OUT" \
  --seed 2024 \
  --gpu 3 \
  --folds HOTEL UNIV \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 16 \
  --precision bf16 \
  > "$LOG/M2_history_armA_hotel_univ_seed2024.out" 2>&1
rc=$?
echo "[$(date '+%F %T')] M2 臂-A(HOTEL+UNIV) rc=$rc" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
