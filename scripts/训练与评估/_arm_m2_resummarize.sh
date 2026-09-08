#!/bin/bash
# M2 五折汇总看护：等全部 fold_result.json 落盘后重汇总（拆分臂各自只汇总自己的折），
# 使 results.json 达到 folds_ok=5 / status=complete，解锁分析臂。
set -uo pipefail
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=/home/lbh/DeMo/outputs/missing_aware
EXP="$LOG/ethucy/train_adapt/M2_history/random_fixed4_ng/seed_2024"

for i in $(seq 1 2880); do  # 最多 24h
  n=0
  for f in ETH HOTEL UNIV ZARA1 ZARA2; do
    [ -f "$EXP/fold_$f/fold_result.json" ] && n=$((n+1))
  done
  [ "$n" = "5" ] && break
  sleep 60
done
if [ "$n" != "5" ]; then
  echo "[$(date '+%F %T')] M2 汇总看护超时: 仅 $n/5 折完成" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
  exit 5
fi

# 等可能仍在写 results.json 的臂多喘 2 分钟再覆盖
sleep 120
"$PY" scripts/训练与评估/resummarize_missing_aware.py "$EXP" M2_history >> "$LOG/chain_random_fixed4_ng_seed2024.log" 2>&1
"$PY" -c "
import json
p='$EXP/experiment_meta.json'
d=json.load(open(p)); d['status']='complete'
json.dump(d, open(p,'w'), indent=2, ensure_ascii=False)
print('M2 meta -> complete')" >> "$LOG/chain_random_fixed4_ng_seed2024.log" 2>&1
echo "[$(date '+%F %T')] M2 五折重汇总完成" >> "$LOG/chain_random_fixed4_ng_seed2024.log"
