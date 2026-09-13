#!/bin/bash
# M0-uni easy-ckpt → Easy test 零缺失块评估（补 Table1 上半区 Clean 行，与 MoFlow Clean 列同口径）
# 零训练，纯评估；GPU0 短时共卡
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/uni_encoder_round/zero_missing_eval.log
echo "[$(date '+%F %T')] start zero-missing eval (easy-ckpt, uni)" >> "$LOG"
for SC in ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M; do
  CKPT=$(ls -t outputs/trajimpute_retrain_uni/M0-current_${SC}_easy-direct_seed2024_uni/train/checkpoints/epoch=*.ckpt | head -1)
  echo "[$(date '+%F %T')] $SC ckpt=$CKPT" >> "$LOG"
  env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY -u scripts/结果分析/evaluate_trajimpute_direct.py \
    --data-root /home/lbh/TrajImpute/dataset/TrajImpute --scene $SC --difficulty Easy --split test \
    --variant M0-current --K 20 --seed 2024 --checkpoint "$CKPT" \
    --output-root outputs/trajimpute_retrain_uni/M0-current_${SC}_easy-direct_seed2024_uni/eval_zero \
    --zero-missing-only >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] $SC rc=$?" >> "$LOG"
done
echo "[$(date '+%F %T')] ALL DONE" >> "$LOG"
