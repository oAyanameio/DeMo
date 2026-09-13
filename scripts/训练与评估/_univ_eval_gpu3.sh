#!/bin/bash
# UNIV eval 重拉（GPU3，17:35 又被杀后）——eval 无断点，但 checkpoint 完好，纯推理重跑
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/uni_encoder_round/univ_tail.log
ts(){ date '+%F %T'; }
echo "[$(ts)] relaunch UNIV eval on GPU3" >> "$LOG"
setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $PY -u scripts/结果分析/evaluate_trajimpute_direct.py \
  --data-root /home/lbh/TrajImpute/dataset/TrajImpute --scene UNIV-M --difficulty Easy --split test \
  --variant B1 --K 20 --seed 2024 \
  --checkpoint outputs/trajimpute_retrain_uni/B1_UNIV-M_easy-direct_seed2024_uni/train/checkpoints/epoch=98.ckpt \
  --output-root outputs/trajimpute_retrain_uni/B1_UNIV-M_easy-direct_seed2024_uni/eval >> "$LOG" 2>&1 < /dev/null
echo "[$(ts)] UNIV eval rc=$?" >> "$LOG"
