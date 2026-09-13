#!/bin/bash
# B1-clean-uni 迁 GPU3 重拉（用户指示优先 3/0 卡）— 2026-09-13
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol clean-direct --variant B1 --gpu 3 \
  --output-root outputs/clean_ethucy_uni/B1_k20 --screening \
  > outputs/uni_encoder_round/runner_clean_B1_uni3.log 2>&1 < /dev/null &
echo "relaunched B1-clean on GPU3"
