#!/bin/bash
# B1-uni 重拉（setsid 防会话清理连带杀）— 2026-09-13 事故恢复
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol easy-direct --variant B1 --gpu 0 \
  --output-root outputs/trajimpute_retrain_uni --screening \
  > outputs/uni_encoder_round/runner_easy_B1_uni2.log 2>&1 < /dev/null &
setsid env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
  --protocol clean-direct --variant B1 --gpu 1 \
  --output-root outputs/clean_ethucy_uni/B1_k20 --screening \
  > outputs/uni_encoder_round/runner_clean_B1_uni2.log 2>&1 < /dev/null &
echo "launched: $(jobs -p | tr '\n' ' ')"
