#!/bin/bash
# M2-social 正式链（GPU3 串行）：Easy 五场景 → Hard 五场景
# M0-uni 对照已有（outputs/trajimpute_retrain_uni/ + 8/28 Clean 轮），不重跑
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/m2_social_round/chain.log
mkdir -p outputs/m2_social_round
ts(){ date '+%F %T'; }

run(){
  local proto=$1 scenes=$2
  echo "[$(ts)] START $proto ($scenes)" >> "$LOG"
  env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol $proto --variant M2-social --scenes $scenes --gpu 3 \
    --output-root outputs/trajimpute_retrain_m2 --screening \
    >> "$LOG.$proto" 2>&1
  echo "[$(ts)] DONE $proto rc=$?" >> "$LOG"
}

run easy-direct "ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M"
run hard-direct "ETH-M HOTEL-M UNIV-M ZARA1-M ZARA2-M"
echo "[$(ts)] M2 ALL DONE" >> "$LOG"
