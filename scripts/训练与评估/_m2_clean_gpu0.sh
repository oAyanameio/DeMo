#!/bin/bash
# M2-social Clean 臂（GPU0 共卡）：原始 ETH/UCY LOO 五折，复刻 8/28 管线
set -u
cd /home/lbh/DeMo
LOG=outputs/m2_social_round/chain.log
ts(){ date '+%F %T'; }
echo "[$(ts)] START clean LOO M2-social (GPU0, run_m2_clean)" >> "$LOG"
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
/home/lbh/.conda/envs/DeMo/bin/python -u scripts/训练与评估/run_m2_clean.py 100 \
  >> "$LOG.clean" 2>&1
echo "[$(ts)] clean LOO rc=$?" >> "$LOG"
