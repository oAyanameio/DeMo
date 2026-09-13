#!/bin/bash
# UNIV train（父runner已杀）完成后：选点 + eval + 落 results，替代被杀的 runner 尾部流程
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/uni_encoder_round/univ_tail.log
ts(){ date '+%F %T'; }
D=outputs/trajimpute_retrain_uni/B1_UNIV-M_easy-direct_seed2024_uni

while pgrep -f 'train.py.*scene=UNIV' >/dev/null 2>&1; do sleep 120; done
echo "[$(ts)] UNIV train ended" >> "$LOG"

CKPT=$($PY - <<'PYEOF' 2>>"$LOG"
import sys; sys.path.insert(0, 'scripts/训练与评估')
from run_trajimpute_experiments import select_best_checkpoint
ep, val, ck = select_best_checkpoint('outputs/trajimpute_retrain_uni/B1_UNIV-M_easy-direct_seed2024_uni/train', 'val_minFDE20')
print(ck)
PYEOF
)
echo "[$(ts)] best ckpt: $CKPT" >> "$LOG"

env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY -u scripts/结果分析/evaluate_trajimpute_direct.py \
  --data-root /home/lbh/TrajImpute/dataset/TrajImpute --scene UNIV-M --difficulty Easy --split test \
  --variant B1 --K 20 --seed 2024 --checkpoint "$CKPT" \
  --output-root $D/eval >> "$LOG" 2>&1
echo "[$(ts)] UNIV eval rc=$?" >> "$LOG"
