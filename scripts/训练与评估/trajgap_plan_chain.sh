#!/bin/bash
# M1 三卡提速链：workers=8，场景互斥，checkpoint 续训。
cd /home/lbh/DeMo
P=/home/lbh/.conda/envs/DeMo/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH=.
PLAN=outputs/trajgap_plan
ROOT=$PLAN/M1_calibration_seed2024
ARM=${1:?usage: trajgap_plan_chain.sh <gpu0|gpu1|gpu2>}
case "$ARM" in
  gpu0) GPU=0; SCENES="ZARA1-M" ;;
  gpu1) GPU=1; SCENES="ETH-M UNIV-M" ;;
  gpu2) GPU=2; SCENES="HOTEL-M ZARA2-M" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
$P -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol mixed-direct --variant M1 \
    --scenes $SCENES --seed 2024 --gpu "$GPU" --num-workers 8 \
    --output-root "$ROOT" \
    >> "$PLAN/chain_M1_${ARM}.log" 2>&1
RC=$?
echo "ARM_DONE ${ARM} rc=${RC} $(date +%F_%T)" >> "$PLAN/chain_${ARM}.log"
exit $RC
