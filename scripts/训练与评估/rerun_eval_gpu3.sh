#!/bin/bash
# GPU3 worker: v2 uni 组 + v1 非 bimamba 组 eval 重跑
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
RUNNER="scripts/训练与评估/run_ethucy_benchmark.py"
LOGDIR=outputs/rerun_eval_0903
mkdir -p "$LOGDIR"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOGDIR/progress_gpu3.log"; }

run_group(){
  local name=$1 cfg=$2 data=$3; shift 3
  $PY "$RUNNER" --config-name=$cfg --data-root=$data --output-root=outputs/$name --gpu 3 --skip-train --folds "$@" >> "$LOGDIR/$name.eval.log" 2>&1
  log "$name eval rc=$?"
}

log "GPU3 worker start"
run_group ethucy_missing_v2_uni_random_block3 config_ethucy_benchmark data/ETHUCY_missing_v2_high/random_block3 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_uni_random_block4 config_ethucy_benchmark data/ETHUCY_missing_v2_high/random_block4 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_uni_random_block6 config_ethucy_benchmark data/ETHUCY_missing_v2_high/random_block6 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_uni_random_fixed3 config_ethucy_benchmark data/ETHUCY_missing_v2_high/random_fixed3 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_uni_random_fixed4 config_ethucy_benchmark data/ETHUCY_missing_v2_high/random_fixed4 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v1_complete config_ethucy_benchmark data/ETHUCY_missing_v1/complete ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v1_random_block2 config_ethucy_benchmark data/ETHUCY_missing_v1/random_block2 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v1_random_single config_ethucy_benchmark data/ETHUCY_missing_v1/random_single ETH HOTEL UNIV ZARA1 ZARA2
log "GPU3 worker DONE"
