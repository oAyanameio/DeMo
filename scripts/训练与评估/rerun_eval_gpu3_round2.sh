#!/bin/bash
# GPU3 worker 第二轮：清残留 model.log/eval 目录后重跑 v1 全部 + v2 两个缺折组
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
RUNNER="scripts/训练与评估/run_ethucy_benchmark.py"
LOGDIR=outputs/rerun_eval_0903
log(){ echo "[$(date '+%F %T')] $*" >> "$LOGDIR/progress_gpu3.log"; }

# eval.py 用 open(model.log,'x')，旧 eval 目录必须清掉
for d in outputs/ethucy_missing_v1_*/fold_*/eval outputs/ethucy_missing_v2_bi_random_fixed4/fold_HOTEL/eval outputs/ethucy_missing_v2_uni_random_block6/fold_HOTEL/eval; do
  rm -rf "$d" 2>/dev/null
done
log "stale eval dirs removed"

run_group(){
  local name=$1 cfg=$2 data=$3; shift 3
  $PY "$RUNNER" --config-name=$cfg --data-root=$data --output-root=outputs/$name --gpu 3 --skip-train --folds "$@" >> "$LOGDIR/$name.eval.log" 2>&1
  log "$name eval rc=$?"
}

run_group ethucy_missing_v1_complete config_ethucy_benchmark data/ETHUCY_missing_v1/complete ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v1_random_block2 config_ethucy_benchmark data/ETHUCY_missing_v1/random_block2 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v1_random_single config_ethucy_benchmark data/ETHUCY_missing_v1/random_single ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v1_bimamba_complete config_ethucy_benchmark_bimamba data/ETHUCY_missing_v1/complete ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v1_bimamba_random_block2 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v1/random_block2 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_bi_random_fixed4 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v2_high/random_fixed4 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_uni_random_block6 config_ethucy_benchmark data/ETHUCY_missing_v2_high/random_block6 ETH HOTEL UNIV ZARA1 ZARA2
log "GPU3 worker round2 DONE"
