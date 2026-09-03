#!/bin/bash
# GPU2 worker: v2 bi 组 eval 重跑 + fixed3-bi ZARA1/ZARA2 补训 + 收尾汇总
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
RUNNER="scripts/训练与评估/run_ethucy_benchmark.py"
LOGDIR=outputs/rerun_eval_0903
mkdir -p "$LOGDIR"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOGDIR/progress_gpu2.log"; }

run_group(){ # $1=out_root_name $2=config $3=data_root $4..=folds
  local name=$1 cfg=$2 data=$3; shift 3
  $PY "$RUNNER" --config-name=$cfg --data-root=$data --output-root=outputs/$name --gpu 2 --skip-train --folds "$@" >> "$LOGDIR/$name.eval.log" 2>&1
  log "$name eval rc=$?"
}

log "GPU2 worker start"
run_group ethucy_missing_v2_bi_random_block3 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v2_high/random_block3 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_bi_random_block4 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v2_high/random_block4 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_bi_random_block6 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v2_high/random_block6 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_bi_random_fixed4 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v2_high/random_fixed4 ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v2_bi_random_fixed3 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v2_high/random_fixed3 ETH HOTEL UNIV
run_group ethucy_missing_v1_bimamba_complete config_ethucy_benchmark_bimamba data/ETHUCY_missing_v1/complete ETH HOTEL UNIV ZARA1 ZARA2
run_group ethucy_missing_v1_bimamba_random_block2 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v1/random_block2 ETH HOTEL UNIV ZARA1 ZARA2
log "phase1 evals done (fixed3-bi ZARA1/ZARA2 pending retrain)"

# 补训 fixed3-bi 的 ZARA1/ZARA2（先把损坏的 train 目录挪走，保证 version_0 是新日志）
for F in ZARA1 ZARA2; do
  D=outputs/ethucy_missing_v2_bi_random_fixed3/fold_$F
  if [ -d "$D/train" ] && [ ! -d "$D/train_corrupt_0903" ]; then
    mv "$D/train" "$D/train_corrupt_0903"
  fi
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY train.py --config-name=config_ethucy_benchmark_bimamba fold=$F \
    data_root=data/ETHUCY_missing_v2_high/random_fixed3 epochs=100 batch_size=64 seed=2024 \
    precision=bf16 num_workers=16 output=runs hydra.run.dir=$D/train \
    > "$LOGDIR/retrain_fixed3_$F.log" 2>&1
  log "retrain $F rc=$?"
done

# 补训完成后全组 5 折重新汇总（重跑全部 eval，10s/折，便宜）
run_group ethucy_missing_v2_bi_random_fixed3 config_ethucy_benchmark_bimamba data/ETHUCY_missing_v2_high/random_fixed3 ETH HOTEL UNIV ZARA1 ZARA2
log "GPU2 worker DONE"
