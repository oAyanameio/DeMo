#!/bin/bash
# 并行补训 fixed3-bi 的 ZARA1(续训) + ZARA2(重训)，各自独立进程，完成自动 eval+汇总
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
RUNNER="scripts/训练与评估/run_ethucy_benchmark.py"
LOGDIR=outputs/rerun_eval_0903
mkdir -p "$LOGDIR"
log(){ echo "[$(date '+%F %T')] $*" >> "$LOGDIR/progress_gpu2.log"; }

# --- ZARA1: 从 last.ckpt 续训 ---
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY train.py --config-name=config_ethucy_benchmark_bimamba fold=ZARA1 \
  data_root=data/ETHUCY_missing_v2_high/random_fixed3 epochs=100 batch_size=64 seed=2024 \
  precision=bf16 num_workers=16 output=runs \
  hydra.run.dir=outputs/ethucy_missing_v2_bi_random_fixed3/fold_ZARA1/train \
  checkpoint="'$(pwd)/outputs/ethucy_missing_v2_bi_random_fixed3/fold_ZARA1/train/checkpoints/last.ckpt'" \
  >> "$LOGDIR/retrain_fixed3_ZARA1.log" 2>&1
log "ZARA1 续训完成 rc=$?"
rm -rf outputs/ethucy_missing_v2_bi_random_fixed3/fold_ZARA1/eval
$PY "$RUNNER" --config-name=config_ethucy_benchmark_bimamba --data-root=data/ETHUCY_missing_v2_high/random_fixed3 \
  --output-root=outputs/ethucy_missing_v2_bi_random_fixed3 --gpu 2 --skip-train --folds ZARA1 >> "$LOGDIR/ethucy_missing_v2_bi_random_fixed3.eval.log" 2>&1
log "ZARA1 eval rc=$?"

# --- ZARA2: 全新训练 ---
D=outputs/ethucy_missing_v2_bi_random_fixed3/fold_ZARA2
[ -d "$D/train" ] && [ ! -d "$D/train_corrupt_0903" ] && mv "$D/train" "$D/train_corrupt_0903"
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY train.py --config-name=config_ethucy_benchmark_bimamba fold=ZARA2 \
  data_root=data/ETHUCY_missing_v2_high/random_fixed3 epochs=100 batch_size=64 seed=2024 \
  precision=bf16 num_workers=16 output=runs hydra.run.dir=$D/train \
  >> "$LOGDIR/retrain_fixed3_ZARA2.log" 2>&1
log "ZARA2 重训完成 rc=$?"
rm -rf "$D/eval"
$PY "$RUNNER" --config-name=config_ethucy_benchmark_bimamba --data-root=data/ETHUCY_missing_v2_high/random_fixed3 \
  --output-root=outputs/ethucy_missing_v2_bi_random_fixed3 --gpu 2 --skip-train --folds ZARA2 >> "$LOGDIR/ethucy_missing_v2_bi_random_fixed3.eval.log" 2>&1
log "ZARA2 eval rc=$?"

# --- 全组重汇总（覆盖 3/5 旧结果） ---
rm -rf outputs/ethucy_missing_v2_bi_random_fixed3/fold_*/eval
$PY "$RUNNER" --config-name=config_ethucy_benchmark_bimamba --data-root=data/ETHUCY_missing_v2_high/random_fixed3 \
  --output-root=outputs/ethucy_missing_v2_bi_random_fixed3 --gpu 2 --skip-train \
  --folds ETH HOTEL UNIV ZARA1 ZARA2 >> "$LOGDIR/ethucy_missing_v2_bi_random_fixed3.eval.log" 2>&1
log "fixed3-bi 全组汇总 DONE"
