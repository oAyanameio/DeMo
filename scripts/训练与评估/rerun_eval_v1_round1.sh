#!/bin/bash
# 重跑 ethucy_missing_v1 三条件 x 5折的 eval（修复 checkpoint 路径含 '=' 需加引号的问题）
cd /home/lbh/DeMo
OUT=outputs/missing_v1_round1_logs/eval_rerun.log
: > "$OUT"
for c in complete random_single random_block2; do
  for f in ETH HOTEL UNIV ZARA1 ZARA2; do
    ckpt=$(ls outputs/ethucy_missing_v1_$c/fold_$f/train/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
      echo "{\"cond\": \"$c\", \"fold\": \"$f\", \"status\": \"no_ckpt\"}" >> "$OUT"
      continue
    fi
    echo "===== $c / $f ($ckpt) =====" >> "$OUT"
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home/lbh/.conda/envs/DeMo/bin/python eval.py \
      --config-name=config_ethucy_benchmark fold=$f test=true \
      data_root=data/ETHUCY_missing_v1/$c precision=bf16 \
      "checkpoint='$ckpt'" \
      hydra.run.dir=/home/lbh/DeMo/outputs/ethucy_missing_v1_$c/fold_$f/eval \
      >> "$OUT" 2>&1
    rc=$?
    echo "{\"cond\": \"$c\", \"fold\": \"$f\", \"rc\": $rc}" >> "$OUT"
  done
done
echo "ALL_EVAL_RERUN_DONE" >> "$OUT"
