#!/bin/bash
# 重跑 bimamba 轮 ETH/UCY 的 eval（round1 同款事故：checkpoint 路径含 '=' 未加内层引号）
# 两臂 × 5 折 = 10 个 eval；跑前 rm -rf eval 目录，防 model.log 的 open("x") 崩溃
cd /home/lbh/DeMo
GPU=${GPU:-3}
OUT=outputs/missing_v1_bimamba_logs/eval_rerun.log
: > "$OUT"
ok=0; fail=0
for c in complete random_block2; do
  for f in ETH HOTEL UNIV ZARA1 ZARA2; do
    ckpt=$(ls outputs/ethucy_missing_v1_bimamba_$c/fold_$f/train/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
      echo "{\"cond\": \"$c\", \"fold\": \"$f\", \"status\": \"no_ckpt\"}" >> "$OUT"
      fail=$((fail+1)); continue
    fi
    rm -rf outputs/ethucy_missing_v1_bimamba_$c/fold_$f/eval
    echo "===== $c / $f ($ckpt) =====" >> "$OUT"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. /home/lbh/.conda/envs/DeMo/bin/python eval.py \
      --config-name=config_ethucy_benchmark_bimamba fold=$f test=true \
      data_root=data/ETHUCY_missing_v1/$c precision=bf16 \
      "checkpoint='$ckpt'" \
      hydra.run.dir=/home/lbh/DeMo/outputs/ethucy_missing_v1_bimamba_$c/fold_$f/eval \
      >> "$OUT" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    echo "{\"cond\": \"$c\", \"fold\": \"$f\", \"rc\": $rc}" >> "$OUT"
  done
done
echo "DONE ok=$ok fail=$fail" >> "$OUT"
echo "BIMAMBA_EVAL_RERUN_DONE ok=$ok fail=$fail"
