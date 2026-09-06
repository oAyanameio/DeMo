#!/bin/bash
# v3_noguard 零样本臂（方案 §5.1.6）：v1/v2 complete ckpt 直测 v3 test 集。
# ckpt 选择规则与 v2 zeroshot 完全一致：ETH/UCY save_top_k=1 → 目录内唯一
# epoch=*.ckpt（由 complete 训练自身的 val 指标选出）——绝不读取 v3 test 指标。
# 输出目录 outputs/ethucy_zeroshot_v3_*，不覆盖 v1/v2。
# 用法: bash scripts/训练与评估/run_zeroshot_v3.sh <gpu> [conds...]
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
GPU=${1:-1}; shift || true
CONDS=${@:-random_block6_ng uniform_hard_ng random_fixed4_ng}
OUT=outputs/missing_v3_zeroshot_logs
mkdir -p "$OUT"
SUMMARY="$OUT/zeroshot_v3_summary.txt"
: > "$SUMMARY"
V3E=data/ETHUCY_missing_v3_noguard

echo "[$(date '+%F %T')] v3 ZEROSHOT start gpu=$GPU conds=$CONDS" >> "$OUT/run.log"

for DIRECTION in uni bi; do
  if [ "$DIRECTION" = uni ]; then
    ECFG=config_ethucy_benchmark;      EROOT=outputs/ethucy_missing_v1_complete
  else
    ECFG=config_ethucy_benchmark_bimamba;      EROOT=outputs/ethucy_missing_v1_bimamba_complete
  fi
  for COND in $CONDS; do
    for FOLD in ETH HOTEL UNIV ZARA1 ZARA2; do
      CKPT=$(ls "$EROOT"/fold_$FOLD/train/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
      if [ -z "$CKPT" ]; then
        echo "ETHUCY $DIRECTION $COND $FOLD no_ckpt" >> "$SUMMARY"
        continue
      fi
      EDIR=outputs/ethucy_zeroshot_v3_${DIRECTION}_${COND}/fold_$FOLD
      rm -rf "$EDIR"; mkdir -p "$EDIR"
      rc=0
      CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY eval.py \
        --config-name=$ECFG fold=$FOLD test=true \
        data_root=$V3E/$COND precision=bf16 \
        "checkpoint='$CKPT'" \
        hydra.run.dir=/home/lbh/DeMo/$EDIR >> "$EDIR/eval.log" 2>&1 || rc=$?
      MET=$(grep -o "TEST METRICS: {.*}" "$EDIR/eval.log" 2>/dev/null | tail -1)
      echo "ETHUCY $DIRECTION $COND $FOLD rc=$rc $MET" >> "$SUMMARY"
      echo "[$(date '+%F %T')] ETHUCY $DIRECTION $COND $FOLD rc=$rc" >> "$OUT/run.log"
    done
  done
done

echo "[$(date '+%F %T')] v3 ZEROSHOT done" >> "$OUT/run.log"
echo "summary -> $SUMMARY"
