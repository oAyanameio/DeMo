#!/bin/bash
# 零样本高缺失鲁棒性臂（方案 §6.2）：v1 complete 训练的 ckpt 直接测 v2_high 五条件 held-out test。
# 不训练、不用 v2 数据选点；ckpt 选择复刻训练臂规则：
#   ETH/UCY save_top_k=1 → 目录内唯一 epoch=*.ckpt 即 val 最优
#   SDD    save_top_k=3  → 从 metrics.csv 取 val_minFDE20 最小 epoch
# 用法: bash scripts/训练与评估/run_zeroshot_v2.sh <gpu>
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
GPU=${1:-3}
OUT=outputs/missing_v2_zeroshot_logs
mkdir -p "$OUT"
SUMMARY="$OUT/zeroshot_summary.txt"
: > "$SUMMARY"

V2E=data/ETHUCY_missing_v2_high
V2S=data/SDD_missing_v2_high
CONDS="random_fixed3 random_fixed4 random_block3 random_block4 random_block6"

echo "[$(date '+%F %T')] ZEROSHOT start gpu=$GPU" >> "$OUT/run.log"

sdd_best_ckpt () {
    # $1 = outputs/sdd_missing(_bimamba)_complete；打印 val_minFDE20 最小的 epoch ckpt 路径
    $PY - "$1" <<'EOF'
import csv, glob, os, sys
root = sys.argv[1]
runs = sorted(glob.glob(os.path.join(root, "*", "logs", "version_*", "metrics.csv")),
              key=os.path.getmtime)
assert runs, f"no metrics.csv under {root}"
best = None
for c in runs:
    with open(c) as fh:
        for r in csv.DictReader(fh):
            v = r.get("val_minFDE20")
            e = r.get("epoch")
            if not v or e is None or v == "":
                continue
            v, e = float(v), int(float(e))
            if best is None or v < best[0]:
                best = (v, e)
assert best, "no val_minFDE20 rows"
run_dir = os.path.dirname(os.path.dirname(os.path.dirname(runs[-1])))
ck = os.path.join(run_dir, "checkpoints", f"epoch={best[1]}.ckpt")
if not os.path.exists(ck):  # save_top_k=3 兜底：任一非 last ckpt
    cands = [c for c in glob.glob(os.path.join(run_dir, "checkpoints", "epoch=*.ckpt"))]
    assert cands, f"no epoch ckpt under {run_dir}"
    ck = cands[0]
    print(f"WARN best epoch {best[1]} ckpt missing, fallback {ck}", file=sys.stderr)
print(ck)
EOF
}

for DIRECTION in uni bi; do
  if [ "$DIRECTION" = uni ]; then
    ECFG=config_ethucy_benchmark;      SCFG=config_sdd_missing
    EROOT=outputs/ethucy_missing_v1_complete;  SROOT=outputs/sdd_missing_complete
  else
    ECFG=config_ethucy_benchmark_bimamba;      SCFG=config_sdd_missing_bimamba
    EROOT=outputs/ethucy_missing_v1_bimamba_complete;  SROOT=outputs/sdd_missing_bimamba_complete
  fi

  # --- ETH/UCY 五折 × 五条件 ---
  for COND in $CONDS; do
    for FOLD in ETH HOTEL UNIV ZARA1 ZARA2; do
      CKPT=$(ls "$EROOT"/fold_$FOLD/train/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
      if [ -z "$CKPT" ]; then
        echo "ETHUCY $DIRECTION $COND $FOLD no_ckpt" >> "$SUMMARY"
        continue
      fi
      EDIR=outputs/ethucy_zeroshot_v2_${DIRECTION}_${COND}/fold_$FOLD
      rm -rf "$EDIR/eval"; mkdir -p "$EDIR"
      rc=0
      CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY eval.py \
        --config-name=$ECFG fold=$FOLD test=true \
        data_root=$V2E/$COND precision=bf16 \
        "checkpoint='$CKPT'" \
        hydra.run.dir=/home/lbh/DeMo/$EDIR/eval \
        >> "$EDIR/eval.log" 2>&1 || rc=$?
      if [ $rc -eq 0 ] && MET=$(grep -o "TEST METRICS: {.*}" "$EDIR/eval.log" | tail -1) && [ -n "$MET" ]; then
        echo "ETHUCY $DIRECTION $COND $FOLD rc=$rc $MET" >> "$SUMMARY"
      else
        echo "ETHUCY $DIRECTION $COND $FOLD rc=$rc FAILED_OR_NO_METRICS" >> "$SUMMARY"
      fi
      echo "[$(date '+%F %T')] ETHUCY $DIRECTION $COND $FOLD rc=$rc" >> "$OUT/run.log"
    done
  done

  # --- SDD 五条件 ---
  for COND in $CONDS; do
    CKPT=$(sdd_best_ckpt "$SROOT")
    if [ -z "$CKPT" ]; then
      echo "SDD $DIRECTION $COND no_ckpt" >> "$SUMMARY"
      continue
    fi
    SLOG=outputs/sdd_zeroshot_v2_${DIRECTION}_${COND}.log
    rc=0
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY eval.py \
      --config-name=$SCFG gpus=1 test=true \
      "datamodule.target.condition=$COND" "datamodule.target.data_root=$V2S" \
      "checkpoint='$CKPT'" \
      hydra.run.dir=/home/lbh/DeMo/outputs/sdd_zeroshot_v2_${DIRECTION}_${COND} \
      > "$SLOG" 2>&1 || rc=$?
    if [ $rc -eq 0 ] && MET=$(grep -o "TEST METRICS: {.*}" "$SLOG" | tail -1) && [ -n "$MET" ]; then
      echo "SDD $DIRECTION $COND rc=$rc $MET" >> "$SUMMARY"
    else
      echo "SDD $DIRECTION $COND rc=$rc FAILED_OR_NO_METRICS" >> "$SUMMARY"
    fi
    echo "[$(date '+%F %T')] SDD $DIRECTION $COND rc=$rc" >> "$OUT/run.log"
  done
done

echo "[$(date '+%F %T')] ZEROSHOT DONE" >> "$OUT/run.log"
echo "ZEROSHOT_DONE"
