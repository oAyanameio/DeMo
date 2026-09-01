#!/bin/bash
# 补跑零样本 SDD 侧 10 条（首次运行时被编排重启的孤儿实例撞 model.log 污染）
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
GPU=${1:-3}
OUT=outputs/missing_v2_zeroshot_logs
SUMMARY="$OUT/zeroshot_summary.txt"
V2S=data/SDD_missing_v2_high
CONDS="random_fixed3 random_fixed4 random_block3 random_block4 random_block6"

sdd_best_ckpt () {
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
            v = r.get("val_minFDE20"); e = r.get("epoch")
            if not v or e is None or v == "": continue
            v, e = float(v), int(float(e))
            if best is None or v < best[0]: best = (v, e)
assert best, "no val rows"
run_dir = os.path.dirname(os.path.dirname(os.path.dirname(runs[-1])))
ck = os.path.join(run_dir, "summary_ckpt_dummy")
cands = [c for c in glob.glob(os.path.join(run_dir, "checkpoints", "epoch=*.ckpt"))
         if f"epoch={best[1]}." in c] or \
        [c for c in glob.glob(os.path.join(run_dir, "checkpoints", "epoch=*.ckpt"))]
assert cands, f"no ckpt under {run_dir}"
print(cands[0])
EOF
}

for DIRECTION in uni bi; do
  if [ "$DIRECTION" = uni ]; then
    SCFG=config_sdd_missing;      SROOT=outputs/sdd_missing_complete
  else
    SCFG=config_sdd_missing_bimamba; SROOT=outputs/sdd_missing_bimamba_complete
  fi
  for COND in $CONDS; do
    CKPT=$(sdd_best_ckpt "$SROOT")
    [ -z "$CKPT" ] && { echo "SDD $DIRECTION $COND no_ckpt" >> "$SUMMARY"; continue; }
    # 从 summary 移除旧行（避免重复）
    grep -v "^SDD $DIRECTION $COND " "$SUMMARY" > /tmp/zs_tmp && mv /tmp/zs_tmp "$SUMMARY"
    SLOG=outputs/sdd_zeroshot_v2_${DIRECTION}_${COND}.log
    rm -rf "outputs/sdd_zeroshot_v2_${DIRECTION}_${COND}"   # 防 model.log open(x) 撞
    rc=0
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY eval.py \
      --config-name=$SCFG gpus=1 test=true \
      "datamodule.target.condition=$COND" "datamodule.target.data_root=$V2S" \
      "checkpoint='$CKPT'" \
      hydra.run.dir=/home/lbh/DeMo/outputs/sdd_zeroshot_v2_${DIRECTION}_${COND} \
      > "$SLOG" 2>&1 || rc=$?
    if [ $rc -eq 0 ] && MET=$(grep -o "TEST METRICS: {.*}" "$SLOG" | tail -1) && [ -n "$MET" ]; then
      echo "SDD $DIRECTION $COND rc=$rc $MET" >> "$SUMMARY"
      echo "[RERUN $(date '+%F %T')] SDD $DIRECTION $COND rc=$rc" >> "$OUT/run.log"
    else
      echo "SDD $DIRECTION $COND rc=$rc FAILED_OR_NO_METRICS" >> "$SUMMARY"
    fi
  done
done
echo "SDD_ZEROSHOT_RERUN_DONE"
