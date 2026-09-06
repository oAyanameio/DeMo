#!/bin/bash
# 多种子复验 + block6反常排查（2026-09-03 晚批准范围：仅这两项）
# 链：block6-uni 种子复验（反常优先）→ complete-uni/bi 新种子 → 零样本 eval（block6/fixed4）
# 用法: nohup bash scripts/训练与评估/run_multiseed_0903.sh 1 > outputs/multiseed_0903/nohup.log 2>&1 &
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
GPU=${1:-1}
OUT=outputs/multiseed_0903
mkdir -p "$OUT"
LOG="$OUT/progress.log"
V2E=data/ETHUCY_missing_v2_high

run_bench () {  # $1=config $2=data_root $3=output_root $4=seed
    if [ -f "outputs/$3/results.json" ] && grep -q '"status": "ok"' "outputs/$3/results.json" 2>/dev/null; then
        echo "[$(date '+%F %T')] SKIP $3 (done)" >> "$LOG"; return 0
    fi
    echo "[$(date '+%F %T')] TRAIN-START $3 gpu=$GPU" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY -u scripts/训练与评估/run_ethucy_benchmark.py \
        --config-name "$1" --data-root "$2" --output-root "outputs/$3" \
        --epochs 100 --batch-size 64 --gpu "$GPU" --seed "$4" \
        >> "$OUT/$3.train.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] TRAIN-DONE $3 rc=$rc" >> "$LOG"
    return $rc
}

zeroshot () {  # $1=config $2=ckpt_root(ms输出) $3=tag
    for COND in random_block6 random_fixed4; do
      for FOLD in ETH HOTEL UNIV ZARA1 ZARA2; do
        CKPT=$(ls "outputs/$2"/fold_$FOLD/train/checkpoints/epoch=*.ckpt 2>/dev/null | head -1)
        [ -z "$CKPT" ] && { echo "ZS $3 $COND $FOLD no_ckpt" >> "$OUT/zeroshot_ms.txt"; continue; }
        EDIR=outputs/ethucy_zeroshot_ms_$3/$COND/fold_$FOLD
        rm -rf "$EDIR"; mkdir -p "$EDIR"
        rc=0
        CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PY eval.py \
          --config-name="$1" fold=$FOLD test=true \
          data_root=$V2E/$COND precision=bf16 \
          "checkpoint='$CKPT'" \
          hydra.run.dir=/home/lbh/DeMo/$EDIR >> "$EDIR/eval.log" 2>&1 || rc=$?
        MET=$(grep -o "TEST METRICS: {.*}" "$EDIR/eval.log" 2>/dev/null | tail -1)
        echo "ETHUCY $3 $COND $FOLD rc=$rc $MET" >> "$OUT/zeroshot_ms.txt"
      done
    done
    echo "[$(date '+%F %T')] ZS-DONE $3" >> "$LOG"
}

echo "[$(date '+%F %T')] === multiseed chain start gpu=$GPU ===" >> "$LOG"

# ── 第一优先：block6 反常排查（uni × 2 新种子，训练适应口径）──
run_bench config_ethucy_benchmark           data/ETHUCY_missing_v2_high/random_block6 ms_block6_uni_s2020 2020
run_bench config_ethucy_benchmark           data/ETHUCY_missing_v2_high/random_block6 ms_block6_uni_s2021 2021

# ── 第二：complete 新种子（uni + bi，反常对照 + 零样本 ckpt 源）──
run_bench config_ethucy_benchmark           data/ETHUCY_missing_v1/complete        ms_complete_uni_s2020 2020
run_bench config_ethucy_benchmark           data/ETHUCY_missing_v1/complete        ms_complete_uni_s2021 2021
run_bench config_ethucy_benchmark_bimamba   data/ETHUCY_missing_v1/complete        ms_complete_bi_s2020 2020
run_bench config_ethucy_benchmark_bimamba   data/ETHUCY_missing_v1/complete        ms_complete_bi_s2021 2021

# ── 第三：零样本多种子（complete ckpt → v2 block6/fixed4）──
zeroshot config_ethucy_benchmark         ms_complete_uni_s2020 uni_s2020
zeroshot config_ethucy_benchmark         ms_complete_uni_s2021 uni_s2021
zeroshot config_ethucy_benchmark_bimamba ms_complete_bi_s2020  bi_s2020
zeroshot config_ethucy_benchmark_bimamba ms_complete_bi_s2021  bi_s2021

echo "[$(date '+%F %T')] === multiseed chain ALL DONE ===" >> "$LOG"
