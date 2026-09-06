#!/bin/bash
# 多种子链 GPU0 并行补充臂（2026-09-04 提速方案）
# 只跑 complete_bi 两臂（链尾部），避开与 GPU1 主链的目录竞态：
# GPU1 主链按正序到达 bi 臂时本脚本已完成，SKIP 逻辑自动跳过。
# GPU0 前提：qyl vLLM 显存占位但 util=0%，剩余 ~12G，本链 2 进程 × ~3G。
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
OUT=outputs/multiseed_0903
mkdir -p "$OUT"
LOG="$OUT/progress_gpu0.log"

run_bench () {  # $1=config $2=data_root $3=output_root $4=seed
    if [ -f "outputs/$3/results.json" ] && grep -q '"status": "ok"' "outputs/$3/results.json" 2>/dev/null; then
        echo "[$(date '+%F %T')] SKIP $3 (done)" >> "$LOG"; return 0
    fi
    if [ -d "outputs/$3" ] && [ ! -f "outputs/$3/results.json" ]; then
        echo "[$(date '+%F %T')] SKIP $3 (dir exists, possibly running elsewhere)" >> "$LOG"; return 0
    fi
    echo "[$(date '+%F %T')] TRAIN-START $3 gpu=0" >> "$LOG"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY -u scripts/训练与评估/run_ethucy_benchmark.py \
        --config-name "$1" --data-root "$2" --output-root "outputs/$3" \
        --epochs 100 --batch-size 64 --gpu 0 --seed "$4" \
        >> "$OUT/$3.train.log" 2>&1
    echo "[$(date '+%F %T')] TRAIN-DONE $3 rc=$?" >> "$LOG"
}

echo "[$(date '+%F %T')] === gpu0 parallel arms start ===" >> "$LOG"

# 两臂同时在 GPU0 上并行（同一脚本两份，各自目录独立）
case "${1:-}" in
  bi_s2020) run_bench config_ethucy_benchmark_bimamba data/ETHUCY_missing_v1/complete ms_complete_bi_s2020 2020 ;;
  bi_s2021) run_bench config_ethucy_benchmark_bimamba data/ETHUCY_missing_v1/complete ms_complete_bi_s2021 2021 ;;
  *) echo "usage: $0 bi_s2020|bi_s2021" >&2; exit 1 ;;
esac
