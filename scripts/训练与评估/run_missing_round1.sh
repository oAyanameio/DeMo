#!/bin/bash
# 第一轮缺失历史实验 · 3 GPU 并行编排
# GPU0: complete | GPU1: random_single | GPU2: random_block2
# 每卡链：SDD 单条件（~45min）→ ETH/UCY 同条件五折（各 ~1-2h）
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG_DIR=/home/lbh/DeMo/outputs/missing_v1_round1_logs
mkdir -p "$LOG_DIR"

run_condition () {
    local cond=$1
    local gpu=$2
    local log="$LOG_DIR/${cond}.log"

    echo "[$(date '+%F %T')] START cond=$cond gpu=$gpu" >> "$log"

    # --- SDD 单条件 ---
    echo "[$(date '+%F %T')] SDD $cond start" >> "$log"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/训练与评估/run_sdd_missing.py "$cond" 100 \
        >> "$log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] SDD $cond done rc=$rc" >> "$log"

    # --- ETH/UCY 五折（同条件） ---
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%F %T')] ETHUCY $cond start (5 folds)" >> "$log"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY -u scripts/训练与评估/run_ethucy_benchmark.py \
            --config-name config_ethucy_benchmark \
            --data-root "data/ETHUCY_missing_v1/$cond" \
            --output-root "outputs/ethucy_missing_v1_$cond" \
            --epochs 100 --batch-size 64 --gpu "$gpu" --seed 2024 \
            >> "$log" 2>&1
        rc=$?
        echo "[$(date '+%F %T')] ETHUCY $cond done rc=$rc" >> "$log"
    fi

    echo "[$(date '+%F %T')] FINISH cond=$cond final_rc=$rc" >> "$log"
}

run_condition complete       0 &
run_condition random_single  1 &
run_condition random_block2  2 &
wait
echo "[$(date '+%F %T')] ALL CONDITIONS DONE" >> "$LOG_DIR/orchestrator.log"
