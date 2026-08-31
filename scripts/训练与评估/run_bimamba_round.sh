#!/bin/bash
# 双向 Mamba 对照臂 · 与第一轮单向协议完全一致，仅 bimamba=true
# GPU1: complete 臂(SDD→ETH/UCY五折) | GPU2: random_single 臂 → 完成后接 random_block2 臂
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG_DIR=/home/lbh/DeMo/outputs/missing_v1_bimamba_logs
mkdir -p "$LOG_DIR"

run_condition () {
    local cond=$1
    local gpu=$2
    local log="$LOG_DIR/${cond}.log"

    echo "[$(date '+%F %T')] START bimamba cond=$cond gpu=$gpu" >> "$log"

    # --- SDD 单条件（bimamba 版脚本） ---
    echo "[$(date '+%F %T')] SDD $cond start" >> "$log"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/训练与评估/run_sdd_missing_bimamba.py "$cond" 100 \
        >> "$log" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] SDD $cond done rc=$rc" >> "$log"

    # --- ETH/UCY 五折（同条件, bimamba 配置） ---
    if [ $rc -eq 0 ]; then
        echo "[$(date '+%F %T')] ETHUCY $cond start (5 folds)" >> "$log"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY -u scripts/训练与评估/run_ethucy_benchmark.py \
            --config-name config_ethucy_benchmark_bimamba \
            --data-root "data/ETHUCY_missing_v1/$cond" \
            --output-root "outputs/ethucy_missing_v1_bimamba_$cond" \
            --epochs 100 --batch-size 64 --gpu "$gpu" --seed 2024 \
            >> "$log" 2>&1
        rc=$?
        echo "[$(date '+%F %T')] ETHUCY $cond done rc=$rc" >> "$log"
    fi

    echo "[$(date '+%F %T')] FINISH cond=$cond final_rc=$rc" >> "$log"
}

case "${1:-fast}" in
  fast)
    # 核心对照: complete 臂(GPU1) + block2 臂(GPU2); SDD random_single 只跑 SDD(45min)
    run_condition complete 1 &
    (
      CUDA_VISIBLE_DEVICES=2 $PY -u scripts/训练与评估/run_sdd_missing_bimamba.py random_single 100 \
          >> "$LOG_DIR/random_single.log" 2>&1
      echo "[$(date '+%F %T')] SDD random_single done rc=$?" >> "$LOG_DIR/random_single.log"
      run_condition random_block2 2
    ) &
    wait
    ;;
  full)
    run_condition complete 1 &
    run_condition random_single 2
    run_condition random_block2 2
    wait
    ;;
  cond)
    run_condition "$2" "$3"
    ;;
esac
echo "[$(date '+%F %T')] BIMAMBA ROUND DONE" >> "$LOG_DIR/orchestrator.log"
