#!/bin/bash
# 第二轮高缺失实验（v2_high）编排 · v3 三卡布局
# GPU2 释放后的迁移版：GPU0(3链,uni) + GPU2(4链) + GPU3(3链,bi,零样本臂若未完成则在旁继续)
# 零样本臂由独立脚本管理，本编排启动前检测其是否已完成(60 行)，未完成则顺带补跑
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG_DIR=/home/lbh/DeMo/outputs/missing_v2_round2_logs
mkdir -p "$LOG_DIR"
V2E=data/ETHUCY_missing_v2_high
V2S=data/SDD_missing_v2_high

run_chain () {
    # $1=direction(uni|bi) $2=gpu $3=链名 其后=条件列表
    local dirn=$1 gpu=$2 chain=$3; shift 3
    local log="$LOG_DIR/${dirn}_${chain}.log"

    for cond in "$@"; do
        echo "[$(date '+%F %T')] START ${dirn}/${chain} cond=$cond gpu=$gpu" >> "$log"

        if [ "$dirn" = bi ]; then
            SDDSCRIPT=scripts/训练与评估/run_sdd_missing_bimamba.py
            ECFG=config_ethucy_benchmark_bimamba
        else
            SDDSCRIPT=scripts/训练与评估/run_sdd_missing.py
            ECFG=config_ethucy_benchmark
        fi

        # --- SDD ---
        echo "[$(date '+%F %T')] SDD $cond start" >> "$log"
        CUDA_VISIBLE_DEVICES=$gpu $PY -u "$SDDSCRIPT" "$cond" 100 "$V2S" \
            >> "$log" 2>&1
        local rc=$?
        echo "[$(date '+%F %T')] SDD $cond done rc=$rc" >> "$log"
        if [ $rc -ne 0 ]; then
            echo "[$(date '+%F %T')] SKIP ETHUCY $cond (SDD rc=$rc)" >> "$log"
            continue
        fi

        # --- ETH/UCY 五折 ---
        echo "[$(date '+%F %T')] ETHUCY $cond start (5 folds)" >> "$log"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY -u scripts/训练与评估/run_ethucy_benchmark.py \
            --config-name "$ECFG" \
            --data-root "$V2E/$cond" \
            --output-root "outputs/ethucy_missing_v2_${dirn}_$cond" \
            --epochs 100 --batch-size 64 --gpu "$gpu" --seed 2024 --num-workers 16 \
            >> "$log" 2>&1
        rc=$?
        echo "[$(date '+%F %T')] ETHUCY $cond done rc=$rc" >> "$log"
    done
    echo "[$(date '+%F %T')] CHAIN_${dirn}_${chain}_DONE" >> "$log"
}

# 零样本臂：60 行(50 ETH + 10 SDD)即完成；未完成则本编排补跑(幂等覆盖)
ZS_SUMMARY=outputs/missing_v2_zeroshot_logs/zeroshot_summary.txt
ZS_LINES=$( [ -f "$ZS_SUMMARY" ] && wc -l < "$ZS_SUMMARY" || echo 0 )
if [ "$ZS_LINES" -lt 60 ]; then
    echo "[$(date '+%F %T')] zeroshot incomplete ($ZS_LINES/60), rerunning" >> "$LOG_DIR/zeroshot.log"
    bash scripts/训练与评估/run_zeroshot_v2.sh 3 >> "$LOG_DIR/zeroshot.log" 2>&1
    echo "[$(date '+%F %T')] ZEROSHOT_ARM_DONE rc=$?" >> "$LOG_DIR/zeroshot.log"
else
    echo "[$(date '+%F %T')] zeroshot already complete ($ZS_LINES/60), skip" >> "$LOG_DIR/zeroshot.log"
fi &

# ---- GPU0: uni 3 链 (与常驻 vLLM 共卡) ----
run_chain uni 0 A random_fixed3 &
run_chain uni 0 B random_fixed4 &
run_chain uni 0 C random_block3 &

# ---- GPU2: 4 链 (刚释放的整卡) ----
run_chain uni 2 D random_block4 &
run_chain uni 2 E random_block6 &
run_chain bi  2 A random_fixed3 &
run_chain bi  2 B random_fixed4 &

# ---- GPU3: bi 3 链 ----
run_chain bi 3 C random_block3 &
run_chain bi 3 D random_block4 &
run_chain bi 3 E random_block6 &

wait
echo "[$(date '+%F %T')] ROUND2 ALL DONE" >> "$LOG_DIR/orchestrator.log"
