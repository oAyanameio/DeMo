#!/bin/bash
# 第二轮高缺失实验（v2_high）编排 · v2 提速版：同卡多链并行
# 背景: 单臂 GPU util 仅 ~30%（kernel-launch bound），同卡叠链近线性提速且不改数值
# 布局(10 臂):
#   GPU0(余12G, 与常驻 vLLM 共卡): uni 链A(fixed3,fixed4) | 链B(block3,block4) | 链C(block6)
#   GPU3(余46G): 零样本臂先行 → bi 链A(fixed3,fixed4) | 链B(block3,block4) | 链C(block6)
# 每臂: SDD(train→best ckpt→held-out test) rc=0 才接 ETH/UCY 五折
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

# ---- GPU0: uni 三链 ----
run_chain uni 0 A random_fixed3 random_fixed4 &
run_chain uni 0 B random_block3 random_block4 &
run_chain uni 0 C random_block6 &
U_PIDS="$!"

# ---- GPU3: 零样本先行, bi 三链 ----
(
  bash scripts/训练与评估/run_zeroshot_v2.sh 3 >> "$LOG_DIR/zeroshot.log" 2>&1
  echo "[$(date '+%F %T')] ZEROSHOT_ARM_DONE rc=$?" >> "$LOG_DIR/zeroshot.log"
) &
run_chain bi 3 A random_fixed3 random_fixed4 &
run_chain bi 3 B random_block3 random_block4 &
run_chain bi 3 C random_block6 &
B_PIDS="$!"

wait
echo "[$(date '+%F %T')] ROUND2 ALL DONE" >> "$LOG_DIR/orchestrator.log"
