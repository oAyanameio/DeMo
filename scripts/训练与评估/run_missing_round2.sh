#!/bin/bash
# 第二轮高缺失实验（v2_high）编排：5 掩码条件 × uni/bi 双向 + 零样本臂
# 数据: data/ETHUCY_missing_v2_high, data/SDD_missing_v2_high（complete 基线复用 v1）
# GPU3: 零样本臂 → bi 链(SDD→ETH/UCY 五折串行)
# GPU0: uni 链(SDD→ETH/UCY 五折串行)
# 每臂内部: SDD(train→best ckpt→held-out test) rc=0 才接 ETH/UCY 五折
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG_DIR=/home/lbh/DeMo/outputs/missing_v2_round2_logs
mkdir -p "$LOG_DIR"
CONDS="random_fixed3 random_fixed4 random_block3 random_block4 random_block6"
V2E=data/ETHUCY_missing_v2_high
V2S=data/SDD_missing_v2_high

run_chain () {
    # $1=direction(uni|bi) $2=gpu
    local dirn=$1 gpu=$2
    local log="$LOG_DIR/${dirn}.log"

    for cond in $CONDS; do
        echo "[$(date '+%F %T')] START ${dirn} cond=$cond gpu=$gpu" >> "$log"

        # --- SDD ---
        if [ "$dirn" = bi ]; then
            SDDSCRIPT=scripts/训练与评估/run_sdd_missing_bimamba.py
        else
            SDDSCRIPT=scripts/训练与评估/run_sdd_missing.py
        fi
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
        if [ "$dirn" = bi ]; then
            ECFG=config_ethucy_benchmark_bimamba
        else
            ECFG=config_ethucy_benchmark
        fi
        echo "[$(date '+%F %T')] ETHUCY $cond start (5 folds)" >> "$log"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY -u scripts/训练与评估/run_ethucy_benchmark.py \
            --config-name "$ECFG" \
            --data-root "$V2E/$cond" \
            --output-root "outputs/ethucy_missing_v2_${dirn}_$cond" \
            --epochs 100 --batch-size 64 --gpu "$gpu" --seed 2024 \
            >> "$log" 2>&1
        rc=$?
        echo "[$(date '+%F %T')] ETHUCY $cond done rc=$rc" >> "$log"
    done
    echo "[$(date '+%F %T')] CHAIN_${dirn}_DONE" >> "$log"
}

# GPU3: 零样本臂(纯 eval, ~1-2h) → bi 链
(
  bash scripts/训练与评估/run_zeroshot_v2.sh 3 >> "$LOG_DIR/zeroshot.log" 2>&1
  echo "[$(date '+%F %T')] ZEROSHOT_ARM_DONE rc=$?" >> "$LOG_DIR/zeroshot.log"
  run_chain bi 3
) &
BI_PID=$!

# GPU0: uni 链直接开跑
run_chain uni 0 &
UNI_PID=$!

wait $BI_PID $UNI_PID
echo "[$(date '+%F %T')] ROUND2 ALL DONE" >> "$LOG_DIR/orchestrator.log"
