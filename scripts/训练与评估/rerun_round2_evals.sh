#!/bin/bash
# round2 全量 eval 补跑（10 链 × 5 折 = 50 eval）
# 前提：run_ethucy_benchmark.py 已修复 checkpoint 引号 + tensor() 指标解析
# 每链先 rm -rf fold_*/eval（防 eval.py model.log 的 open("x") 二次崩溃），再 --skip-train 重跑
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG_DIR=outputs/missing_v2_round2_logs
V2E=data/ETHUCY_missing_v2_high

run_chain () {
    # $1=gpu $2=dirn(uni|bi) $3=cond
    local gpu=$1 dirn=$2 cond=$3
    local out="outputs/ethucy_missing_v2_${dirn}_${cond}"
    if [ "$dirn" = bi ]; then
        local cfg=config_ethucy_benchmark_bimamba
    else
        local cfg=config_ethucy_benchmark
    fi
    echo "[$(date '+%F %T')] RERUN ${dirn}_${cond} gpu=$gpu" >> "$LOG_DIR/eval_rerun.log"
    rm -rf "$out"/fold_{ETH,HOTEL,UNIV,ZARA1,ZARA2}/eval
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. $PY scripts/训练与评估/run_ethucy_benchmark.py \
        --config-name "$cfg" \
        --data-root "$V2E/$cond" \
        --output-root "$out" \
        --gpu "$gpu" --seed 2024 --skip-train \
        >> "$LOG_DIR/eval_rerun.log" 2>&1
    echo "[$(date '+%F %T')] RERUN ${dirn}_${cond} rc=$?" >> "$LOG_DIR/eval_rerun.log"
}

lane () { # $1=gpu, 其后=链描述 dirn:cond
    local gpu=$1; shift
    for spec in "$@"; do
        run_chain "$gpu" "${spec%%:*}" "${spec##*:}"
    done
}

# 3 卡分道（eval 仅 ~2GB/个，与常驻任务错峰）
lane 1 uni:block3 uni:block6 bi:block3 &
lane 2 uni:fixed3 uni:fixed4 bi:fixed3 bi:fixed4 &
lane 3 uni:block4 bi:block4 bi:block6 &
wait

# 汇总各链 results.json 的 ok 数
ok=0; fail=0
for j in outputs/ethucy_missing_v2_{uni,bi}_{fixed3,fixed4,block3,block4,block6}/results.json; do
    n=$(grep -c '"status": "ok"' "$j" 2>/dev/null || true); n=${n:-0}
    ok=$((ok+n)); fail=$((fail+5-n))
done
echo "[$(date '+%F %T')] ROUND2_EVAL_RERUN_DONE ok=$ok fail=$fail" >> "$LOG_DIR/eval_rerun.log"
echo "ROUND2_EVAL_RERUN_DONE ok=$ok fail=$fail"
