#!/bin/bash
# 单场景完成监听：任一场景的正式 results.json 出现即退出（Hermes 通知→汇报）。
# 完成定义 = 训练完成 + Mixed test 评估落盘 results.json。
PLAN=/home/lbh/DeMo/outputs/trajgap_plan
BASE=$(find $PLAN -name results.json 2>/dev/null | wc -l)
echo "baseline_results=$BASE $(date +%F_%T)"
while true; do
    sleep 300
    N=$(find $PLAN -name results.json 2>/dev/null | wc -l)
    if [ "$N" -gt "$BASE" ]; then
        echo "NEW_SCENE_RESULTS:"
        find $PLAN -name results.json 2>/dev/null
        exit 0
    fi
    # 双臂都结束却无新结果（异常路径）也退出上报
    if grep -q "ARM_DONE gpu3" $PLAN/chain_gpu3.log 2>/dev/null \
       && grep -q "ARM_DONE gpu2" $PLAN/chain_gpu2.log 2>/dev/null; then
        echo "ALL_ARMS_DONE_NO_NEW_RESULTS (baseline=$BASE now=$N)"
        exit 1
    fi
done
