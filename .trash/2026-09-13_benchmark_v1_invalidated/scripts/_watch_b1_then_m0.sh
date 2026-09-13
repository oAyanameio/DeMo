#!/bin/bash
# 看护链：B1-uni 两链收官 → 归档 M0 半成品目录 → 原参重拉 M0 三臂（单向）
# 触发：B1-easy runner (pid $EASY_PID, GPU0) / B1-clean runner (pid $CLEAN_PID, GPU1) 退出
# 归档原因：runner 目录守卫拒绝复用已有目录；hydra 复用旧 run_dir 会产生 version_1
# 混叠 metrics，选点脚本会扫全部 version_*，存在选中过期 checkpoint 的风险。
set -u
cd /home/lbh/DeMo
PY=/home/lbh/.conda/envs/DeMo/bin/python
LOG=outputs/uni_encoder_round/watch_b1_then_m0.log
ts() { date '+%F %T'; }
echo "[$(ts)] watcher start (B1-easy pid=$1, B1-clean pid=$2)" >> "$LOG"
EASY_PID=$1
CLEAN_PID=$2
ARCH=outputs/uni_encoder_round/M0_partial_archived_$(date +%Y%m%d)
mkdir -p "$ARCH"

# ---- 槽位1: B1-easy 退出 → GPU0 → M0-easy 五场景 ----
(
  while kill -0 "$EASY_PID" 2>/dev/null; do sleep 300; done
  sleep 60
  python3 - <<'PY' >> "$LOG" 2>&1
import json
try:
    rs = json.load(open('outputs/trajimpute_retrain_uni/runs_B1_easy-direct_seed2024.json'))
    print('[B1-easy final]', [(r['scene'], r['status']) for r in rs])
except Exception as e:
    print('[B1-easy final] runs json unreadable:', e)
PY
  mv outputs/trajimpute_retrain_uni/M0-current_ETH-M_easy-direct_seed2024_uni "$ARCH/" 2>>"$LOG"
  echo "[$(ts)] launch M0-easy-uni GPU0 (archived old partial)" >> "$LOG"
  env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol easy-direct --variant M0-current --no-bimamba --gpu 0 \
    --output-root outputs/trajimpute_retrain_uni --screening \
    > outputs/uni_encoder_round/runner_easy_uni2.log 2>&1
  echo "[$(ts)] M0-easy-uni finished rc=$?" >> "$LOG"
) &

# ---- 槽位2: B1-clean 退出 → GPU1 → M0-hard；GPU3 → M0-clean 五折 ----
(
  while kill -0 "$CLEAN_PID" 2>/dev/null; do sleep 300; done
  sleep 60
  python3 -c "import json;r=json.load(open('outputs/clean_ethucy_uni/B1_k20/B1/complete/seed_2024/results.json'));print('[B1-clean final]',r.get('status'))" >> "$LOG" 2>&1 || echo "[B1-clean final] results.json unreadable" >> "$LOG"
  mv outputs/trajimpute_retrain_uni/M0-current_ETH-M_hard-direct_seed2024_uni "$ARCH/" 2>>"$LOG"
  echo "[$(ts)] launch M0-hard-uni GPU1" >> "$LOG"
  env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol hard-direct --variant M0-current --no-bimamba --gpu 1 \
    --output-root outputs/trajimpute_retrain_uni --screening \
    > outputs/uni_encoder_round/runner_hard_uni2.log 2>&1 &
  HARD_PID=$!
  mv outputs/clean_ethucy_uni/M0-current_k20 "$ARCH/" 2>>"$LOG"
  echo "[$(ts)] launch M0-clean-uni GPU3 (shared)" >> "$LOG"
  env PYTHONNOUSERSITE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 $PY -u scripts/训练与评估/run_trajimpute_experiments.py \
    --protocol clean-direct --variant M0-current --no-bimamba --gpu 3 \
    --output-root outputs/clean_ethucy_uni/M0-current_k20 --screening \
    > outputs/uni_encoder_round/runner_clean_uni2.log 2>&1
  CLEAN_RC=$?
  wait "$HARD_PID"
  HARD_RC=$?
  echo "[$(ts)] M0-clean-uni rc=$CLEAN_RC M0-hard-uni rc=$HARD_RC" >> "$LOG"
) &

wait
echo "[$(ts)] ALL DONE (B1 two arms + M0 three arms)" >> "$LOG"
