#!/bin/bash
# 5-fold Leave-One-Out CV for ETH/UCY
# Standard protocol: train on 3 scenes, val on 1, test on 1
# 5 folds, each scene is test once. Results averaged at the end.
#
# Usage: bash scripts/训练与评估/run_5fold_cv.sh [GPU_ID] [EPOCHS] [BATCH_SIZE]
#   Default: GPU=2, EPOCHS=100, BATCH_SIZE=64
set -uo pipefail

# Activate conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate DeMo
export PYTHONUNBUFFERED=1

GPU=${1:-2}
EPOCHS=${2:-100}
BATCH_SIZE=${3:-64}

# 5 folds: (train, val, test)
# Each fold: 3 train + 1 val + 1 test, all disjoint
FOLDS=(
  "HOTEL,UNIV,ZARA2 ZARA1 ETH"
  "ETH,UNIV,ZARA2 ZARA1 HOTEL"
  "ETH,HOTEL,ZARA2 ZARA1 UNIV"
  "ETH,HOTEL,UNIV ZARA2 ZARA1"
  "ETH,HOTEL,UNIV ZARA1 ZARA2"
)

RESULTS_FILE="5fold_cv_results.txt"
echo "=== 5-Fold CV Results ===" > "$RESULTS_FILE"
echo "Started at: $(date)" >> "$RESULTS_FILE"
echo "GPU: $GPU, Epochs: $EPOCHS, Batch: $BATCH_SIZE" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

FOLD_NUM=0
for FOLD in "${FOLDS[@]}"; do
  FOLD_NUM=$((FOLD_NUM + 1))
  read -r TRAIN VAL TEST <<< "$FOLD"

  echo ""
  echo "============================================"
  echo "  Fold $FOLD_NUM/5: test=$TEST  val=$VAL  train=$TRAIN"
  echo "============================================"

  # --- Train ---
  echo "[Fold $FOLD_NUM] Training..."
  set +e
  python -u train.py \
    --config-name=config_ethucy \
    "gpus=[$GPU]" \
    "epochs=$EPOCHS" \
    "batch_size=$BATCH_SIZE" \
    "datamodule.target.train_scenes=[$TRAIN]" \
    "datamodule.target.val_scenes=[$VAL]" \
    "datamodule.target.test_scenes=[$TEST]" \
    "output=ethucy_ethucy_ethucy_test-${TEST}"
  TRAIN_EXIT=$?
  set -e

  if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "[Fold $FOLD_NUM] Training FAILED (exit=$TRAIN_EXIT)" | tee -a "$RESULTS_FILE"
    continue
  fi

  # Find the latest output directory for this fold
  OUTPUT_DIR=$(ls -td outputs/ethucy_ethucy_ethucy_test-"$TEST"/*/ 2>/dev/null | head -1)
  if [ -z "$OUTPUT_DIR" ]; then
    echo "[Fold $FOLD_NUM] Cannot find output dir" | tee -a "$RESULTS_FILE"
    continue
  fi

  # Find best checkpoint
  BEST_CKPT=$(ls "$OUTPUT_DIR/checkpoints/"epoch=*.ckpt 2>/dev/null | grep -v last.ckpt | tail -1)
  if [ -z "$BEST_CKPT" ]; then
    BEST_CKPT="$OUTPUT_DIR/checkpoints/last.ckpt"
  fi
  echo "[Fold $FOLD_NUM] Best checkpoint: $BEST_CKPT"

  # --- Evaluate on test scene ---
  echo "[Fold $FOLD_NUM] Evaluating on test scene $TEST..."
  set +e
  python -u eval.py \
    --config-name=config_ethucy \
    "gpus=[$GPU]" \
    "test=false" \
    "checkpoint=$BEST_CKPT" \
    "datamodule.target.val_scenes=[$TEST]" \
    "output=ethucy_ethucy_ethucy_test-${TEST}"
  EVAL_EXIT=$?
  set -e

  if [ "$EVAL_EXIT" -ne 0 ]; then
    echo "[Fold $FOLD_NUM] Evaluation FAILED (exit=$EVAL_EXIT)" | tee -a "$RESULTS_FILE"
    continue
  fi

  echo "[Fold $FOLD_NUM] Done: test=$TEST" | tee -a "$RESULTS_FILE"
done

echo "" | tee -a "$RESULTS_FILE"
echo "=== All folds completed at $(date) ===" | tee -a "$RESULTS_FILE"
echo "Results saved to $RESULTS_FILE"