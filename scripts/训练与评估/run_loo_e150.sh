#!/bin/bash
cd /home/lbh/DeMo
source activate DeMo 2>/dev/null
conda activate DeMo
exec python -u scripts/训练与评估/run_5fold_loo.py 150 128
