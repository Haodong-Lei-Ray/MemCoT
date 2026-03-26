#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=locomo_qwen_base

# ==================== Qwen2.5-14B-Instruct base (full context) ====================
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

OUT_DIR=/mnt/petrelfs/leihaodong/ICML/locomo/outputs/qwen2.5-14b/base
DATA_FILE_PATH=/mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json

mkdir -p $OUT_DIR

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH \
    --out-file $OUT_DIR/locomo10_base_qa.json \
    --model "Qwen/Qwen2.5-14B-Instruct" \
    --batch-size 1
