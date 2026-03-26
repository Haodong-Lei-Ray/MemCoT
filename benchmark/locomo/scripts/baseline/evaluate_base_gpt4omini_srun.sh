#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=locomo_base_gpt4omini
#SBATCH -p DataFrontier_Knowledge

# ==================== gpt-4o-mini base (no RAG) ====================
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

OUT_DIR=/mnt/petrelfs/leihaodong/ICML/locomo/outputs/gpt-4o-mini/base
DATA_FILE_PATH=/mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json

mkdir -p $OUT_DIR

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH \
    --out-file $OUT_DIR/locomo10_base_qa.json \
    --model gpt-4o-mini \
    --batch-size 1
