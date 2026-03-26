#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=locomo_rag_top5

# ==================== gpt-4o + RAG (dialog, top-k=5) ====================
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

OUT_DIR=/mnt/petrelfs/leihaodong/ICML/locomo/outputs/gpt-4o/rag
EMB_DIR=/mnt/petrelfs/leihaodong/ICML/exp/memory/locomo
DATA_FILE_PATH=/mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json

mkdir -p $OUT_DIR

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH \
    --out-file $OUT_DIR/locomo10_rag_qa.json \
    --model gpt-4o-mini \
    --batch-size 1 \
    --use-rag \
    --retriever openai \
    --top-k 5 \
    --emb-dir $EMB_DIR \
    --rag-mode dialog
