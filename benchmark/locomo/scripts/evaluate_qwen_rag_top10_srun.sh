#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=locomo_qwen_rag10

# ==================== Qwen2.5-14B-Instruct + LightRAG (rag_event_conv26, top-k=10) ====================
# 跑 199 个 conv-26 问题，结果输出到 ablation/exp
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

OUT_DIR=/mnt/petrelfs/leihaodong/ICML/locomo/module_version/ablation/exp
RAG_WORKING_DIR=/mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_event_conv26
DATA_FILE_PATH=/mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json

mkdir -p $OUT_DIR

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 task_eval/evaluate_qa.py \
    --data-file $DATA_FILE_PATH \
    --out-file $OUT_DIR/conv26_lightrag_top10_qa.json \
    --model "Qwen/Qwen2.5-14B-Instruct" \
    --batch-size 1 \
    --use-rag \
    --rag-mode lightrag \
    --rag-working-dir $RAG_WORKING_DIR \
    --top-k 10 \
    --sample-id conv-26
