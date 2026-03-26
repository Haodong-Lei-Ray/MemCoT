#!/bin/bash
# Evaluate gpt-4o with standard RAG (dialog mode)
source scripts/env.sh

OUT_DIR=/mnt/petrelfs/leihaodong/ICML/locomo/outputs/gpt-4o/rag
EMB_DIR=/mnt/petrelfs/leihaodong/ICML/exp/memory/locomo
DATA_FILE_PATH=/mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json

mkdir -p $OUT_DIR

# dialog as database (standard RAG)
for TOP_K in 5 10 25; do
    echo "========================================"
    echo "Running RAG (dialog) evaluation with top-k=$TOP_K..."
    echo "========================================"
    python3 task_eval/evaluate_qa.py \
        --data-file $DATA_FILE_PATH \
        --out-file $OUT_DIR/locomo10_rag_qa.json \
        --model gpt-4o \
        --batch-size 1 \
        --use-rag \
        --retriever openai \
        --top-k $TOP_K \
        --emb-dir $EMB_DIR \
        --rag-mode dialog
done

echo "========================================"
echo "RAG evaluation completed!"
echo "Results saved to: $OUT_DIR/locomo10_rag_qa.json"
echo "Stats saved to: $OUT_DIR/locomo10_rag_qa_stats.json"
echo "========================================"
