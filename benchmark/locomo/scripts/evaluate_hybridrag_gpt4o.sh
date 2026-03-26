#!/bin/bash
# sets necessary environment variables
source scripts/env.sh

# Evaluate gpt-4o with Hybrid RAG
OUT_DIR=/mnt/petrelfs/leihaodong/ICML/locomo/outputs/gpt-4o/hybirdrag
mkdir -p $OUT_DIR

DATA_FILE_PATH=/mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json
EMB_DIR=/mnt/petrelfs/leihaodong/ICML/exp/memory

# 使用不同的top-k值进行测试
for TOP_K in 5 10 25; do
    echo "Running Hybrid RAG evaluation with top-k=$TOP_K..."
    python3 task_eval/evaluate_qa.py \
        --data-file $DATA_FILE_PATH \
        --out-file $OUT_DIR/locomo10_qa.json \
        --model gpt-4o \
        --batch-size 1 \
        --use-rag \
        --retriever openai \
        --top-k $TOP_K \
        --emb-dir $EMB_DIR \
        --rag-mode hybridrag \
        --overwrite
done

echo "Evaluation completed! Results saved to: $OUT_DIR"
