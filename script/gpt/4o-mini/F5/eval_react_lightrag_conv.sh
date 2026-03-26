#!/bin/bash
#SBATCH -N 1
#SBATCH -p DataFrontier_Knowledge

# ReAct+LightRAG 评估 (CONV 由 wrapper 通过环境变量传入)
# 使用 wrapper: ./sbatch_eval_react_lightrag_7b.sh [CONV] [RAG_TOPK]
# 直接 sbatch 时 CONV 默认 conv-26

if [[ $# -ge 1 ]]; then
  RAG_TOPK=$1
  shift
else
  RAG_TOPK=10
fi

NUM_QA=-1
CONV=${CONV:-conv-26}
if [[ -z "$EVAL_OUT" ]]; then
    echo "Error: EVAL_OUT environment variable is not set"
    exit 1
fi

cd /mnt/petrelfs/leihaodong/ICML/locomo
# source scripts/env.sh

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 module_version/version2/eval_react_lightrag.py \
    --lightrag-base "/mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag1/rag_storage" \
    --conv $CONV \
    -m gpt-4o-mini \
    -n $NUM_QA \
    -k $RAG_TOPK \
    --agent-flag 10001 \
    --max-step 8 \
    --resume \
    -o "$EVAL_OUT" \
    --skip-category 5