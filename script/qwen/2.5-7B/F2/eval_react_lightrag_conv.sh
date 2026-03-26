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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
if [[ -f "${PROJECT_ROOT}/scripts/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/env.sh"
fi
LIGHTRAG_BASE="${LIGHTRAG_BASE:-${PROJECT_ROOT}/tool/rag_storage}"

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 "${PROJECT_ROOT}/eval_react_lightrag.py" \
    --lightrag-base "${LIGHTRAG_BASE}" \
    --conv $CONV \
    -m "Qwen/Qwen2.5-7B-Instruct" \
    -n $NUM_QA \
    -k $RAG_TOPK \
    --agent-flag 11000 \
    --middle-scale 4 \
    --max-step 8 \
    --resume \
    -o "$EVAL_OUT" \
    --skip-category 5