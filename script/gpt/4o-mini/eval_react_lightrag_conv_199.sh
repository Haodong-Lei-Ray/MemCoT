#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=eval_react_conv199
#SBATCH -p DataFrontier_Knowledge

# ReAct+LightRAG 评估 conv-26 全 199 QA
# Usage: sbatch eval_react_lightrag_conv_199_srun.sh [RAG_TOPK] [skip_categories...]
#   RAG_TOPK=10 (default)
#   skip_categories: 可选；默认 --skip-category 5
#     不跳过任何: 传 no-skip，如 sbatch ... 10 no-skip
#     自定义: 如 sbatch ... 10 1 2 3 4

if [[ $# -ge 1 ]]; then
  RAG_TOPK=$1
  shift
else
  RAG_TOPK=10
fi

NUM_QA=199
EVAL_OUT="/mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/eval_output/gpt/4o-mini/gpt-4o-mini_reactmem_conv${NUM_QA}"

cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 module_version/version2/eval_react_lightrag_top20.py \
    -m "gpt-4o-mini" \
    -n $NUM_QA \
    -k $RAG_TOPK \
    --max-step 8 \
    --resume \
    -o "$EVAL_OUT" \
    --skip-category 5