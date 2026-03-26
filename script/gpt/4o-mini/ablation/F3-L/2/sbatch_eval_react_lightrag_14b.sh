#!/bin/bash
# Wrapper: 动态传参 CONV，log 输出到 EVAL_OUT 下
# Usage: ./sbatch_eval_react_lightrag_7b.sh [CONV] [RAG_TOPK]
#   例: ./sbatch_eval_react_lightrag_7b.sh conv-26
#       ./sbatch_eval_react_lightrag_7b.sh conv-30 10

CONV=${1:-conv-26}
RAG_TOPK=${2:-10}
EVAL_BASE="/mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/eval_output/gpt/4o-mini/F2/"
EVAL_OUT="${EVAL_BASE}/${CONV}"
mkdir -p "$EVAL_OUT"

export CONV
export EVAL_OUT
sbatch \
  -J "4F2G14B${CONV##*-}" \
  -o "${EVAL_OUT}/slurm_%j.out" \
  -e "${EVAL_OUT}/slurm_%j.err" \
  "eval_react_lightrag_conv.sh" "$RAG_TOPK"
