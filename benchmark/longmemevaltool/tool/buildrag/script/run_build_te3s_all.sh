#!/bin/bash
#SBATCH --job-name=LME_all
#SBATCH --output=/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/run_build_te3s_all_%j.out
#SBATCH --error=/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/run_build_te3s_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p DataFrontier_Knowledge

# 批量构建全部 500 个问题的 LightRAG（text-embedding-3-small）。
# 并发 20，已构建的自动跳过。
# Output: /mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/te3s/{name}/

DATASET_DIR="/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/dataset"
RAG_OUT="/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/m/te3s"
BATCH_SIZE=20

LOG_DIR="/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag"
mkdir -p "$LOG_DIR" "$RAG_OUT"

cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh 2>/dev/null || true

LLM_MODE="${LLM_MODE:-vllm}"
VLLM_CONFIG="${VLLM_CONFIG:-/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/tool/buildrag/config/config.json}"

srun python benchmark/LongMemEval/tool/buildrag/build_lightrag_longmemeval.py \
  --input "$DATASET_DIR" \
  -o "$RAG_OUT" \
  --batch-size "$BATCH_SIZE" \
  --embedding-model text-embedding-3-small \
  --llm-mode "$LLM_MODE" \
  --vllm-config "$VLLM_CONFIG"
