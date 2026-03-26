#!/bin/bash
#SBATCH --job-name=400-449-01
#SBATCH --output=/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/run_build_te3s_all_%j.out
#SBATCH --error=/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/run_build_te3s_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p DataFrontier_Knowledge

DATASET_DIR="/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/m_split/400-449"
RAG_OUT="/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/m/te3s"
BATCH_SIZE=100

LOG_DIR="/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag"
mkdir -p "$LOG_DIR" "$RAG_OUT"

cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh 2>/dev/null || true

LLM_MODE="${LLM_MODE:-vllm}"
VLLM_CONFIG="${VLLM_CONFIG:-/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/tool/buildrag/config/config1_8101.json}"

srun python benchmark/LongMemEval/tool/buildrag/build_lightrag_longmemeval.py \
  --input "$DATASET_DIR" \
  -o "$RAG_OUT" \
  --batch-size "$BATCH_SIZE" \
  --embedding-model text-embedding-3-small \
  --llm-mode "$LLM_MODE" \
  --vllm-config "$VLLM_CONFIG"
