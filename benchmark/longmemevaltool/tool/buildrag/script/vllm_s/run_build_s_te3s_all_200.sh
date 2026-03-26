#!/bin/bash
#SBATCH --job-name=100-149-o
#SBATCH --output=../memory/longmemeval/lightrag/log/s/run_build_te3s_all_%j.out
#SBATCH --error=../memory/longmemeval/lightrag/log/s/run_build_te3s_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p DataFrontier_Knowledge

# 批量构建全部 500 个问题的 LightRAG（text-embedding-3-small）。
# 并发 20，已构建的自动跳过。
# Output: ../memory/longmemeval/lightrag/te3s/{name}/

DATASET_DIR="benchmark/longmemeval/data/s_split/200-249/0200_0ea62687.json"
RAG_OUT="../memory/longmemeval/lightrag/te3s"
BATCH_SIZE=1

LOG_DIR="../memory/longmemeval/lightrag/log/s"
mkdir -p "$LOG_DIR" "$RAG_OUT"

cd /home/lei/Project/DMSMem
source scripts/env.sh 2>/dev/null || true

VLLM_CONFIG="${VLLM_CONFIG:-benchmark/longmemevaltool/tool/buildrag/config/config8101.json}"

srun python benchmark/longmemevaltool/tool/buildrag/build_lightrag_longmemeval.py \
  --input "$DATASET_DIR" \
  -o "$RAG_OUT" \
  --batch-size "$BATCH_SIZE" \
  --embedding-model text-embedding-3-small \
  --llm-mode "openai" \
  --vllm-config "$VLLM_CONFIG"