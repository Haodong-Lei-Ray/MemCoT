#!/bin/bash
#SBATCH --job-name=LME0
#SBATCH --output=/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/run_build_te3s_%j.out
#SBATCH --error=/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/run_build_te3s_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p DataFrontier_Knowledge

# Build LightRAG for LongMemEval with text-embedding-3-small.
# Output: /mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/te3s/{name}/

INPUT="/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/dataset/0000_e47becba.json"
NAME=$(basename "$INPUT" .json | sed 's/^[0-9]*_//')

LOG_DIR="/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag"
mkdir -p "$LOG_DIR"

cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh 2>/dev/null || true

# OUT_DIR must be set after env.sh (env.sh sets OUT_DIR=./outputs and would overwrite)
RAG_OUT="/mnt/petrelfs/leihaodong/ICML/exp/longmemeval/lightrag/te3s/${NAME}"
mkdir -p "$RAG_OUT"

srun python benchmark/LongMemEval/tool/buildrag/build_lightrag_longmemeval.py \
  --input "$INPUT" \
  -o "$RAG_OUT" \
  --embedding-model text-embedding-3-small
