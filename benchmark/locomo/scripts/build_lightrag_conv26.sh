#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=build_lightrag_conv26
#SBATCH -p DataFrontier_Knowledge
#SBATCH --gres=gpu:0
#SBATCH -o /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/build_lightrag_conv26_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/build_lightrag_conv26_%j.err
#
# Build LightRAG index for LoCoMo conv-26 (first conversation)
# Storage: /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_storage/conv26
# Supports resume: already-processed docs in kv_store_doc_status.json are skipped
#
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh
# LightRAG uses OPENAI_API_BASE; locomo uses OPENAI_BASE_URL
export OPENAI_API_BASE="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"

srun -N 1 --ntasks-per-node 1 python3 -u task_eval/build_lightrag_locomo.py
