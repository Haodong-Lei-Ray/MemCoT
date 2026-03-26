#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=build_lightrag_conv50
#SBATCH -p DataFrontier_Knowledge
#SBATCH --gres=gpu:0
#SBATCH -o /mnt/petrelfs/leihaodong/ICML/locomo/docs/lightrag_log/build_lightrag_conv50_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/ICML/locomo/docs/lightrag_log/build_lightrag_conv50_%j.err
#
# Build LightRAG index for LoCoMo conv-50
# Storage: /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_storage/conv50
#
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh
export OPENAI_API_BASE="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"

srun -N 1 --ntasks-per-node 1 python3 -u task_eval/build_lightrag_locomo.py --conv-id conv-50
