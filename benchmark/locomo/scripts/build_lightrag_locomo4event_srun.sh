#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=build_lightrag_event
#SBATCH -p DataFrontier_Knowledge
#SBATCH --gres=gpu:0
#SBATCH -o /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/build_lightrag_event_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/build_lightrag_event_%j.err
#
# Build LightRAG index from events in module_test/events_output/conv-26
# Output: /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_event/conv26
#
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh
export OPENAI_API_BASE="${OPENAI_BASE_URL:-$OPENAI_API_BASE}"

EVENTS_INPUT="${1:-/mnt/petrelfs/leihaodong/ICML/locomo/module_test/events_output/conv-26}"
OUTPUT_DIR="${2:-/mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_event}"

srun -N 1 -p DataFrontier_Knowledge --ntasks-per-node 1 python3 -u task_eval/build_lightrag_locomo4event.py \
  --events-dir "$EVENTS_INPUT" \
  --output-dir "$OUTPUT_DIR"
