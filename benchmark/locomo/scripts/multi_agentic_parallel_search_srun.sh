#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=multi_agent_search
#SBATCH -p DataFrontier_Knowledge

# Usage: sbatch multi_agentic_parallel_search_srun.sh [query] [output_dir]
# Example: sbatch multi_agentic_parallel_search_srun.sh "Where did Caroline move from 4 years ago?"
# Example: sbatch multi_agentic_parallel_search_srun.sh "Where did Caroline move from 4 years ago?" module_version/version1/eval_output/Qwen/multi_agentic_search

QUERY="${1:-Where did Caroline move from 4 years ago?}"
MAPS_OUT_DIR="${2:-module_version/version1/eval_output/Qwen/UT/202603041736}"
mkdir -p "$MAPS_OUT_DIR"

cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

echo "Query: $QUERY"
echo "Output: $MAPS_OUT_DIR"
echo "---"

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 module_version/version1/multi_agentic_parallel_search.py \
    -c conv-26 \
    -m "Qwen/Qwen2.5-14B-Instruct" \
    -o "$MAPS_OUT_DIR" \
    "$QUERY"
