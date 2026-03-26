#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=fix_conv26_f1
# Usage: sbatch scripts/sbatch_fix_conv26_f1.sh
# 修正 conv26_event_search_full_f1.json 的 gold_answer 和 F1

cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh 2>/dev/null || true

BASE="/mnt/petrelfs/leihaodong/ICML/locomo/module_version/version1/eval_output/Qwen/qwen2.5-14B-Instruct-all"
INPUT_JSON="${BASE}/conv26_event_search_full_f1 copy.json"
OUTPUT_JSON="${BASE}/conv26_event_search_full_f1.json"

srun -N 1 --ntasks-per-node 1 \
  python3 module_version/version1/fix_conv26_f1_json_v2.py --input "$INPUT_JSON" --output "$OUTPUT_JSON"
