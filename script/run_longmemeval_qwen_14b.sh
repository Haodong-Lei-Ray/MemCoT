#!/usr/bin/env bash
set -euo pipefail

cd "/home/lei/Project/MemCoT"

export PYTHONUNBUFFERED=1
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""

conda run -n memcot python run_eval_in_benchmark.py \
  --output-dir "/home/lei/Project/exp/Qwen/longmemeval-test" \
  --rag-config "/home/lei/Project/MemCoT/config/rag/longmemevallightrag.json" \
  --memcot-config "/home/lei/Project/MemCoT/config/memcot-qwen2.5-14b.json" \
  -n 2
