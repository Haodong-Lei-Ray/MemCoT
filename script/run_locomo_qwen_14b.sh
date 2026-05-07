#!/usr/bin/env bash
set -euo pipefail

cd "/home/lei/Project/MemCoT"

export PYTHONUNBUFFERED=1
export OPENAI_API_KEY="sk-M9LyabPjGlgnNyj68nuGXmCkAkJOD25Fsmg6sAMQB8WbQVh9"
export OPENAI_BASE_URL="http://35.220.164.252:3888/v1"

# 使用 memcot 环境运行
conda run -n memcot python run_eval_in_benchmark.py \
  --output-dir "/home/lei/Project/exp/Qwen/locomo-test" \
  --rag-config "/home/lei/Project/MemCoT/config/rag/locomolightrag.json" \
  --memcot-config "/home/lei/Project/MemCoT/config/memcot-qwen2.5-14b.json" \
  -c "conv-26" \
  -n 2
