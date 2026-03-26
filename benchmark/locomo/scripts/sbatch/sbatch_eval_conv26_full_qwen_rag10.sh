#!/bin/bash

LOG_DIR=/mnt/petrelfs/leihaodong/ICML/locomo/module_test/log/eval
mkdir -p "$LOG_DIR"

sbatch \
  -p DataFrontier_Knowledge \
  --gres=gpu:0 \
  -o "$LOG_DIR/conv26_full_qwen_rag10_%j.out" \
  -e "$LOG_DIR/conv26_full_qwen_rag10_%j.err" \
  /mnt/petrelfs/leihaodong/ICML/locomo/scripts/eval_conv26_full_qwen_rag10_srun.sh
