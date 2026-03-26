#!/bin/bash
# sbatch 提交 Qwen2.5-14B-Instruct RAG (dialog, top-k=10) 评估任务

output_root=/mnt/petrelfs/leihaodong/ICML/locomo/outputs/qwen2.5-14b/rag
method=qwen_rag_top10
mkdir -p $output_root

sbatch \
  -p DataFrontier_Knowledge \
  --gres=gpu:0 \
  -o ${output_root}/${method}_log.out \
  -e ${output_root}/${method}_log.err \
  /mnt/petrelfs/leihaodong/ICML/locomo/scripts/evaluate_qwen_rag_top10_srun.sh
