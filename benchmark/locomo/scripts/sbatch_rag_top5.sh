#!/bin/bash
# sbatch 提交 locomo RAG (dialog, top-k=5) 评估任务

output_root=/mnt/petrelfs/leihaodong/ICML/locomo/outputs/gpt-4o/rag
method=rag_dialog_top5
mkdir -p $output_root

sbatch \
  -p DataFrontier_Knowledge \
  --gres=gpu:0 \
  -o ${output_root}/${method}_log.out \
  -e ${output_root}/${method}_log.err \
  /mnt/petrelfs/leihaodong/ICML/locomo/scripts/evaluate_rag_gpt4o_top5_srun.sh
