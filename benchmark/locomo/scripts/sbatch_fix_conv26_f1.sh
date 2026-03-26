#!/bin/bash
# sbatch 提交 fix_conv26_f1 修正任务

output_root=/mnt/petrelfs/leihaodong/ICML/locomo/module_version/version1/eval_output/Qwen/qwen2.5-14B-Instruct-all
mkdir -p $output_root

sbatch \
  -p DataFrontier_Knowledge \
  --gres=gpu:0 \
  -o ${output_root}/fix_conv26_f1_log.out \
  -e ${output_root}/fix_conv26_f1_log.err \
  /mnt/petrelfs/leihaodong/ICML/locomo/scripts/fix_conv26_f1_srun.sh
