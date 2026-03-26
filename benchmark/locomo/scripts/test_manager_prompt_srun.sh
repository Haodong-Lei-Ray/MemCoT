#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=test_manager_prompt
#SBATCH -p DataFrontier_Knowledge

# 单元测试：输出 Manager 完整 prompt 到 eval_output/Qwen/UT/

cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

srun -N 1 --ntasks-per-node 1 python3 module_version/version1/tests/test_manager_prompt_dump.py
