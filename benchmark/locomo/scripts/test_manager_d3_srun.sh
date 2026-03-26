#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=test_manager_d3
#SBATCH -p DataFrontier_Knowledge

# 单元测试：Manager Agent 对 D3 的搜索结果
# Query: Where did Caroline move from 4 years ago?
# 结果保存到 module_version/version1/eval_output/Qwen/manager_d3_result.json

cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

srun \
  -N 1 \
  --ntasks-per-node 1 \
  python3 module_version/version1/tests/test_manager_d3_search.py
