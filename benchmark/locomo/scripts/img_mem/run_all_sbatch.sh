#!/bin/bash
# sbatch 提交全部 9 个 build 任务
# Usage: ./run_all_sbatch.sh

cd "$(dirname "$0")"
mkdir -p /mnt/petrelfs/leihaodong/ICML/exp/memory/img_mem/vit_rag
CONVS=(30 41 42 43 44 47 48 49 50)
for conv in "${CONVS[@]}"; do
    echo "Submitting build for conv-${conv} ..."
    sbatch sbatch_build_one.sh "$conv" -J "BIM${conv}"
done
echo "Done. Submitted ${#CONVS[@]} jobs."
