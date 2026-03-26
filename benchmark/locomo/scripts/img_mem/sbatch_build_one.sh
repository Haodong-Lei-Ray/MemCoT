#!/bin/bash
#SBATCH -N 1
#SBATCH -p DataFrontier_Knowledge
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/petrelfs/leihaodong/ICML/exp/memory/img_mem/vit_rag/build_%j.out
#SBATCH -e /mnt/petrelfs/leihaodong/ICML/exp/memory/img_mem/vit_rag/build_%j.err
# Usage: sbatch sbatch_build_one.sh <conv_id>
# Example: sbatch sbatch_build_one.sh 30

CONV=${1:?Usage: sbatch sbatch_build_one.sh <conv_id>}
python /mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/tool/t2i/build_img_index.py \
    -i /mnt/petrelfs/leihaodong/ICML/locomo/data/img_con/img/conv-${CONV} \
    -o /mnt/petrelfs/leihaodong/ICML/exp/memory/img_mem/vit_rag/conv-${CONV}
