#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p DataFrontier_Knowledge
#SBATCH --gres=gpu:0
#SBATCH --job-name=genimg26
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source /mnt/petrelfs/leihaodong/anaconda3/etc/profile.d/conda.sh
conda activate qwen3

cd /mnt/petrelfs/leihaodong/ICML/locomo

python module_version/version2/tool/t2i/batch_genimg.py \
    -i /mnt/petrelfs/leihaodong/ICML/locomo/data/con/conv-26.json \
    -o /mnt/petrelfs/leihaodong/ICML/locomo/data/img_con/img_genp/conv-26 \
    --batch-size 100 \
    --model doubao-seedream-5-0-260128 \
    --size 1920x1920
