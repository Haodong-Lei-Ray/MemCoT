srun -N 1 --ntasks-per-node 1 -p DataFrontier_Knowledge --gres=gpu:1 --job-name=build_idx \
    python /mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/tool/t2i/build_img_index.py \
        -i /mnt/petrelfs/leihaodong/ICML/locomo/data/img_con/img/conv-26 \
        -o /mnt/petrelfs/leihaodong/ICML/exp/memory/img_mem/vit_rag/conv-26