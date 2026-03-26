#!/bin/bash
# 检查Hybrid RAG评估进度

OUT_DIR=/mnt/petrelfs/leihaodong/ICML/locomo/outputs/gpt-4o/hybirdrag

if [ -f "$OUT_DIR/locomo10_qa.json" ]; then
    echo "检查评估结果..."
    python3 << EOF
import json
import os

out_file = "$OUT_DIR/locomo10_qa.json"
if os.path.exists(out_file):
    with open(out_file, 'r') as f:
        data = json.load(f)
    
    print(f"已处理的samples数量: {len(data)}")
    
    # 检查每个sample的QA数量
    total_qa = 0
    completed_qa = 0
    for sample in data:
        if 'qa' in sample:
            qa_list = sample['qa']
            total_qa += len(qa_list)
            for qa in qa_list:
                # 检查是否有hybridrag预测结果
                if any('hybridrag' in key for key in qa.keys()):
                    completed_qa += 1
    
    print(f"总QA数量: {total_qa}")
    print(f"已完成预测的QA数量: {completed_qa}")
    print(f"完成进度: {completed_qa/total_qa*100:.1f}%")
else:
    print("结果文件尚未生成")
EOF
else
    echo "结果文件尚未生成，评估可能还在进行中..."
    echo "检查是否有正在运行的评估进程..."
    ps aux | grep "evaluate_qa.py" | grep -v grep
fi
