#!/usr/bin/env python3
"""
检测所有 result.json 文件中 answer == answer_thinking 的情况，按会话汇总。

Usage:
    python3 check_answer_thinking_duplicate.py
"""
import json
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("/mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/eval_output/Qwen/2.5-14B/qwen2.5-14B-Instruct_reactmem")
OUTPUT_FILE = BASE_DIR / "answer_thinking_duplicate_summary.json"


def main():
    """检测所有 result.json 中 answer == answer_thinking 的情况"""
    results_by_conv = defaultdict(list)
    total_count = 0
    
    # 遍历所有 conv-* 目录
    for conv_dir in sorted(BASE_DIR.iterdir()):
        if not conv_dir.is_dir() or not conv_dir.name.startswith("conv-"):
            continue
        
        conv_id = conv_dir.name
        debug_dir = conv_dir / "debug"
        
        if not debug_dir.exists():
            continue
        
        # 遍历所有 qa_* 目录
        for qa_dir in sorted(debug_dir.iterdir()):
            if not qa_dir.is_dir() or not qa_dir.name.startswith("qa_"):
                continue
            
            qa_id = qa_dir.name
            result_file = qa_dir / "result.json"
            
            if not result_file.exists():
                continue
            
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                answer = data.get("answer", "")
                answer_thinking = data.get("answer_thinking", "")
                
                # 检测 answer == answer_thinking
                if answer and answer == answer_thinking:
                    total_count += 1
                    results_by_conv[conv_id].append({
                        "qa_id": qa_id,
                        "answer": answer[:200] + "..." if len(answer) > 200 else answer,  # 截断过长的答案
                        "answer_length": len(answer),
                    })
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                continue
    
    # 汇总结果
    summary = {
        "total_count": total_count,
        "by_conv": {}
    }
    
    for conv_id in sorted(results_by_conv.keys()):
        conv_results = results_by_conv[conv_id]
        summary["by_conv"][conv_id] = {
            "count": len(conv_results),
            "qa_ids": [r["qa_id"] for r in conv_results],
            "details": conv_results
        }
    
    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"Total cases found: {total_count}")
    print(f"\nBy conversation:")
    for conv_id in sorted(summary["by_conv"].keys()):
        count = summary["by_conv"][conv_id]["count"]
        print(f"  {conv_id}: {count} cases")
        if count > 0:
            print(f"    QAs: {', '.join(summary['by_conv'][conv_id]['qa_ids'])}")
    
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
