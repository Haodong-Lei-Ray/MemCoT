#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取 event_search_f1 == 0.0 的所有情况，并统计类别分布。

用法:
  python extract_event_search_f1_zero.py <input_json>
  python extract_event_search_f1_zero.py --input /path/to/conv26_react_lightrag_top20_f1.json
  python extract_event_search_f1_zero.py --input ... --output zero_cases.json
"""

import argparse
import json
import os
from collections import defaultdict

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="提取 event_search_f1==0.0 的情况并统计类别"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="输入的 JSON 路径 (如 conv26_react_lightrag_top20_f1.json)",
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_file",
        help="输入的 JSON 路径",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="将 zero 情况输出到该 JSON 文件（不指定则只打印到 stdout）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印每条 zero 情况的详情",
    )
    return parser.parse_args()


def load_details(data):
    """从 JSON 中提取 details 或 qa 列表"""
    if isinstance(data, dict) and "details" in data:
        return data["details"]
    if isinstance(data, dict) and "qa" in data:
        return data["qa"]
    if isinstance(data, list):
        all_details = []
        for item in data:
            if isinstance(item, dict) and "qa" in item:
                all_details.extend(item["qa"])
            elif isinstance(item, dict) and "details" in item:
                all_details.extend(item["details"])
            elif isinstance(item, dict) and "qa_id" in item:
                all_details.append(item)
        return all_details
    return []


def main():
    args = parse_args()
    json_path = args.input or args.input_file
    if not json_path:
        print("请指定输入的 JSON 文件:")
        print("  python extract_event_search_f1_zero.py <input_json>")
        print("  python extract_event_search_f1_zero.py -i /path/to/conv26_react_lightrag_top20_f1.json")
        return
    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = load_details(data)
    if not details:
        print("未找到 details/qa 列表，请检查 JSON 格式")
        return

    # 筛选 event_search_f1 == 0.0 或 0
    zero_cases = []
    for item in details:
        f1_val = item.get("event_search_f1")
        if f1_val is not None and (f1_val == 0 or f1_val == 0.0):
            zero_cases.append(item)

    # 按 category 统计
    cat_counts = defaultdict(int)
    for item in zero_cases:
        cat = item.get("category", "unknown")
        cat_counts[cat] += 1

    total = len(zero_cases)
    n_total = len(details)

    print("=" * 60)
    print(f"输入文件: {json_path}")
    print(f"总 QA 数: {n_total}")
    print(f"event_search_f1 == 0.0 的数量: {total} ({100*total/n_total:.1f}%)")
    print("=" * 60)

    print("\n【类别统计】")
    print("-" * 40)
    for cat in sorted(cat_counts.keys(), key=lambda x: (0 if isinstance(x, int) else 99, x)):
        cnt = cat_counts[cat]
        name = CATEGORY_NAMES.get(cat, str(cat))
        print(f"  category {cat} ({name}): {cnt} 条 ({100*cnt/total:.1f}%)")
    print("-" * 40)
    print(f"  合计: {total} 条\n")

    if args.verbose or args.output:
        output_data = {
            "input_file": os.path.abspath(json_path),
            "total_qa": n_total,
            "zero_count": total,
            "category_stats": {f"cat_{k}": v for k, v in sorted(cat_counts.items())},
            "zero_cases": zero_cases,
        }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"已写入 {args.output}")

    if args.verbose:
        print("\n【Zero 情况详情】")
        print("-" * 60)
        for i, item in enumerate(zero_cases, 1):
            qa_id = item.get("qa_id", "?")
            cat = item.get("category", "?")
            name = CATEGORY_NAMES.get(cat, str(cat))
            question = (item.get("question", "") or "")[:60]
            gold = (str(item.get("gold_answer", "")) or "")[:40]
            pred = (str(item.get("prediction", "")) or "")[:40]
            print(f"{i}. [{qa_id}] cat={cat}({name})")
            print(f"   Q: {question}...")
            print(f"   gold: {gold}")
            print(f"   pred: {pred}")
            print()


if __name__ == "__main__":
    main()
