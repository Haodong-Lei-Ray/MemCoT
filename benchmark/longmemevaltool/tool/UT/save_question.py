#!/usr/bin/env python3
"""将 longmemeval_s_cleaned.json 第 i 个问题的相关信息保存到 dataset 目录。

用法:
    # 保存第 i 个问题（0-based，i=0 表示第 1 题）
    python save_question.py -i 0

    # 保存第 i 个问题（1-based，i=1 表示第 1 题）
    python save_question.py -i 1 --one-based

    # 保存全部 500 个问题
    python save_question.py --all

    # 指定输入/输出路径
    python save_question.py -i 0 --input path/to/input.json --output path/to/dataset
"""

import argparse
import json
import os
from pathlib import Path

# 默认路径（相对于脚本所在目录）
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = BENCHMARK_DIR / "data" / "longmemeval_s_cleaned.json"
# DEFAULT_INPUT = BENCHMARK_DIR / "data" / "longmemeval_m_cleaned.json"
DEFAULT_OUTPUT = BENCHMARK_DIR / "s"
# DEFAULT_OUTPUT = BENCHMARK_DIR / "m"


def main():
    parser = argparse.ArgumentParser(description="保存 LongMemEval 第 i 个问题到 dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--index", type=int, help="问题索引（0-based）")
    group.add_argument("--all", action="store_true", help="保存全部 500 个问题")
    parser.add_argument("--one-based", action="store_true", help="index 为 1-based（1 表示第 1 题）")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 json 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出目录")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 应为数组")

    n = len(data)
    if args.all:
        indices = range(n)
    else:
        idx = args.index
        if args.one_based:
            idx = idx - 1
        if idx < 0 or idx >= n:
            raise ValueError(f"索引 {args.index} 超出范围 [0, {n-1}]")
        indices = [idx]

    saved = 0
    skipped = 0
    for i in indices:
        entry = data[i]
        qid = entry.get("question_id", f"idx_{i}")
        out_path = args.output / f"{i:04d}_{qid}.json"
        new_content = json.dumps(entry, ensure_ascii=False, indent=2)
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                if f.read() == new_content:
                    skipped += 1
                    print(f"跳过（内容相同）: {out_path}")
                    continue
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        saved += 1
        print(f"已保存: {out_path}")

    print(f"共处理 {len(indices)} 个问题: 保存 {saved} 个, 跳过 {skipped} 个（内容相同）")


if __name__ == "__main__":
    main()
