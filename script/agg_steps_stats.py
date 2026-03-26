#!/usr/bin/env python3
"""
遍历 base_dir 下所有 result.json，提取 steps 字段，按 conv 聚合统计。
输出格式参考 agg_reactmem_stats.py：表格形式，per-conv + sum 行。

Usage:
    python agg_steps_stats.py -i /path/to/eval_output/Qwen/2.5-14B/F1
    python agg_steps_stats.py -i /path/to/F1 -o /path/to/steps_table.txt
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate steps from result.json per conv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agg_steps_stats.py -i /mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/eval_output/Qwen/2.5-14B/F1
  python agg_steps_stats.py -i /path/to/F1 -o steps_table.txt
        """,
    )
    parser.add_argument("-i", "--input-dir", required=True, help="eval 根目录，如 .../F1")
    parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        help="输出文件 (默认: <input-dir>/steps_table.txt)",
    )
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return

    out_file = Path(args.output_file) if args.output_file else base_dir / "steps_table.txt"

    # 收集所有 result.json
    result_paths = list(base_dir.glob("**/result.json"))
    print(f"Found {len(result_paths)} result.json under {base_dir}")

    # conv_name -> list of steps
    conv_steps: dict[str, list[int]] = defaultdict(list)
    max_steps = 0
    parse_fail = 0

    for p in result_paths:
        # 从路径解析 conv: base/conv-26/debug/qa_1/result.json -> conv-26
        try:
            rel = p.relative_to(base_dir)
            parts = rel.parts
            conv_name = None
            for part in parts:
                if part.startswith("conv-"):
                    conv_name = part
                    break
            if conv_name is None:
                parse_fail += 1
                continue
        except ValueError:
            parse_fail += 1
            continue

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            steps = data.get("steps")
            if steps is None:
                parse_fail += 1
                continue
            s = int(steps)
            conv_steps[conv_name].append(s)
            if s > max_steps:
                max_steps = s
        except (json.JSONDecodeError, TypeError, ValueError):
            parse_fail += 1

    if parse_fail:
        print(f"Warning: {parse_fail} files skipped (parse/read error or missing 'steps')")

    if not conv_steps:
        print("No valid result.json with 'steps' field found.")
        return

    def _conv_num(name):
        try:
            return int(name.replace("conv-", "").split("-")[0])
        except ValueError:
            return 999

    rows = sorted(conv_steps.keys(), key=_conv_num)

    # 确定 step 列范围：1..max_steps
    step_cols = list(range(1, max_steps + 1))
    col_w = 8
    conv_id_w = 12

    def fmt_row(vals):
        return "".join(str(v).rjust(col_w) for v in vals)

    def fmt_header(vals):
        return "".join(str(v).rjust(col_w) for v in vals)

    # ------ Steps count table ------
    lines = []
    headers = [f"step_{s}" for s in step_cols] + ["mean", "total"]
    lines.append("Steps distribution (各步数数量)")
    lines.append("conv_id".ljust(conv_id_w) + fmt_header(headers))
    lines.append("-" * (conv_id_w + col_w * len(headers)))

    count_by_step = defaultdict(lambda: defaultdict(int))
    total_by_conv = {}
    mean_by_conv = {}
    sum_by_step = {s: 0 for s in step_cols}

    for conv_name in rows:
        steps_list = conv_steps[conv_name]
        total = len(steps_list)
        total_by_conv[conv_name] = total
        mean_val = sum(steps_list) / total if total else 0
        mean_by_conv[conv_name] = mean_val

        row_vals = []
        for s in step_cols:
            cnt = steps_list.count(s)
            count_by_step[conv_name][s] = cnt
            sum_by_step[s] += cnt
            row_vals.append(cnt)
        row_vals.append(round(mean_val, 2))
        row_vals.append(total)
        lines.append(conv_name.ljust(conv_id_w) + fmt_row(row_vals))

    # sum row
    total_all = sum(total_by_conv.values())
    sum_row = []
    for s in step_cols:
        sum_row.append(sum_by_step[s])
    if total_all > 0:
        sum_mean = sum(s * sum_by_step[s] for s in step_cols) / total_all
    else:
        sum_mean = 0
    sum_row.append(round(sum_mean, 2))
    sum_row.append(total_all)
    lines.append("sum".ljust(conv_id_w) + fmt_row(sum_row))

    out = "\n".join(lines)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
