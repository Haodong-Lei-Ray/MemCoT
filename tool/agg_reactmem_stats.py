#!/usr/bin/env python3
"""
Aggregate F1, recall, count by category across all conv in qwen2.5-14B-Instruct_reactmem.
Output: stats_table.txt
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

CATEGORIES = [1, 2, 3, 4]

# 每个 conv 目标 QA 数，用于显示进度（now/sum）
TARGET_QA_PER_CONV = {
    26: 152,
    30: 81,
    41: 152,
    42: 199,
    43: 178,
    44: 123,
    47: 150,
    48: 191,
    49: 156,
    50: 158,
}

# Default values
DEFAULT_BASE = Path("/mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/eval_output/Qwen/2.5-14B/qwen2.5-14B-Instruct_reactmem")


def main(base_dir=None, out_file=None):
    # Set default values if not provided
    if base_dir is None:
        base_dir = DEFAULT_BASE
    else:
        base_dir = Path(base_dir)
    
    if out_file is None:
        out_file = base_dir / "stats_table.txt"
    else:
        out_file = Path(out_file)
    
    BASE = base_dir
    OUT_FILE = out_file
    
    # Find all conv*_react_lightrag_f1.json
    conv_data = {}
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        conv_name = d.name  # e.g. conv-26
        conv_num = conv_name.replace("conv-", "").replace("-", "")
        try:
            conv_num_int = int(conv_num)
        except ValueError:
            continue
        # File: conv26_react_lightrag_f1.json
        json_path = d / f"conv{conv_num}_react_lightrag_f1.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        conv_data[conv_num_int] = {
            "path": json_path,
            "data": data,
            "conv_name": conv_name,
        }

    if not conv_data:
        print(f"No JSON files found under {BASE}")
        return

    # 进度统计：当前已覆盖多少 QA，相对于 sum.txt 中的期望数量
    total_target = 0
    total_now = 0
    per_conv_progress = []  # (conv_num, now, target or None)
    for conv_num, info in sorted(conv_data.items()):
        target = TARGET_QA_PER_CONV.get(conv_num)
        d = info["data"]
        # 优先使用文件中的 n 字段，否则用 details 长度
        n_now = d.get("n")
        if n_now is None:
            n_now = len(d.get("details", []))

        per_conv_progress.append((conv_num, n_now, target))
        if target is not None:
            total_target += target
            total_now += n_now

    # 打印每个 conv 的进度
    print("Per-conv QA progress (now/sum):")
    for conv_num, n_now, target in per_conv_progress:
        if target is not None:
            print(f"  conv-{conv_num}: {n_now}/{target}")
        else:
            print(f"  conv-{conv_num}: {n_now}/?")

    # 打印总进度
    if total_target > 0:
        print(f"\nProgress (QA count): {total_now}/{total_target}")
    else:
        print("\nProgress (QA count): 未配置 TARGET_QA_PER_CONV，无法计算总目标数")

    # Aggregate by category per conv
    rows = sorted(conv_data.keys())
    f1_by_cat = {c: {} for c in CATEGORIES}
    recall_by_cat = {c: {} for c in CATEGORIES}
    count_by_cat = {c: {} for c in CATEGORIES}

    for conv_num in rows:
        d = conv_data[conv_num]["data"]
        details = d.get("details", [])
        cat_f1 = defaultdict(list)
        cat_recall = defaultdict(list)
        for qa in details:
            cat = qa.get("category")
            if cat is None:
                continue
            if "event_search_f1" in qa:
                cat_f1[cat].append(qa["event_search_f1"])
            if "recall" in qa:
                cat_recall[cat].append(qa["recall"])

        for c in CATEGORIES:
            if cat_f1[c]:
                f1_by_cat[c][conv_num] = sum(cat_f1[c]) / len(cat_f1[c])
            else:
                f1_by_cat[c][conv_num] = 0.0
            if cat_recall[c]:
                recall_by_cat[c][conv_num] = sum(cat_recall[c]) / len(cat_recall[c])
            else:
                recall_by_cat[c][conv_num] = 0.0
            count_by_cat[c][conv_num] = len(cat_f1[c])

    # Build tables
    lines = []
    col_w = 12  # Column width for data columns
    conv_id_w = 8  # Width for conv_id column

    def fmt_row(vals):
        return "".join(str(v).rjust(col_w) for v in vals)
    
    def fmt_header(vals):
        """Format header with right alignment to match data alignment"""
        return "".join(str(v).rjust(col_w) for v in vals)

    # ------ Count table (compute first for weighted avg) ------
    lines.append("Count (数量)")
    lines.append("conv_id".rjust(conv_id_w) + fmt_header(["Multi hop", "Temporal", "Open-domain", "Single hop", "sum"]))
    lines.append("-" * (conv_id_w + col_w * 5))

    count_sums = [0] * 5
    for conv_num in rows:
        row_vals = []
        for i, c in enumerate(CATEGORIES):
            v = count_by_cat[c].get(conv_num, 0)
            row_vals.append(v)
            count_sums[i] += v
        row_sum = sum(count_by_cat[c].get(conv_num, 0) for c in CATEGORIES)
        row_vals.append(row_sum)
        count_sums[4] += row_sum
        lines.append(str(conv_num).rjust(conv_id_w) + fmt_row(row_vals))

    lines.append("sum".rjust(conv_id_w) + fmt_row(count_sums))
    lines.append("")

    # Weighted avg: sum(count * value) / sum(count) for each category
    def weighted_sum_row(metric_by_cat):
        sum_row = []
        for i, c in enumerate(CATEGORIES):
            numer = sum(
                count_by_cat[c].get(conv_num, 0) * metric_by_cat[c].get(conv_num, 0.0)
                for conv_num in rows
            )
            denom = count_sums[i]
            sum_row.append(round(numer / denom, 4) if denom > 0 else "-")
        # avg column: sum(count_c * value_c) / sum(count_c) over all convs
        numer = 0.0
        denom = 0
        for conv_num in rows:
            for c in CATEGORIES:
                cnt = count_by_cat[c].get(conv_num, 0)
                val = metric_by_cat[c].get(conv_num, 0.0)
                numer += cnt * val
                denom += cnt
        sum_row.append(round(numer / denom, 4) if denom > 0 else "-")
        return sum_row

    # ------ F1 table ------
    lines.append("F1")
    lines.append("conv_id".rjust(conv_id_w) + fmt_header(["Multi hop", "Temporal", "Open-domain", "Single hop", "avg"]))
    lines.append("-" * (conv_id_w + col_w * 5))

    for conv_num in rows:
        row_vals = []
        numer, denom = 0.0, 0
        for i, c in enumerate(CATEGORIES):
            v = f1_by_cat[c].get(conv_num, 0.0)
            cnt = count_by_cat[c].get(conv_num, 0)
            row_vals.append(round(v, 4) if v else "-")
            numer += cnt * (v or 0)
            denom += cnt
        avg = round(numer / denom, 4) if denom > 0 else "-"
        row_vals.append(avg)
        lines.append(str(conv_num).rjust(conv_id_w) + fmt_row(row_vals))

    lines.append("sum".rjust(conv_id_w) + fmt_row(weighted_sum_row(f1_by_cat)))
    lines.append("")

    # ------ Recall table ------
    lines.append("Recall")
    lines.append("conv_id".rjust(conv_id_w) + fmt_header(["Multi hop", "Temporal", "Open-domain", "Single hop", "avg"]))
    lines.append("-" * (conv_id_w + col_w * 5))

    for conv_num in rows:
        row_vals = []
        numer, denom = 0.0, 0
        for i, c in enumerate(CATEGORIES):
            v = recall_by_cat[c].get(conv_num, 0.0)
            cnt = count_by_cat[c].get(conv_num, 0)
            row_vals.append(round(v, 4) if v else "-")
            numer += cnt * (v or 0)
            denom += cnt
        avg = round(numer / denom, 4) if denom > 0 else "-"
        row_vals.append(avg)
        lines.append(str(conv_num).rjust(conv_id_w) + fmt_row(row_vals))

    lines.append("sum".rjust(conv_id_w) + fmt_row(weighted_sum_row(recall_by_cat)))

    out = "\n".join(lines)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate F1, recall, count by category across all conv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Interactive mode (will prompt for base directory)
  python3 {__file__}
  
  # Specify output file only
  python3 {__file__} --out-file /path/to/output.txt
        """
    )
    parser.add_argument(
        "--out-file", "-o",
        type=str,
        default=None,
        help="Output file path (default: <base-dir>/stats_table.txt)"
    )
    
    args = parser.parse_args()
    
    # Interactive input for base directory
    print(f"Enter base directory (press Enter for default):")
    print(f"Default: {DEFAULT_BASE}")
    user_input = input("Base directory: ").strip()
    
    if user_input:
        base_dir = user_input
    else:
        base_dir = None  # Will use default in main()
    
    main(base_dir=base_dir, out_file=args.out_file)
