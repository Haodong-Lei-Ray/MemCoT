#!/usr/bin/env python3
"""
Count QA per category (1-4) for each conv in locomo10.json.
Output: table to stdout and optionally to file.
"""
import json
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "locomo10.json"
CATEGORIES = [1, 2, 3, 4]


def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Aggregate: conv -> {cat: count}
    conv_counts = defaultdict(lambda: {c: 0 for c in CATEGORIES})

    for sample in data:
        sample_id = sample.get("sample_id", "")
        if not sample_id or not sample_id.startswith("conv-"):
            continue
        conv_num = sample_id.replace("conv-", "")
        try:
            conv_num_int = int(conv_num)
        except ValueError:
            continue

        qa_list = sample.get("qa", [])
        for qa in qa_list:
            cat = qa.get("category")
            if cat in CATEGORIES:
                conv_counts[conv_num_int][cat] += 1

    rows = sorted(conv_counts.keys())
    col_w = 10

    def fmt_row(vals):
        return "".join(str(v).rjust(col_w) for v in vals)

    lines = []
    lines.append("conv_id" + fmt_row(["1", "2", "3", "4", "sum"]))
    lines.append("-" * (6 + col_w * 5))

    totals = [0] * 5
    for conv_num in rows:
        cnt = conv_counts[conv_num]
        row_vals = [cnt[c] for c in CATEGORIES]
        row_sum = sum(row_vals)
        row_vals.append(row_sum)
        for i, v in enumerate(row_vals):
            totals[i] += v
        lines.append(str(conv_num).rjust(6) + fmt_row(row_vals))

    lines.append("sum".rjust(6) + fmt_row(totals))

    out = "\n".join(lines)
    print(out)


if __name__ == "__main__":
    main()
