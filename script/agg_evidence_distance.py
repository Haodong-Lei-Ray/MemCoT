#!/usr/bin/env python3
"""
Aggregate evidence distance from conv*_react_lightrag_f1.json files.

Distance between two evidence points:
  D{x1}:{y1} and D{x2}:{y2}
  dis = |y1 - y2| + |x1 - x2| * 30

For one QA item:
  P1 = final_evidence, P2 = gold_evidence
  For each p1_i in P1:
    d_i = average_{p2_j in P2} dis(p1_i, p2_j)
  distance(P1, P2) = average_{p1_i in P1} d_i

Rules:
  - If P1 is empty, skip this QA.
"""

import argparse
import json
import re
from pathlib import Path


EVIDENCE_RE = re.compile(r"^D(\d+):(\d+)$")


def parse_evidence_point(evidence):
    """Parse 'D23:16' -> (23, 16). Return None if invalid."""
    if not isinstance(evidence, str):
        return None
    m = EVIDENCE_RE.match(evidence.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def pair_distance(p1, p2):
    """dis = |y1-y2| + |x1-x2|*30"""
    x1, y1 = p1
    x2, y2 = p2
    return abs(y1 - y2) + abs(x1 - x2) * 30


def qa_distance(final_evidence, gold_evidence):
    """
    Calculate distance(P1, P2) for a single QA.
    Return None if this QA should be skipped.
    """
    p1 = [parse_evidence_point(x) for x in (final_evidence or [])]
    p2 = [parse_evidence_point(x) for x in (gold_evidence or [])]
    p1 = [x for x in p1 if x is not None]
    p2 = [x for x in p2 if x is not None]

    # User requirement: skip if P1 empty.
    if not p1:
        return None
    # If P2 empty, no meaningful distance can be computed.
    if not p2:
        return None

    d_values = []
    for p1_i in p1:
        distances = [pair_distance(p1_i, p2_j) for p2_j in p2]
        d_i = sum(distances) / len(distances)
        d_values.append(d_i)

    return sum(d_values) / len(d_values)


def find_conv_jsons(base_dir):
    """Find files like conv-26/conv26_react_lightrag_f1.json"""
    files = []
    for conv_dir in sorted(base_dir.iterdir()):
        if not conv_dir.is_dir():
            continue
        conv_name = conv_dir.name
        if not conv_name.startswith("conv-"):
            continue
        conv_num = conv_name.replace("conv-", "")
        json_path = conv_dir / f"conv{conv_num}_react_lightrag_f1.json"
        if json_path.exists():
            files.append((conv_name, json_path))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Compute evidence distance for each conv*_react_lightrag_f1.json"
    )
    parser.add_argument(
        "base_dir",
        help="Directory containing conv-* subdirs, e.g. .../eval_output/Qwen/2.5-14B/F1",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output txt file path (default: <base_dir>/evidence_distance_stats.txt)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"base_dir not found: {base_dir}")

    out_path = Path(args.out) if args.out else (base_dir / "evidence_distance_stats.txt")

    conv_files = find_conv_jsons(base_dir)
    if not conv_files:
        raise SystemExit(f"No conv*_react_lightrag_f1.json found under: {base_dir}")

    lines = []
    lines.append(f"Base dir: {base_dir}")
    lines.append("")
    lines.append(
        "conv_id".rjust(10)
        + "used_qas".rjust(12)
        + "skip_p1_empty".rjust(16)
        + "skip_p2_empty".rjust(16)
        + "avg_distance".rjust(16)
    )
    lines.append("-" * 70)

    global_distances = []
    total_used = 0
    total_skip_p1 = 0
    total_skip_p2 = 0

    for conv_name, json_path in conv_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        details = data.get("details", [])

        conv_distances = []
        skip_p1 = 0
        skip_p2 = 0

        for qa in details:
            p1 = qa.get("final_evidence", [])
            p2 = qa.get("gold_evidence", [])

            # Count skip reasons for observability
            p1_valid = [parse_evidence_point(x) for x in (p1 or [])]
            p1_valid = [x for x in p1_valid if x is not None]
            if not p1_valid:
                skip_p1 += 1
                continue
            p2_valid = [parse_evidence_point(x) for x in (p2 or [])]
            p2_valid = [x for x in p2_valid if x is not None]
            if not p2_valid:
                skip_p2 += 1
                continue

            d = qa_distance(p1, p2)
            if d is not None:
                conv_distances.append(d)
                global_distances.append(d)

        used = len(conv_distances)
        conv_avg = (sum(conv_distances) / used) if used > 0 else None

        total_used += used
        total_skip_p1 += skip_p1
        total_skip_p2 += skip_p2

        lines.append(
            conv_name.rjust(10)
            + str(used).rjust(12)
            + str(skip_p1).rjust(16)
            + str(skip_p2).rjust(16)
            + (f"{conv_avg:.4f}" if conv_avg is not None else "N/A").rjust(16)
        )

    overall = (sum(global_distances) / len(global_distances)) if global_distances else None
    lines.append("-" * 70)
    lines.append(
        "TOTAL".rjust(10)
        + str(total_used).rjust(12)
        + str(total_skip_p1).rjust(16)
        + str(total_skip_p2).rjust(16)
        + (f"{overall:.4f}" if overall is not None else "N/A").rjust(16)
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {out_path}")
    if overall is not None:
        print(f"Overall avg distance: {overall:.4f} (used_qas={total_used})")
    else:
        print("Overall avg distance: N/A (no valid QA pairs)")


if __name__ == "__main__":
    main()
