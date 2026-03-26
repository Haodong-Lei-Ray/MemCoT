#!/usr/bin/env python3
"""
将 skip.json 按 sample_id 拆分成独立 JSON 文件，输出到 data/skip/temp/

Usage:
  python scripts/split_skip_by_sample.py
"""

import json
from pathlib import Path

SKIP_JSON = Path(__file__).parent.parent / "data" / "skip.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "skip" / "temp"


def main():
    data = json.load(open(SKIP_JSON, encoding="utf-8"))

    by_sample = {}
    for item in data:
        sid = item.get("sample_id", "unknown")
        by_sample.setdefault(sid, []).append(item)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample_id, items in sorted(by_sample.items()):
        out_path = OUT_DIR / f"{sample_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  {sample_id}: {len(items)} items -> {out_path}")

    print(f"\nDone. {len(by_sample)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
