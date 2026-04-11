#!/usr/bin/env python3
"""
Extract per-conversation JSON files from locomo10.json.

Input:
  DMSMem/benchmark/locomo/data/locomo10.json

Output:
  DMSMem/benchmark/locomo/data/con/conv-*.json
"""

import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "benchmark" / "locomo" / "data" / "locomo10.json"
    out_dir = project_root / "benchmark" / "locomo" / "data" / "con"
    out_dir.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for item in data:
        sample_id = item.get("sample_id")
        conversation = item.get("conversation")
        if not sample_id or conversation is None:
            continue

        out_path = out_dir / f"{sample_id}.json"
        with out_path.open("w", encoding="utf-8") as fw:
            json.dump(conversation, fw, ensure_ascii=False, indent=2)
        count += 1

    print(f"wrote={count}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()

