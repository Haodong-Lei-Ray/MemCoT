#!/usr/bin/env python3
"""
将 LongMemEval/s 下的 500 个 json 文件等分为 10 份，输出到 s_split。

默认：
  src_dir  = /mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/s
  out_base = /mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/s_split

输出目录命名：
  0-49, 50-99, ..., 450-499
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SRC = Path("/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/s")
DEFAULT_OUT = Path("/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/s_split")
TOTAL = 500
SPLITS = 10
EACH = TOTAL // SPLITS  # 50


def split_500_to_10(src_dir: Path, out_base: Path) -> None:
    files = sorted([p for p in src_dir.glob("*.json") if p.is_file()])
    if len(files) != TOTAL:
        raise ValueError(f"{src_dir} 期望 {TOTAL} 个 .json，实际 {len(files)} 个")

    out_base.mkdir(parents=True, exist_ok=True)
    created_dirs: list[Path] = []

    for i in range(SPLITS):
        start = i * EACH
        end = start + EACH - 1
        part_dir = out_base / f"{start}-{end}"

        if part_dir.exists():
            shutil.rmtree(part_dir)
        part_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.append(part_dir)

        chunk = files[start : start + EACH]
        for fp in chunk:
            shutil.copy2(fp, part_dir / fp.name)

    print("拆分完成：")
    for d in created_dirs:
        n = len(list(d.glob("*.json")))
        print(f"{d} -> {n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=Path, default=DEFAULT_SRC, help="输入目录（500 个 json）")
    parser.add_argument("--out_base", type=Path, default=DEFAULT_OUT, help="输出根目录（生成 10 个子目录）")
    args = parser.parse_args()

    if not args.src_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {args.src_dir}")

    split_500_to_10(args.src_dir, args.out_base)


if __name__ == "__main__":
    main()

