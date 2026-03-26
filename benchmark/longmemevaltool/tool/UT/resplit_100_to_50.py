#!/usr/bin/env python3
import shutil
from pathlib import Path


def split_dir_to_two_50(src_dir: Path, out_base: Path) -> tuple[Path, Path]:
    files = sorted(src_dir.glob("*.json"))
    if len(files) != 100:
        raise ValueError(f"{src_dir} 文件数不是 100，而是 {len(files)}")

    start, end = map(int, src_dir.name.split("-"))
    mid = start + 49
    left_dir = out_base / f"{start}-{mid}"
    right_dir = out_base / f"{mid + 1}-{end}"

    if left_dir.exists():
        shutil.rmtree(left_dir)
    if right_dir.exists():
        shutil.rmtree(right_dir)
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    for f in files[:50]:
        shutil.copy2(f, left_dir / f.name)
    for f in files[50:]:
        shutil.copy2(f, right_dir / f.name)

    return left_dir, right_dir


def main():
    out_base = Path("/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/m_split")
    targets = ["100-199", "200-299", "300-399", "400-499"]
    created = []
    for name in targets:
        left, right = split_dir_to_two_50(out_base / name, out_base)
        created.extend([left, right])

    print("完成，新增/覆盖以下目录：")
    for d in created:
        n = len(list(d.glob("*.json")))
        print(f"{d} -> {n}")


if __name__ == "__main__":
    main()
