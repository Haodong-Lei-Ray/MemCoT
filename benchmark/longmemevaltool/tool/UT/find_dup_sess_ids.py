#!/usr/bin/env python3
"""
找出并（可选）修复 JSON 文件中的重复 `haystack_session_ids`。

默认仅 dry-run：打印哪些文件存在重复、以及按规则准备的去重效果（不覆盖原文件）。
若加 `--apply`：会直接写回原 JSON（可选带备份）。
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from collections import Counter

_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<i>\d+)$")


def _parse_session_id(session_id: str):
    """
    解析 session_id：
    - 若形如 {base}_{i}，返回 (base, i, True)
    - 若无数字后缀，返回 (session_id, None, False)

    兼容“可能出现名字_名字_i”的情况：
    - 若 base 由两个相同 token 拼成（如 a_a），视为 base=a（即 a_a_3 -> a_3）
    """
    m = _SUFFIX_RE.match(session_id)
    if not m:
        return session_id, None, False

    base = m.group("base")
    i = int(m.group("i"))

    # 只做简单启发式：base 只有两个 token 且相同，则合并为一个 token
    parts = base.split("_")
    if len(parts) == 2 and parts[0] == parts[1]:
        base = parts[0]

    return base, i, True


def dedupe_haystack_session_ids(ids):
    """
    按用户给的规则进行去重：
    - 遇到 {base}_{i} 重复：寻找最小的 k >= i+1，使得 {base}_{k} 未出现，则替换为 {base}_{k}
    - 遇到 {base} 重复：寻找最小的 k >= 1，使得 {base}_{k} 未出现，则替换为 {base}_{k}
    """
    used = set()
    out = []
    changed = 0

    for sid in ids:
        if sid not in used:
            out.append(sid)
            used.add(sid)
            continue

        base, i, has_suffix = _parse_session_id(sid)
        if has_suffix:
            candidate = i + 1
            while f"{base}_{candidate}" in used:
                candidate += 1
            new_sid = f"{base}_{candidate}"
        else:
            candidate = 1
            while f"{base}_{candidate}" in used:
                candidate += 1
            new_sid = f"{base}_{candidate}"

        out.append(new_sid)
        used.add(new_sid)
        changed += 1

    return out, changed


def main():
    # root = Path("/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/con20-250")
    default_root = Path("/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/m")

    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=str(default_root), help="JSON 所在目录")
    parser.add_argument("--apply", action="store_true", help="是否直接覆盖写回 JSON（谨慎使用）")
    parser.add_argument("--backup-dir", default=None, help="--apply 时备份目录（将原文件拷贝进去）")
    parser.add_argument("--output-dir", default=None, help="将修复后的 JSON 写到该目录（不覆盖原文件）")
    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        print(f"目录不存在: {root}")
        sys.exit(1)

    found = []
    json_files = sorted(root.glob("*.json"))
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                entry = json.load(fp)
        except Exception as e:
            print(f"SKIP {f.name}: {e}")
            continue
        ids = entry.get("haystack_session_ids", [])
        counts = Counter(ids)
        dupes = [k for k, v in counts.items() if v > 1]
        if dupes:
            new_ids, changed = dedupe_haystack_session_ids(ids)
            found.append((f.name, dupes, changed, ids, new_ids))

    if not found:
        print(f"未发现重复 session_id，共检查 {len(json_files)} 个文件")
        return

    print(f"以下 {len(found)} 个文件存在重复 haystack_session_ids（共检查 {len(json_files)} 个文件）：\n")
    for fname, dupes, changed, _, _ in found:
        print(f"  {fname}")
        for d in dupes:
            print(f"    重复: {d}")
        print(f"    去重准备替换次数: {changed}")
        print()

    if not args.apply and not args.output_dir:
        print("未开启写回（dry-run）。如果需要实际修复：")
        print("  --apply                覆盖原文件（可配 --backup-dir）")
        print("  --output-dir <dir>    写到新目录（不覆盖原文件）")
        return

    for item in found:
        fname, _, _, old_ids, new_ids = item
        src = root / fname

        if args.output_dir:
            dst_dir = Path(args.output_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / fname
            with open(src, "r", encoding="utf-8") as fp:
                entry = json.load(fp)
            entry["haystack_session_ids"] = new_ids
            with open(dst, "w", encoding="utf-8") as fp:
                json.dump(entry, fp, ensure_ascii=False, indent=2)
        elif args.apply:
            if args.backup_dir:
                backup_dir = Path(args.backup_dir)
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, backup_dir / fname)

            with open(src, "r", encoding="utf-8") as fp:
                entry = json.load(fp)
            entry["haystack_session_ids"] = new_ids
            with open(src, "w", encoding="utf-8") as fp:
                json.dump(entry, fp, ensure_ascii=False, indent=2)

    print("去重写回完成。")

if __name__ == "__main__":
    main()
