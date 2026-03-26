#!/usr/bin/env python3
"""
检查 LightRAG 构建失败时对应的输入 json 是否还能被正常 json.load()。

会从日志里抽取形如：
  [x/100] FAIL: 0015_488d3006.json - ...
的 question_id：0015_488d3006

然后检查：
  ${m_split_dir}/<question_id>.json

输出：
  - 命中失败任务的去重后 question_id 列表数量
  - json.load() 成功/失败原因统计
  - 明细报告（TSV）
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


FAIL_QID_RE = re.compile(r"FAIL:\s+([0-9]{4}_[0-9a-f]+)\.json\b")


def extract_failed_question_ids(log_path: Path) -> list[str]:
    if not log_path.exists():
        raise FileNotFoundError(f"log 不存在: {log_path}")

    ids: set[str] = set()
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = FAIL_QID_RE.search(line)
            if not m:
                continue
            ids.add(m.group(1))
    return sorted(ids)


def check_one_json(fp: Path) -> tuple[str, str, int]:
    """
    返回:
      (status, message, file_size_bytes)
    status in {"ok","empty","json_decode_error","unicode_decode_error","missing","other_error"}
    """
    if not fp.exists():
        return ("missing", "file not found", 0)

    try:
        sz = fp.stat().st_size
    except OSError:
        sz = 0

    if sz == 0:
        return ("empty", "file size == 0", sz)

    try:
        with open(fp, "r", encoding="utf-8") as f:
            json.load(f)
        return ("ok", "ok", sz)
    except json.JSONDecodeError as e:
        return ("json_decode_error", str(e), sz)
    except UnicodeDecodeError as e:
        return ("unicode_decode_error", str(e), sz)
    except Exception as e:
        return ("other_error", f"{type(e).__name__}: {e}", sz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="run_build_*.out 日志文件路径",
    )
    parser.add_argument(
        "--m_split_dir",
        type=Path,
        required=True,
        help="m_split/0-99 目录路径（放 question_id json 的目录）",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="输出 TSV 报告路径（默认：与 log 同目录同名）",
    )
    args = parser.parse_args()

    log_path: Path = args.log
    m_split_dir: Path = args.m_split_dir

    if not m_split_dir.exists():
        raise FileNotFoundError(f"m_split_dir 不存在: {m_split_dir}")

    qids = extract_failed_question_ids(log_path)
    print(f"从日志抽取到 {len(qids)} 个失败 question_id（已去重）。")
    print(f"将逐个检查: {m_split_dir}/<question_id>.json")

    if args.report is None:
        args.report = log_path.with_suffix("") .with_name(log_path.stem + "_jsonload_report.tsv")
    report_path: Path = args.report

    counts: dict[str, int] = {}
    details: list[tuple[str, str, int]] = []

    # TSV header: question_id \t status \t message \t bytes
    for qid in qids:
        fp = m_split_dir / f"{qid}.json"
        status, msg, sz = check_one_json(fp)
        counts[status] = counts.get(status, 0) + 1
        details.append((qid, status, msg, sz))
        # 进度：每检查一个就打印一行（73 个规模很小）
        print(f"  {qid}: {status} ({sz} bytes)")

    print("\n统计结果:")
    for k in sorted(counts.keys()):
        print(f"  - {k}: {counts[k]}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("question_id\tstatus\tmessage\tbytes\n")
        for qid, status, msg, sz in details:
            # TSV 转义：把换行和 tab 干掉，避免格式破坏
            msg_clean = str(msg).replace("\t", " ").replace("\n", "\\n")
            f.write(f"{qid}\t{status}\t{msg_clean}\t{sz}\n")

    print(f"\n报告已写入: {report_path}")


if __name__ == "__main__":
    main()

