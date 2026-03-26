#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 conv-26 debug 结果中 trajectory 里的 query_queue：
1. query_queue 为空列表 [] 的情况
2. query_queue 中存在长度为 2 的字符串（例如 "Wh"）的情况

用法：
  cd /mnt/petrelfs/leihaodong/ICML/locomo
  python3 ICML/locomo/module_version/version2/script/debug/check_empty_or_short_query_queue.py
"""

import json
from pathlib import Path


def bad_query_queue(qq) -> bool:
    """判定 query_queue 是否“异常”：为空，或包含长度为 2 的字符串元素。"""
    # 条件1：完全空列表 []
    if isinstance(qq, list) and len(qq) == 0:
        return True
    # 条件2：列表中存在长度为2的字符串元素
    if isinstance(qq, list):
        for x in qq:
            if isinstance(x, str) and len(x) == 2:
                return True
    return False


def file_has_bad_query_queue(path: Path) -> bool:
    """检查单个 result.json 是否存在满足条件的 query_queue。"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    traj = data.get("trajectory", [])
    for step in traj:
        act = step.get("Act", {})
        qq = act.get("query_queue", None)
        if bad_query_queue(qq):
            return True
    return False


def build_f1_map(eval_path: Path) -> dict[int, float]:
    """从 convXX_react_lightrag_f1.json 中构建 qa_id -> event_search_f1 映射。"""
    with eval_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    f1_map = {}
    for item in data.get("details", []):
        qa_id = item.get("qa_id")
        if qa_id is None:
            continue
        f1_map[int(qa_id)] = item.get("event_search_f1", 0.0)
    return f1_map


def check_one_conv(conv_dir: Path) -> None:
    """检查单个 conv-i 目录下的 debug 和 eval。"""
    conv_name = conv_dir.name  # 例如 "conv-26"
    # 提取数字部分，如 "26"
    try:
        conv_num = conv_name.split("-")[1]
    except IndexError:
        print(f"跳过目录（命名不符合 conv-XX）：{conv_dir}")
        return

    debug_root = conv_dir / "debug"
    eval_path = conv_dir / f"conv{conv_num}_react_lightrag_f1.json"

    if not debug_root.exists():
        print(f"[{conv_name}] debug 目录不存在: {debug_root}")
        return
    if not eval_path.exists():
        print(f"[{conv_name}] eval 文件不存在: {eval_path}")
        return

    print(f"\n===== {conv_name} =====")

    # 读取 eval 文件中的 F1 映射
    f1_map = build_f1_map(eval_path)

    # 遍历 debug 下所有 qa_*
    ids_with_bad_qq = []
    for qa_dir in sorted(debug_root.glob("qa_*")):
        if not qa_dir.is_dir():
            continue
        result_file = qa_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            qa_id = int(qa_dir.name.split("_")[1])
        except Exception:
            continue
        if file_has_bad_query_queue(result_file):
            ids_with_bad_qq.append(qa_id)

    ids_with_bad_qq = sorted(ids_with_bad_qq)
    print("满足条件 (query_queue 为空 或 含长度为2字符串) 的 qa_id 列表:")
    print(ids_with_bad_qq)

    # 打印这些 qa_id 在 eval 文件中的 event_search_f1
    print("对应的 event_search_f1：")
    for qa_id in ids_with_bad_qq:
        f1 = f1_map.get(qa_id, None)
        if f1 is None:
            print(f"  qa_id={qa_id}: 未在 eval 文件中找到")
        else:
            print(f"  qa_id={qa_id}: event_search_f1={f1}")


def main():
    """
    遍历 middle-scale-2 下所有 conv-* 目录，逐个检查：
      1) debug/qa_*/result.json 中 query_queue 是否为空或含长度为2的字符串
      2) 打印这些 qa_id 在对应 convXX_react_lightrag_f1.json 中的 event_search_f1
    """
    base_dir = Path(
        "ICML/locomo/module_version/version2/eval_output/"
        "Qwen/2.5-14B/ablation/middle-scale-2"
    )
    if not base_dir.exists():
        print(f"Error: 基础目录不存在: {base_dir}")
        return

    conv_dirs = sorted(d for d in base_dir.glob("conv-*") if d.is_dir())
    if not conv_dirs:
        print(f"在 {base_dir} 下未找到任何 conv-* 目录")
        return

    for conv_dir in conv_dirs:
        check_one_conv(conv_dir)


if __name__ == "__main__":
    main()

