#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 debug/qa_* 目录下的结果，重新构建单个 conv 的 *_react_lightrag_f1.json。

逻辑与 eval_react_lightrag.py 相同：
- prediction: 来自 debug/qa_k/result.json 中的 "answer"
- final_evidence: 来自 debug/qa_k/result.json 中的 "final_evidence"
- gold_answer / category / gold_evidence: 来自原始 eval 文件（或 locomo10.json）
- F1: 使用 eval_react_lightrag.compute_f1_for_qa
- recall: 与 eval_react_lightrag 一致

可以针对任意 conv-i 使用，例如：
  python3 rebuild_conv_from_debug.py --conv-dir /.../ablation/middle-scale-2/conv-26
"""

import json
import sys
import argparse
from pathlib import Path

# 确保可以从 version2 根目录导入 eval_react_lightrag
CURRENT_DIR = Path(__file__).resolve().parent
VERSION2_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(VERSION2_DIR))

from eval_react_lightrag import compute_f1_for_qa, DATA_PATH as LOCOMO_DATA_PATH


def _load_sample_from_locomo(sample_id: str) -> dict:
    """从 locomo10.json 加载指定 sample_id 的样本（包含 qa 列表）"""
    with open(LOCOMO_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for s in data:
        if s.get("sample_id") == sample_id:
            return s
    raise ValueError(f"Sample {sample_id} not found in {LOCOMO_DATA_PATH}")


def _build_eval_from_locomo_and_debug(
    sample_id: str,
    conv_dir: Path,
    debug_root: Path,
) -> dict:
    """
    eval_file 不存在时，从 locomo10.json + debug 目录构建一份新的 eval 结构。
    仅依赖：
      - locomo10.json 中的 qa 列表（问题 / 答案 / 类别 / evidence）
      - debug/qa_k/result.json 中的 answer / final_evidence
    """
    print(f"从 locomo10.json 和 {debug_root} 构建新的 eval 数据: sample_id={sample_id}")
    sample = _load_sample_from_locomo(sample_id)
    qa_all = sample.get("qa", [])

    # 找出有哪些 qa_k 有 debug 结果
    available_qas = []
    for qa_dir in sorted(debug_root.glob("qa_*")):
        if not qa_dir.is_dir():
            continue
        try:
            k = int(qa_dir.name.split("_")[1])
        except Exception:
            continue
        available_qas.append(k)

    if not available_qas:
        raise RuntimeError(f"{debug_root} 下没有 qa_* debug 结果，无法重建")

    details = []
    for k in sorted(available_qas):
        idx = k - 1
        if idx < 0 or idx >= len(qa_all):
            print(f"  警告: qa_{k} 超出 locomo10.json 的 qa 范围，跳过")
            continue
        qa = qa_all[idx]
        question = qa.get("question", "")
        # 与 eval_react_lightrag 一致：优先 answer，否则 adversarial_answer
        gold_answer = qa.get("answer", qa.get("adversarial_answer", ""))
        category = qa.get("category", 0)

        gold_evidence = qa.get("evidence", [])
        if isinstance(gold_evidence, str):
            gold_evidence = [gold_evidence] if gold_evidence else []
        elif not isinstance(gold_evidence, list):
            gold_evidence = []

        debug_dir = debug_root / f"qa_{k}"
        result_path = debug_dir / "result.json"
        prediction = ""
        final_evidence = []
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            prediction = res.get("answer", "")
            final_evidence = res.get("final_evidence", [])

        f1 = compute_f1_for_qa(prediction, gold_answer, category)

        if gold_evidence:
            pred_set = set(str(x) for x in final_evidence)
            gold_set = set(str(x) for x in gold_evidence)
            recall = round(len(pred_set & gold_set) / len(gold_set), 3)
        else:
            recall = 1.0

        details.append(
            {
                "qa_id": k,
                "question": question,
                "gold_answer": str(gold_answer),
                "prediction": prediction,
                "category": category,
                "event_search_f1": f1,
                "recall": recall,
                "final_evidence": final_evidence,
                "gold_evidence": gold_evidence,
            }
        )

    n = len(details)
    f1_scores = [d.get("event_search_f1", 0.0) for d in details]
    recall_scores = [d.get("recall", 1.0) for d in details]
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    eval_data = {
        "sample_id": sample_id,
        "model": "",  # 无法从 debug 恢复，留空或后续填充
        "method": "react_lightrag",
        "n": n,
        "mean_f1": round(mean_f1, 4),
        "mean_recall": round(mean_recall, 4),
        "details": details,
    }
    return eval_data


def rebuild_conv_from_debug(
    sample_id: str,
    eval_file: Path,
    debug_root: Path,
    output_file: Path | None = None,
):
    """
    根据 debug 目录重建单个 conv 的 *_react_lightrag_f1.json。

    Args:
        sample_id: 如 'conv-26'
        eval_file: 现有的 eval json（用于提供 question / gold_answer / category / gold_evidence）
        debug_root: debug 目录，例如 .../conv-26/debug
        output_file: 输出文件路径，默认覆盖 eval_file
    """
    print(f"加载 eval 文件: {eval_file}")
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    details = eval_data.get("details", [])
    if not details:
        print(" eval 文件中没有 details，直接返回")
        return

    new_details = []

    for item in details:
        qa_id = item.get("qa_id")
        question = item.get("question", "")
        gold_answer = item.get("gold_answer", item.get("answer", ""))
        category = item.get("category", 0)
        gold_evidence = item.get("gold_evidence", [])

        debug_dir = debug_root / f"qa_{qa_id}"
        result_path = debug_dir / "result.json"

        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                res = json.load(f)
            prediction = res.get("answer", item.get("prediction", ""))
            final_evidence = res.get("final_evidence", item.get("final_evidence", []))
        else:
            prediction = item.get("prediction", "")
            final_evidence = item.get("final_evidence", [])

        f1 = compute_f1_for_qa(prediction, gold_answer, category)

        if gold_evidence:
            pred_set = set(str(x) for x in final_evidence)
            gold_set = set(str(x) for x in gold_evidence)
            recall = round(len(pred_set & gold_set) / len(gold_set), 3)
        else:
            recall = 1.0

        new_item = {
            "qa_id": qa_id,
            "question": question,
            "gold_answer": str(gold_answer),
            "prediction": prediction,
            "category": category,
            "event_search_f1": f1,
            "recall": recall,
            "final_evidence": final_evidence,
            "gold_evidence": gold_evidence,
        }
        new_details.append(new_item)

    # 重新计算统计值
    n = len(new_details)
    f1_scores = [d.get("event_search_f1", 0.0) for d in new_details]
    recall_scores = [d.get("recall", 1.0) for d in new_details]
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    eval_data["sample_id"] = sample_id
    eval_data["n"] = n
    eval_data["mean_f1"] = round(mean_f1, 4)
    eval_data["mean_recall"] = round(mean_recall, 4)
    eval_data["details"] = new_details

    if output_file is None:
        output_file = eval_file

    print(f"\n写回文件: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"  n = {n}, mean_f1 = {eval_data['mean_f1']}, mean_recall = {eval_data['mean_recall']}")


def main():
    """
    通用主程序：针对任意 conv-i 目录重建 eval 文件。

    典型用法：
      cd module_version/version2
      python3 script/rebuild_conv_from_debug.py \\
        --conv-dir /mnt/.../ablation/middle-scale-2/conv-26

    参数说明：
      --conv-dir:  指向某个 conv-i 结果目录（里面包含 debug/ 子目录）
      --eval-file: 可选，eval 文件路径；默认在 conv-dir 下自动推断
      --sample-id: 可选，样本 ID，默认取 conv-dir 的目录名（例如 conv-26）
    """

    parser = argparse.ArgumentParser(
        description="Rebuild conv-i *_react_lightrag_f1.json from debug/qa_* results"
    )
    parser.add_argument(
        "--conv-dir",
        required=True,
        help="conv-i 结果目录路径，例如 /.../ablation/middle-scale-2/conv-26",
    )
    parser.add_argument(
        "--eval-file",
        help="eval JSON 文件路径（默认在 conv-dir 下自动推断）",
    )
    parser.add_argument(
        "--sample-id",
        help="样本 ID（默认取 conv 目录名，如 conv-26）",
    )

    args = parser.parse_args()

    conv_dir = Path(args.conv_dir)
    if not conv_dir.exists():
        print(f"Error: conv-dir {conv_dir} 不存在")
        return

    sample_id = args.sample_id or conv_dir.name

    # 确定 eval_file
    if args.eval_file:
        eval_file = Path(args.eval_file)
    else:
        # 默认命名：conv26_react_lightrag_f1.json 或 conv26_react_lightrag_f1 copy.json
        stem = sample_id.replace("-", "")
        candidate = conv_dir / f"{stem}_react_lightrag_f1.json"
        if candidate.exists():
            eval_file = candidate
        else:
            candidate_copy = conv_dir / f"{stem}_react_lightrag_f1 copy.json"
            eval_file = candidate_copy

    debug_root = conv_dir / "debug"

    if not eval_file.exists():
        # 如果 eval_file 不存在，则完全从 locomo10.json + debug 构建一份新的 eval 数据
        print(f"eval_file {eval_file} 不存在，将从 locomo10.json + debug 直接重建")
        eval_data = _build_eval_from_locomo_and_debug(sample_id, conv_dir, debug_root)
        print(f"\n写回文件: {eval_file}")
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(
            f"  n = {eval_data['n']}, mean_f1 = {eval_data['mean_f1']}, mean_recall = {eval_data['mean_recall']}"
        )
        return
    if not debug_root.exists():
        print(f"Error: debug 目录 {debug_root} 不存在")
        return

    rebuild_conv_from_debug(sample_id, eval_file, debug_root, output_file=eval_file)


if __name__ == "__main__":
    main()

