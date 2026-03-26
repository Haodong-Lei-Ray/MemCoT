#!/usr/bin/env python3
"""
比对 conv30_react_lightrag_f1.json 与 data/skip/conv-30.json，
分析在 skip 中的问题，以及使用 correct_answer 后 F1 变化。
结果输出到 data/skip/com/

Usage:
  python scripts/analyze_skip_vs_f1_conv30.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task_eval.evaluation import f1_score, f1

F1_JSON = PROJECT_ROOT / "module_version/version2/eval_output/Qwen/2.5-14B/qwen2.5-14B-Instruct_reactmem/conv-30/conv30_react_lightrag_f1.json"
SKIP_JSON = PROJECT_ROOT / "data/skip/conv-30.json"
OUT_DIR = PROJECT_ROOT / "data/skip/com"


def compute_f1_for_item(prediction: str, answer: str, category: int) -> float:
    prediction = str(prediction).strip()
    answer = str(answer).strip()
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return f1_score(prediction, answer)
    if category == 1:
        return f1(prediction, answer)
    return 0.0


def main():
    f1_data = json.load(open(F1_JSON))
    skip_data = json.load(open(SKIP_JSON))

    skip_by_qa = {item["qa_id"]: item for item in skip_data}

    details = f1_data.get("details", [])

    overlap = []
    higher_with_correct = []

    for d in details:
        qa_id = d.get("qa_id")
        if qa_id not in skip_by_qa:
            continue

        skip_item = skip_by_qa[qa_id]
        gold_answer = d.get("gold_answer", "")
        prediction = d.get("prediction", "")
        category = d.get("category", 0)
        orig_f1 = d.get("event_search_f1", 0)
        correct_answer = skip_item.get("correct_answer", "")

        f1_with_correct = compute_f1_for_item(prediction, correct_answer, category)

        item = {
            "qa_id": qa_id,
            "question": d.get("question", ""),
            "golden_answer": gold_answer,
            "correct_answer": correct_answer,
            "prediction": prediction,
            "category": category,
            "error_type": skip_item.get("error_type", ""),
            "f1_golden": round(orig_f1, 4),
            "f1_correct": round(f1_with_correct, 4),
            "improvement": round(f1_with_correct - orig_f1, 4),
        }
        overlap.append(item)
        if f1_with_correct > orig_f1:
            higher_with_correct.append(item)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "sample_id": "conv-30",
        "overlap_count": len(overlap),
        "higher_with_correct_count": len(higher_with_correct),
        "overlap": overlap,
        "higher_with_correct": higher_with_correct,
    }

    out_path = OUT_DIR / "conv-30_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("conv-30: 在 skip 中的问题")
    print("=" * 60)
    print(f"重叠数量: {len(overlap)} / {len(details)} (f1) , {len(skip_data)} (skip)")
    print(f"使用 correct_answer 后 F1 更高: {len(higher_with_correct)} / {len(overlap)}")
    print(f"输出: {out_path}")
    for item in sorted(higher_with_correct, key=lambda x: -x["improvement"]):
        print(f"  qa_id={item['qa_id']} [{item['error_type']}] +{item['improvement']:.4f}")


if __name__ == "__main__":
    main()
