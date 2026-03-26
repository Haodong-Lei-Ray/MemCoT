#!/usr/bin/env python3
"""
分析 conv26_react_lightrag_f1.json 中哪些问题在 skip.json 里，
以及使用 correct_answer 替代 golden_answer 后 F1 是否会提高。

Usage:
  python scripts/analyze_skip_vs_f1.py
"""

import json
import sys
from pathlib import Path

# Add project root for evaluation import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task_eval.evaluation import f1_score, f1

F1_JSON = PROJECT_ROOT / "module_version/version2/eval_output/Qwen/2.5-14B/qwen2.5-14B-Instruct_reactmem/conv-26/conv26_react_lightrag_f1.json"
SKIP_JSON = PROJECT_ROOT / "data/skip.json"


def compute_f1_for_item(prediction: str, answer: str, category: int) -> float:
    """按 eval_question_answering 的逻辑计算 F1"""
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

    # Build skip lookup: (sample_id, qa_id) -> item
    skip_by_key = {}
    for item in skip_data:
        if item.get("sample_id") != "conv-26":
            continue
        key = (item["sample_id"], item["qa_id"])
        skip_by_key[key] = item

    # f1.json structure: details[] with qa_id, question, gold_answer, prediction, event_search_f1
    details = f1_data.get("details", [])

    overlap = []
    higher_with_correct = []

    for d in details:
        qa_id = d.get("qa_id")
        question = d.get("question", "")
        gold_answer = d.get("gold_answer", "")
        prediction = d.get("prediction", "")
        category = d.get("category", 0)
        orig_f1 = d.get("event_search_f1", 0)

        key = ("conv-26", qa_id)
        if key not in skip_by_key:
            continue

        skip_item = skip_by_key[key]
        correct_answer = skip_item.get("correct_answer", "")

        # Compute F1 with correct_answer
        f1_with_correct = compute_f1_for_item(prediction, correct_answer, category)

        overlap.append({
            "qa_id": qa_id,
            "question": question,
            "golden_answer": gold_answer,
            "correct_answer": correct_answer,
            "prediction": prediction,
            "category": category,
            "error_type": skip_item.get("error_type", ""),
            "f1_golden": round(orig_f1, 4),
            "f1_correct": round(f1_with_correct, 4),
            "improvement": round(f1_with_correct - orig_f1, 4),
        })

        if f1_with_correct > orig_f1:
            higher_with_correct.append(overlap[-1])

    # Summary
    print("=" * 60)
    print("conv-26: 在 skip.json 中的问题")
    print("=" * 60)
    print(f"重叠数量: {len(overlap)} / {len(details)} (f1.json) , {len(skip_by_key)} (skip conv-26)")
    print()

    print("使用 correct_answer 后 F1 更高的项:")
    print("-" * 60)
    print(f"数量: {len(higher_with_correct)} / {len(overlap)}")
    print()

    for item in sorted(higher_with_correct, key=lambda x: -x["improvement"])[:20]:
        print(f"  qa_id={item['qa_id']} [{item['error_type']}] improvement={item['improvement']:.4f}")
        print(f"    golden F1: {item['f1_golden']} -> correct F1: {item['f1_correct']}")
        print(f"    Q: {item['question'][:70]}...")
        print(f"    golden: {item['golden_answer'][:60]}...")
        print(f"    correct: {item['correct_answer'][:60]}...")
        print(f"    pred: {item['prediction'][:60]}...")
        print()

    # Output full overlap to JSON
    out_path = PROJECT_ROOT / "data/skip_overlap_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "overlap_count": len(overlap),
            "higher_with_correct_count": len(higher_with_correct),
            "overlap": overlap,
            "higher_with_correct": higher_with_correct,
        }, f, ensure_ascii=False, indent=2)
    print(f"Full result saved to {out_path}")


if __name__ == "__main__":
    main()
