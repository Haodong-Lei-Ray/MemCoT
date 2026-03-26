#!/usr/bin/env python3
"""
计算 QA 结果 JSON 的 F1 分数。
Usage:
  python task_eval/calc_f1.py /path/to/result.json
  python task_eval/calc_f1.py /path/to/result.json --prediction-key "Qwen/Qwen2.5-14B-Instruct_lightrag_top_10_prediction"
  srun -p DataFrontier_Knowledge --gres=gpu:0 python task_eval/calc_f1.py /path/to/result.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_eval.evaluation import eval_question_answering


def _find_prediction_key(qa_item: dict) -> str:
    """自动查找 *_prediction 的 key。"""
    for k in qa_item.keys():
        if k.endswith("_prediction") and not k.endswith("_prediction_context"):
            return k
    return ""


def calc_f1(in_file: str, prediction_key: str = "", out_file: str = "") -> dict:
    data = json.load(open(in_file, encoding="utf-8"))
    if isinstance(data, list):
        samples = data
    else:
        samples = [data]

    all_qa = []
    for s in samples:
        all_qa.extend(s.get("qa", []))

    if not all_qa:
        raise ValueError("No QA items found in %s" % in_file)

    if not prediction_key:
        prediction_key = _find_prediction_key(all_qa[0])
    if not prediction_key:
        raise ValueError("Cannot find prediction key (expect *(_prediction)). Keys: %s" % list(all_qa[0].keys()))

    f1_scores, _, recall = eval_question_answering(all_qa, eval_key=prediction_key)
    n = len(f1_scores)
    mean_f1 = sum(f1_scores) / n if n else 0

    # by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for i, qa in enumerate(all_qa):
        if i < len(f1_scores):
            by_cat[qa.get("category", 0)].append(f1_scores[i])
    cat_stats = {k: (sum(v) / len(v) if v else 0, len(v)) for k, v in by_cat.items()}

    result = {
        "file": in_file,
        "prediction_key": prediction_key,
        "n": n,
        "mean_f1": round(mean_f1, 4),
        "f1_scores": [round(x, 4) for x in f1_scores],
        "by_category": {str(k): {"mean_f1": round(v[0], 4), "count": v[1]} for k, v in sorted(cat_stats.items())},
    }
    if recall:
        result["mean_recall"] = round(sum(recall) / len(recall), 4)

    print("=== F1 Stats: %s ===" % in_file)
    print("prediction_key: %s" % prediction_key)
    print("n: %d" % n)
    print("mean_f1: %.4f" % mean_f1)
    for k in sorted(cat_stats.keys()):
        m, c = cat_stats[k]
        print("  category %s: mean_f1=%.4f, count=%d" % (k, m, c))
    if recall:
        print("mean_recall: %.4f" % (sum(recall) / len(recall)))

    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("Saved to %s" % out_file)

    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate F1 for QA result JSON")
    parser.add_argument("in_file", type=str, help="Input JSON (e.g. conv26_lightrag_top10_qa_27.json)")
    parser.add_argument("--prediction-key", "-k", type=str, default="", help="Prediction key (auto-detect if empty)")
    parser.add_argument("--out", "-o", type=str, default="", help="Output stats JSON path")
    args = parser.parse_args()
    calc_f1(args.in_file, args.prediction_key, args.out)


if __name__ == "__main__":
    main()
