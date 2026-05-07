#!/usr/bin/env python3
"""
Evaluate MemCoT on locomo or longmemeval benchmarks.
Adapted to use the MemCoT class from memcot.py directly.
"""

import argparse
import json
import os
import string
import sys
import traceback
from collections import Counter
from pathlib import Path

import regex
import numpy as np
from nltk.stem import PorterStemmer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from memcot import MemCoT, answer_memcot_exit

ps = PorterStemmer()

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
NUM_QA = 20

# ──────────────── F1 computation (for Locomo) ────────────

def normalize_answer(s):
    s = str(s).replace(",", "")
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score_single(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_multi(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return float(np.mean([
        max([f1_score_single(pred, gt) for pred in predictions])
        for gt in ground_truths
    ]))


def compute_f1_for_qa(prediction, answer, category):
    prediction = str(prediction or "")
    answer = str(answer or "")
    if category == 3:
        answer = answer.split(';')[0].strip()

    if category in [2, 3, 4]:
        return round(f1_score_single(prediction, answer), 3)
    elif category == 1:
        return round(f1_multi(prediction, answer), 3)
    elif category == 5:
        pred_lower = prediction.lower()
        if "no information" in pred_lower or "not mentioned" in pred_lower:
            return 1.0
        return 0.0
    else:
        return round(f1_score_single(prediction, answer), 3)


# ──────────────── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate MemCoT on Locomo/Longmemeval")
    parser.add_argument("-n", "--num-qa", type=int, default=NUM_QA, help=f"Number of QA pairs (default: {NUM_QA})")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON")
    
    # New configuration arguments
    parser.add_argument("--rag-config", default=str(PROJECT_ROOT / "config" / "rag" / "locomolightrag.json"))
    parser.add_argument("--memcot-config", default=str(PROJECT_ROOT / "config" / "memcot.json"))
    
    # Optional arguments depending on benchmark
    parser.add_argument("--sample-id", "-c", "--conv", dest="sample_id", default="conv-26",
                        help="Sample ID for locomo (e.g. conv-26)")
    parser.add_argument("--skip-category", type=int, nargs="*", default=[], help="Skip QA categories for locomo")
    args = parser.parse_args()

    print(args)

    with open(args.rag_config, 'r', encoding='utf-8') as f:
        rag_config_data = json.load(f)
        
    with open(args.memcot_config, 'r', encoding='utf-8') as f:
        memcot_config_data = json.load(f)
    
    model = memcot_config_data.get("agent_config", {}).get("model_name", "Qwen/Qwen2.5-14B-Instruct")
    benchmark = rag_config_data.get("benchmark", "locomo")
    rag_topk = rag_config_data.get("rag_topk", 10)
    print(f"Detected benchmark: {benchmark}, model: {model}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        safe_model = model.replace("/", os.sep)
        safe_model = f"{safe_model}_rag_topk{rag_topk}"
        output_dir = DEFAULT_OUTPUT_DIR / safe_model

    debug_root = output_dir / "debug"

    # Initialize MemCoT
    print("Initializing MemCoT...")
    memcot = MemCoT(
        model=model,
        memcot_file_path=args.memcot_config,
        rag_file_path=args.rag_config,
        rag_top_k=rag_topk,
        conv_id=None,
    )

    # Load Dataset
    if benchmark == "locomo":
        DATA_PATH = PROJECT_ROOT / "benchmark" / "locomo" / "data" / "locomo10.json"
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        sample = next((s for s in data if s.get("sample_id") == args.sample_id), None)
        if sample is None:
            raise SystemExit(f"sample_id '{args.sample_id}' not found.")
        qa_all = sample["qa"]
        
        # apply skipping
        skip_categories = set(args.skip_category)
        if skip_categories:
            qa_all = [qa for qa in qa_all if qa.get("category") not in skip_categories]
            print("Skipping QA categories: %s -> %d QA remaining" % (sorted(skip_categories), len(qa_all)))
            
        result_file_name = f"{args.sample_id.replace('-', '')}_eval.json"
        
    elif benchmark == "longmemeval":
        DATA_PATH = PROJECT_ROOT / "benchmark" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            qa_all = json.load(f)
        result_file_name = "longmemeval_eval.json"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    qa_list = qa_all if args.num_qa < 0 else qa_all[:args.num_qa]
    output_path = output_dir / result_file_name

    results = []
    already_ids = set()

    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        results = prev.get("details", [])
        already_ids.update([r.get('qa_id', r.get('question_id')) for r in results])
        print(f"Resuming: {len(results)} done, {len(qa_list) - len(results)} remaining")

    print(f"Evaluating {len(qa_list)} QA pairs on {benchmark}")
    print("=" * 70)

    # Important: avoid switching sessions per-QA for locomo (same conv),
    # otherwise LightRAG's asyncio locks may be bound to different event loops.
    if benchmark == "locomo":
        memcot.switch_session(conv_id=args.sample_id)

    for i, qa in enumerate(qa_list):
        qa_id = qa.get("qa_id") if benchmark == "locomo" else qa.get("question_id")
        effective_qa_id = qa_id if qa_id is not None else i + 1
        
        if effective_qa_id in already_ids:
            continue

        question = qa["question"]
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        category = qa.get("category", 0)

        print(f"\n[{benchmark} QA {i+1}/{len(qa_list)}] ID: {effective_qa_id} | Q: {question}")
        print(f"  Gold answer: {answer}")

        # Set conversation context for MemCoT (only needed for longmemeval)
        if benchmark == "longmemeval":
            idx = qa_all.index(qa)  # conv_id is usually idx_question_id
            conv_target = f"{idx:04d}_{qa['question_id']}"
            memcot.switch_session(conv_id=conv_target)
        debug_dir = str(debug_root / f"qa_{effective_qa_id}")

        try_step = 2
        prediction = ""
        final_evidence = []
        last_error = None
        
        while try_step > 0:
            try:
                exit_state = memcot.run(query=question, output_dir=debug_dir, category=category)
                res = answer_memcot_exit(exit_state)
                prediction = res.get("answer", "")
                final_evidence = res.get("final_evidence", [])
                if prediction:
                    break
                last_error = "Prediction is empty"
            except Exception as e:
                last_error = e
                print(f"  [QA {effective_qa_id}] ERROR: {e}")
                traceback.print_exc()
            try_step -= 1

        if not prediction:
            print(f"  [QA {effective_qa_id}] Empty prediction after retries. Last error: {last_error}")
            prediction = ""

        # Compute metrics
        f1 = compute_f1_for_qa(prediction, answer, category) if benchmark == "locomo" else f1_score_single(prediction, answer)
        
        gold_evidence = qa.get("evidence", []) if benchmark == "locomo" else qa.get("answer_session_ids", [])
        if isinstance(gold_evidence, str):
            gold_evidence = [gold_evidence] if gold_evidence else []
        elif not isinstance(gold_evidence, list):
            gold_evidence = []

        if gold_evidence:
            pred_set = set(str(x) for x in final_evidence)
            gold_set = set(str(x) for x in gold_evidence)
            recall = round(len(pred_set & gold_set) / len(gold_set), 3) if len(gold_set) > 0 else 1.0
        else:
            recall = 1.0

        print(f"  [QA {effective_qa_id}] Prediction: {prediction}")
        print(f"  [QA {effective_qa_id}] Match/F1: {f1} | Recall: {recall}")

        result_dict = {
            "qa_id": effective_qa_id if benchmark == "locomo" else None,
            "question_id": effective_qa_id if benchmark == "longmemeval" else None,
            "question": question,
            "gold_answer": str(answer),
            "prediction": prediction,
            "category": category,
            "event_search_f1": f1,
            "recall": recall,
            "final_evidence": final_evidence,
            "gold_evidence": gold_evidence,
        }
        
        results.append(result_dict)
        already_ids.add(effective_qa_id)
        
        # Save incrementally
        f1_scores = [d.get("event_search_f1", 0) for d in results]
        recall_scores = [d.get("recall", 1.0) for d in results]
        mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        
        output = {
            "sample_id": args.sample_id if benchmark == "locomo" else "longmemeval",
            "model": model,
            "method": "memcot_unified",
            "n": len(results),
            "mean_f1": round(mean_f1, 4),
            "mean_recall": round(mean_recall, 4),
            "details": results,
        }
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
