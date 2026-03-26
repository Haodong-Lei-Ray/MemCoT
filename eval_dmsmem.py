#!/usr/bin/env python3
"""
Evaluate ReAct+LightRAG on the first 20 QA pairs of conv-26.
Same F1 logic as eval_event_search_top20.py.
Output: eval_output/<model>_rag_topk{N}/conv26_react_lightrag_top20_f1.json
"""

import argparse
import asyncio
import json
import os
import string
import sys
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import regex
import numpy as np
from nltk.stem import PorterStemmer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from module_version.version2.dmsmem import (
    run_react_lightrag,
    run_react_lightrag_async,
    create_lightrag,
    finalize_lightrag,
    create_img_retriever,
    _conv_id_to_workspace,
    _get_rag_event_loop,
    DEFAULT_LIGHTRAG_WORKING_BASE,
    DEFAULT_IMG_INDEX_BASE,
    FULL_VIEW_SESS_NUM,
    RAG_TYPE_LIGHTRAG,
    RAG_TYPE_NAIVE,
    RAG_TYPE_CHOICES,
    DEFAULT_AGENT_FLAG,
)
from module_version.version2.agent.agent import _build_full_conv_context

ps = PorterStemmer()

DATA_PATH = PROJECT_ROOT / "data" / "locomo10.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "eval_output"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
NUM_QA = 20
DEFAULT_RAG_TOPK = 10


# ──────────────── F1 computation (from task_eval/evaluation.py) ────────────

def normalize_answer(s):
    s = s.replace(",", "")
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
    """
    F1 计算与 task_eval/evaluation.py 一致，参考 docs/locomo_category5_f1_adversarial.md：
    - Category 1: multi-hop，f1_multi（子答案拆分）
    - Category 2,3,4: single-hop，f1_score_single
    - Category 5: 对抗题，二值判断，不看 answer，仅看 prediction 是否表示"未提及"
    """
    prediction = str(prediction or "")
    answer = str(answer or "")

    if category == 3:
        answer = answer.split(';')[0].strip()

    if category in [2, 3, 4]:
        return round(f1_score_single(prediction, answer), 3)
    elif category == 1:
        return round(f1_multi(prediction, answer), 3)
    elif category == 5:
        # Category 5: 正确答案为 "Not mentioned" / "No information available"
        # 二值判断：含以下任一子串 → 1，否则 0（参考 locomo_category5_f1_adversarial.md）
        pred_lower = prediction.lower()
        if "no information" in pred_lower or "not mentioned" in pred_lower:
            return 1.0
        return 0.0
    else:
        return round(f1_score_single(prediction, answer), 3)


# ──────────────── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ReAct+LightRAG on conv-26 top 20 QA")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("-n", "--num-qa", type=int, default=NUM_QA, help=f"Number of QA pairs (default: {NUM_QA})")
    parser.add_argument("-k", "--rag-topk", type=int, default=DEFAULT_RAG_TOPK, help=f"LightRAG top-k (default: {DEFAULT_RAG_TOPK})")
    parser.add_argument(
        "--rag-type",
        choices=RAG_TYPE_CHOICES,
        default=RAG_TYPE_LIGHTRAG,
        help=f"RAG 类型: naive=向量检索(pkl@naiverag), lightrag=LightRAG 知识图谱 (default: {RAG_TYPE_LIGHTRAG})",
    )
    parser.add_argument("--max-step", type=int, default=10, help="Max ReAct loop steps (default: 10)")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON")
    parser.add_argument("--skip-category", type=int, nargs="*", default=[], help="Skip QA categories (e.g. --skip-category 1 2 3 4)")
    parser.add_argument("--lightrag-base", default=DEFAULT_LIGHTRAG_WORKING_BASE,
                        help=f"LightRAG 索引根目录 (default: {DEFAULT_LIGHTRAG_WORKING_BASE})")
    parser.add_argument("--sample-id", "--conv", dest="sample_id", default="conv-26",
                        help="样本 ID，如 conv-26, conv-30 (default: conv-26)")
    parser.add_argument("--middle-scale", type=int, default=3,
                        help="Middle view agent scale K (default: 3)")
    parser.add_argument("--agent-flag", default=DEFAULT_AGENT_FLAG,
                        help=f"5-bit flag: rag_view/middle_view/full_view/agentic_graph/visual_search (default: {DEFAULT_AGENT_FLAG})")
    parser.add_argument("--img-index-base", default=DEFAULT_IMG_INDEX_BASE,
                        help=f"图片索引根目录 (default: {DEFAULT_IMG_INDEX_BASE})")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="QA 并发数（默认 1 即串行；>1 时 asyncio 并发评测）")
    args = parser.parse_args()

    print(args)
    
    model = args.model
    num_qa = args.num_qa
    rag_topk = args.rag_topk

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        safe_model = model.replace("/", os.sep)
        safe_model = f"{safe_model}_rag_topk{rag_topk}"
        output_dir = DEFAULT_OUTPUT_DIR / safe_model

    sample_id = args.sample_id
    conv_prefix = sample_id.replace("-", "")
    result_file_name = f"{conv_prefix}_react_lightrag_f1.json"
    output_path = output_dir / (result_file_name if num_qa < 0 else result_file_name)
    debug_root = output_dir / "debug"

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = next((s for s in data if s.get("sample_id") == sample_id), None)
    if sample is None:
        available = [s.get("sample_id") for s in data if s.get("sample_id")]
        raise SystemExit(f"sample_id '{sample_id}' not found. Available: {available}")
    qa_all = sample["qa"]
    qa_list = qa_all if num_qa < 0 else qa_all[:num_qa]

    skip_categories = set(args.skip_category)
    if skip_categories:
        qa_list = [qa for qa in qa_list if qa.get("category") not in skip_categories]
        print("Skipping QA categories: %s -> %d QA remaining" % (sorted(skip_categories), len(qa_list)))

    max_step = args.max_step
    print(f"Model: {model}")
    print(f"RAG top-k: {rag_topk}")
    print(f"Max step: {max_step}")
    print(f"Output: {output_dir}")
    print(f"Sample: {sample_id}")
    print(f"Evaluating {len(qa_list)} QA pairs via ReAct+LightRAG")
    print("=" * 70)

    results = []
    already_ids = set()

    if args.resume:
        load_path = output_path
        if not output_path.exists() and num_qa < 0:
            fallback = output_dir / result_file_name
            if fallback.exists():
                load_path = fallback
                print(f"Full output not found, resuming from {load_path.name}")
        if load_path.exists():
            with open(load_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            results = prev.get("details", [])
            already_ids_list = [r['qa_id'] for r in results]
            already_ids.update(already_ids_list)
            print(f"Resuming: {len(results)} done, {len(qa_list) - len(results)} remaining")
    skip_flag = True
    if skip_flag:
        skip_path = "/mnt/petrelfs/leihaodong/ICML/locomo/data/skip"
        skip_path = Path(skip_path) / f"{sample_id}.json"
        if skip_path.exists():
            with open(skip_path, "r", encoding="utf-8") as f:
                skip_list = json.load(f)
            skip_ids_list = [qa['qa_id'] for qa in skip_list]
            already_ids.update(skip_ids_list)
            print(f"Skipping: {len(skip_list)} done, {len(qa_list) - len(skip_list)} remaining")

    # 构建 working_dir
    workspace = _conv_id_to_workspace(sample_id)
    working_dir = os.path.join(args.lightrag_base, workspace)
    
    # 载入 LightRAG（仅初始化一次，后续检索复用同一实例）
    rag = None
    if args.rag_type == RAG_TYPE_LIGHTRAG:
        rag = create_lightrag(working_dir)
    
    # 构建会话
    full_conv = _build_full_conv_context(sample_id)
    # 初始化视觉搜索
    img_retriever = None
    if args.agent_flag[4] == "1":
        print("初始化视觉搜索...")
        img_retriever = create_img_retriever(sample_id, img_index_base=args.img_index_base)
    # ── 单个 QA 处理函数（供串行 / 并行共用） ──────────────────────────
    def _process_one_qa(i: int, qa: dict) -> dict | None:
        effective_qa_id = i + 1
        if effective_qa_id in already_ids:
            return None

        question = qa["question"]
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        category = qa["category"]

        print(f"\n[QA {i+1}/{len(qa_list)}] cat={category} | Q: {question}")
        print(f"  Gold answer: {answer}")

        debug_dir = str(debug_root / f"qa_{i+1}")

        gold_evidence = qa.get("evidence", [])
        if isinstance(gold_evidence, str):
            gold_evidence = [gold_evidence] if gold_evidence else []
        elif not isinstance(gold_evidence, list):
            gold_evidence = []

        try_step = 2
        prediction = ""
        final_evidence = []
        last_error = None
        while try_step > 0:
            try:
                res = run_react_lightrag(
                    query=question,
                    conv_id=sample_id,
                    category=category,
                    model=model,
                    output_dir=debug_dir,
                    rag_top_k=rag_topk,
                    max_step=max_step,
                    rag_type=args.rag_type,
                    working_dir=working_dir,
                    middle_scale=args.middle_scale,
                    agent_flag_str=args.agent_flag,
                    rag=rag,
                    full_conv=full_conv,
                    img_retriever=img_retriever,
                )
                prediction = res.get("answer", "")
                final_evidence = res.get("final_evidence", [])
                if prediction:
                    break
                last_error = "Prediction is empty"
            except Exception as e:
                last_error = e
                print(f"  [QA {i+1}] ERROR: {e}")
                traceback.print_exc()
            try_step -= 1

        if not prediction:
            raise RuntimeError(f"[QA {i+1}] Empty prediction after retries. Last error: {last_error}")

        f1 = compute_f1_for_qa(prediction, answer, category)

        if gold_evidence:
            pred_set = set(str(x) for x in final_evidence)
            gold_set = set(str(x) for x in gold_evidence)
            recall = round(len(pred_set & gold_set) / len(gold_set), 3)
        else:
            recall = 1.0

        print(f"  [QA {i+1}] Prediction: {prediction}")
        print(f"  [QA {i+1}] F1: {f1} | Recall: {recall}")

        return {
            "qa_id": effective_qa_id,
            "question": question,
            "gold_answer": str(answer),
            "prediction": prediction,
            "category": category,
            "event_search_f1": f1,
            "recall": recall,
            "final_evidence": final_evidence,
            "gold_evidence": gold_evidence,
        }

    # ── async 版 QA 处理（并发路径使用） ─────────────────────────────────
    async def _process_one_qa_async(i: int, qa: dict) -> dict | None:
        effective_qa_id = i + 1
        if effective_qa_id in already_ids:
            return None

        question = qa["question"]
        answer = qa.get("answer", qa.get("adversarial_answer", ""))
        category = qa["category"]

        print(f"\n[QA {i+1}/{len(qa_list)}] cat={category} | Q: {question}")
        print(f"  Gold answer: {answer}")

        debug_dir = str(debug_root / f"qa_{i+1}")

        gold_evidence = qa.get("evidence", [])
        if isinstance(gold_evidence, str):
            gold_evidence = [gold_evidence] if gold_evidence else []
        elif not isinstance(gold_evidence, list):
            gold_evidence = []

        try_step = 2
        prediction = ""
        final_evidence = []
        last_error = None
        while try_step > 0:
            try:
                res = await run_react_lightrag_async(
                    query=question,
                    conv_id=sample_id,
                    category=category,
                    model=model,
                    output_dir=debug_dir,
                    rag_top_k=rag_topk,
                    max_step=max_step,
                    rag_type=args.rag_type,
                    working_dir=working_dir,
                    middle_scale=args.middle_scale,
                    agent_flag_str=args.agent_flag,
                    rag=rag,
                    full_conv=full_conv,
                    img_retriever=img_retriever,
                )
                prediction = res.get("answer", "")
                final_evidence = res.get("final_evidence", [])
                if prediction:
                    break
                last_error = "Prediction is empty"
            except Exception as e:
                last_error = e
                print(f"  [QA {i+1}] ERROR: {e}")
                traceback.print_exc()
            try_step -= 1

        if not prediction:
            raise RuntimeError(f"[QA {i+1}] Empty prediction after retries. Last error: {last_error}")

        f1 = compute_f1_for_qa(prediction, answer, category)

        if gold_evidence:
            pred_set = set(str(x) for x in final_evidence)
            gold_set = set(str(x) for x in gold_evidence)
            recall = round(len(pred_set & gold_set) / len(gold_set), 3)
        else:
            recall = 1.0

        print(f"  [QA {i+1}] Prediction: {prediction}")
        print(f"  [QA {i+1}] F1: {f1} | Recall: {recall}")

        return {
            "qa_id": effective_qa_id,
            "question": question,
            "gold_answer": str(answer),
            "prediction": prediction,
            "category": category,
            "event_search_f1": f1,
            "recall": recall,
            "final_evidence": final_evidence,
            "gold_evidence": gold_evidence,
        }

    # ── 收集结果 & 落盘（单线程事件循环内无竞争，无需锁） ────────────────
    def _collect_result(result_dict: dict | None) -> None:
        if result_dict is None:
            return
        results.append(result_dict)
        already_ids.add(result_dict["qa_id"])
        f1_scores = [d.get("event_search_f1", 0) for d in results]
        recall_scores = [d.get("recall", 1.0) for d in results]
        _save_output(output_path, sample_id, model, qa_list, results, f1_scores, recall_scores)

    # ── 执行（串行 or 并发） ──────────────────────────────────────────
    concurrency = max(1, args.concurrency)
    pending = [(i, qa) for i, qa in enumerate(qa_list) if (i + 1) not in already_ids]
    print(f"\n待评测 {len(pending)} 个 QA，concurrency={concurrency}")

    if concurrency == 1:
        for i, qa in pending:
            result = _process_one_qa(i, qa)
            _collect_result(result)
    else:
        async def _run_concurrent():
            sem = asyncio.Semaphore(concurrency)
            completed_count = 0
            total = len(pending)

            async def _bounded(i, qa):
                async with sem:
                    try:
                        return i, await asyncio.wait_for(
                            _process_one_qa_async(i, qa),
                            timeout=600,
                        )
                    except asyncio.TimeoutError:
                        print(f"  [QA {i+1}] TIMEOUT (600s)")
                        return i, None
                    except Exception as e:
                        print(f"  [QA {i+1}] FATAL: {e}")
                        traceback.print_exc()
                        return i, None

            tasks = [asyncio.create_task(_bounded(i, qa)) for i, qa in pending]
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                _collect_result(result)
                completed_count += 1
                print(f"  [Progress] {completed_count}/{total} done, {len(results)} saved")

        loop = _get_rag_event_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=max(32, concurrency * 5)))
        loop.run_until_complete(_run_concurrent())

    finalize_lightrag(rag)

    f1_scores = [d.get("event_search_f1", 0) for d in results]
    recall_scores = [d.get("recall", 1.0) for d in results]
    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    _save_output(output_path, sample_id, model, qa_list, results, f1_scores, recall_scores)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_path}")
    print(f"F1 scores: {f1_scores}")
    print(f"Mean F1: {mean_f1:.4f}")
    print(f"Recall scores: {recall_scores}")
    print(f"Mean Recall: {mean_recall:.4f}")


def _qa_id_sort_key(x):
    return int(str(x or ""))

def _save_output(output_path, sample_id, model, qa_list, results, f1_scores, recall_scores=None):
    if recall_scores is None:
        recall_scores = [d.get("recall", 1.0) for d in results if "recall" in d]
        if len(recall_scores) < len(results):
            recall_scores = [d.get("recall", 1.0) for d in results]
    if f1_scores is None:
        f1_scores = [d.get("event_search_f1", 0) for d in results]
    details = []
    for r in results:
        d = dict(r)
        if "qa_id" not in d:
            d["qa_id"] = ""
        details.append(d)
    details.sort(key=lambda d: _qa_id_sort_key(d.get("qa_id")))

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    output = {
        "sample_id": sample_id,
        "model": model,
        "method": "react_lightrag",
        "n": len(results),
        "mean_f1": round(mean_f1, 4),
        "mean_recall": round(mean_recall, 4),
        "details": details,
    }
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
