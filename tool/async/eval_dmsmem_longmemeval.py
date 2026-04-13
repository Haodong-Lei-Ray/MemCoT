#!/usr/bin/env python3
"""
Evaluate DMSMem(ReAct+LightRAG) on LongMemEval.

【已优化】改为纯顺序执行 + 独立 asyncio.run() 避免异步 event loop bug
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import dmsmem_longmemeval as dmsmem_mod
from dmsmem_longmemeval import (
    run_react_lightrag,
    finalize_lightrag,
    RAG_TYPE_LIGHTRAG,
    RAG_TYPE_NAIVE,
    RAG_TYPE_CHOICES,
    DEFAULT_AGENT_FLAG,
    DEFAULT_LIGHTRAG_WORKING_BASE,
    DEFAULT_IMG_INDEX_BASE,
)
from benchmark.longmemeval.src.retrieval.eval_utils import evaluate_retrieval


DEFAULT_DATA_PATH = PROJECT_ROOT / "benchmark" / "longmemeval" / "data" / "longmemeval_s_cleaned.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exp" / "longmemeval_eval"
DEFAULT_MODEL = "gpt-4o-mini"


def _extract_qid_from_filename(stem: str) -> str:
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem


def _build_rankings_from_session_order(
    ranked_sessions: list[str], corpus_ids: list[str]
) -> list[int]:
    index = {sid: i for i, sid in enumerate(corpus_ids)}
    used = set()
    rankings: list[int] = []
    for sid in ranked_sessions:
        if sid in index and sid not in used:
            rankings.append(index[sid])
            used.add(sid)
    for sid in corpus_ids:
        if sid not in used:
            rankings.append(index[sid])
    return rankings


def _dia_id_to_session_id(dia_id: str, haystack_session_ids: list[str]) -> str | None:
    for sid in haystack_session_ids:
        if dia_id == sid or dia_id.startswith(sid + "_"):
            return sid
    return None


def _compute_retrieval_metrics_for_one(
    entry: dict[str, Any], final_evidence: list[str], ks: list[int]
) -> tuple[dict[str, float], list[str]]:
    corpus_ids = list(entry.get("haystack_session_ids", []))
    correct_docs = list(entry.get("answer_session_ids", []))

    ranked_sessions: list[str] = []
    seen = set()
    for dia_id in final_evidence:
        sid = _dia_id_to_session_id(str(dia_id), corpus_ids)
        if sid and sid not in seen:
            ranked_sessions.append(sid)
            seen.add(sid)

    rankings = _build_rankings_from_session_order(ranked_sessions, corpus_ids)
    out: dict[str, float] = {}
    for k in ks:
        recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
        out[f"recall_any@{k}"] = float(round(recall_any, 4))
        out[f"recall_all@{k}"] = float(round(recall_all, 4))
        out[f"ndcg_any@{k}"] = float(round(ndcg_any, 4))
    seen_list = list(seen)
    return out, seen_list


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_llm_judge(metric_model: str, pred_jsonl: Path, ref_file: Path) -> Path:
    script = PROJECT_ROOT / "benchmark" / "longmemeval" / "src" / "evaluation" / "evaluate_qa.py"
    pred_abs = Path(pred_jsonl).resolve()
    ref_abs = Path(ref_file).resolve()
    cmd = [sys.executable, str(script), metric_model, str(pred_abs), str(ref_abs)]
    subprocess.run(cmd, check=True, cwd=str(script.parent))
    return Path(str(pred_jsonl) + f".eval-results-{metric_model}")


def _parse_llm_judge_results(eval_log_path: Path, qid_to_qtype: dict[str, str]) -> dict[str, Any]:
    rows = []
    with eval_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    acc = []
    by_type: dict[str, list[int]] = {}
    for row in rows:
        qid = row.get("question_id", "")
        label = bool(row.get("autoeval_label", {}).get("label", False))
        acc.append(1 if label else 0)
        qtype = qid_to_qtype.get(qid, "unknown")
        by_type.setdefault(qtype, []).append(1 if label else 0)

    out = {
        "overall_accuracy": round(float(np.mean(acc)) if acc else 0.0, 4),
        "n": len(acc),
        "by_question_type": {
            k: {
                "accuracy": round(float(np.mean(v)) if v else 0.0, 4),
                "n": len(v),
            }
            for k, v in sorted(by_type.items())
        },
    }
    return out

def _get_rag_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("loop closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
        except ImportError:
            pass
    return loop

def create_lightrag_longmemeval(working_dir: str):
    """创建并初始化 LightRAG 实例（使用 asyncio.run + 全新 event loop，避免异步 bug）"""
    print(f"[create_lightrag] working_dir: {working_dir}")
    if not os.path.exists(working_dir):
        raise FileNotFoundError(f"LightRAG working dir not found: {working_dir}")

    if os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

    from lightrag import LightRAG
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
    )

    
    loop = _get_rag_event_loop()
    print(f"运行状态: {'运行中' if loop.is_running() else '已停止'}")
    print(f"关闭状态: {'已关闭' if loop.is_closed() else '未关闭'}")
    if loop.is_running():
        print("\n步骤1: 停止循环...")
        loop.stop()
        time.sleep(1)
    
    print("\n步骤2: 取消所有任务...")
    tasks = asyncio.all_tasks(loop)
    print(f"找到 {len(tasks)} 个任务")
    
    for i, task in enumerate(tasks):
        print(f"  任务 {i+1}: {task.get_name() if hasattr(task, 'get_name') else task}")
        if not task.done():
            task.cancel()
            print(f"    已取消")
    
    # 给任务一些时间来处理取消
    max_wait=5
    if tasks and not loop.is_closed():
        print("\n步骤3: 等待任务取消...")
        try:
            # 设置超时
            loop.run_until_complete(
                asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=max_wait
                )
            )
        except asyncio.TimeoutError:
            print(f"  等待超时 ({max_wait}秒)")
        except RuntimeError:
            print("  循环已关闭，跳过等待")
    
    print("\n步骤4: 关闭循环...")
    if not loop.is_closed():
        loop.close()
        print("  循环已关闭")
    
    loop = _get_rag_event_loop()
    try:
        loop.run_until_complete(rag.finalize_storages())
    except Exception:
        pass
    loop.run_until_complete(rag.initialize_storages())
    return rag


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DMSMem on LongMemEval (顺序执行版)")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--lightrag-base", type=Path, default=Path(DEFAULT_LIGHTRAG_WORKING_BASE))
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("--rag-type", choices=RAG_TYPE_CHOICES, default=RAG_TYPE_LIGHTRAG)
    parser.add_argument("-k", "--rag-topk", type=int, default=10)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--middle-scale", type=int, default=3)
    parser.add_argument("--agent-flag", default=DEFAULT_AGENT_FLAG)
    parser.add_argument("--img-index-base", default=DEFAULT_IMG_INDEX_BASE)
    parser.add_argument("--benchmark", type=str, default="longmemeval")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-n", "--num-qa", type=int, default=-1)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-llm-judge", action="store_true", default=False)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--retrieval-ks", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--skip-abstention-for-retrieval", action="store_true", default=True)
    args = parser.parse_args()

    print(args)

    # 加载数据（保持不变）
    def _load_entries() -> list[dict[str, Any]]:
        if not args.data_file.exists():
            raise FileNotFoundError(f"data_file not found: {args.data_file}")
        with args.data_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"data_file must be a list JSON: {args.data_file}")
        return data

    all_entries: list[dict[str, Any]] = _load_entries()
    qid_to_conv_id: dict[str, str] = {}
    for idx, e in enumerate(all_entries):
        if args.num_qa >= 0 and idx >= args.num_qa+args.start_idx:
            break
        qid = str(e.get("question_id", ""))
        if qid:
            qid_to_conv_id[qid] = f"{idx:04d}_{qid}"

    entries = all_entries
    if args.start_idx > 0:
        entries = entries[args.start_idx:]
    if args.num_qa >= 0:
        entries = entries[: args.num_qa]
    print("entries:",len(entries))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "longmemeval_details.json"
    pred_path = output_dir / "longmemeval_predictions.jsonl"
    metrics_path = output_dir / "longmemeval_metrics.json"

    # resume 支持
    details: list[dict[str, Any]] = []
    done_qids = set()
    if args.resume and details_path.exists():
        with details_path.open("r", encoding="utf-8") as f:
            prev = json.load(f)
        print(prev)
        details = prev.get("details", [])
        done_qids = {d.get("question_id") for d in details if d.get("question_id")}
        print(f"Resuming: {len(done_qids)} done, {len(entries) - len(done_qids)} remaining")

    # Monkey patch（保持不变）
    qid_to_entry = {e.get("question_id", ""): e for e in entries}

    def _patched_get_haystack_session_ids(conv_id: str) -> list[str]:
        qid = conv_id.split("_", 1)[1] if "_" in conv_id and conv_id.split("_", 1)[0].isdigit() else conv_id
        ent = qid_to_entry.get(qid)
        return ent.get("haystack_session_ids", []) if ent else []

    dmsmem_mod._get_haystack_session_ids = _patched_get_haystack_session_ids

    failures = 0
    qid_to_qtype = {str(e.get("question_id", "")): str(e.get("question_type", "")) for e in entries}
    per_item_judge_rows: list[dict[str, Any]] = []

    # ==================== 核心循环：纯顺序执行 ====================
    for i, entry in enumerate(entries, start=1):
        qid = str(entry.get("question_id", ""))
        if qid in done_qids:
            continue

        question = str(entry.get("question", ""))
        answer = str(entry.get("answer", ""))
        qtype = str(entry.get("question_type", ""))
        category = 5 if qid.endswith("_abs") else 2

        conv_id = qid_to_conv_id.get(qid)
        if conv_id is None:
            failures += 1
            details.append({"question_id": qid, "question_type": qtype, "question": question,
                            "gold_answer": answer, "error": "conv_id_not_found"})
            continue

        workspace = conv_id.replace("-", "")
        working_dir = str(args.lightrag_base / workspace)
        debug_dir = str(output_dir / "debug" / conv_id)

        rag = None
        try:
            if args.rag_type == RAG_TYPE_LIGHTRAG:
                rag = create_lightrag_longmemeval(working_dir)

            res = run_react_lightrag(
                query=question,
                conv_id=conv_id,
                category=category,
                model=args.model,
                output_dir=debug_dir,
                max_step=args.max_step,
                rag_top_k=args.rag_topk,
                rag_type=args.rag_type,
                working_dir=working_dir,
                middle_scale=args.middle_scale,
                agent_flag_str=args.agent_flag,
                rag=rag,
                full_conv=entry,
                img_retriever=None,
                benchmark=args.benchmark,
            )

            pred = str(res.get("answer", ""))
            final_evidence = [str(x) for x in res.get("final_evidence", [])]
            retrieval_metrics, seen_ids = _compute_retrieval_metrics_for_one(entry, final_evidence, args.retrieval_ks)

            llm_judge_item = None
            if args.run_llm_judge:
                with tempfile.TemporaryDirectory(prefix=f"judge_{qid}_", dir=str(output_dir)) as td:
                    tmp_pred = Path(td) / "predictions.jsonl"
                    with tmp_pred.open("w", encoding="utf-8") as f:
                        f.write(json.dumps({"question_id": qid, "hypothesis": pred}, ensure_ascii=False) + "\n")
                    eval_log_path = _run_llm_judge(args.judge_model, tmp_pred, args.data_file)
                    with eval_log_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            if str(row.get("question_id", "")) == qid:
                                per_item_judge_rows.append(row)
                                label = bool(row.get("autoeval_label", {}).get("label", False))
                                llm_judge_item = {"label": label, "raw": row, "llm_score": 1.0 if label else 0.0}
                                break

            details.append({
                "question_id": qid,
                "conv_id": conv_id,
                "question_type": qtype,
                "question": question,
                "gold_answer": answer,
                "evidence": entry['answer_session_ids'],
                "prediction": pred,
                "final_evidence": final_evidence,
                "seen_ids": seen_ids,
                "retrieval_metrics": retrieval_metrics,
                "llm_judge": llm_judge_item,
            })
            print(f"[{i}/{len(entries)}] OK {qid}")
        finally:
            if rag is not None:
                finalize_lightrag(rag)

        # 每处理完一个 QA 就落盘（防止中断丢失）
        _save_json(
            details_path,
            {
                "benchmark": args.benchmark,
                "model": args.model,
                "n": len(details),
                "failures": failures,
                "details": details,
            },
        )

    # ==================== 后续处理（不变） ====================
    with pred_path.open("w", encoding="utf-8") as f:
        for d in details:
            qid = d.get("question_id", "")
            hyp = d.get("prediction", "")
            if qid:
                f.write(json.dumps({"question_id": qid, "hypothesis": hyp}, ensure_ascii=False) + "\n")

    # 检索指标汇总
    valid_for_retrieval = [d["retrieval_metrics"] for d in details
                           if "retrieval_metrics" in d and not (args.skip_abstention_for_retrieval and str(d.get("question_id", "")).endswith("_abs"))]

    retrieval_summary: dict[str, float] = {}
    if valid_for_retrieval:
        keys = sorted(valid_for_retrieval[0].keys())
        for k in keys:
            retrieval_summary[k] = round(float(np.mean([x.get(k, 0.0) for x in valid_for_retrieval])), 4)

    metrics: dict[str, Any] = {
        "benchmark": args.benchmark,
        "model": args.model,
        "data_file": str(args.data_file),
        "n_total": len(entries),
        "n_done": len(details),
        "n_failures": failures,
        "retrieval_summary": retrieval_summary,
        "predictions_file": str(pred_path),
        "details_file": str(details_path),
    }

    if args.run_llm_judge:
        # ...（LLM judge 统计逻辑保持不变）
        acc = []
        by_type: dict[str, list[int]] = {}
        for row in per_item_judge_rows:
            qid = str(row.get("question_id", ""))
            label = bool(row.get("autoeval_label", {}).get("label", False))
            acc.append(1 if label else 0)
            qtype = qid_to_qtype.get(qid, "unknown")
            by_type.setdefault(qtype, []).append(1 if label else 0)
        metrics["qa_judge"] = {
            "overall_accuracy": round(float(np.mean(acc)) if acc else 0.0, 4),
            "n": len(acc),
            "by_question_type": {
                k: {"accuracy": round(float(np.mean(v)) if v else 0.0, 4), "n": len(v)}
                for k, v in sorted(by_type.items())
            },
            "eval_log_file": "per-item-inline",
        }

    _save_json(metrics_path, metrics)
    print(f"Saved details: {details_path}")
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()