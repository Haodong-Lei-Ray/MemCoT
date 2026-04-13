#!/usr/bin/env python3
"""
ReAct + RAG — LoCoMo 应用场景。支持 NaiveRAG 与 LightRAG 两种检索，通过超参数 --rag-type 选择。

输入 query -> 入 query queue
Loop (max step=10):
  Act: 对 query queue 中所有 query 执行 RAG 检索（naive 或 lightrag），得到 RAG 结果
  Observation: 用 query 比对 RAG 结果
    1. 可直接回答则停止
    2. 否则保留可能有效的 RAG 结果
  Thought:
    1. 当前 RAG 缺失什么信息？
    2. 为缺失信息生成新的 RAG query（可多个），入 query queue
  下一轮 Act...

轨迹保存为 Act-Observation-Thought 格式，便于调试。

Usage:
  python react_naiverag.py -c conv-26 "When did Caroline go to the LGBTQ support group?"
  python react_naiverag.py -c conv-26 --rag-type naive -o output_dir "query"   # NaiveRAG (pkl@naiverag)
  python react_naiverag.py -c conv-26 --rag-type lightrag -o output_dir "query"  # LightRAG (default)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional
from global_methods import with_agent_llm_config, get_openai_config_source

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from tool.longmemevaltool import rewrite_first_person_to_user as _rewrite_first_person_to_user

LIGHTRAG_PATH = PROJECT_ROOT / "mem" / "LightRAG"
sys.path.insert(0, str(LIGHTRAG_PATH))

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Agent 模块：rag_view, full_view, answer, thought, observation
from agent.agent import (
    ZoomInFocalRetrieve,
    zoom_out_context_expansion_agent,
    answer_agent,
    guess_answer_agent,
    judge_agent,
    panoramic_visual_grounding,
    conv_answer_agent,
    _build_full_conv_context,
    _build_full_conv_context_longmemeval,
    _get_haystack_session_ids,
    WRONG_LIST,
)

DEFAULT_LIGHTRAG_WORKING_BASE = "/mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_storage"
NAIVERAG_BASE = "/mnt/petrelfs/leihaodong/ICML/exp/memory/naiverag"
DEFAULT_IMG_INDEX_BASE = "/mnt/petrelfs/leihaodong/ICML/exp/memory/img_mem/vit_rag"
DATA_PATH = PROJECT_ROOT / "data" / "locomo10.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUTPUT_DIR = "/mnt/petrelfs/leihaodong/ICML/locomo/module_version/version2/eval_output"
MAX_LOOP_STEP = 10
RAG_TOP_K = 5
FULL_VIEW_SESS_NUM = 1  # Full View agent 每次选择的 session 数量
DEFAULT_AGENT_FLAG = "11000"
AGENT_LLM_CONFIG_PATH = PROJECT_ROOT / "config" / "configqwen1.json"


@with_agent_llm_config(AGENT_LLM_CONFIG_PATH)
def rag_view_agent_with_cfg(
    root_query: str,
    query_queue: list[str],
    rag_results: list[dict],
    last_thought_information: str,
    known_info_rag: list[dict],
    model: str,
    benchmark: str = "locomo",
    haystack_session_ids: list[str] | None = None,
) -> dict:
    return ZoomInFocalRetrieve(model, temperature=1.0).run(
        root_query=root_query,
        query_queue=query_queue,
        rag_results=rag_results,
        last_thought_information=last_thought_information,
        known_info_rag=known_info_rag,
        benchmark=benchmark,
        haystack_session_ids=haystack_session_ids,
    )


middle_view_agent_with_cfg = with_agent_llm_config(AGENT_LLM_CONFIG_PATH)(zoom_out_context_expansion_agent)
visual_ocr_agent_with_cfg = with_agent_llm_config(AGENT_LLM_CONFIG_PATH)(panoramic_visual_grounding)
observation_agent_with_cfg = with_agent_llm_config(AGENT_LLM_CONFIG_PATH)(judge_agent)
answer_agent_with_cfg = with_agent_llm_config(AGENT_LLM_CONFIG_PATH)(answer_agent)
conv_answer_agent_with_cfg = with_agent_llm_config(AGENT_LLM_CONFIG_PATH)(conv_answer_agent)



def _conv_id_to_workspace(conv_id: str) -> str:
    return conv_id.replace("-", "")


def _get_chunk_id_to_doc_id(working_dir: str) -> dict:
    path = os.path.join(working_dir, "kv_store_text_chunks.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        cid: info.get("full_doc_id", cid)
        for cid, info in data.items()
        if isinstance(info, dict)
    }


def _lightrag_retrieve_sync(query: str, conv_id: str, top_k: int = RAG_TOP_K,
                            working_dir: str = None, rag=None) -> list[dict]:
    """同步版 LightRAG 检索（关键修复点）"""
    if rag is None or working_dir is None:
        raise ValueError("rag instance and working_dir must be provided")

    from lightrag import QueryParam
    param = QueryParam(mode="hybrid", top_k=top_k, chunk_top_k=top_k)

    result = rag.query_data(query, param)   # ← 改成同步调用！

    if result.get("status") != "success":
        raise ValueError(f"LightRAG status != success for conv_id: {conv_id}")
    if not result.get("data", {}).get("chunks"):
        raise ValueError(f"LightRAG returned no chunks for conv_id: {conv_id}")

    chunks = result["data"]["chunks"]
    chunk_id_to_doc = _get_chunk_id_to_doc_id(working_dir)
    results = []
    for i, c in enumerate(chunks):
        chunk_id = c.get("chunk_id", "")
        dia_id = chunk_id_to_doc.get(chunk_id, chunk_id)
        results.append({
            "dia_id": dia_id,
            "date_time": c.get("file_path", ""),
            "context": c.get("content", ""),
            "score": float(1.0 - i * 0.01),
            "from_query": query,
        })
    return results


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

import time
def create_lightrag(working_dir: str):
    """创建并初始化 LightRAG 实例（同步接口），避免每次检索都重新创建。"""
    print(f"[create_lightrag] working_dir: {working_dir}")
    if not os.path.exists(working_dir):
        raise FileNotFoundError(f"LightRAG working dir not found: {working_dir}")

    if os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

    from lightrag import LightRAG
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    normalized_working_dir = os.path.normpath(working_dir)
    workspace = os.path.basename(normalized_working_dir)
    working_root = os.path.dirname(normalized_working_dir)
    rag = LightRAG(
        working_dir=working_root,
        workspace=workspace,
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


def finalize_lightrag(rag):
    """关闭 LightRAG 实例的存储。"""
    if rag is None:
        return
    try:
        loop = _get_rag_event_loop()
        loop.run_until_complete(rag.finalize_storages())
    except Exception as e:
        print(f"[finalize_lightrag] Warning: {e}")


def create_img_retriever(conv_id: str, img_index_base: str = DEFAULT_IMG_INDEX_BASE, top_k: int = 3):
    """创建 CLIP text→image 检索器。仅在 visual search agent 开启时调用。"""
    index_dir = os.path.join(img_index_base, conv_id)
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Image index dir not found: {index_dir}")

    from module_version.version2.tool.t2i.query_img_index import CLIPTextEmbedding
    from llama_index.core import StorageContext, load_index_from_storage, Settings

    clip_embed = CLIPTextEmbedding()
    Settings.embed_model = clip_embed
    Settings.llm = None
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context, embed_model=clip_embed)
    retriever = index.as_retriever(similarity_top_k=top_k)
    print(f"[create_img_retriever] Loaded from {index_dir}, top_k={top_k}")
    return retriever


def lightrag_retrieve_multi(
    queries: list[str], conv_id: str, top_k: int = RAG_TOP_K,
    working_dir: str = None,
    rag=None,
) -> list[dict]:
    """对多个 query 执行 LightRAG 检索，合并去重（按 dia_id+context）。"""

    loop = _get_rag_event_loop()

    async def _run_all():
        all_results = []
        result_list = []
        for q in queries:
            res = await _lightrag_retrieve_async(q, conv_id, top_k, working_dir, rag=rag)
            one_result = {
                "query": q,
                "res": res,
            }
            all_results.extend(res)
            result_list.append(one_result)
        return all_results, result_list

    results, result_list = loop.run_until_complete(_run_all())
    # 去重：dia_id 唯一，按 dia_id 保留第一次出现
    seen = set()
    deduped = []
    for r in results:
        did = r.get("dia_id", "")
        if did and did not in seen:
            seen.add(did)
            deduped.append(r)
    return deduped, result_list


def lightrag_retrieve_multi(
    queries: list[str], conv_id: str, top_k: int = RAG_TOP_K,
    working_dir: str = None, rag=None,
) -> tuple[list[dict], list[dict]]:
    """同步多 query 检索 + 去重"""
    all_results = []
    result_list = []
    for q in queries:
        res = _lightrag_retrieve_sync(q, conv_id, top_k, working_dir, rag)
        all_results.extend(res)
        result_list.append({"query": q, "res": res})

    # 去重
    seen = set()
    deduped = []
    for r in all_results:
        did = r.get("dia_id", "")
        if did and did not in seen:
            seen.add(did)
            deduped.append(r)
    return deduped, result_list


def naiverag_retrieve_multi(
    queries: list[str], conv_id: str, top_k: int = RAG_TOP_K
) -> tuple[list[dict], list[dict]]:
    """
    Naive RAG: 从 naiverag 目录加载 pkl，向量检索。返回 (deduped, result_list)，格式与 lightrag_retrieve_multi 一致。
    """
    import pickle
    import numpy as np
    from global_methods import get_openai_embedding

    pkl_path = os.path.join(NAIVERAG_BASE, f"locomo10_dialog_{conv_id}.pkl")
    if not os.path.exists(pkl_path):
        return [], [{"query": q, "res": []} for q in queries]

    with open(pkl_path, "rb") as f:
        db = pickle.load(f)

    embeddings = db.get("embeddings")
    dia_ids = db.get("dia_id", [])
    date_times = db.get("date_time", [])
    contexts = db.get("context", [])

    if embeddings is None or len(dia_ids) == 0:
        return [], [{"query": q, "res": []} for q in queries]

    all_results = []
    result_list = []
    for q in queries:
        query_emb = get_openai_embedding([q])
        scores = np.dot(query_emb, np.array(embeddings).T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        res = [
            {
                "dia_id": dia_ids[idx] if idx < len(dia_ids) else "",
                "date_time": date_times[idx] if idx < len(date_times) else "",
                "context": contexts[idx] if idx < len(contexts) else "",
                "score": float(scores[idx]),
                "from_query": q,
            }
            for idx in top_indices
        ]
        all_results.extend(res)
        result_list.append({"query": q, "res": res})

    # 去重：dia_id 唯一
    seen = set()
    deduped = []
    for r in all_results:
        did = r.get("dia_id", "")
        if did and did not in seen:
            seen.add(did)
            deduped.append(r)
    return deduped, result_list


RAG_TYPE_NAIVE = "naive"
RAG_TYPE_LIGHTRAG = "lightrag"
RAG_TYPE_CHOICES = (RAG_TYPE_NAIVE, RAG_TYPE_LIGHTRAG)

def add_category_information(category: int) -> str:
    """根据 QA category 返回额外提示信息，用于引导各 Agent 的推理方向。"""
    if category == 2:
        return "Use DATE of CONVERSATION to answer with an approximate date."
    if category == 5:
        return 'If the answer is not mentioned in the conversation, answer "Not mentioned in the conversation".'
    return ""

def rag_retrieve_multi(
    queries: list[str],
    conv_id: str,
    top_k: int = RAG_TOP_K,
    rag_type: str = RAG_TYPE_LIGHTRAG,
    working_dir: str = None,
    rag=None,
) -> tuple[list[dict], list[dict]]:
    """根据 rag_type 选择 NaiveRAG 或 LightRAG 检索。返回 (deduped, result_list)。"""
    if rag_type == RAG_TYPE_NAIVE:
        return naiverag_retrieve_multi(queries, conv_id, top_k)
    if rag_type == RAG_TYPE_LIGHTRAG:
        return lightrag_retrieve_multi(queries, conv_id, top_k, working_dir, rag=rag)
    raise ValueError(f"rag_type must be one of {RAG_TYPE_CHOICES}, got {rag_type!r}")


def rag_retrieve_multi(
    queries: list[str],
    conv_id: str,
    top_k: int = RAG_TOP_K,
    rag_type: str = RAG_TYPE_LIGHTRAG,
    working_dir: str = None,
    rag=None,
) -> tuple[list[dict], list[dict]]:
    """统一同步检索入口"""
    if rag_type == RAG_TYPE_NAIVE:
        return naiverag_retrieve_multi(queries, conv_id, top_k)
    if rag_type == RAG_TYPE_LIGHTRAG:
        return lightrag_retrieve_multi(queries, conv_id, top_k, working_dir, rag)
    raise ValueError(f"rag_type must be one of {RAG_TYPE_CHOICES}")


def run_react_lightrag_sync(
    query: str,
    conv_id: str,
    category: int,
    model: str = DEFAULT_MODEL,
    output_dir: Optional[str] = None,
    max_step: int = MAX_LOOP_STEP,
    rag_top_k: int = RAG_TOP_K,
    rag_type: str = RAG_TYPE_LIGHTRAG,
    working_dir: str = None,
    agent_flag_str: str = DEFAULT_AGENT_FLAG,
    middle_scale: int = 3,
    rag=None,
    full_conv=None,
    img_retriever=None,
    benchmark: str = "locomo",
) -> dict:
    """
    运行 ReAct + LightRAG 流水线（async 版本，支持并发调度）。
    同步入口请使用 run_react_lightrag()。
    返回: {answer, stopped_reason, steps, trajectory, final_query_queue}
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"[global-llm] source={get_openai_config_source()}")

    haystack_session_ids = None
    if benchmark == "longmemeval":
        query = _rewrite_first_person_to_user(query)
        question_date = full_conv['question_date']
        query = f"Today is {question_date}. {query}"
        haystack_session_ids = _get_haystack_session_ids(conv_id)
    root_query = query
    query_queue = [query]
    short_memory: list[dict] = []
    trajectory_memory: list[dict] = []
    trajectory: list[dict] = []
    fail_queue_trajectory: list[dict] = []
    agent_flag = [c == "1" for c in agent_flag_str]
    additional_information = add_category_information(category)
    conv_memory = []
    interrupt_step = 3
    visual_seen_session = []

    for step in range(1, max_step + 1):
        temp_useful_rag: list[dict] = []
        short_memory_dia_ids = [m.get("dia_id", "") for m in short_memory]
        print(f"Query_queue: {query_queue}")
        assert len(query_queue) != 0
        # ─── Act: RAG 检索（naive 或 lightrag）query queue 中所有 query ───
        rag_results, rag_result_list = rag_retrieve_multi(
            query_queue, conv_id, top_k=rag_top_k, rag_type=rag_type,
            working_dir=working_dir, rag=rag,
        )
        assert len(rag_results) != 0
        
        if trajectory:
            last_observation = trajectory[-1].get("Observation") or {}
            last_thinking = trajectory[-1].get("Thought") or {}
            last_action = last_observation.get("action", "")
            last_query_queue = last_observation.get("query_queue", "")
            last_queries = f"Last queries: {', '.join(last_query_queue)}"
        else:
            last_queries = ''
            last_action=''
        rag_record = []
        # ─── RAG View agent: 观察 rag_results，选出 useful_dia_ids，写 report ───
        if agent_flag[0]:
            rag_view_result = rag_view_agent_with_cfg(
                root_query=root_query,
                query_queue=query_queue,
                rag_results=rag_results,
                last_thought_information=last_queries,
                known_info_rag=short_memory,
                model=model,
                benchmark=benchmark,
                haystack_session_ids=haystack_session_ids,
            )
            rag_dia = rag_view_result.get("useful_evidence", [])
            rag_view_thinking = rag_view_result.get("thinking", [])
            rag_view_missing_information = rag_view_result.get("missing_information", [])
            rag_dia_ids = [r.get("dia_id", "") for r in rag_dia]
            temp_useful_rag.extend(rag_dia)
            rag_view_record = {
                "rag_view_useful_dia_ids": rag_dia_ids,
                "rag_view_thinking": rag_view_thinking,
                "rag_view_missing_information": rag_view_missing_information,
            }
            rag_record.append(rag_view_record)

        # ─── middle View agent: ───
        if agent_flag[1] and len(temp_useful_rag) > 0:
            K = middle_scale
            middle_view_result = middle_view_agent_with_cfg(
                root_query=root_query,
                query_queue=query_queue,
                temp_useful_rag=temp_useful_rag,
                known_info_rag=short_memory,
                model=model,
                conv_id=conv_id,
                K=middle_scale,
                temperature=0.0,
                benchmark=benchmark,
                haystack_session_ids=haystack_session_ids,
                full_conv=full_conv
            )
            middle_dia = middle_view_result.get("useful_evidence", [])
            middle_view_thinking = middle_view_result.get("thinking", "")
            middle_view_missing = middle_view_result.get("missing_information", "")
            existing_ids = {r.get("dia_id", "") for r in temp_useful_rag}
            for r in middle_dia:
                if r.get("dia_id", "") not in existing_ids:
                    temp_useful_rag.append(r)
                    existing_ids.add(r.get("dia_id", ""))
            middle_view_record = {
                "middle_view_useful_dia_ids": [r.get("dia_id", "") for r in middle_dia],
                "middle_view_thinking": middle_view_thinking,
                "middle_view_missing_information": middle_view_missing,
            }
            rag_record.append(middle_view_record)

        # ─── visual_ocr_agent: ───
        if agent_flag[2] and benchmark != 'longmemeval':
            visual_ocr_result = visual_ocr_agent_with_cfg(
                root_query=root_query,
                query_queue=query_queue,
                temp_useful_rag=temp_useful_rag,
                known_info_rag=short_memory,
                model=model,
                conv_id=conv_id,
                seen_session=visual_seen_session,
                max_view_sessions=6,
                temperature=0.0,
                benchmark=benchmark,
                haystack_session_ids=haystack_session_ids,
            )
            visual_ocr_dia = visual_ocr_result.get("useful_evidence", [])
            visual_ocr_thinking = visual_ocr_result.get("thinking", "")
            visual_ocr_missing = visual_ocr_result.get("missing_information", "")
            viewed = visual_ocr_result.get("add_view_sessions", [])
            sessions_to_view = visual_ocr_result.get("sessions_to_view", [])
            # visual_seen_session.extend(s for s in viewed if s not in visual_seen_session)
            existing_ids = {r.get("dia_id", "") for r in temp_useful_rag}
            for r in visual_ocr_dia:
                if r.get("dia_id", "") not in existing_ids:
                    temp_useful_rag.append(r)
                    existing_ids.add(r.get("dia_id", ""))
            visual_ocr_record = {
                "visual_ocr_useful_dia_ids": [r.get("dia_id", "") for r in visual_ocr_dia],
                "visual_ocr_thinking": visual_ocr_thinking,
                "visual_ocr_missing_information": visual_ocr_missing,
                "add_view_sessions": viewed,
                "visual_ocr_viewed_sessions": sessions_to_view,
            }
            rag_record.append(visual_ocr_record)

        # ─── 更新 known_info_rag ───For thought agent
        temp_useful_rag_dia_ids = [r.get("dia_id", "") for r in temp_useful_rag]
        act_record = {
            "step": step,
            "query_queue": str(query_queue),
            "rag_record": rag_record,
            "temp_useful_rag_dia_ids": temp_useful_rag_dia_ids,
        }

        # ─── Observation ───给出是否能回答的结果
        obs = observation_agent_with_cfg(
            query, temp_useful_rag, short_memory, model,
            fail_queue_trajectory=fail_queue_trajectory,
            temperature=0.0, conv_memory=[], benchmark=benchmark,
            haystack_session_ids=haystack_session_ids
        )
        useful_evidence = obs.get("useful_evidence", [])#TODO:这里可以做个知识更新
        new_queries = obs.get("new_queries", [])
        action = obs.get("action", [])
        fail_query_flag = True
        if len(useful_evidence) > 0:#增加new short memory
            for i in temp_useful_rag:# 遍历临时记忆
                if i['dia_id'] in useful_evidence and i['dia_id'] not in short_memory_dia_ids:#如果临时记忆有的，但是短期记忆没有的
                    short_memory.append(i)
                    fail_query_flag = False
        print(f"No.{step}:")
        if fail_query_flag:
            fail_queue_trajectory.append({"last_action":last_action,"query_queue":query_queue,"rag_view_report":rag_view_missing_information})#TODO:判断真不能靠这个判断obs agent来判断
            fail_queue_trajectory=fail_queue_trajectory[:3]
        obs_record = {
            "can_answer": obs["can_answer"],
            "useful_evidence": useful_evidence,
            "short_memory": [i['dia_id'] for i in short_memory],
            "action": action,
            "new_queries": new_queries,
            # "answer": obs['answer'],
        }

        obs_thinking = obs["thinking"]
        trajectory.append({
            "Act": act_record,
            "Observation": obs_record,
            "Thought": obs_thinking,
        })
        # 轨迹记忆
        trajectory_memory_dia_ids = [r.get("dia_id", "") for r in trajectory_memory]
        for i in short_memory:# 遍历临时记忆
            if i['dia_id'] not in trajectory_memory_dia_ids:#如果临时记忆有的，但是短期记忆没有的
                trajectory_memory.append(i)
        # ─── 机械判断 ───
        if obs["can_answer"] or step == max_step - 1:
            short_memory_dia_ids = [m.get("dia_id", "") for m in short_memory]
            ans = answer_agent_with_cfg(query, short_memory, model=model,
                obs_report=obs_thinking, additional_information=additional_information,
                benchmark=benchmark, haystack_session_ids=haystack_session_ids)
            if ans["answer"] != '':
                answer = ans["answer"]
                answer_thinking = ans["thinking"]
            else:
                continue
            if answer in WRONG_LIST:
                continue
            result = {
                "answer": answer,
                "answer_thinking": answer_thinking,
                "stopped_reason": "can_answer",
                "steps": step,
                "final_evidence": short_memory_dia_ids,
                "final_query_queue": query_queue,
                "trajectory": trajectory,
            }
            _save_json(result, os.path.join(output_dir, "result.json"))
            return result
        # 更新
        assert len(new_queries) != 0
        query_queue = new_queries
    short_memory_dia_ids = [m.get("dia_id", "") for m in short_memory]
    conv_view_result = conv_answer_agent_with_cfg(
        root_query=root_query,
        model=model,
        full_conv=full_conv,
        temperature=0.0,
        benchmark=benchmark,
    )
    conv_view_answer = conv_view_result.get("answer", "")
    conv_view_thinking = conv_view_result.get("thinking", "")
    if benchmark == 'longmemeval':
        conv_view_answer = conv_view_thinking+" Answer is: "+ conv_view_answer
    ans = {
            "answer": conv_view_answer,
            "report": conv_view_thinking
        }
    answer = ans["answer"]
    answer_thinking = ans["report"]

    result = {
        "answer": answer,
        "answer_thinking": answer_thinking,
        "stopped_reason": "max_step",
        "steps": max_step,
        "final_evidence": short_memory_dia_ids,
        "final_query_queue": query_queue,
        "trajectory": trajectory,
    }
    _save_json(result, os.path.join(output_dir, "result.json"))
    print(f"Answer: {answer}")
    return result

def run_react_lightrag(
    query: str, conv_id: str, category: int, model: str = DEFAULT_MODEL,
    output_dir: Optional[str] = None, max_step: int = MAX_LOOP_STEP,
    rag_top_k: int = RAG_TOP_K, rag_type: str = RAG_TYPE_LIGHTRAG,
    working_dir: str = None, agent_flag_str: str = DEFAULT_AGENT_FLAG,
    middle_scale: int = 3, rag=None, full_conv=None, img_retriever=None,
    benchmark: str = "locomo",
) -> dict:
    """Sync wrapper — delegates to run_react_lightrag_async via event-loop bridge."""
    loop = _get_rag_event_loop()
    return run_react_lightrag_sync(   # ← 新增一个 sync 函数，或者直接把下面内容粘贴
        query=query, conv_id=conv_id, category=category, model=model,
        output_dir=output_dir, max_step=max_step, rag_top_k=rag_top_k,
        rag_type=rag_type, working_dir=working_dir,
        agent_flag_str=agent_flag_str, middle_scale=middle_scale,
        rag=rag, full_conv=full_conv, img_retriever=img_retriever,
        benchmark=benchmark,
    )


def _save_json(data, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct + RAG (naive/lightrag) on LoCoMo")
    parser.add_argument("query", nargs="*", help="Query")
    parser.add_argument("-c", "--conv", default="conv-26", help="Conversation ID")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--max-step", type=int, default=MAX_LOOP_STEP)
    parser.add_argument("-k", "--rag-topk", type=int, default=RAG_TOP_K, help=f"RAG top-k (default: {RAG_TOP_K})")
    parser.add_argument(
        "--rag-type",
        choices=RAG_TYPE_CHOICES,
        default=RAG_TYPE_LIGHTRAG,
        help=f"RAG 类型: naive=向量检索(pkl@naiverag), lightrag=LightRAG 知识图谱 (default: {RAG_TYPE_LIGHTRAG})",
    )
    parser.add_argument(
        "--lightrag-base",
        default=DEFAULT_LIGHTRAG_WORKING_BASE,
        help=f"LightRAG 索引根目录 (default: {DEFAULT_LIGHTRAG_WORKING_BASE})",
    )
    parser.add_argument(
        "--agent-flag",
        default=DEFAULT_AGENT_FLAG,
        help=f"5 位 0/1 串控制 agent 开关: rag_view/middle_view/full_view/agentic_graph/visual_search (default: {DEFAULT_AGENT_FLAG})",
    )
    parser.add_argument(
        "--middle-scale",
        type=int,
        default=3,
        help=f"Middle view agent scale K (default: 3)",
    )
    parser.add_argument(
        "--img-index-base",
        default=DEFAULT_IMG_INDEX_BASE,
        help=f"图片索引根目录 (default: {DEFAULT_IMG_INDEX_BASE})",
    )
    # benchmark
    parser.add_argument(
        "--benchmark",
        default="locomo",
        type=str,
        help="Benchmark name for prompt selection (default: locomo)",
    )
    args = parser.parse_args()

    q = " ".join(args.query) if args.query else "When did Caroline go to the LGBTQ support group?"

    # 构建 working_dir
    conv_id = args.conv
    workspace = _conv_id_to_workspace(conv_id)
    working_dir = os.path.join(args.lightrag_base, workspace)
    print(f"working_dir: {working_dir}")

    # 载入 LightRAG（仅初始化一次，后续检索复用同一实例）
    rag = None
    if args.rag_type == RAG_TYPE_LIGHTRAG:
        rag = create_lightrag(working_dir)
    category = 0
    agent_flag_str = args.agent_flag
    if args.benchmark == "longmemeval":
        full_conv = _build_full_conv_context_longmemeval(conv_id)
    else:
        full_conv = _build_full_conv_context(conv_id)
    img_retriever = None
    if len(agent_flag_str) > 4 and agent_flag_str[4] == "1":
        print("初始化视觉搜索...")
        img_retriever = create_img_retriever(conv_id, img_index_base=args.img_index_base)
    try:
        result = run_react_lightrag(
            query=q,
            conv_id=conv_id,
            category=category,
            model=args.model,
            output_dir=args.output_dir,
            max_step=args.max_step,
            rag_top_k=args.rag_topk,
            rag_type=args.rag_type,
            working_dir=working_dir,
            agent_flag_str=agent_flag_str,
            middle_scale=args.middle_scale,
            rag=rag,
            full_conv=full_conv,
            img_retriever=img_retriever,
            benchmark=args.benchmark,
        )
    finally:
        finalize_lightrag(rag)
