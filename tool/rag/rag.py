"""
LightRAG / NaiveRAG 检索与实例生命周期。供 memcot、评测脚本等复用。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# MemCoT 项目根（本文件位于 tool/rag/rag.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIGHTRAG_PATH = PROJECT_ROOT / "mem" / "LightRAG"
sys.path.insert(0, str(LIGHTRAG_PATH))

try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass

RAG_TOP_K = 5
RAG_TYPE_NAIVE = "naive"
RAG_TYPE_LIGHTRAG = "lightrag"
RAG_TYPE_CHOICES = (RAG_TYPE_NAIVE, RAG_TYPE_LIGHTRAG)

DEFAULT_IMG_INDEX_BASE = "/mnt/petrelfs/leihaodong/ICML/exp/memory/img_mem/vit_rag"


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


async def _lightrag_retrieve_async(
    query: str,
    conv_id: str,
    top_k: int = RAG_TOP_K,
    working_dir: str = None,
    rag=None,
) -> list[dict]:
    """LightRAG aquery_data 检索。返回 [{dia_id, date_time, context, score, from_query}, ...]"""
    if rag is None:
        raise ValueError("rag instance must be provided")
    if working_dir is None:
        raise ValueError("working_dir must be provided")

    from lightrag import QueryParam

    param = QueryParam(mode="hybrid", top_k=top_k, chunk_top_k=top_k)
    result = await rag.aquery_data(query, param)

    if result.get("status") != "success" or not result.get("data", {}).get("chunks"):
        return []

    chunks = result["data"]["chunks"]
    chunk_id_to_doc = _get_chunk_id_to_doc_id(working_dir)
    results = []
    for i, c in enumerate(chunks):
        chunk_id = c.get("chunk_id", "")
        dia_id = chunk_id_to_doc.get(chunk_id, chunk_id)
        date_time = c.get("file_path", "")
        context = c.get("content", "")
        score = 1.0 - (i * 0.01)
        results.append(
            {
                "dia_id": dia_id,
                "date_time": date_time,
                "context": context,
                "score": float(score),
                "from_query": query,
            }
        )
    return results


def get_rag_event_loop():
    """供检索与 LightRAG 初始化使用的 event loop（与历史 _get_rag_event_loop 行为一致）。"""
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


def create_lightrag(working_dir: str):
    """创建并初始化 LightRAG 实例（同步接口），避免每次检索都重新创建。"""
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
    loop = get_rag_event_loop()
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
            print("    已取消")

    max_wait = 5
    if tasks and not loop.is_closed():
        print("\n步骤3: 等待任务取消...")
        try:
            loop.run_until_complete(
                asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=max_wait,
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

    loop = get_rag_event_loop()
    try:
        loop.run_until_complete(rag.finalize_storages())
    except Exception:
        pass
    loop.run_until_complete(rag.initialize_storages())
    return rag


def create_img_retriever(
    conv_id: str, img_index_base: str = DEFAULT_IMG_INDEX_BASE, top_k: int = 3
):
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
    queries: list[str],
    conv_id: str,
    top_k: int = RAG_TOP_K,
    working_dir: str = None,
    rag=None,
) -> tuple[list[dict], list[dict]]:
    """对多个 query 执行 LightRAG 检索，合并去重（按 dia_id）。"""

    loop = get_rag_event_loop()

    async def _run_all():
        all_results = []
        result_list = []
        for q in queries:
            res = await _lightrag_retrieve_async(q, conv_id, top_k, working_dir, rag=rag)
            one_result = {"query": q, "res": res}
            all_results.extend(res)
            result_list.append(one_result)
        return all_results, result_list

    results, result_list = loop.run_until_complete(_run_all())
    seen = set()
    deduped = []
    for r in results:
        did = r.get("dia_id", "")
        if did and did not in seen:
            seen.add(did)
            deduped.append(r)
    return deduped, result_list


def naiverag_retrieve_multi(
    queries: list[str],
    conv_id: str,
    top_k: int = RAG_TOP_K,
    working_dir: str = None,
) -> tuple[list[dict], list[dict]]:
    """
    Naive RAG: 从 naiverag 目录加载 pkl，向量检索。返回 (deduped, result_list)。
    """
    import pickle

    import numpy as np
    from global_methods import get_openai_embedding

    NAIVERAG_BASE = os.path.dirname(working_dir)
    pkl_path = os.path.join(NAIVERAG_BASE, f"locomo10_dialog_{conv_id}.pkl")
    print(f"[rag workdir] {pkl_path}")

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
    if isinstance(queries, str):
        queries = [queries]
    for q in queries:
        query_emb = get_openai_embedding([q])
        scores = np.dot(query_emb, np.array(embeddings).T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        res = [
            {
                "dia_id": dia_ids[idx],
                "date_time": date_times[idx],
                "context": contexts[idx],
                "score": float(scores[idx]),
                "from_query": q,
            }
            for idx in top_indices
        ]
        all_results.extend(res)
        result_list.append({"query": q, "res": res})

    seen = set()
    deduped = []
    for r in all_results:
        did = r.get("dia_id", "")
        if did and did not in seen:
            seen.add(did)
            deduped.append(r)
    return deduped, result_list


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
        return naiverag_retrieve_multi(queries, conv_id, top_k, working_dir)
    if rag_type == RAG_TYPE_LIGHTRAG:
        return lightrag_retrieve_multi(queries, conv_id, top_k, working_dir, rag=rag)
    raise ValueError(f"rag_type must be one of {RAG_TYPE_CHOICES}, got {rag_type!r}")


def finalize_lightrag(rag):
    """关闭 LightRAG 实例的存储。"""
    if rag is None:
        return
    try:
        loop = get_rag_event_loop()
        loop.run_until_complete(rag.finalize_storages())
    except Exception as e:
        print(f"[finalize_lightrag] Warning: {e}")


# 兼容旧名（若外部仍引用 _get_rag_event_loop）
_get_rag_event_loop = get_rag_event_loop
