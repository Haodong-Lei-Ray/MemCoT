"""
LightRAG / NaiveRAG 检索与实例生命周期。供 memcot、评测脚本等复用。
"""
from __future__ import annotations

import asyncio
import json
import os
import pickle
import sys
import time
from pathlib import Path
from agent.conversation import Conversation

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


class LightRagRetriever:
    """LightRAG 单次检索配置：`top_k`、`working_dir`、已初始化的 `rag` 实例与 `QueryParam`。"""

    def __init__(self, working_dir: str, rag, top_k: int = RAG_TOP_K):
        if working_dir is None:
            raise ValueError("working_dir must be provided")
        self.working_dir = working_dir
        self.rag_type='lightrag'
        self.top_k = top_k
        self.light_rag = rag
        from lightrag import QueryParam
        self.param = QueryParam(mode="hybrid", top_k=top_k, chunk_top_k=top_k)
    
    async def _lightrag_retrieve_async(
        self,
        query: str,
    ) -> list[dict]:
        """LightRAG aquery_data 检索。返回 [{dia_id, date_time, context, score, from_query}, ...]"""
        working_dir = self.working_dir
        light_rag = self.light_rag
        if light_rag is None:
            raise ValueError("rag instance must be provided")

        result = await light_rag.aquery_data(query, self.param)

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
    
    def retrieve_multi(self, queries: list[str]) -> tuple[list[dict], list[dict]]:
        """对多个 query 执行 LightRAG 检索，合并去重（按 dia_id）。"""

        loop = get_rag_event_loop()

        async def _run_all():
            all_results = []
            result_list = []
            for q in queries:
                res = await self._lightrag_retrieve_async(q)
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


from tool.show import cil
class NaiveRagRetriever:
    def __init__(self, working_dir: str, conversation_base: str):
        self.working_dir = working_dir
        self.rag_type = "naive"
        # openclaw
        self.conversation_base = conversation_base
    # openclaw
    def get_session_list(self):
        import subprocess
        import json
        import os

        session_file = os.path.join(self.working_dir, "session.json")
        if os.path.exists(session_file):
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            try:
                result = subprocess.run(["openclaw", "sessions", "--json"], capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
            except Exception as e:
                print(f"Error running openclaw sessions --json: {e}")
                data = {"sessions": []}
            idx = 0
            for item in data.get("sessions", []):
                item["rag_status"] = "fail"
                item["index"] = idx
                idx += 1
            os.makedirs(self.working_dir, exist_ok=True)
            data["session_file"] = session_file
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        cil.show_session_list(data)
        return data
    
    def wash_data(self, idx=0):
        import json
        import os
        import re
        from datetime import datetime, timezone, timedelta
        
        # ==========================================
        # 1. 获取目标 session 的基本信息
        # 从 session.json 中读取并找到对应 idx 的 sessionId
        # ==========================================
        session_file = os.path.join(self.working_dir, "session.json")
        if not os.path.exists(session_file):
            raise FileNotFoundError(f"Session file not found: {session_file}")
            
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        sessions = data.get("sessions", [])
        target_session = None
        for s in sessions:
            if s.get("index") == idx:
                target_session = s
                break
                
        if not target_session:
            raise ValueError(f"Session with index {idx} not found in {session_file}")
            
        session_id = target_session.get("sessionId")
        if not session_id:
            raise ValueError(f"Session with index {idx} has no sessionId")
            
        # ==========================================
        # 2. 定位原始 .jsonl 文件并准备输出的清洗文件路径
        # ==========================================
        raw_file = os.path.join(self.conversation_base, f"{session_id}.jsonl")
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw session file not found: {raw_file}")
            
        out_dir = os.path.join(self.working_dir, "session")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{session_id}.json")
        
        # ==========================================
        # 3. 读取已存在的清洗文件，保留历史的 rag_status 状态
        # 避免重复 embedding 已经处理过的数据
        # ==========================================
        existing_sessions = {}
        if os.path.exists(out_file):
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                for item in existing_data.get("session", []):
                    existing_sessions[item.get("dia_id")] = item
            except Exception:
                pass
            
        # ==========================================
        # 4. 逐行解析原始 .jsonl 文件，提取并格式化有效的 message 数据
        # 处理 user 和 assistant 的时间、内容等字段
        # ==========================================
        washed_session = []
        total_num = self.get_washable_count(raw_file)
        wash_num = 0
        
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                    
                if item.get("type") != "message":
                    continue
                    
                msg = item.get("message", {})
                role = msg.get("role")
                dia_id = item.get("id", "")
                
                if role == "user":
                    content_list = msg.get("content", [])
                    text = ""
                    for c in content_list:
                        if c.get("type") == "text":
                            text += c.get("text", "")
                    
                    match = re.search(r'\n\n\[(.*?)\]\s*(.*)', text, re.DOTALL)
                    if match:
                        date_time = match.group(1)
                        context = match.group(2).strip()
                    else:
                        date_time = ""
                        context = text.strip()
                        
                    rag_status = "fail"
                    if dia_id in existing_sessions:
                        rag_status = existing_sessions[dia_id].get("rag_status", "fail")
                        
                    washed_session.append({
                        "dia_id": dia_id,
                        "role": role,
                        "date_time": date_time,
                        "context": context,
                        "rag_status": rag_status,
                    })
                    wash_num += 1
                    
                elif role == "assistant":
                    ts_ms = msg.get("timestamp")
                    if ts_ms:
                        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone(timedelta(hours=8)))
                        date_time = dt.strftime("%a %Y-%m-%d %H:%M GMT+8")
                    else:
                        date_time = ""
                        
                    content_list = msg.get("content", [])
                    text = ""
                    for c in content_list:
                        if c.get("type") == "text":
                            text += c.get("text", "")
                    
                    if not text:
                        text = msg.get("errorMessage", "")
                        
                    context = text.strip()
                    rag_status = "fail"
                    if dia_id in existing_sessions:
                        rag_status = existing_sessions[dia_id].get("rag_status", "fail")
                        
                    washed_session.append({
                        "dia_id": dia_id,
                        "role": role,
                        "date_time": date_time,
                        "context": context,
                        "rag_status": rag_status,
                    })
                    wash_num += 1
                    
        # ==========================================
        # 5. 组装最终数据并保存到清洗后的 json 文件中
        # ==========================================
        result = {
            "session_id": session_id,
            "wash_num": wash_num,
            "session": washed_session
        }
        
        out_dir = os.path.join(self.working_dir, "session")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{session_id}.json")
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"[🦉 MemCoT] Washed data saved to {out_file}")
        return out_file

    def build_rag(self, idx=0):
        import json
        import os
        import pickle
        import numpy as np
        from global_methods import get_openai_embedding
        
        # ==========================================
        # 1. 获取目标 session 的基本信息
        # 从 session.json 中读取并找到对应 idx 的 sessionId
        # ==========================================
        session_file = os.path.join(self.working_dir, "session.json")
        if not os.path.exists(session_file):
            raise FileNotFoundError(f"Session file not found: {session_file}")
            
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        sessions = data.get("sessions", [])
        target_session = None
        for s in sessions:
            if s.get("index") == idx:
                target_session = s
                break
                
        if not target_session:
            raise ValueError(f"Session with index {idx} not found in {session_file}")
            
        session_id = target_session.get("sessionId")
        if not session_id:
            raise ValueError(f"Session with index {idx} has no sessionId")
            
        # ==========================================
        # 2. 检查数据是否需要清洗 (同步状态)
        # 对比原始文件的可清洗数量与已清洗文件中的 wash_num
        # ==========================================
        raw_file = os.path.join(self.conversation_base, f"{session_id}.jsonl")
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw session file not found: {raw_file}")
            
        total_num = self.get_washable_count(raw_file)
        
        out_dir = os.path.join(self.working_dir, "session")
        out_file = os.path.join(out_dir, f"{session_id}.json")
        
        needs_wash = True
        if os.path.exists(out_file):
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    washed_data = json.load(f)
                if washed_data.get("wash_num", 0) == total_num:
                    needs_wash = False
            except Exception:
                pass
                
        if needs_wash:
            print(f"[🦉 MemCoT] Wash data not synced or missing. Running wash_data...")
            self.wash_data(idx)
            with open(out_file, "r", encoding="utf-8") as f:
                washed_data = json.load(f)
                
        # ==========================================
        # 3. 筛选出尚未 embedding 的数据 (rag_status == "fail")
        # ==========================================
        fail_items = [item for item in washed_data.get("session", []) if item.get("rag_status") == "fail"]
        
        if not fail_items:
            print("[🦉 MemCoT] All data is embedded.")
            return
            
        print(f"[🦉 MemCoT] Found {len(fail_items)} items to embed.")
        
        # ==========================================
        # 4. 加载已存在的 embedding 数据库 (.pkl)
        # ==========================================
        emb_dir = os.path.join(self.working_dir, "emb")
        os.makedirs(emb_dir, exist_ok=True)
        pkl_path = os.path.join(emb_dir, f"{session_id}.pkl")
        
        database = {'embeddings': [], 'date_time': [], 'dia_id': [], 'context': []}
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    database = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load existing pkl {pkl_path}: {e}")
                
        # ==========================================
        # 5. 分批调用 OpenAI API 生成新的 embedding
        # ==========================================
        texts_to_embed = []
        for item in fail_items:
            role = item.get("role", "")
            context = item.get("context", "")
            embedding_text = f"{role} said, {context}"
            texts_to_embed.append(embedding_text)
            
        batch_size = 100
        new_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch_inputs = texts_to_embed[i:i+batch_size]
            batch_emb = get_openai_embedding(batch_inputs)
            new_embeddings.extend(batch_emb)
            
        # ==========================================
        # 6. 更新数据库内容并将状态标记为 "success"
        # ==========================================
        if len(database['embeddings']) > 0:
            database['embeddings'] = np.vstack([database['embeddings'], np.array(new_embeddings)])
        else:
            database['embeddings'] = np.array(new_embeddings)
            
        for item in fail_items:
            database['date_time'].append(item.get("date_time", ""))
            database['dia_id'].append(item.get("dia_id", ""))
            database['context'].append(item.get("context", ""))
            item["rag_status"] = "success"
            
        # ==========================================
        # 7. 保存更新后的数据库 (.pkl) 和清洗数据 (.json)
        # ==========================================
        with open(pkl_path, "wb") as f:
            pickle.dump(database, f)
            
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(washed_data, f, ensure_ascii=False, indent=2)
            
        print(f"[🦉 MemCoT] Successfully embedded {len(fail_items)} items and saved to {pkl_path}")

    def get_washable_count(self, raw_file: str) -> int:
        import json

        total_num = 0
        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("type") == "message":
                    total_num += 1
        return total_num

    def load_rag(self,conv_id,top_k):
        self.top_k = top_k
        self.conv_id = conv_id
        pkl_path = os.path.join(self.working_dir, f"{self.conv_id}.pkl")
        with open(pkl_path, "rb") as f:
            self.db = pickle.load(f)
        print(f"[🦉 MemCoT] rag workdir: {pkl_path}")

    def retrieve_multi(
        self,
        queries: list[str],
    ) -> tuple[list[dict], list[dict]]:
        """
        Naive RAG: 从 naiverag 目录加载 pkl，向量检索。返回 (deduped, result_list)。
        """
        import pickle

        import numpy as np
        from global_methods import get_openai_embedding

        embeddings = self.db.get("embeddings")
        dia_ids = self.db.get("dia_id", [])
        date_times = self.db.get("date_time", [])
        contexts = self.db.get("context", [])

        if embeddings is None or len(dia_ids) == 0:
            return [], [{"query": q, "res": []} for q in queries]

        all_results = []
        result_list = []
        if isinstance(queries, str):
            queries = [queries]
        for q in queries:
            query_emb = get_openai_embedding([q])
            scores = np.dot(query_emb, np.array(embeddings).T).flatten()
            top_indices = np.argsort(scores)[::-1][:self.top_k]
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

def _load_rag_config(path: str | Path) -> dict:
    defaults = {
        "rag_topk": 10,
        "rag_type": RAG_TYPE_LIGHTRAG,
        "rag_base": '../memory/rag_storage',
        "benchmark": "locomo",
        "conversation_base": str(PROJECT_ROOT / "benchmark" / "locomo" / "data" / "con"),
    }
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rag config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = defaults.copy()
    cfg.update(raw)
    return cfg

def load_rag_retrieve(
    rag_file_path: str,
    conv_id: str | None = None,
    top_k: int | None = None,
):
    """根据 rag.json + conv_id 创建检索器实例。"""
    rag_cfg = _load_rag_config(rag_file_path)
    if not conv_id:
        raise ValueError("conv_id is required (pass it or set 'conv-id' in rag config)")

    conversation_base = rag_cfg["conversation_base"]
    benchmark = str(rag_cfg['benchmark'])
    rag_type = str(rag_cfg["rag_type"])
    rag_base = str(rag_cfg["rag_base"])
    if top_k is None:
        top_k = rag_cfg["rag_topk"]

    print(f"[🦉 MemCoT] Rag Base: {rag_base}")

    if rag_type == RAG_TYPE_NAIVE:
        working_dir = rag_base
        ragretriever = NaiveRagRetriever(working_dir, conversation_base)
        ragretriever.load_rag(conv_id)
    elif rag_type == RAG_TYPE_LIGHTRAG:
        workspace = conv_id.replace("-", "")
        working_dir = os.path.join(rag_base, workspace)
        rag = create_lightrag(working_dir)
        ragretriever = LightRagRetriever(working_dir, rag, top_k)
    else:
        raise ValueError(f"rag_type must be one of {RAG_TYPE_CHOICES}, got {rag_type!r}")
    conversation = Conversation(conv_id, benchmark, conversation_base)
    return ragretriever, conversation

def build_rag_retrieve(
    rag_file_path: str
):
    """根据 rag.json + conv_id 创建检索器实例。"""
    rag_cfg = _load_rag_config(rag_file_path)

    conversation_base = rag_cfg["conversation_base"]
    benchmark = str(rag_cfg['benchmark'])
    rag_type = str(rag_cfg["rag_type"])
    rag_base = str(rag_cfg["rag_base"])

    print(f"[🦉 MemCoT] Rag Base: {rag_base}")

    if rag_type == RAG_TYPE_NAIVE:
        working_dir = rag_base
        ragretriever = NaiveRagRetriever(working_dir, conversation_base)
    elif rag_type == RAG_TYPE_LIGHTRAG:
        workspace = conv_id.replace("-", "")
        working_dir = os.path.join(rag_base, workspace)
        rag = create_lightrag(working_dir)
        ragretriever = LightRagRetriever(working_dir, rag, top_k)
    else:
        raise ValueError(f"rag_type must be one of {RAG_TYPE_CHOICES}, got {rag_type!r}")
    # conversation = Conversation(conv_id, benchmark, conversation_base)
    conversation = None
    return ragretriever, conversation

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
