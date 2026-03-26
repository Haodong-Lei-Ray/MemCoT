#!/usr/bin/env python3
"""
Build LightRAG index for LongMemEval haystack.

支持两种模式:
  1. 单文件: --input path/to/0000_e47becba.json
  2. 批量目录: --input path/to/dataset/ --batch-size 20
     自动扫描目录下所有 *.json，每个问题独立建库，已建成的自动跳过。

Output: LightRAG working_dir with KV, graph, vector stores.

Chunking: User and assistant turns use different limits (user short, assistant long):
  - User: --user-chunk-token-size (default 4096) to keep turns atomic.
  - Assistant: --assistant-chunk-token-size (default 800), --assistant-chunk-overlap (default 100).

Usage:
  cd /mnt/petrelfs/leihaodong/ICML/locomo
  source scripts/env.sh

  # 单文件
  python benchmark/LongMemEval/tool/buildrag/build_lightrag_longmemeval.py \
    --input benchmark/LongMemEval/dataset/0000_e47becba.json \
    --output-dir /path/to/rag_storage/0000_e47becba

  # 批量（并发 20）
  python benchmark/LongMemEval/tool/buildrag/build_lightrag_longmemeval.py \
    --input benchmark/LongMemEval/dataset/ \
    --output-dir /path/to/rag_output/ \
    --batch-size 20
"""

import argparse
import json
import os
import json as _json
import re
import sys
import asyncio
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
# /home/lei/Project/DMSMem/benchmark/longmemevaltool/tool/buildrag -> /home/lei/Project/DMSMem
PROJECT_ROOT = SCRIPT_DIR.parents[3]
LIGHTRAG_PATH = PROJECT_ROOT / "mem" / "LightRAG"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIGHTRAG_PATH))

if os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

from global_methods import set_openai_key
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc
from vllm_llm import _make_llm_model_func

# Defaults
RAG_STORAGE_BASE = "../memory/lightrag/rag_storage"
# User turns are typically short (30-150 tokens); use large limit to avoid splitting.
USER_CHUNK_TOKEN_SIZE = 4096
# Assistant turns are longer (300-800+ tokens); use moderate size for entity/relation extraction.
ASSISTANT_CHUNK_TOKEN_SIZE = 800
ASSISTANT_CHUNK_OVERLAP = 100
ENTITY_TYPES = ["Person", "Event", "Location", "Organization", "Activity", "Concept", "Date", "Other"]


def _get_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _split_text_by_tokens(text: str, tokenizer, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """Split text into chunks of at most max_tokens. Returns list of strings."""
    if not text:
        return []
    if tokenizer is None:
        # Fallback: ~4 chars per token for English
        approx_chars = max_tokens * 4
        if len(text) <= approx_chars:
            return [text]
        chunks = []
        start = 0
        overlap_chars = overlap_tokens * 4
        while start < len(text):
            end = min(start + approx_chars, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap_chars
        return chunks

    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - overlap_tokens
    return chunks


def format_haystack_as_docs(
    entry: dict,
    user_chunk_token_size: int = USER_CHUNK_TOKEN_SIZE,
    assistant_chunk_token_size: int = ASSISTANT_CHUNK_TOKEN_SIZE,
    user_chunk_overlap: int = 0,
    assistant_chunk_overlap: int = ASSISTANT_CHUNK_OVERLAP,
) -> tuple[list[str], list[str], list[str]]:
    """
    Format haystack as list of doc strings for LightRAG.

    User and assistant turns use different chunk sizes:
    - User: typically short; large limit (4096) to keep turns atomic.
    - Assistant: typically long; moderate limit (800) for entity/relation extraction.

    - doc_ids: from haystack_session_ids, format sess_id + "_" + turn_idx [ + "_c" + chunk_idx ]
    - file_paths: haystack_dates (time) for citation
    - speaker: use role as-is ("user", "assistant")
    """
    tokenizer = _get_tokenizer()
    max_chunk = max(user_chunk_token_size, assistant_chunk_token_size)
    if not tokenizer and max_chunk < 10000:
        print("Warning: tiktoken not available, using character-based approx (4 chars/token)")

    session_ids = entry.get("haystack_session_ids", [])
    dates = entry.get("haystack_dates", [])
    sessions = entry.get("haystack_sessions", [])
    sess_id_counts = Counter(session_ids)
    has_duplicate_sess_ids = any(c > 1 for c in sess_id_counts.values())

    doc_texts = []
    doc_ids = []
    file_paths = []

    for si, (sess_id, date_time, session) in enumerate(zip(session_ids, dates, sessions)):
        ts = date_time or "unknown"
        for ti, turn in enumerate(session):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            speaker = role

            # Role-specific chunk size and overlap
            if role == "assistant":
                chunk_size = assistant_chunk_token_size
                overlap = assistant_chunk_overlap
            else:
                chunk_size = user_chunk_token_size
                overlap = user_chunk_overlap

            # 当 haystack_session_ids 中有重复 sess_id 时，用 si 保证 doc_id 唯一
            if has_duplicate_sess_ids:# 不可能了，不要在乎这个
                base_id = f"{si}_{ti + 1}"
            else:
                base_id = f"{sess_id}_{ti + 1}"

            if not tokenizer:
                n_tokens_approx = len(content) // 4
                need_chunk = n_tokens_approx > chunk_size
                chunks = _split_text_by_tokens(content, None, chunk_size, overlap) if need_chunk else [content]
            else:
                chunks = _split_text_by_tokens(content, tokenizer, chunk_size, overlap)

            for ci, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                escaped = chunk_text.replace("\\", "\\\\").replace('"', '\\"')
                line = speaker + ' said, "' + escaped + '"'
                doc_texts.append(line)
                doc_ids.append(f"{base_id}_c{ci}" if len(chunks) > 1 else base_id)
                file_paths.append(ts)

    return doc_texts, doc_ids, file_paths


def _load_existing_doc_ids(working_dir: str) -> set[str]:
    """从 kv_store_full_docs.json 读取已入库的 doc_id 集合。
    文件顶层 key 即 doc_id（与 _id 字段一致），格式为
    {session_id}_{turn_idx}[_c{chunk_idx}]。
    """
    kv_path = os.path.join(working_dir, "kv_store_full_docs.json")
    if not os.path.isfile(kv_path):
        return set()
    try:
        with open(kv_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        if isinstance(data, dict):
            return set(data.keys())
        return set()
    except Exception:
        return set()


async def build_single(
    input_path: Path,
    working_dir: str,
    user_chunk_token_size: int = USER_CHUNK_TOKEN_SIZE,
    assistant_chunk_token_size: int = ASSISTANT_CHUNK_TOKEN_SIZE,
    assistant_chunk_overlap: int = ASSISTANT_CHUNK_OVERLAP,
    embedding_model: str = "text-embedding-3-small",
    llm_mode: str = "openai",
    vllm_config: Path = Path(__file__).resolve().parent / "config" / "config.json",
    llm_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
):
    """为单个问题构建 LightRAG 索引。"""
    set_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Run: source scripts/env.sh")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        entry = json.load(f)

    if verbose:
        print(f"Formatting haystack from {input_path}...")
    doc_texts, doc_ids, file_paths = format_haystack_as_docs(
        entry,
        user_chunk_token_size=user_chunk_token_size,
        assistant_chunk_token_size=assistant_chunk_token_size,
        assistant_chunk_overlap=assistant_chunk_overlap,
    )
    if verbose:
        print(f"  {len(doc_texts)} documents")

    os.makedirs(working_dir, exist_ok=True)
    lightrag_chunk_size = max(user_chunk_token_size, assistant_chunk_token_size)

    if llm_kwargs is None:
        llm_kwargs = {}

    if llm_mode == "vllm":
        llm_model_func = _make_llm_model_func(
            vllm_config_path=vllm_config,
            llm_kwargs=llm_kwargs,
        )
    else:
        llm_model_func = gpt_4o_mini_complete

    print(f"\nLightRAG working_dir: {working_dir}")
    print("Hyperparameters:")
    print(f"  - user_chunk_token_size: {user_chunk_token_size}")
    print(f"  - assistant_chunk_token_size: {assistant_chunk_token_size}")
    print(f"  - assistant_chunk_overlap: {assistant_chunk_overlap}")
    print(f"  - LightRAG chunk_token_size: {lightrag_chunk_size}")
    print(f"  - embedding: {embedding_model}")
    print(f"  - llm_mode: {llm_mode}")
    if llm_mode == "vllm":
        print(f"  - vllm_config: {vllm_config}")
        if llm_kwargs:
            print(f"  - llm_kwargs: {llm_kwargs}")
    print("")

    # LightRAG requires EmbeddingFunc (dataclass), not functools.partial.
    if embedding_model == "text-embedding-3-small":
        _embedding_func = openai_embed
    # 每个实例必须用唯一 workspace，否则共享 pipeline_status 导致并发时互相阻塞
    rag = LightRAG(
        working_dir=working_dir,
        workspace=working_dir,
        embedding_func=_embedding_func,
        llm_model_func=llm_model_func,
        chunk_token_size=lightrag_chunk_size,
        chunk_overlap_token_size=assistant_chunk_overlap,
        addon_params={"entity_types": ENTITY_TYPES, "language": "English"},
    )

    print("Initializing LightRAG storages...")
    await rag.initialize_storages()

    # ── 增量去重：读取已有 kv_store_full_docs.json，跳过已入库文档 ──
    existing_ids = _load_existing_doc_ids(working_dir)
    if existing_ids:
        before = len(doc_ids)
        keep = [(t, d, p) for t, d, p in zip(doc_texts, doc_ids, file_paths)
                if d not in existing_ids]
        doc_texts = [x[0] for x in keep]
        doc_ids   = [x[1] for x in keep]
        file_paths = [x[2] for x in keep]
        print(f"{working_dir}/n增量去重: 已有 {len(existing_ids)} 条, "
              f"本次需插入 {len(doc_ids)}/{before} 条 "
              f"(跳过 {before - len(doc_ids)} 条)")

    if not doc_ids:
        print("所有文档已入库，无需插入，跳过 ainsert。")
    else:
        print("Inserting haystack into LightRAG...")
        await rag.ainsert(doc_texts, ids=doc_ids, file_paths=file_paths)
        print(f"{working_dir}  Done.")

    q = entry.get("question", "What did the user mention?")
    if verbose:
        print(f"Testing query: {q[:80]}...")
    from lightrag import QueryParam
    result = await rag.aquery(q, param=QueryParam(mode="hybrid", top_k=10))
    if verbose:
        print(f"  Answer (excerpt): {result[:200]}...")

    await rag.finalize_storages()

    # 写完成标记，供断点续跑时判断
    total_in_store = len(_load_existing_doc_ids(working_dir))
    done_marker = os.path.join(working_dir, "_BUILD_DONE")
    with open(done_marker, "w") as f:
        f.write(f"input={input_path.name}\n"
                f"total_docs_in_store={total_in_store}\n"
                f"inserted_this_run={len(doc_ids)}\n")
    if verbose:
        print(f"LightRAG index built: {input_path.name}")


# ── 批量并发 ──────────────────────────────────────────────────────

def _name_from_stem(stem: str) -> str:
    """保留原始 stem（包含前缀数字）。"""
    return stem


_DONE_MARKER = "_BUILD_DONE"


def _is_already_built(out_dir: Path) -> bool:
    """严格判定：输出目录存在且含 _BUILD_DONE 标记文件才视为已构建。
    目录存在但无标记（中途崩溃的残库）会被视为未构建，触发重建。
    """
    return (out_dir / _DONE_MARKER).is_file()


def _clean_incomplete(out_dir: Path) -> None:
    """清除残库目录，为重建做准备。"""
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)


async def _build_one_wrapped(
    input_path: Path,
    out_dir: Path,
    sem: asyncio.Semaphore,
    clean_incomplete: bool = False,
    **kwargs,
) -> tuple[str, bool, str]:
    """带信号量的单任务包装，返回 (filename, success, message)。
    clean_incomplete=True 时先删残库再重建（非增量）；
    clean_incomplete=False（默认）时依赖 build_single 内的增量去重。
    """
    async with sem:
        try:
            if clean_incomplete and out_dir.exists() and not (out_dir / _DONE_MARKER).is_file():
                _clean_incomplete(out_dir)
            await build_single(input_path, str(out_dir), verbose=False, **kwargs)
            return (input_path.name, True, "OK")
        except Exception as e:
            return (input_path.name, False, f"{e}\n{traceback.format_exc()[-300:]}")


async def build_batch(
    dataset_dir: Path,
    output_base: Path,
    batch_size: int = 20,
    **kwargs,
):
    """扫描 dataset_dir 下所有 *.json，并发构建，已建成的自动跳过。"""
    set_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Run: source scripts/env.sh")
        sys.exit(1)

    all_files = sorted(
        f for f in dataset_dir.iterdir()
        if f.is_file() and f.suffix == ".json"
    )

    to_build = []
    skipped = 0
    for f in all_files:
        name = _name_from_stem(f.stem)
        out_dir = output_base / name
        if _is_already_built(out_dir):
            skipped += 1
        else:
            to_build.append((f, out_dir))

    print(f"共 {len(all_files)} 个问题, 已构建跳过 {skipped} 个, 待构建 {len(to_build)} 个")
    if not to_build:
        print("无待构建任务，退出")
        return

    sem = asyncio.Semaphore(batch_size)
    tasks = [
        _build_one_wrapped(inp, out, sem, **kwargs)
        for inp, out in to_build
    ]

    success = 0
    fail_list = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        name, ok, msg = await coro
        if ok:
            success += 1
            print(f"[{success + len(fail_list)}/{len(to_build)}] OK: {name}")
        else:
            fail_list.append((name, msg))
            print(f"[{success + len(fail_list)}/{len(to_build)}] FAIL: {name} - {msg.splitlines()[0]}")

    print(f"\n完成: 成功 {success}, 失败 {len(fail_list)}")
    if fail_list:
        print("失败列表:")
        for n, m in fail_list:
            print(f"  {n}: {m.splitlines()[0]}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build LightRAG index for LongMemEval haystack. "
        "支持单文件或批量目录（--batch-size 并发）。"
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="输入: 单个 JSON 文件 或 包含多个 JSON 的目录",
    )
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="输出 RAG 目录（批量时为父目录）")
    parser.add_argument(
        "--batch-size", "-b", type=int, default=1,
        help="并发数（仅目录模式生效，默认 1 即串行）",
    )
    parser.add_argument(
        "--user-chunk-token-size",
        type=int,
        default=USER_CHUNK_TOKEN_SIZE,
        help=f"Max tokens per user turn (default: {USER_CHUNK_TOKEN_SIZE})",
    )
    parser.add_argument(
        "--assistant-chunk-token-size",
        type=int,
        default=ASSISTANT_CHUNK_TOKEN_SIZE,
        help=f"Max tokens per assistant chunk (default: {ASSISTANT_CHUNK_TOKEN_SIZE})",
    )
    parser.add_argument(
        "--assistant-chunk-overlap",
        type=int,
        default=ASSISTANT_CHUNK_OVERLAP,
        help=f"Overlap tokens for assistant chunks (default: {ASSISTANT_CHUNK_OVERLAP})",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model")
    parser.add_argument(
        "--llm-mode",
        choices=["openai", "vllm"],
        default="openai",
        help="LLM 后端模式：openai 使用 gpt_4o_mini，vllm 使用 create_vllm_complete",
    )
    parser.add_argument(
        "--vllm-config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "config.json",
        help="vLLM 参数配置文件（默认读取本目录 config/config.json）",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        print(f"Error: input not found: {args.input}")
        sys.exit(1)

    common_kwargs = dict(
        user_chunk_token_size=args.user_chunk_token_size,
        assistant_chunk_token_size=args.assistant_chunk_token_size,
        assistant_chunk_overlap=args.assistant_chunk_overlap,
        embedding_model=args.embedding_model,
        llm_mode=args.llm_mode,
        vllm_config=args.vllm_config,
        llm_kwargs={},
    )

    if args.input.is_dir():
        # ── 批量模式 ──
        output_base = args.output_dir
        if output_base is None:
            output_base = Path(RAG_STORAGE_BASE) / "longmemeval"
        output_base.mkdir(parents=True, exist_ok=True)
        asyncio.run(
            build_batch(
                args.input,
                output_base,
                batch_size=args.batch_size,
                **common_kwargs,
            )
        )
    else:
        # ── 单文件模式（向后兼容） ──
        out_dir = args.output_dir
        if out_dir is None:
            out_dir = Path(RAG_STORAGE_BASE) / f"longmemeval_{args.input.stem}"
        asyncio.run(
            build_single(args.input, str(out_dir), **common_kwargs)
        )
