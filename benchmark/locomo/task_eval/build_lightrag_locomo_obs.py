#!/usr/bin/env python3
"""
Build LightRAG index for LoCoMo conversation (conv-26, conv-30, etc.).

Adapted from build_hybrid_rag_db.py (naive RAG / Hybrid RAG script).
Uses rag_mode == 'dialog': each dialog turn is a separate document/chunk,
aligned with evaluate_qa.py naive RAG (gpt_utils.prepare_for_rag rag_mode='dialog').

Usage:
  pip install -e module_unit/LightRAG   # one-time: install LightRAG deps
  cd /mnt/petrelfs/leihaodong/ICML/locomo
  source scripts/env.sh
  python3 task_eval/build_lightrag_locomo.py [--conv-id conv-26]
  # or: bash scripts/build_lightrag_conv26.sh
  # or: bash scripts/build_lightrag_conv30.sh

Output: LightRAG working_dir at /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_storage/{conv_id}/
  - KV, graph, vector stores
  - chunks: one per dialog turn (rag_mode == 'dialog')
"""

import argparse
import os
import sys
import json
import asyncio
from pathlib import Path

# Add project root and LightRAG to path
PROJECT_ROOT = Path(__file__).parent.parent
LIGHTRAG_PATH = PROJECT_ROOT / "module_unit" / "LightRAG"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIGHTRAG_PATH))

# Ensure OPENAI_API_BASE is set for LightRAG (uses this instead of OPENAI_BASE_URL)
if os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

from global_methods import set_openai_key

# LightRAG imports (after path setup)
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed


# ──────────────────────────── Configuration ────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "locomo10.json"
RAG_STORAGE_BASE = "/mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_storage"

# LightRAG hyperparameters (rag_mode == 'dialog': one chunk per dialog turn)
# chunk_token_size 需足够大，确保单条对话不被拆分（单条通常 < 500 tokens）
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100
ENTITY_TYPES = ["Person", "Event", "Location", "Organization", "Activity", "Concept", "Date", "Other"]


def load_conversation(sample_id: str) -> dict:
    """Load conversation by sample_id from locomo10.json."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        if sample.get("sample_id") == sample_id:
            return sample
    raise ValueError(f"Sample {sample_id} not found in {DATA_PATH}")


def format_conversation_as_dialogs(sample: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Format conversation as list of dialog strings (rag_mode == 'dialog').
    与 gpt_utils.prepare_for_rag(rag_mode='dialog') 完全一致：
    - 每条 dialog 文本格式: Speaker said, "text" 或 Speaker said, "text" and shared blip_caption
    - 无 [dia_id]、无 date_time 前缀（date_time 存入 file_paths 用于 citation）
    """
    conv = sample.get("conversation", {})
    session_nums = [int(k.split("_")[-1]) for k in conv.keys() if "session" in k and "date_time" not in k]
    if not session_nums:
        return [], [], []

    dialog_texts = []
    doc_ids = []
    file_paths = []

    for i in range(min(session_nums), max(session_nums) + 1):
        date_time = conv.get(f"session_{i}_date_time", "")
        for dialog in conv[f"session_{i}"]:
            dia_id = dialog.get("dia_id", "")
            speaker = dialog.get("speaker", "Unknown")
            text = dialog.get("text", dialog.get("clean_text", ""))

            if "blip_caption" in dialog:
                line = speaker + ' said, "' + text + '"' + " and shared " + dialog["blip_caption"]
            else:
                line = speaker + ' said, "' + text + '"'

            dialog_texts.append(line)
            doc_ids.append(dia_id if dia_id else f"s{i}_{len(dialog_texts)}")
            file_paths.append(date_time or "unknown")

    return dialog_texts, doc_ids, file_paths


def _session_num_from_dia_id(dia_id: str) -> int | None:
    """从 dia_id (e.g. 'D1:3') 提取 session 编号 (e.g. 1)。"""
    if ":" not in dia_id:
        return None
    prefix = dia_id.split(":")[0]  # 'D1'
    num_str = prefix.lstrip("D")
    try:
        return int(num_str)
    except ValueError:
        return None


def format_observation_as_docs(sample: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Format observation entries as documents (rag_mode == 'obs').
    observation 结构: sample["observation"]["session_X_observation"][speaker] = [[summary, dia_id], ...]
    每条 obs 文本: "[dia_id] (date_time) speaker: summary"
    doc_id = dia_id, file_path = session date_time
    """
    obs_data = sample.get("observation", {})
    conv = sample.get("conversation", {})
    if not obs_data:
        return [], [], []

    obs_texts = []
    doc_ids = []
    file_paths = []
    seen_content = set()
    dia_id_counter: dict[str, int] = {}

    session_keys = sorted(obs_data.keys(), key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0)
    for sess_key in session_keys:
        sess_num_str = sess_key.split("_")[1]
        try:
            sess_num = int(sess_num_str)
        except ValueError:
            continue
        date_time = conv.get(f"session_{sess_num}_date_time", "unknown")
        speaker_dict = obs_data[sess_key]
        for speaker, entries in speaker_dict.items():
            for entry in entries:
                if not isinstance(entry, list) or len(entry) < 2:
                    continue
                summary, dia_id = entry[0], entry[1]
                content_key = f"{dia_id}_{speaker}_{summary[:30]}"
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                # LightRAG 要求 id 唯一，同一 dia_id 可能有多条 obs，加序号区分
                cnt = dia_id_counter.get(dia_id, 0)
                dia_id_counter[dia_id] = cnt + 1
                unique_id = dia_id if cnt == 0 else f"{dia_id}_obs{cnt}"
                text = f"[{dia_id}] ({date_time}) {speaker}: {summary}"
                obs_texts.append(text)
                doc_ids.append(unique_id)
                file_paths.append(date_time)

    return obs_texts, doc_ids, file_paths


async def main(sample_id: str, working_dir: str, rag_mode: str = "dialog"):
    set_openai_key()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Run: source scripts/env.sh")
        sys.exit(1)

    print(f"Loading {sample_id} from {DATA_PATH}...")
    sample = load_conversation(sample_id)

    if rag_mode == "obs":
        print("Formatting observations as docs (rag_mode == 'obs')...")
        dialog_texts, doc_ids, file_paths = format_observation_as_docs(sample)
        print(f"  {len(dialog_texts)} observation entries")
    else:
        print("Formatting conversation as dialogs (rag_mode == 'dialog')...")
        dialog_texts, doc_ids, file_paths = format_conversation_as_dialogs(sample)
        print(f"  {len(dialog_texts)} dialog turns (one chunk per turn)")

    os.makedirs(working_dir, exist_ok=True)
    print(f"\nLightRAG working_dir: {working_dir}")
    print("Hyperparameters:")
    print(f"  - chunking: rag_mode == '{rag_mode}'")
    print(f"  - chunk_token_size: {CHUNK_TOKEN_SIZE}")
    print(f"  - chunk_overlap_token_size: {CHUNK_OVERLAP_TOKEN_SIZE}")
    print(f"  - entity_types: {ENTITY_TYPES}")
    print("  - embedding: openai default (text-embedding-3-small)")
    print("  - llm: gpt-4o-mini")
    print("  - storage: default (JsonKV, NetworkX, NanoVector)")
    print("")

    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        addon_params={
            "entity_types": ENTITY_TYPES,
            "language": "English",
        },
    )

    print("Initializing LightRAG storages...")
    await rag.initialize_storages()

    print("Inserting dialogs into LightRAG (each dialog = one document)...")
    await rag.ainsert(dialog_texts, ids=doc_ids, file_paths=file_paths)
    print("  Done.")

    print("\nTesting query...")
    from lightrag import QueryParam
    test_query = "When did the conversation mention a specific event?"
    result = await rag.aquery(test_query, param=QueryParam(mode="hybrid", top_k=20))
    print(f"  Query: {test_query}")
    print(f"  Answer (excerpt): {result[:300]}...")

    await rag.finalize_storages()
    print("\nLightRAG index built successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build LightRAG index for LoCoMo conversation")
    parser.add_argument("--conv-id", "-c", default="conv-26", help="Sample ID (e.g. conv-26, conv-30)")
    parser.add_argument("--output-dir", "-o", default=None, help=f"Output dir (default: {RAG_STORAGE_BASE}/<conv_id>)")
    parser.add_argument("--rag-mode", "-m", default="dialog", choices=["dialog", "obs"],
                        help="'dialog': one doc per dialog turn; 'obs': one doc per observation entry")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_id = args.conv_id
    conv_name = sample_id.replace("-", "")
    suffix = f"_{args.rag_mode}" if args.rag_mode != "dialog" else ""
    working_dir = args.output_dir or os.path.join(RAG_STORAGE_BASE, f"{conv_name}{suffix}")
    asyncio.run(main(sample_id, working_dir, rag_mode=args.rag_mode))
