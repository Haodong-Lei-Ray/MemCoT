#!/usr/bin/env python3
"""
Build LightRAG index from LoCoMo events (events_output structure).

Each event JSON (event_description, time, people, reasoning) becomes one document/chunk.
Input: module_test/events_output/conv-26/ (or --events-dir)
  - D1/1.json, D1/2.json, ... (per-session event files)
  - D2/, D3/, ...
Output: /mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_event/{conv_id}/

Usage:
  python3 task_eval/build_lightrag_locomo4event.py [--conv-id conv-26] [--events-dir ...] [--output-dir ...]
  # or: bash scripts/build_lightrag_locomo4event_srun.sh
"""

import argparse
import os
import sys
import json
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LIGHTRAG_PATH = PROJECT_ROOT / "module_unit" / "LightRAG"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIGHTRAG_PATH))

if os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
    os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

from global_methods import set_openai_key
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed


# ──────────────────────────── Configuration ────────────────────────────
EVENTS_OUTPUT_BASE = PROJECT_ROOT / "module_test" / "events_output"
RAG_EVENT_BASE = "/mnt/petrelfs/leihaodong/ICML/exp/memory/lightrag/rag_event"

CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100
ENTITY_TYPES = ["Person", "Event", "Location", "Organization", "Activity", "Concept", "Date", "Other"]


def load_events_from_dir(events_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """
    Load all events from events_output/conv-26 structure.
    Returns (texts, doc_ids, file_paths) for LightRAG ainsert.
    Each event becomes one document: event_description + time + people + reasoning.
    """
    texts = []
    doc_ids = []
    file_paths = []

    for session_dir in sorted(events_dir.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith("D"):
            continue
        session_id = session_dir.name
        for event_file in sorted(session_dir.glob("*.json")):
            if event_file.name in ("events_summary.json", "full_conversation.json"):
                continue
            try:
                with open(event_file, "r", encoding="utf-8") as f:
                    ev = json.load(f)
            except Exception:
                continue
            event_id = ev.get("event_id", event_file.stem)
            if isinstance(event_id, str) and event_id.isdigit():
                event_id = int(event_id)
            doc_id = f"{session_id}:{event_id}"
            time_str = ev.get("time", "")
            people = ev.get("people", [])
            people_str = ", ".join(people) if isinstance(people, list) else str(people)
            desc = ev.get("event_description", "")
            reasoning = ev.get("reasoning", "")
            text_parts = [f"Event: {desc}"]
            if time_str:
                text_parts.append(f"Time: {time_str}")
            if people_str:
                text_parts.append(f"People: {people_str}")
            if reasoning:
                text_parts.append(f"Context: {reasoning}")
            text = "\n".join(text_parts)
            texts.append(text)
            doc_ids.append(doc_id)
            file_paths.append(time_str or session_id)

    return texts, doc_ids, file_paths


async def main(events_dir: Path, working_dir: str):
    set_openai_key()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Run: source scripts/env.sh")
        sys.exit(1)

    print(f"Loading events from {events_dir}...")
    texts, doc_ids, file_paths = load_events_from_dir(events_dir)
    print(f"  {len(texts)} events (one chunk per event)")

    os.makedirs(working_dir, exist_ok=True)
    print(f"\nLightRAG working_dir: {working_dir}")
    print("Hyperparameters:")
    print(f"  - chunk_token_size: {CHUNK_TOKEN_SIZE}")
    print(f"  - chunk_overlap_token_size: {CHUNK_OVERLAP_TOKEN_SIZE}")
    print(f"  - entity_types: {ENTITY_TYPES}")
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

    print("Inserting events into LightRAG (each event = one document)...")
    await rag.ainsert(texts, ids=doc_ids, file_paths=file_paths)
    print("  Done.")

    print("\nTesting query...")
    from lightrag import QueryParam
    test_query = "When did Caroline go to the LGBTQ support group?"
    result = await rag.aquery(test_query, param=QueryParam(mode="hybrid", top_k=10))
    print(f"  Query: {test_query}")
    print(f"  Answer (excerpt): {result[:400]}...")

    await rag.finalize_storages()
    print("\nLightRAG event index built successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build LightRAG index from LoCoMo events")
    parser.add_argument("--conv-id", "-c", default="conv-26", help="Conv ID (e.g. conv-26)")
    parser.add_argument("--events-dir", "-e", default=None, help="Events dir (default: module_test/events_output/<conv_id>)")
    parser.add_argument("--output-dir", "-o", default=None, help=f"Output dir (default: {RAG_EVENT_BASE}/<conv_id>)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    conv_id = args.conv_id
    conv_name = conv_id.replace("-", "")
    events_dir = Path(args.events_dir) if args.events_dir else EVENTS_OUTPUT_BASE / conv_id
    working_dir = args.output_dir or os.path.join(RAG_EVENT_BASE, conv_name)

    if not events_dir.exists():
        print(f"Error: events dir not found: {events_dir}")
        sys.exit(1)

    asyncio.run(main(events_dir, working_dir))
