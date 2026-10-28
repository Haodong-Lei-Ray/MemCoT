#!/usr/bin/env python3
"""Reconstruct the observed MemCoT LongMemEval turn-level LightRAG format.

prepare/verify use the standard library. build uses the project's vendored
LightRAG and calls the configured embedding and extraction APIs.
This recovers the input format, not the unknown historical LLM configuration.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import hashlib
import html
import json
import multiprocessing
import os
from pathlib import Path
import re
import sys

ENTITY_TYPES = ["Person", "Event", "Location", "Organization", "Activity", "Concept", "Date", "Other"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sanitize(text):
    """Mirror the vendored LightRAG sanitizer for deterministic offline checks."""
    text = text.strip()
    text.encode("utf-8")  # LightRAG also rejects unencodable source text.
    text = re.sub(r"[\ufffe\uffff\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", html.unescape(text)).strip()


def chunk_id(content):
    return "chunk-" + hashlib.md5(content.encode("utf-8")).hexdigest()


def format_documents(entry):
    sessions = entry["haystack_sessions"]
    ids, dates = entry["haystack_session_ids"], entry["haystack_dates"]
    if not (len(sessions) == len(ids) == len(dates)):
        raise ValueError("haystack_sessions / session_ids / dates lengths differ")
    # Positional IDs are supported by agent._parse_longmemeval_dia_id.
    # This branch follows the surviving code comments; it is not observed in 0484.
    positional = len(ids) != len(set(ids))
    documents = []
    for si, (sid, date, turns) in enumerate(zip(ids, dates, sessions)):
        for ti, turn in enumerate(turns, 1):
            role, text = turn["role"], turn["content"]
            if role not in ("user", "assistant") or not isinstance(text, str):
                raise ValueError(f"Unsupported turn at {sid}_{ti}")
            # Escaping embedded quotes is essential for the historical hashes.
            text = text.strip().replace('"', '\\"')
            documents.append({
                "id": f"{si if positional else sid}_{ti}",
                "content": sanitize(f'{role} said, "{text}"'),
                "file_path": date,
            })
    if not documents or len({d["id"] for d in documents}) != len(documents):
        raise ValueError("Empty history or colliding turn IDs")
    return documents


def unique_documents(documents):
    # LightRAG hashes chunks by content. Repeated text has one stored chunk.
    by_content = {}
    for doc in documents:
        by_content.setdefault(doc["content"], doc)
    return list(by_content.values())


def verify_index(index_dir, documents):
    """Check full coverage, allowing historical content-hash deduplication."""
    docs = json.loads((index_dir / "kv_store_full_docs.json").read_text())
    chunks = json.loads((index_dir / "kv_store_text_chunks.json").read_text())
    statuses = json.loads((index_dir / "kv_store_doc_status.json").read_text())
    expected = {d["id"]: d for d in documents}
    bad_docs = [key for key, d in expected.items() if key in docs and any(
        docs[key].get(field) != d[field] for field in ("content", "file_path"))]
    expected_hashes = {chunk_id(d["content"]) for d in documents}
    missing = sorted(expected_hashes - chunks.keys())
    extra = sorted(chunks.keys() - expected_hashes)
    bad_chunks = []
    for cid, stored in chunks.items():
        original = expected.get(stored.get("full_doc_id"))
        if (not original or stored.get("full_doc_id") not in docs
                or stored.get("content") != original["content"]
                or stored.get("file_path") != original["file_path"]
                or cid != chunk_id(stored.get("content", ""))):
            bad_chunks.append(cid)
    # Some historical full_docs records share identical content, hence no unique
    # chunk. Only require a processed owner for each actual stored chunk.
    bad_status = [cid for cid, c in chunks.items() if
                  statuses.get(c.get("full_doc_id"), {}).get("status") != "processed"]
    bad_vectors = []
    for name in ("chunks", "entities", "relationships"):
        vector_path = index_dir / f"vdb_{name}.json"
        if not vector_path.exists():
            bad_vectors.append(f"missing {vector_path.name}")
            continue
        vector = json.loads(vector_path.read_text())
        if vector.get("embedding_dim") != 1536:
            bad_vectors.append(f"{name}: expected 1536 dimensions")
        if name == "chunks" and {x["__id__"] for x in vector["data"]} != set(chunks):
            bad_vectors.append("chunk vector IDs differ from text chunk IDs")
    graph_exists = (index_dir / "graph_chunk_entity_relation.graphml").is_file()
    duplicates = sum(bool(s.get("metadata", {}).get("is_duplicate")) for s in statuses.values())
    return {
        "ok": not any((bad_docs, missing, extra, bad_chunks, bad_status, bad_vectors)) and graph_exists,
        "source_turns": len(documents), "unique_source_texts": len(expected_hashes),
        "stored_full_docs": len(docs), "matching_full_doc_ids": len(docs.keys() & expected.keys()),
        "stored_text_chunks": len(chunks), "missing_chunk_ids": missing,
        "extra_chunk_ids": extra, "mismatched_full_docs": bad_docs,
        "mismatched_chunks": bad_chunks, "unprocessed_chunk_owners": bad_status,
        "vector_errors": bad_vectors, "graph_exists": graph_exists,
        "status_counts": dict(Counter(s["status"] for s in statuses.values())),
        "historical_duplicate_status_records": duplicates,
    }


def one_turn_one_chunk(tokenizer, content, *args, **kwargs):
    # The released sample includes an unsplit 2915-token turn. A default 1200
    # token LightRAG chunker cannot reproduce those hashes.
    tokens = tokenizer.encode(content)
    return [{"tokens": len(tokens), "content": content.strip(), "chunk_order_index": 0}]


async def build_index(index_dir, documents, args):
    sys.path.insert(0, str(args.repo_root / "mem" / "LightRAG"))
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY before building")
    if os.getenv("OPENAI_BASE_URL") and not os.getenv("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]

    async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            args.llm_model, prompt, system_prompt=system_prompt,
            history_messages=history_messages or [], base_url=args.llm_base_url, **kwargs)

    # Preserve the metadata interface required by current NanoVectorDBStorage.
    embedding = EmbeddingFunc(
        embedding_dim=1536, max_token_size=8192,
        func=partial(openai_embed.func, model="text-embedding-3-small",
                     base_url=args.embedding_base_url))
    rag = LightRAG(
        working_dir=str(index_dir), workspace="", embedding_func=embedding,
        llm_model_func=complete, llm_model_name=args.llm_model,
        llm_model_max_async=args.llm_concurrency,
        entity_extract_max_gleaning=1,
        chunking_func=one_turn_one_chunk,
        addon_params={"entity_types": ENTITY_TYPES, "language": "English"},
    )
    # No implicit truncation of unusually large turns. The historical long-turn
    # split threshold/overlap cannot be recovered from the sampled index.
    import tiktoken
    embed_tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
    too_long = [d["id"] for d in documents if len(embed_tokenizer.encode(d["content"])) > 8192]
    if too_long:
        raise ValueError(f"Turns exceed the embedding limit; choose and document a splitting policy: {too_long}")
    await rag.initialize_storages()
    try:
        work = unique_documents(documents)
        await rag.ainsert([d["content"] for d in work], ids=[d["id"] for d in work],
                          file_paths=[d["file_path"] for d in work])
    finally:
        await rag.finalize_storages()


def build_and_verify(index_dir, documents, args):
    # LightRAG shared namespaces are process-global. Isolate question indexes
    # in fresh processes to prevent caches/locks leaking across working dirs.
    marker = index_dir / "_BUILD_DONE"
    marker.unlink(missing_ok=True)
    asyncio.run(build_index(index_dir, documents, args))
    report = verify_index(index_dir, documents)
    write_json(index_dir / "verification.json", report)
    if not report["ok"]:
        raise RuntimeError(f"Incomplete index; see {index_dir / 'verification.json'}")
    write_json(marker, {"verified_at": datetime.now(timezone.utc).isoformat(), **report})


def select_entries(args):
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if isinstance(source, dict):
        if args.index_offset is None:
            raise ValueError("A single-row source requires --index-offset (original dataset position)")
        source = [source]
    if not isinstance(source, list):
        raise ValueError("Source must be a dataset array or a single entry object")
    selected = [(i + (args.index_offset or 0), e) for i, e in enumerate(source)
                if i + (args.index_offset or 0) >= args.start_index
                and (not args.question_id or e["question_id"] == args.question_id)]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("No selected entries")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "build"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, default=Path("memory/longmemeval/lightrag/te3s"))
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, help="Also export per-question raw JSON files here")
    parser.add_argument("--index-offset", type=int, help="Original index of first source entry; required for a single object")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--question-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--llm-model", help="Required for build: historical extraction model is unknown")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--embedding-base-url", default=None)
    parser.add_argument("--llm-concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.action == "build" and not args.llm_model:
        parser.error("build requires --llm-model; it cannot be inferred from the released cache")
    if (args.limit is not None and args.limit <= 0) or args.llm_concurrency <= 0:
        parser.error("--limit and --llm-concurrency must be positive")
    if args.repo_root is None:
        args.repo_root = next((p for p in Path(__file__).resolve().parents
                               if (p / "mem" / "LightRAG").is_dir()), Path.cwd())
    reports = []
    for idx, entry in select_entries(args):
        name = f'{idx:04d}_{entry["question_id"]}'
        if Path(name).name != name or "\\" in name:
            raise ValueError("Unsafe question ID")
        index_dir = args.output_base / name
        documents = format_documents(entry)
        if args.action == "verify":
            report = verify_index(index_dir, documents)
            reports.append({"index": name, **report})
            continue
        fingerprint = hashlib.sha256(json.dumps(documents, ensure_ascii=False,
                                  sort_keys=True).encode()).hexdigest()
        manifest_path = index_dir / "reconstruction_manifest.json"
        previous = {}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["source_fingerprint"] != fingerprint:
                raise ValueError(f"Source changed: use a fresh output directory: {index_dir}")
            if args.action == "build" and previous.get("llm_model") not in (None, args.llm_model):
                raise ValueError("Extraction model changed: use a fresh output directory")
        elif (index_dir / "kv_store_doc_status.json").exists():
            raise ValueError("Refusing to mix a reconstructed build into an existing historical index")
        manifest = {
            "index": idx, "question_id": entry["question_id"],
            "source_fingerprint": fingerprint, "source_turns": len(documents),
            "unique_texts": len(unique_documents(documents)),
            "format": "escaped-role-said; one complete turn per chunk",
            "embedding_model": "text-embedding-3-small", "embedding_dim": 1536,
            "llm_model": args.llm_model if args.action == "build" else previous.get("llm_model"),
            "historical_llm_model_known": False,
        }
        write_json(manifest_path, manifest)
        write_json(index_dir / "source.json", entry)
        with (index_dir / "documents.jsonl").open("w", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
        if args.dataset_dir:
            write_json(args.dataset_dir / f"{name}.json", entry)
        if args.action == "build":
            worker = multiprocessing.get_context("spawn").Process(
                target=build_and_verify, args=(index_dir, documents, args))
            worker.start()
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(f"Index build failed: {index_dir}; worker exit {worker.exitcode}")
        reports.append(manifest)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if any(r.get("ok") is False for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
