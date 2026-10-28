#!/usr/bin/env python3
"""Exercise the real vendored LightRAG against a local, fake OpenAI API.

Uses no paid API. Run with the same environment as the build script.
"""
import argparse
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    calls = Counter()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls[self.path] += 1
            if self.path.endswith("/embeddings"):
                items = request["input"]
                if isinstance(items, str):
                    items = [items]
                data = []
                for i, item in enumerate(items):
                    digest = hashlib.sha256(str(item).encode()).digest()
                    vector = [(digest[j % len(digest)] - 127.5) / 128 for j in range(1536)]
                    data.append({"object": "embedding", "index": i, "embedding": vector})
                result = {"object": "list", "model": request["model"], "data": data,
                          "usage": {"prompt_tokens": 1, "total_tokens": 1}}
            elif self.path.endswith("/chat/completions"):
                content = (
                    "entity<|#|>User<|#|>Person<|#|>The user discusses an archive.\n"
                    "entity<|#|>Archive<|#|>Concept<|#|>An archive stores memories.\n"
                    "relation<|#|>User<|#|>Archive<|#|>discussion<|#|>The user discusses the archive.\n"
                    "<|COMPLETE|>"
                )
                result = {"id": "offline", "object": "chat.completion", "created": 0,
                          "model": request["model"], "choices": [{"index": 0,
                          "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            else:
                self.send_error(404)
                return
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="memcot-offline-") as folder:
            root = Path(folder)
            entries = []
            for i in range(2):
                entries.append({"question_id": f"offline{i}", "question": "Where?", "answer": "archive",
                                "haystack_session_ids": [f"session_{i}"],
                                "haystack_dates": [f"2026/03/0{i+1} (Sun) 10:00"],
                                "haystack_sessions": [[
                                    {"role": "user", "content": 'Remember the "archive".'},
                                    {"role": "assistant", "content": "The archive contains memories. " * 350},
                                    {"role": "user", "content": 'Remember the "archive".'}]]})
            source = root / "source.json"
            source.write_text(json.dumps(entries))
            base = f"http://127.0.0.1:{server.server_port}/v1"
            environment = {**os.environ, "OPENAI_API_KEY": "offline-test-only", "OPENAI_API_BASE": base,
                           "OPENAI_BASE_URL": base, "NO_PROXY": "127.0.0.1,localhost"}
            command = [sys.executable, str(Path(__file__).with_name("build_lightrag_longmemeval.py")),
                       "build", "--source", str(source), "--output-base", str(root / "indexes"),
                       "--repo-root", str(args.repo_root), "--llm-model", "offline-extractor",
                       "--llm-base-url", base, "--embedding-base-url", base]
            result = subprocess.run(command, env=environment, cwd=root, capture_output=True, text=True)
            if result.returncode:
                print(result.stdout[-6000:])
                print(result.stderr[-12000:])
                raise SystemExit(result.returncode)
            for i in range(2):
                index = root / "indexes" / f"{i:04d}_offline{i}"
                report = json.loads((index / "verification.json").read_text())
                assert report["ok"] and report["source_turns"] == 3 and report["stored_text_chunks"] == 2, report
                assert (index / "_BUILD_DONE").exists()
                docs = json.loads((index / "kv_store_full_docs.json").read_text())
                assert all(d["file_path"] == entries[i]["haystack_dates"][0] for d in docs.values())
                chunks = json.loads((index / "kv_store_text_chunks.json").read_text())
                assert max(c["tokens"] for c in chunks.values()) > 1200
            assert calls["/v1/embeddings"] and calls["/v1/chat/completions"], calls
            print(json.dumps({"ok": True, "indexes_built": 2, "source_turns_per_index": 3,
                              "unique_chunks_per_index": 2, "local_api_calls": dict(calls)}, indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
