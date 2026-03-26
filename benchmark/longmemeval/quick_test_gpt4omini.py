#!/usr/bin/env python3
"""
Quick smoke test: ask gpt-4o-mini to answer ONE LongMemEval_S question.

Usage (在 locomo 根目录下，先 source env.sh，保证 OPENAI_API_KEY / OPENAI_BASE_URL 就绪):

    cd /mnt/petrelfs/leihaodong/ICML/locomo
    source scripts/env.sh
    python module_unit/LongMemEval/quick_test_gpt4omini.py
"""

import json
import os
import sys
from pathlib import Path


def _load_one_example(idx: int = 0):
    """从 longmemeval_s_cleaned.json 里取一条样本。"""
    base = Path("/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval")
    data_path = base / "data" / "longmemeval_s_cleaned.json"
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if idx < 0 or idx >= len(data):
        raise IndexError(f"idx {idx} out of range, dataset size = {len(data)}")
    return data[idx]


def main():
    # 把 locomo 根目录加入 sys.path，重用项目里的 run_chatgpt 封装
    project_root = Path(__file__).resolve().parents[3]
    longmemeval_src = Path(__file__).resolve().parent / "src"
    for p in (project_root, str(longmemeval_src)):
        if p not in sys.path:
            sys.path.insert(0, str(p))

    from global_methods import run_chatgpt  # type: ignore
    from generation.format_history import format_history_for_prompt

    example = _load_one_example(0)
    question_id = example["question_id"]
    question = example["question"]
    gold_answer = example["answer"]

    print(f"[Example] question_id = {question_id}")
    print(f"Question_type = {example.get('question_type')}")
    print(f"Question_date = {example.get('question_date')}")
    print(f"Question      = {question}")
    print(f"Gold answer   = {gold_answer}")
    print(f"#sessions     = {len(example.get('haystack_sessions', []))}")

    # 为了运行快一点，可以只取前 N 个 session；如果想要完整 long context，把 max_sessions=None
    # 使用 run_generation 的 format_history_for_prompt（con=False，orig-session + nl 格式）
    history_text = format_history_for_prompt(example, max_sessions=None)

    prompt = f"""You are a helpful assistant.

Below is the user's full chat history (chronological). Each block is a session.

{history_text}

Now, based ONLY on the information in the history above, answer the final question.

Question: {question}

Answer in one short phrase or sentence.
If the answer is clearly given in the history, copy it verbatim as much as possible.
"""

    model_name = os.environ.get("LOCAMO_GPT_MODEL", "gpt-4o-mini")
    print(f"\n[Info] Using model: {model_name}")

    resp = run_chatgpt(
        prompt,
        model=model_name,
        num_tokens_request=512,
        temperature=0.0,
    )

    print("\n=== Model Answer ===")
    print(resp.strip())
    print("\n=== Gold Answer ===")
    print(gold_answer)


if __name__ == "__main__":
    main()

