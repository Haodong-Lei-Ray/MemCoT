## LongMemEval 数据格式速览（本地拷贝）

本说明文档只针对你在本项目中的本地数据拷贝：

- 根目录：`/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval`
- 数据目录：`benchmark/LongMemEval/data/`
  - `longmemeval_oracle.json`
  - `longmemeval_s_cleaned.json`
  - `longmemeval_m_cleaned.json`

这里重点看你已经打开的 `longmemeval_s_cleaned.json`。

---

## 顶层结构

`longmemeval_s_cleaned.json` 是一个 **list**，长度为 500，每个元素是一个 evaluation instance，结构大致如下：

```json
{
  "question_id": "e47becba",
  "question_type": "single-session-user",
  "question": "What degree did I graduate with?",
  "question_date": "2023/05/30 (Tue) 23:40",
  "answer": "Business Administration",
  "answer_session_ids": [
    "answer_280352e9"
  ],
  "haystack_dates": [
    "2023/05/20 (Sat) 02:21",
    "2023/05/20 (Sat) 02:57",
    "... 省略若干 ..."
  ],
  "haystack_session_ids": [
    "sharegpt_yywfIrx_0",
    "85a1be56_1",
    "... 省略若干 ...",
    "answer_280352e9",
    "84503ce4_1"
  ],
  "haystack_sessions": [
    [
      { "role": "user", "content": "..." },
      { "role": "assistant", "content": "..." }
    ],
    [
      { "role": "user", "content": "..." },
      { "role": "assistant", "content": "..." }
    ],
    "... 共若干个 session，每个 session 是一个对话轮列表 ..."
  ]
}
```

关键点：

- **一个 instance = 一个问题 + 一整段长对话历史**  
  - `question`：要问的问题  
  - `answer`：标准答案（string）  
  - `question_type`：题目类型（单 session / multi-session / knowledge-update 等）
- **haystack 部分**：长记忆
  - `haystack_dates`：每个会话对应的时间戳（字符串列表）
  - `haystack_session_ids`：每个会话的 ID，和 `dates`、`sessions` 一一对应
  - `haystack_sessions`：长度与 `haystack_dates` 一致的 list，每个元素是一个 **对话 session**：
    - 每个 session 是 `[{ "role": "user"/"assistant", "content": "..." }, ...]`
    - 对话中包含了真正的 evidences，有些 turn 带有 `has_answer: true` 字段（用于评估记忆召回）
- **answer_session_ids**：哪些 session ID 里包含回答该问题的证据（session 级 ground truth）。

简化理解：

- 你可以把 `haystack_sessions` 看成很多天的聊天记录，按时间排好。  
- 模型必须在这些聊天记录中“翻找记忆”，才能回答 `question`。  
- `answer_session_ids` 告诉你：真正需要记住的是哪几段 session。

---

## 用 Python 读一条样本

下面是一个最小 demo，读取第一条样本，并打印问题、答案以及前两个 session 的内容，方便你在 REPL 或脚本里快速 inspect：

```python
import json
from pathlib import Path

BASE = Path("/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval")
data_path = BASE / "data" / "longmemeval_s_cleaned.json"

with data_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

item = data[0]

print("question_id:", item["question_id"])
print("question_type:", item["question_type"])
print("question_date:", item["question_date"])
print("question:", item["question"])
print("gold answer:", item["answer"])
print("num haystack sessions:", len(item["haystack_sessions"]))

# 打印前两个 session 的前若干轮
for idx, (date, sess_id, sess) in enumerate(
    zip(item["haystack_dates"], item["haystack_session_ids"], item["haystack_sessions"])
):
    if idx >= 2:
        break
    print(f"\n=== Session {idx} | id={sess_id} | date={date} ===")
    for turn in sess[:4]:
        role = turn.get("role")
        content = turn.get("content", "").replace("\n", " ")[:200]
        print(f"{role}: {content}...")
```

---

## 小测试：用 gpt-4o-mini 回答一个问题

> 这里只是说明思路，真正的测试脚本在同目录的 `quick_test_gpt4omini.py` 中，你可以直接运行。

核心思路：

1. 从 `longmemeval_s_cleaned.json` 读一条样本（比如第 0 条）。  
2. 把 `haystack_sessions` 展开成一个长的对话文本（带上日期/role 信息）。  
3. 拼一个 prompt：先给历史，再给问题，让 `gpt-4o-mini` 回答。  
4. 对比模型回答和 `answer`。

伪代码大致是：

```python
history = []
for date, sess in zip(item["haystack_dates"], item["haystack_sessions"]):
    history.append(f"[{date}]")
    for turn in sess:
        history.append(f"{turn['role']}: {turn['content']}")
history_text = "\n".join(history)

prompt = f\"\"\"You are a helpful assistant.

Here is the user's chat history (chronological):

{history_text}

Now answer the user's final question based ONLY on the history above.

Question: {item['question']}
Answer in one short sentence.\"
\"\"\"

resp = run_chatgpt(prompt, model="gpt-4o-mini", num_tokens_request=512, temperature=0.0)
print("Model answer:", resp.strip())
print("Gold answer:", item["answer"])
```

真实可运行的版本见：`quick_test_gpt4omini.py`。

