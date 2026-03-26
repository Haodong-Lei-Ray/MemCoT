# locomo10 无 RAG 版本评估技术说明

本文档说明 `task_eval/evaluate_qa.py` 在 **不使用 RAG**（`--use-rag` 未设置）时，如何对 `locomo10.json` 进行问答评估，重点以 conv-26 为例。

---

## 1. 数据格式概览

### 1.1 locomo10.json 结构

每个 sample 包含：

- `sample_id`：如 `"conv-26"`
- `conversation`：多轮会话
  - `session_1`, `session_2`, ... `session_N`：各 session 的对话列表
  - `session_1_date_time`, `session_2_date_time`, ...：各 session 的时间
- `qa`：问答对列表，每项含 `question`、`answer`、`evidence`（dia_id）、`category` 等

### 1.2 conv-26 的 session 规模

conv-26 共 **19 个 session**（session_1 至 session_19），每个 session 多轮对话，跨数月（如 2023-05 至 2023-10）。

---

## 2. 无 RAG 时的上下文构建

### 2.1 核心逻辑：`get_input_context`

无 RAG 时，上下文由 `gpt_utils.get_input_context` 从**完整对话**中按 token 预算截取：

```python
# gpt_utils.py: 281-314
def get_input_context(data, num_question_tokens, encoding, args):
    query_conv = ''
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    for i in range(min(session_nums), max(session_nums) + 1):
        # 按 session 顺序遍历
        for dialog in data['session_%s' % i][::-1]:
            turn = dialog['speaker'] + ' said, "' + dialog['text'] + '"' + ...
            # 若加入该 turn 后不超过 token 预算，则加入
            if (num_tokens + ...) < (MAX_LENGTH - PER_QA_TOKEN_BUDGET * batch_size):
                query_conv = turn + query_conv
            else:
                stop = True
                break
        if stop:
            break
    return query_conv
```

要点：

1. **遍历顺序**：从 session_1 到 session_N，按时间顺序
2. **是否包含全部 19 个 session**：**不必然**。是否包含由 token 预算决定
3. **Token 预算**：
   - `MAX_LENGTH`：模型最大长度（如 Qwen2.5-14B: 128000）
   - `PER_QA_TOKEN_BUDGET`：每个问题预留约 50 tokens
   - 有效预算 ≈ `MAX_LENGTH - PER_QA_TOKEN_BUDGET * batch_size`
4. **截断策略**：从 session_1 开始按时间顺序逐个 session、逐个 turn 加入，超过预算即停止。因此**较早的 session 会被保留，较晚的 session 会被丢弃**（例如只保留到 session_10，则 session_11～19 不会被送入 LLM）

### 2.2 Prompt 结构

最终送入 LLM 的 prompt 由以下部分拼接：

```
[start_prompt]
Below is a conversation between two people: {speaker1} and {speaker2}. 
The conversation takes place over multiple days and the date of each 
conversation is written at the beginning of the conversation.

[query_conv]
DATE: 1:56 pm on 8 May, 2023
CONVERSATION:
Caroline said, "..."

Melanie said, "..."

DATE: 1:14 pm on 25 May, 2023
CONVERSATION:
...

[question_prompt]
Based on the above conversations, write short answers...
0: {question_0}
1: {question_1}
...

Question: {current_question} Short answer:
```

---

## 3. 是否会把 19 个 session 都作为上文？

**不一定。**

- **128k 上下文模型**（如 gpt-4o、Qwen2.5-14B-Instruct）：若 19 个 session 总 token 数小于约 12 万，则**可以全部**作为上文；否则会从**较晚的 session** 开始截断
- **更短的上下文模型**（如 4k、8k）：只会保留从 session_1 起、能塞进预算的**靠前若干 session**，较晚的 session 全部丢弃

因此：无 RAG 版本是**按 token 预算从前向后填充对话**，超出部分（靠后的 session）会被丢弃。

---

## 4. Batch 处理

- `batch_size=1`：每个 question 单独一次 API 调用，但上下文仍按上述方式截取
- `batch_size>1`：多个 question 共用同一段上下文，一次调用返回多个答案
  - 此时 `num_question_tokens` 更大，对话能塞进的数量更少
  - 输出要求为 JSON：`{"0": "ans1", "1": "ans2", ...}`

---

## 5. 相关代码位置

| 功能           | 文件            | 函数/位置            |
|----------------|-----------------|----------------------|
| 主入口         | `evaluate_qa.py` | `main()`             |
| GPT/Qwen 调用  | `gpt_utils.py`   | `get_gpt_answers()`  |
| 无 RAG 上下文  | `gpt_utils.py`   | `get_input_context()`|
| Token 预算     | `gpt_utils.py`   | `MAX_LENGTH`, `PER_QA_TOKEN_BUDGET` |

---

## 6. 运行示例

无 RAG 评估 conv-26（及其他 samples）：

```bash
python task_eval/evaluate_qa.py \
  --data-file data/locomo10.json \
  --out-file outputs/gpt-4o/base/conv26_results.json \
  --model gpt-4o
```

（未加 `--use-rag` 即为无 RAG 模式）
