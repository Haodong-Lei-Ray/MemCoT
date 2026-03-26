# task_eval 中 Agent 回答依赖的提示词说明

本文档说明 `task_eval/` 目录下评估脚本（如 `evaluate_qa.py`）中，LLM 生成回答时所使用的提示词。

---

## 1. 主入口与文件位置

评估入口：`task_eval/evaluate_qa.py`，通过各 `*_utils.py` 的 `get_*_answers()` 调用 LLM。

核心提示词在 **`task_eval/gpt_utils.py`** 第 29–55 行。

---

## 2. 提示词定义

### 2.1 QA_PROMPT（常规问题，category 1–4）

```python
QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""
```

- 用于非 category 5 的问题
- 要求：短句回答，尽可能使用 context 中的原文

### 2.2 QA_PROMPT_CAT_5（category 5 对抗问题）

```python
QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""
```

- 用于 category 5 的对抗/无信息问题
- 问题会追加选项：(a) Not mentioned in the conversation (b) 标准答案

### 2.3 QA_PROMPT_BATCH（批量模式）

```python
QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. 
Write the answers in the form of a json dictionary where each entry contains the question number as "key" and the short answer as "value". 
Use single-quote characters for named entities and double-quote characters for enclosing json elements. Answer with exact words from the conversations whenever possible.

"""
```

- 仅用于 **batch 模式**（`batch_size > 1`）
- 要求：一次回答多个问题，以 JSON 形式返回，如 `{"0": "ans1", "1": "ans2", ...}`

### 2.4 CONV_START_PROMPT（对话开场）

```python
CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"
```

- 在 context 前添加，用于说明对话场景

---

## 3. 完整 Query 拼接方式

`gpt_utils.py` 第 381–428 行：

### 3.1 RAG 模式

- `query_conv` = `get_rag_context(...)` 返回的检索结果（**不含** CONV_START_PROMPT）
- 格式：`date_time: context` 每行一条
- `query = query_conv + '\n\n' + QA_PROMPT.format(question)`（或 `QA_PROMPT_CAT_5`）
- **RAG 模式强制 `batch_size == 1`**，不使用 QA_PROMPT_BATCH

### 3.2 非 RAG 模式

- `query_conv` = `CONV_START_PROMPT` + `get_input_context(...)`（完整对话，按 token 预算截断）
- 若 `batch_size == 1`：`query = query_conv + '\n\n' + QA_PROMPT.format(question)`
- 若 `batch_size > 1`：`query = query_conv + '\n' + question_prompt`，其中 `question_prompt = QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])`

---

## 4. RAG 模式详解

### 4.1 概述

RAG 模式下，context 不再使用完整对话，而是**按 question 检索 top-k 相关片段**，仅将检索结果送入 LLM。评估时使用 `evaluate_qa.py --use-rag`，并指定 `--rag-mode` 和 `--top-k`。

### 4.2 支持的 rag_mode

| rag_mode | 数据源 | 粒度 | 说明 |
|----------|--------|------|------|
| **dialog** | 对话轮次 | 每条 `speaker said, "text"` 一个向量 | 与 LightRAG rag_storage 对齐，每轮对话一个 chunk |
| **summary** | session 摘要 | 每个 session 一个向量 | 摘要级检索 |
| **observation** | observation 单元 | 每个 observation 一个向量 | 结构化事件/观察 |
| **hybridrag** | observation | 同上 | **向量 + 关键词** 混合检索（0.7 语义 + 0.3 关键词） |

### 4.3 检索流程

1. **准备**：`prepare_for_rag()` 加载对应 pkl（embeddings + dia_id + context + date_time）
2. **Question embedding**：对当前 question 计算向量（hybridrag 用 openai retriever）
3. **检索**：
   - `dialog` / `summary` / `observation`：纯向量相似度，取 top-k
   - `hybridrag`：`get_hybrid_rag_context()`，向量 0.7 + 关键词 0.3 加权
4. **拼接**：`date_time: context` 逐行拼接，得到 `query_conv`

### 4.4 Context 格式

```
1:56 pm on 8 May, 2023: Caroline said, "I went to a LGBTQ support group yesterday..."
4:33 pm on 12 July, 2023: Melanie said, "That's wonderful. How did it go?"
...
```

- `dialog` / `observation` / `hybridrag`：行间用 `\n` 分隔
- `summary`：用 `\n\n` 分隔

### 4.5 与 Recall 的关系

RAG 模式下会写入 `{prediction_key}_context`，记录检索到的 dia_id 列表，供 `evaluate_qa_recall_technical.md` 中 recall 计算使用。

### 4.6 运行示例

```bash
# dialog 模式，top-5
python task_eval/evaluate_qa.py \
  --data-file data/locomo10.json \
  --out-file outputs/conv26_rag_results.json \
  --model gpt-4o \
  --use-rag \
  --rag-mode dialog \
  --top-k 5 \
  --emb-dir /path/to/embeddings

# hybridrag 模式，top-10
python task_eval/evaluate_qa.py \
  --use-rag --rag-mode hybridrag --top-k 10 ...
```

---

## 5. Batch 模式说明

### 5.1 含义

**Batch 模式**：多个问题共用同一段 context，在一次 API 调用中一起回答。

- `batch_size = 1`（默认）：逐题回答，每个问题单独一次 API 调用
- `batch_size > 1`：每次取 `batch_size` 个问题，共用同一 context，一次调用返回多个答案

### 5.2 工作方式

| 项目 | batch_size=1 | batch_size>1 |
|------|--------------|--------------|
| API 调用 | 每个问题 1 次 | 每 `batch_size` 个问题 1 次 |
| 使用的 prompt | QA_PROMPT / QA_PROMPT_CAT_5 | QA_PROMPT_BATCH |
| 输出格式 | 单个短答案 | JSON：`{"0": "ans1", "1": "ans2", ...}` |
| token 预算 | 32 per question | batch_size × 50 (PER_QA_TOKEN_BUDGET) |

### 5.3 限制

- **RAG 模式**：必须 `batch_size == 1`（`gpt_utils.py` 第 331 行）
- **非 RAG 模式**：可使用 batch 以节省 API 调用次数；但 batch 越大，每个问题分到的 token 越少，context 能塞进的对话越短

### 5.4 适用场景

- batch_size=1：RAG 评估、需要为每个问题单独检索时
- batch_size>1：非 RAG、长上下文、希望减少 API 调用时

---

## 6. 其他 LLM 后端的提示词

| 文件 | 位置 | 说明 |
|------|------|------|
| `claude_utils.py` | 18–43 行 | QA_PROMPT / QA_PROMPT_CAT_5 / QA_PROMPT_BATCH / CONV_START_PROMPT |
| `gemini_utils.py` | 15–39 行 | 同上 |
| `hf_llm_utils.py` | 41–87 行 | 同上，并包含 Llama/Gemma 等 chat 模板 |

各后端与 gpt_utils 的 prompt 内容基本一致，仅在调用接口和模板格式上有区别。

---

## 7. 与 human_loop_rag 的区别

| 方面 | task_eval | human_loop_rag |
|------|-----------|----------------|
| 核心指令 | 尽可能使用 context 中的原文 | 含 when、yes/no、多轮等复杂规则 |
| "No information" | 无专门规则 | 有 yes/no 问题的 "No information provided." 规则 |
| 风格 | 简洁、评测向 | 更接近 agentic_event_search 的 Responder Agent |

---

## 8. 相关代码位置

| 功能 | 文件 | 行号/位置 |
|------|------|-----------|
| 提示词定义 | `task_eval/gpt_utils.py` | 29–55 |
| Query 拼接 | `task_eval/gpt_utils.py` | 381–428 |
| RAG 强制 batch_size=1 | `task_eval/gpt_utils.py` | 331 |
| RAG 准备 (prepare_for_rag) | `task_eval/gpt_utils.py` | 72–145 |
| RAG 检索 (get_rag_context) | `task_eval/gpt_utils.py` | 247–278 |
| Hybrid RAG 检索 | `task_eval/gpt_utils.py` | 185–244 |
| 评估入口 | `task_eval/evaluate_qa.py` | main() |
