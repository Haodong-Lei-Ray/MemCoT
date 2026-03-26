# evaluate_qa 中 Recall 计算方式技术说明

本文档说明 `task_eval/evaluate_qa.py` 在使用 **RAG 模式**（`--use-rag`）时，如何计算 **retrieval recall**（检索召回率）。

---

## 1. 概述

- **Recall 仅在 RAG 模式下计算**：`evaluate_qa.py` 第 114-115 行，当 `args.use_rag` 为 True 且 `recall` 非空时，将 `model_key + '_recall'` 写入每个 QA 项。
- **Recall 含义**：衡量 RAG 检索到的上下文（context）是否覆盖了支持答案的**证据 dia_id**，即 **evidence recall**。
- **与 F1 的区别**：F1 衡量**答案文本**与标准答案的 token 重叠；Recall 衡量**检索到的 dia_id** 是否命中 ground truth evidence。

---

## 2. 数据依赖

### 2.1 输入字段

每条 QA 需包含：

| 字段 | 说明 |
|------|------|
| `evidence` | ground truth 证据列表，如 `["D1:3", "D2:8"]`，表示支持该答案的对话轮次 |
| `eval_key + '_context'` | RAG 检索到的 context ID 列表，由 `gpt_utils` 等在推理时写入 |

其中 `eval_key` 随模型与 RAG 配置变化，例如：
- `gpt-4o_dialog_top_5` → `gpt-4o_dialog_top_5_context`
- `Qwen/Qwen2.5-14B-Instruct_dialog_top_10` → `..._context`

### 2.2 dia_id 格式

- **evidence**：格式为 `D{session}:{turn}`，如 `D1:3` 表示 session 1 的第 3 轮对话。
- **_context**：可为 **dia_id 格式**（如 `D1:3`）或 **session 格式**（如 `S1`），由 RAG 实现决定。

---

## 3. Recall 计算公式

代码位置：`task_eval/evaluation.py` 第 229-238 行。

### 3.1 条件

Recall 仅在以下条件同时满足时计算：
1. `eval_key + '_context' in line`：该 QA 有 RAG 检索到的 context 记录
2. `len(line['evidence']) > 0`：该 QA 有 ground truth evidence

否则该 QA 的 recall 设为 **1**（视为无 evidence 或无 context 时默认满分）。

### 3.2 按 dia_id 匹配（context 为 dia_id 格式）

当 `line[eval_key + '_context'][0].startswith('S')` 为 **False** 时，假设 context 为 dia_id 列表：

```
recall = (命中的 evidence 数量) / (evidence 总数量)
       = sum(ev in line[eval_key + '_context'] for ev in line["evidence"]) / len(line["evidence"])
```

即：每个 evidence dia_id 若出现在检索到的 context 中则计 1，否则计 0；最终除以 evidence 总数。

**示例**：
- evidence: `["D1:3", "D2:8"]`
- context: `["D1:3", "D1:5", "D3:1"]`
- 命中：D1:3 ✓，D2:8 ✗ → recall = 1/2 = 0.5

### 3.3 按 session 匹配（context 为 session 格式）

当 `line[eval_key + '_context'][0].startswith('S')` 为 **True** 时，假设 context 为 session 列表（如 `["S1", "S2"]`）：

```python
sessions = [e[1:] for e in line[eval_key + '_context']]  # ["1", "2"]
recall_acc = float(sum([
    ev.split(':')[0][1:] in sessions for ev in line["evidence"]
])) / len(line['evidence'])
```

- `ev.split(':')[0]`：从 `D1:3` 得到 `D1`
- `[1:]`：得到 session 编号 `"1"`
- 判断该 session 是否在检索到的 `sessions` 中

即：按 **session 粒度** 计算 recall——只要 evidence 所在 session 被检索到，即算命中。

**示例**：
- evidence: `["D1:3", "D1:5"]`（均属 session 1）
- context (sessions): `["S1", "S3"]` → sessions = `["1", "3"]`
- D1:3、D1:5 的 session 均为 "1" ∈ sessions → recall = 2/2 = 1.0

---

## 4. 输出与聚合

### 4.1 单条 QA

每条 QA 会写入：
- `{model_key}_recall`：该问题的 retrieval recall，保留 3 位小数。

### 4.2 汇总统计

`evaluate_qa.py` 会调用 `analyze_aggr_acc`，在 RAG 模式下会按 category 汇总 recall（`evaluation_stats.py` 第 66-67 行）：

```python
if rag:
    recall_by_category[qa['category']] += qa[model_name + '_recall']
```

---

## 5. 相关代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| Recall 计算 | `task_eval/evaluation.py` | `eval_question_answering()` 第 229-238 行 |
| Recall 写入 QA | `task_eval/evaluate_qa.py` | 第 114-115 行 |
| RAG context 写入 | `task_eval/gpt_utils.py` | 第 421 行 `prediction_key + '_context'` |
| Recall 聚合 | `task_eval/evaluation_stats.py` | 第 66-67 行 |

---

## 6. 注意事项

1. **无 RAG 时**：不计算 recall，`recall` 列表为空，不会写入 `_recall` 字段。
2. **无 evidence 时**：该 QA 的 recall 设为 1，不参与有意义的 recall 评估。
3. **Session 与 dia_id 匹配**：session 匹配较宽松（命中 session 即可），dia_id 匹配更严格（需精确命中具体对话轮）。
