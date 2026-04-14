# MemCoT：`MemCoT.run` 架构说明

本文描述 [`memcot.py`](../memcot.py) 中 **`MemCoT` 类与 `run()`** 的职责、数据流与模块边界，并给出 **MemCoT Process** 与 **Responder Agent Process** 的划分方式，便于后续重构、评测对接或拆进程部署。

---

## 1. 角色概览

`MemCoT.run()` 实现一条 **ReAct 风格** 的长对话问答流水线：在固定上限步数内，反复 **检索（Act）→ 判据与续查（Observation）→ 轨迹记录（Thought）**，必要时进入 **最终作答**。检索后端可为 NaiveRAG（pkl 向量）或 LightRAG（图谱混合检索），由 `rag_type` 选择。

**返回值（`MemCoTExit`）**包括：`prompt`、`kind`（`evidence_ok` 或 `fallback`）、`trajectory` 等，并由 `finalize_memcot_exit` 最终调用 LLM 写入 `output_dir/result.json`。

---

## 2. 依赖与外围设施

| 类别 | 说明 |
|------|------|
| **检索** | `load_rag_retrieve(rag_file_path)` 创建 retriever，循环内统一调用 `retriever.retrieve_multi(query_queue)` |
| **Agent** | 封装为类：`ZoomInFocalRetrieve`、`ZoomOutContextExpansion`、`JudgeAgent`、`ResponderAgent`、`FallbackResponderAgent` |
| **Conversation** | `Conversation` 类封装当前对话的 `conv_id`、`benchmark`、`full_conv`、`haystack_session_ids` 等 |
| **全量对话** | `full_conv` 仅在循环耗尽后由 `FallbackResponderAgent` 使用 |
| **开关** | `agent_flag` 五位 `0/1` 控制各个阶段 agent 是否启用（与 CLI 说明一致） |

---

## 3. `MemCoT.run` 内部状态（跨步维护）

初始化阶段建立：

- **`root_query` / `query`**：根问题；LongMemEval 下会改写 query。
- **`query_queue`**：当前轮待检索的子查询列表；每步由 `JudgeAgent` 的 `new_queries` 整体替换。
- **`short_semantic_memory`**：跨步累积的「已采纳」证据片段（`dia_id` 级），供后续 agent 与最终作答使用。
- **`trajectory`**：每步一条 `{ Act, Observation, Thought }`，对应 ReAct 轨迹。
- **`fail_episondic_queue_trajectory`**、`conv_memory`、`visual_seen_session` 等：辅助 observation 与其它视图 agent。

---

## 4. 主循环（单步语义）

每一步（`for step in range(1, self.max_step + 1)`）顺序为：

1. **Act — 检索**  
   `self.ragretriever.retrieve_multi(query_queue)` → `rag_results`（去重后的证据列表）。当前实现要求 `len(rag_results) != 0`（见断言）。

2. **Act — 证据精化（可选，受 `agent_flag` 控制）**  
   在 `rag_results` 上串联可选模块，结果写入 `temp_short_memory`：  
   - `ZoomInFocalRetrieve`：从粗检索中筛「有用证据」并给出缺失信息描述。  
   - `ZoomOutContextExpansion`：在会话窗口内扩展/精炼证据（`K = middle_scale`）。  
   - `panoramic_visual_grounding`：非 `longmemeval` 时可选，结合图像会话视图。

3. **Act 记录**  
   `act_record`：步号、`query_queue`、`rag_record`、`temp_useful_rag_dia_ids`。

4. **Observation**  
   `JudgeAgent.run(...)` → `can_answer`、`useful_evidence`、`new_queries`、`action`、`thinking` 等；并根据 `useful_evidence` 更新 `short_semantic_memory`。

5. **轨迹落盘**  
   `trajectory.append({ Act, Observation, Thought })`，其中 `Thought` 取自 observation 的 `thinking`。

6. **查询队列推进**  
   `assert len(new_queries) != 0`，然后 `query_queue = new_queries`（进入下一轮检索目标）。若本步判定可答或到达 `step == self.max_step - 1`，则进入 **Responder**（见下节）。

7. **循环结束后的回退**  
   若未提前返回，则调用 `FallbackResponderAgent`，以整段对话为上下文给出答案，`kind` 为 `fallback`。

---

## 5. 过程划分：MemCoT Process vs Responder Agent Process

划分原则：**MemCoT Process 负责「为回答问题而持续检索与整理证据」；Responder Agent Process 负责「在已有约束下生成对用户可见的最终答案」**。二者通过 **结构化的 observation 输出** 与 **`short_semantic_memory` 证据集** 交接。

### 5.1 MemCoT Process（记忆与检索推理过程）

**范围（建议定义）**

- **输入**：`root_query`、`query_queue`、`Conversation` 对象、`short_semantic_memory` 历史、上一轮轨迹摘要（如 `last_queries`）。
- **核心动作**：  
  - 多查询 RAG 检索；  
  - 可选多阶段证据裁剪与扩展（ZoomIn / ZoomOut / visual）；  
  - **Observation (JudgeAgent)**：判断是否已具备作答条件、哪些 `dia_id` 可信、**下一步检索子问题 `new_queries`**；  
  - 更新 `short_semantic_memory`、`trajectory`、`query_queue`。
- **输出（对接 Responder 的契约）**：  
  - `can_answer`（布尔）；  
  - `short_semantic_memory`（证据子图）；  
  - `obs_thinking`（observation 的推理，供作答时引用）；  
  - `new_queries`（MemCoT 继续检索时用；一旦进入 Responder，最终以 `ResponderAgent` 或 `FallbackResponderAgent` 的结果为准）。

**在代码中的对应片段**：从循环开始到 `JudgeAgent` 完成、并更新 `trajectory` 与 `query_queue` 为止；即 **不包含** `ResponderAgent` / `FallbackResponderAgent` 调用。

**本质**：这是一个 **封闭的「检索—判断—再检索」控制回路**，目标是最大化 `short_semantic_memory` 中与问题相关的可验证证据，直到 observation 认为可答或步数逼近上限。

### 5.2 Responder Agent Process（最终作答过程）

**范围（建议定义）**

- **触发条件 1（早停作答）**：`obs["can_answer"] or step == self.max_step - 1`。  
  - 调用 **`ResponderAgent`**：`query`、`short_semantic_memory`、`obs_report=obs_thinking`、`additional_information` 等。  
  - 返回 `MemCoTExit(kind="evidence_ok")`，包含构造好的 `prompt`。
- **触发条件 2（兜底全对话）**：主循环正常结束仍未返回成功答案时。  
  - 调用 **`FallbackResponderAgent`**：`query`、`Conversation.full_conv`、`additional_information` 等。  
  - 返回 `MemCoTExit(kind="fallback")`，包含构造好的 `prompt`。

**本质**：这是 **单次的生成调用**，不再涉及检索或状态修改。它信任 MemCoT Process 传来的 `short_semantic_memory` 或退而求其次使用 `full_conv`。

---

## 6. 结语

当前的 `MemCoT.run()` 已经通过面向对象重构，将 Agent 逻辑封装在各个类中，使得状态管理更加清晰，参数传递更加简洁。`Conversation` 类的引入将对话元数据集中管理，方便不同 Benchmark 的拓展。 CLI Daemon 的引入（`memcot_cil.py`）进一步将 `MemCoT` 实例常驻内存，提供高效的检索服务。
