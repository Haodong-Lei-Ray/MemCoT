# LongMemEval 转 LoCoMo 格式：可行性分析报告

> 对比数据源：`/mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json` 与 `/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/data/longmemeval_s_cleaned.json`

---

## 一、两种格式结构对比

### 1.1 LoCoMo 格式（locomo10.json）

| 字段 | 结构 | 说明 |
|------|------|------|
| 顶层 | `[{...}, {...}]` | 样本数组，每个样本对应一段**共享对话** |
| `qa` | `[{question, answer, evidence, category}, ...]` | 多道 QA 共用一个 conversation |
| `evidence` | `["D1:3", "D2:8"]` | **turn 级**证据，`Dx:y` = 第 x 个 session 的第 y 个 turn |
| `category` | 1–5 | 1=多跳，2=单跳时间，3=推理，4=其它，5=对抗（含 `adversarial_answer`） |
| `conversation` | `{speaker_a, speaker_b, session_N, session_N_date_time}` | 会话结构 |
| `session_N` | `[{speaker, dia_id, text, ...}]` | 每个 turn 有 `speaker`（角色名）、`dia_id`（Dx:y）、`text` |
| 扩展 | `img_url`, `blip_caption`, `query` | 可选的图文/检索相关字段 |

**特点**：Caroline / Melanie 双角色对话，连续叙事，同一 conversation 上有多道 QA。

---

### 1.2 LongMemEval 格式（longmemeval_s_cleaned.json）

| 字段 | 结构 | 说明 |
|------|------|------|
| 顶层 | `[{...}, {...}]` | 500 条独立样本，**每题一个独立 haystack** |
| `question_id` | string | 唯一 ID；以 `_abs` 结尾表示 abstention 题 |
| `question_type` | string | 题目类型（见下表） |
| `question`, `answer` | string | 问题与答案 |
| `answer_session_ids` | `["answer_xxx", ...]` | **session 级**证据（哪些 session 含答案） |
| `haystack_sessions` | `[[{role, content, has_answer?}, ...], ...]` | 模拟用户聊天历史 |
| `haystack_dates` | `["2023/05/20 (Sat) 02:21", ...]` | 每个 session 的时间戳 |
| `has_answer` | bool（可选） | **turn 级**标记，表示该 turn 包含证据 |

**question_type 分布**：

| question_type | 数量 | 对应能力 |
|---------------|------|----------|
| single-session-user | 70 | 单 session，证据在 user 发言 |
| single-session-assistant | 56 | 单 session，证据在 assistant 发言 |
| single-session-preference | 30 | 单 session，个性化偏好 |
| multi-session | 133 | 跨多 session 推理 |
| temporal-reasoning | 133 | 时间推理 |
| knowledge-update | 78 | 知识更新 |
| **Abstention** | 30 | 正确答案为「未提及」 |

---

## 二、核心差异与转换难点

### 2.1 结构差异

| 维度 | LoCoMo | LongMemEval | 影响 |
|------|--------|-------------|------|
| 样本粒度 | 一个 conversation 对应多道 QA | 一个 question 对应一个 haystack | 转换后：每题 = 1 个 LoCoMo 样本，每个样本只有 1 道 QA |
| 证据粒度 | turn 级（Dx:y） | session 级 + turn 级（has_answer） | 必须把 evidence 映射到 turn 级 |
| 角色设定 | 固定角色（Caroline, Melanie） | 通用 user / assistant | 语义不同，需用 User / Assistant 或占位名 |
| 时间格式 | `1:56 pm on 8 May, 2023` | `2023/05/20 (Sat) 02:21` | 需做格式转换 |
| 多媒体 | 支持 img_url、blip_caption、query | 纯文本 | 转换后无图片，可留空或省略 |

### 2.2 语义差异

- **LoCoMo**：两人（Caroline / Melanie）的连续生活叙事，QA 围绕同一时间线展开。
- **LongMemEval**：模拟单一用户的通用对话历史，来源于 ShareGPT、UltraChat 等，无固定 persona。

因此，转换得到的是** structurally LoCoMo-like、 semantically different** 的合成数据，适用于测试检索/推理 pipeline，但不能完全复现 LoCoMo 的 persona 场景。

---

## 三、可行性结论

### 3.1 总体结论

**可以实现部分转换**：非 abstention 题目在技术上可以转为 LoCoMo 结构；abstention 和部分类型在语义上与 LoCoMo 不完全对齐。

### 3.2 按 question_type 的可行性

| question_type | 可转换 | 说明 |
|---------------|--------|------|
| single-session-user | ✅ 是 | 直接映射，category 建议 2 或 3 |
| single-session-assistant | ✅ 是 | 同上 |
| single-session-preference | ⚠️ 部分 | 偏好类与 LoCoMo 推理类不完全一致，可映射到 category 3 |
| multi-session | ✅ 是 | 映射到 category 1（多跳） |
| temporal-reasoning | ✅ 是 | 映射到 category 2（时间） |
| knowledge-update | ✅ 是 | 映射到 category 3（推理） |
| **Abstention** | ⚠️ 特殊 | 与 LoCoMo category 5 不同，见下文 |

### 3.3 Abstention 的特殊性

- **LoCoMo category 5**：对抗题，存在**误导性**证据和 `adversarial_answer`，模型需区分真假证据。
- **LongMemEval Abstention**：题干**无法在上下文中找到答案**，期望回答「未提及」或类似表述。

两者评估目标不同：LoCoMo 5 是「抗干扰」，LongMemEval Abstention 是「识别不可答」。若强行映射到 category 5，会与 LoCoMo 原有 category 5 含义冲突。建议：

- **方案 A**：abstention 单独标记（如 `category: 6` 或 `is_abstention: true`），评估时分开处理。
- **方案 B**：不转换 abstention，仅转换 470 道非 abstention 题。

---

## 四、转换规则（非 abstention）

### 4.1 会话与 turn 映射

```
LongMemEval haystack_sessions[i]  →  LoCoMo session_{i+1}
haystack_dates[i]                 →  session_{i+1}_date_time（格式需转换）
haystack_sessions[i][j]           →  session_{i+1} 的第 j 个 turn
```

### 4.2 Turn 结构映射

| LongMemEval | LoCoMo |
|-------------|--------|
| `role: "user"` | `speaker: "User"`（或占位名） |
| `role: "assistant"` | `speaker: "Assistant"` |
| `content` | `text` |
| — | `dia_id: "D{i+1}:{j+1}"`（按 session、turn 序号生成） |

### 4.3 证据映射

- 优先使用 `has_answer: true` 的 turn 作为证据；
- 对每个 `has_answer: true` 的 turn，根据其所在 session 和 turn 索引生成 `Dx:y`，加入 `evidence` 列表；
- 若某题在 `longmemeval_s_cleaned.json` 中**没有** `has_answer`，则只能依赖 `answer_session_ids`：可将该 session 内所有 turn 的 dia_id 都加入 evidence（粗粒度，有信息损失）。

### 4.4 时间格式转换

```
LongMemEval: "2023/05/20 (Sat) 02:21"
可选 LoCoMo 风格: "2:21 am on 20 May, 2023"
```

### 4.5 category 映射建议

| LongMemEval question_type | LoCoMo category |
|---------------------------|-----------------|
| multi-session | 1 |
| temporal-reasoning | 2 |
| single-session-user, single-session-assistant | 2 或 3 |
| single-session-preference, knowledge-update | 3 |

---

## 五、输出格式示例

转换后的单条样本结构示例：

```json
{
  "qa": [
    {
      "question": "What degree did I graduate with?",
      "answer": "Business Administration",
      "evidence": ["D44:5", "D44:6"],
      "category": 2
    }
  ],
  "conversation": {
    "speaker_a": "User",
    "speaker_b": "Assistant",
    "session_1_date_time": "2:21 am on 20 May, 2023",
    "session_1": [
      {
        "speaker": "User",
        "dia_id": "D1:1",
        "text": "The farmer needs to transport a fox..."
      },
      {
        "speaker": "Assistant",
        "dia_id": "D1:2",
        "text": "To solve this puzzle..."
      }
    ],
    "session_2_date_time": "2:57 am on 20 May, 2023",
    "session_2": [...],
    ...
  }
}
```

（其中 D44 对应包含 `has_answer: true` 的 session 在 haystack_sessions 中的索引 +1）

---

## 六、无法完全对齐的原因总结

1. **数据范式不同**：LoCoMo 为多 QA 共享 conversation；LongMemEval 为单题单 haystack。
2. **角色设定不同**：LoCoMo 为 persona 对话；LongMemEval 为通用 user-assistant。
3. **Abstention 与 category 5 不同**：LoCoMo 5 是对抗/干扰，LongMemEval abstention 是不可答。
4. **多媒体缺失**：LongMemEval 无 `img_url` 等，转换后为纯文本。
5. **evidence 粒度**：部分 LongMemEval 数据可能仅有 session 级标注，转 turn 级会有损失。

---

## 七、建议

- **可转换**：470 道非 abstention 题可按上述规则转为 LoCoMo 结构，用于检索、证据定位、多跳推理等评估。
- **慎用**：abstention 题需单独设计评估逻辑，不建议直接映射到 LoCoMo category 5。
- **建议实现**：编写独立转换脚本，支持过滤 abstention、选择 question_type 子集、输出为 `longmemeval_as_locomo.json`，便于与现有 LoCoMo 评估流程对接。
