# MemCot + LoCoMo 方案讨论

## 〇、LoCoMo 有训练集吗？

**没有。** LoCoMo 官方**不提供训练集**，也没有官方的 train / dev / test 划分。

- 官方定义：LoCoMo 是一个 **evaluation benchmark**（评估基准），不是带训练/测试拆分的通用数据集。
- 当前发布：**一个文件 `locomo10.json`**，内含 **10 个 conversation**（约 10 个 sample_id），每个 conversation 下有大量 QA（例如单个 sample 就有上百条 qa）。
- 用途：所有代码与说明都是「在 LoCoMo 上**评估**模型」（evaluate on the LoCoMo QA task），没有「在 LoCoMo 上训练」的脚本或划分。
- 历史：早期 Arxiv 曾 release 过 50 个 conversation，当前 10 个是子集，用于高质量标注与成本可控的闭源模型评估；依然**没有**给出训练集或划分方案。

**对 MemCot 的直接影响**：  
若你想用 LoCoMo 的 (question, answer) 来**建记忆**，并仍在 LoCoMo 上**测试**，就需要**自己做划分**，例如：
- 按 **sample_id**：例如 8 个 conversation 的 QA 建记忆，2 个 conversation 的 QA 做测试；
- 或按 **题目**：每个 conversation 内按比例划分（如 80% QA 建记忆，20% 做测试），避免同一题既进记忆又被当作测试题。

否则会出现「测试题的标准答案已经出现在记忆里」的泄露问题。

---

## 一、LoCoMo 是否有标准答案？

**有。** LoCoMo 每条 QA 都带标注答案和证据。

- **`answer`**：标准答案（多数题目）
- **`adversarial_answer`**：部分题目只有“对抗式”选项，无标准短答
- **`evidence`**：支撑答案的对话片段 id（如 `["D1:3", "D2:8"]`），可用于检索或做 recall 评估
- **`category`**：题型
  - 1: 多跳 (multi-hop)，F1 按子答案算
  - 2: 时间/日期
  - 3: 开放域，答案可能含 `;` 分隔的多个部分
  - 4: 其他
  - 5: 对抗题（选“文中未提及”或正确选项）

因此：**可以做“模型预测 vs 标准答案”的自动评估**（F1 / exact match 等），也可以在你说的“记忆 + 检索”流程里，用 `(question, answer)` 构建记忆。

---

## 一（补充）、多跳 (multi-hop) 与「F1 按子答案算」

**多跳**：需要从对话的**多处**（多个 session/多轮对话）综合信息才能答对的题，对应 `category == 1`。答案往往是**多个并列要点**，在标注里用**逗号分隔**成多个「子答案」。

**例子（来自 locomo10.json）**：

| 字段 | 内容 |
|------|------|
| **question** | What activities does Melanie partake in? |
| **answer** | pottery, camping, painting, swimming |
| **evidence** | ['D5:4', 'D9:1', 'D1:12', 'D1:18'] |

- **子答案**：把 `answer` 按逗号拆开 → `['pottery', 'camping', 'painting', 'swimming']`，每个词来自不同对话片段（D5、D9、D1 等），所以是「多跳」。
- **F1 按子答案算**：代码里对 category 1 用的是 `f1(prediction, ground_truth)`（见 `task_eval/evaluation.py`）：
  1. 把**标准答案**和**模型预测**都按逗号拆成列表：  
     `ground_truths = ['pottery', 'camping', 'painting', 'swimming']`，  
     `predictions = [模型输出的每个逗号分隔部分]`。
  2. 对**每一个**标准子答案 `gt`，在模型给出的所有子答案里找「和 gt 的 token F1 最高的那个」作为该 gt 的得分。
  3. 把这些得分**平均**，得到这道题的最终 F1。

**数值例子**（标准答案 = `pottery, camping, painting, swimming`）：

| 模型预测 | 多跳 F1 |
|----------|---------|
| pottery, camping, painting, swimming | **1.0**（四个子答案都对齐） |
| pottery, painting, hiking | **≈0.5**（对两个、错一个、漏一个） |
| pottery | **0.25**（只对一个，其余三个无匹配） |

也就是说：**多跳题不是整句一个 F1，而是「每个子答案一个 F1，再取平均」**，漏答或少答会拉低平均，多答错答也会因为匹配不到正确的 gt 而拉低分数。

---

## 二、MemCot 设想（你的表述）

- **记忆形式**：`记忆 = question + thinking + answer`  
  即由 **(question, answer)** 生成或附加上 **thinking**，再存成一条记忆。
- **测试时**：给定 LoCoMo 的 **question** → 用该 question 去**搜索记忆** → 把匹配到的记忆载入**上文** → 再让模型基于“上文 + 当前 question”生成答案。

下面按“记忆从哪来、thinking 从哪来、怎么检索、怎么测”拆开说。

---

## 三、记忆从哪来：question + answer → 记忆

有两种典型来源：

### 方案 A：直接用 LoCoMo 的 (question, answer) 建记忆

- 从 `locomo10.json` 里取出所有 `(question, answer)`（以及可选的 `evidence`）。
- 每条记忆 = **question + thinking + answer**。
  - **thinking** 需要额外生成或从别处来（见下一节）。
- 优点：和测试集同分布，检索时容易匹配到“同题/类似题”的记忆。
- 注意：若测试时用的就是这些 question，要严格区分**训练/建记忆用的 QA** 和**评估用的 QA**（例如按 sample_id 或按题目 id 做 split），避免同一题既建记忆又当测试题，造成信息泄露。

### 方案 B：用外部/其他对话数据建记忆

- 用其他长对话、QA 数据构建 (question, thinking, answer)。
- 测试时只在 LoCoMo 上评估：用 LoCoMo 的 question 去搜这些“外部记忆”，再答题。
- 优点：无 LoCoMo 泄露问题；可考察“跨域记忆”的迁移。
- 缺点：检索匹配度可能不如“同分布”的 A。

也可以 **A + B 混合**：一部分记忆来自 LoCoMo（且做好 train/test 划分），一部分来自外部。

---

## 四、“thinking”从哪来

你定义的是 **记忆 = question + thinking + answer**，所以关键是 **thinking** 怎么得到。几种可能：

1. **用模型生成**  
   输入 (question, answer)，让模型生成“推理过程/依据”，作为 thinking，再拼成记忆。  
   - 可加约束：必须基于 `evidence` 对应的对话内容来写 thinking，这样记忆更贴近 LoCoMo 的 evidence。

2. **用 evidence 当 thinking**  
   把 `evidence` 对应的对话片段拼成一段文字（或结构化描述），当作“依据/推理”，即 thinking = f(evidence)。  
   - 实现简单，且和 LoCoMo 的标注一致。

3. **模板/规则**  
   例如：`thinking = "根据对话中关于...的讨论，可知..."`，再填 (question, answer)。  
   - 可控，但表达力有限。

4. **先有 thinking 再抽 answer**  
   若你的 MemCot 是“先有长推理再得到答案”的设定，也可以从“长推理”里抽/生成 (question, answer)，再一起存成记忆。

建议先定一种（例如 1 或 2），在少量样本上看看“载入记忆后的回答”是否明显好于“无记忆”，再决定是否加复杂度。

---

## 五、检索：用当前 question 搜记忆

- **输入**：当前 LoCoMo 的 **question**。
- **输出**：Top-K 条记忆（每条是 question + thinking + answer）。
- **实现选项**：
  - 稀疏：BM25 等，对 question 和记忆里的 question（或整条记忆）建索引。
  - 稠密：用 encoder 对 question 和“记忆文本”编码，做向量检索（如 FAISS / 你现有的 RAG 管线）。
  - 混合：keyword + 向量一起用。

注意：若记忆条目的“主键”是 question，检索时可以用 **question 相似度** 为主，必要时再考虑 thinking/answer 长度或重要性（例如在排序时加一点 reward）。

---

## 六、测试流程（在 LoCoMo 上）

可以抽象成：

```
对 LoCoMo 中每条测试 QA：
  1. 取 question（以及可选的 conversation/context，若你要和原版 LoCoMo 设定对比）
  2. 用 question 在 MemCot 记忆库里检索 → 得到 Top-K 条记忆
  3. 将 Top-K 条记忆格式化成“上文”（例如：每条一行 "Q: ... Thinking: ... A: ..."）
  4. 模型输入 = [上文中的记忆] + [当前 conversation（若用）] + [当前 question]
  5. 模型输出 = 预测 answer
  6. 用 LoCoMo 的 answer / adversarial_answer 做评估（F1、exact match、category 5 的 0/1）
```

这样就有“标准答案”：**LoCoMo 的 answer 就是标准答案**；你比较的是“模型在 MemCot 检索增强下的预测”和“标注 answer”。

---

## 七、需要你拍板的几个点

1. **记忆来源**  
   - 仅 LoCoMo（需严格 train/test 划分），还是外部数据，还是混合？

2. **thinking 定义**  
   - 模型生成 / evidence 转写 / 模板 / 其他？

3. **测试时上下文**  
   - 只有“检索到的记忆”？  
   - 还是“记忆 + 原始长对话”（即 LoCoMo 的 conversation）？  
   - 若两者都有，顺序和格式怎么设计（例如先记忆后对话，或先对话后记忆）？

4. **检索粒度**  
   - 按“单条记忆”检索，还是按“整个 sample 的若干 QA”做成一条大记忆再检索？

5. **评估指标**  
   - 沿用 LoCoMo 官方的 F1 / category 划分即可；若你做 RAG，还可以加 **recall**：检索到的记忆里是否包含当前题的 evidence 或标准 answer。

---

## 八、和现有 LoCoMo 脚本的关系

- **建记忆 / 生成 thinking**：可以单独写脚本，读 `data/locomo10.json`，输出“记忆库”（jsonl 或带索引的存储）。
- **检索**：可以接在你现有的 RAG 管线（如 `task_eval/rag_utils.py` 的 retriever）上，只是把“文档”从 observation/session summary 换成“MemCot 记忆条”。
- **评估**：仍用 `task_eval/evaluate_qa.py` 的流程，只改“如何构造 model 的 context”（从“长对话截断”或“RAG 文档”改为“MemCot 检索到的 question+thinking+answer”），输出仍写进 `locomo10_qa.json` 之类的，用同一套 `eval_question_answering` 和 `analyze_aggr_acc` 算 F1。

如果你愿意，下一步可以定一个最小方案（例如：记忆只用 LoCoMo 的 question+answer，thinking 先用 evidence 转写；检索用 BM25；测试时只给记忆不给原对话），我可以按你当前 repo 结构写一版“MemCot 记忆构建 + 检索 + 在 LoCoMo 上跑评估”的伪代码或具体脚本接口设计。
