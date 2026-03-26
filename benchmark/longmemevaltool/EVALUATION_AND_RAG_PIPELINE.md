# LongMemEval 评价方法与 RAG 完整流程

> 从建库（索引）到检索、生成、评估的完整说明。  
> 数据：`benchmark/LongMemEval/data/` | 代码：`module_unit/LongMemEval/`

---

## 一、评价方法概览

### 1.1 两类评价指标

| 类型 | 指标 | 含义 | 脚本 |
|------|------|------|------|
| **检索质量** | recall_any@k, recall_all@k, ndcg_any@k | 检索到的证据是否覆盖正确答案所在 session/turn | `print_retrieval_metrics.py` |
| **QA 正确率** | Accuracy（按 question_type 分） | 模型回答是否与标准答案一致 | `evaluate_qa.py` |

### 1.2 检索指标说明

- **recall_any@k**：top-k 检索结果中是否至少命中一个证据文档
- **recall_all@k**：top-k 是否覆盖全部证据文档
- **ndcg_any@k**：NDCG，考虑排序质量

支持的 k：1, 3, 5, 10, 30, 50。  
粒度：`session`（按 session 评估）或 `turn`（按 user turn 评估）。

### 1.3 QA 正确率（LLM-as-Judge）

`evaluate_qa.py` 使用 gpt-4o 或 llama-3.1-70b 作为评判模型，按 `question_type` 使用不同 prompt：

| question_type | 评判逻辑 |
|---------------|----------|
| single-session-user, single-session-assistant, multi-session | 回答是否包含正确答案或等价表述 |
| temporal-reasoning | 同上，且允许 ±1 天等 off-by-one |
| knowledge-update | 允许含旧信息，只要更新后的答案是正确即可 |
| single-session-preference | 是否按 rubric 正确利用用户偏好 |
| Abstention（_abs） | 是否正确判断问题不可答 |

输入：`hypothesis`（模型回答） + `question` + `answer`（或 rubric）。  
输出：`yes` / `no` → 转为 0/1 计算准确率。

---

## 二、RAG 完整流程（建库 → 检索 → 生成 → 评价）

### 2.1 流程概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. 数据准备                                                              │
│     longmemeval_s_cleaned.json 或 longmemeval_m_cleaned.json             │
│     （每题有独立的 haystack_sessions）                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. 检索（不预建全局库，每题按需构建 corpus）                               │
│     run_retrieval.sh → 输出 retrieval_log（含 retrieval_results）        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. 检索增强生成                                                          │
│     run_generation.sh → 取 top-k 检索结果拼 prompt → LLM 生成 hypothesis  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. QA 评估                                                               │
│     evaluate_qa.py → LLM-as-Judge → 输出准确率                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 关于「建库」的说明

LongMemEval 官方实现**不预先建全局向量库**：

- 每题有独立的 haystack（约 40 或 500 个 session）
- 检索时：对**当前题**的 haystack，按 session 或 turn 构建 corpus
- 在 corpus 上用 BM25 或 Dense Retriever 对 query（问题）做检索
- 不跨题共享索引，每题 corpus 只在检索时临时构建

若使用自己的 RAG 系统（如 LightRAG、Faiss 等），则需要自己定义如何为每题建库。

---

## 三、分步操作指南

### 3.1 环境准备

```bash
# 完整环境（含检索、生成）
conda create -n longmemeval python=3.9
conda activate longmemeval
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r module_unit/LongMemEval/requirements-full.txt

# 数据软链接（若 module_unit 需要 data 目录）
cd /mnt/petrelfs/leihaodong/ICML/locomo/module_unit/LongMemEval
ln -sf ../../benchmark/LongMemEval/data data
```

### 3.2 Step 1：检索

**命令**：

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo/module_unit/LongMemEval/src/retrieval

# 基本用法
bash run_retrieval.sh IN_FILE RETRIEVER GRANULARITY

# 示例：在 S 版数据上用 BM25，session 粒度
bash run_retrieval.sh \
  /mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/data/longmemeval_s_cleaned.json \
  flat-bm25 \
  session
```

**参数**：

| 参数 | 取值 | 说明 |
|------|------|------|
| IN_FILE | json 路径 | 含 haystack 的数据文件 |
| RETRIEVER | flat-bm25, flat-contriever, flat-stella, flat-gte, oracle | oracle=理想检索，仅用于上限 |
| GRANULARITY | session, turn | 索引粒度 |

**输出**：

- 目录：`module_unit/LongMemEval/retrieval_logs/{retriever}/{granularity}/`
- 文件：`{文件名}_retrievallog_{granularity}_{retriever}`
- 格式：每行一个 json，含 `retrieval_results.ranked_items` 和 `retrieval_results.metrics`

**查看检索指标**：

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo/module_unit/LongMemEval
python src/evaluation/print_retrieval_metrics.py \
  retrieval_logs/flat-bm25/session/longmemeval_s_cleaned_retrievallog_session_flat-bm25
```

### 3.3 Step 2：检索增强生成

**命令**：

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo/module_unit/LongMemEval/src/generation

# RETRIEVAL_LOG 来自 Step 1
# EXP = RETRIEVER-GRANULARITY，例如 flat-bm25-session
bash run_generation.sh RETRIEVAL_LOG MODEL EXP TOPK [HISTORY_FORMAT] [USERONLY] [READING_METHOD]
```

**示例**：

```bash
bash run_generation.sh \
  /path/to/retrieval_logs/.../longmemeval_s_cleaned_retrievallog_session_flat-bm25 \
  gpt-4o-mini \
  flat-bm25-session \
  50 \
  json \
  false \
  con
```

**参数**：

| 参数 | 示例 | 说明 |
|------|------|------|
| RETRIEVAL_LOG | 上一步输出路径 | 含 retrieval_results 的 jsonl |
| MODEL | gpt-4o-mini, gpt-4o, llama-3.1-8b-instruct 等 | 生成用 LLM |
| EXP | flat-bm25-session | 必须与检索粒度一致 |
| TOPK | 50 | 取检索结果 top-k 拼入 prompt |
| READING_METHOD | con（推荐） | con=先抽取信息再推理 |

**输出**：

- 目录：`generation_logs/{retriever_alias}/{model_alias}/{reading_method}/`
- 文件：jsonl，每行 `{question_id, hypothesis, ...}`

### 3.4 Step 3：QA 评估

**命令**：

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo/module_unit/LongMemEval
source /mnt/petrelfs/leihaodong/ICML/locomo/scripts/env.sh

# 需设置 OPENAI_API_KEY、OPENAI_ORGANIZATION（若用 gpt-4o）
python src/evaluation/evaluate_qa.py gpt-4o HYPOTHESIS_FILE REF_FILE
```

**示例**：

```bash
python src/evaluation/evaluate_qa.py gpt-4o \
  generation_logs/flat-bm25-session/gpt-4o-mini/con/xxx.jsonl \
  /mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/data/longmemeval_oracle.json
```

**注意**：REF_FILE 建议用 `longmemeval_oracle.json`，因其 question_id 与 S/M 版完全对应，且包含标准答案。

**输出**：

- 终端：总体 Accuracy，以及各 question_type 的准确率
- 文件：`HYPOTHESIS_FILE.eval-results-gpt-4o`，每行含 `autoeval_label`

---

## 四、无 RAG 的 Baseline 流程

### 4.1 长上下文直接生成（Full History）

不检索，直接把完整 haystack 喂给模型：

```bash
cd src/generation
bash run_generation.sh DATA_FILE MODEL full-history-session 1000 json false con
```

- `DATA_FILE`：longmemeval_s_cleaned.json 或 longmemeval_oracle.json
- `full-history-session`：按 session 顺序使用全部 history
- S 版约 115k tokens，需 128k 上下文模型；M 版超长，不适合直接输入

### 4.2 Oracle 基线（理想检索）

用 `longmemeval_oracle.json`，其 haystack 仅含证据 session，相当于检索完美：

```bash
bash run_generation.sh \
  /path/to/longmemeval_oracle.json \
  gpt-4o-mini \
  full-history-session \
  1000
```

该分数代表 QA 正确率**上限**（检索无损失时）。

---

## 五、数据与脚本路径速查

| 项目 | 路径 |
|------|------|
| 数据目录 | `/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/data/` |
| 检索脚本 | `module_unit/LongMemEval/src/retrieval/run_retrieval.sh` |
| 生成脚本 | `module_unit/LongMemEval/src/generation/run_generation.sh` |
| QA 评估 | `module_unit/LongMemEval/src/evaluation/evaluate_qa.py` |
| 检索指标 | `module_unit/LongMemEval/src/evaluation/print_retrieval_metrics.py` |
| 检索日志 | `module_unit/LongMemEval/retrieval_logs/` |
| 生成日志 | `module_unit/LongMemEval/generation_logs/` |

---

## 六、可选：Index Expansion 与 Time-Aware 检索

论文中还有 Index Expansion（对 session/turn 做摘要、关键短语等扩展）和 Time-Aware Query Pruning，需额外下载缓存并传入参数，详见 `module_unit/LongMemEval/README.md` 的 “Index Expansion” 和 “Time-Aware Query Expansion” 小节。
