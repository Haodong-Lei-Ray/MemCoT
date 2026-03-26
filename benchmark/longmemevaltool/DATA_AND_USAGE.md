# LongMemEval 数据结构与使用指南

> 参考论文：[longmemeval.pdf](longmemeval.pdf)（ICLR 2025）  
> 数据目录：`/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/data/`  
> 代码模块：`/mnt/petrelfs/leihaodong/ICML/locomo/module_unit/LongMemEval/`

---

## 一、数据目录结构

```
benchmark/LongMemEval/
├── data/
│   ├── longmemeval_oracle.json    # Oracle 检索版（仅证据 session）
│   ├── longmemeval_s_cleaned.json # 小规模版 S（~40 sessions，~115k tokens）
│   └── longmemeval_m_cleaned.json # 大规模版 M（~500 sessions，超长）
├── longmemeval.pdf                # 论文
└── DATA_AND_USAGE.md              # 本文档
```

---

## 二、问题数量与类型

### 2.1 总览

| 指标 | 数值 |
|------|------|
| **总问题数** | 500 |
| **Abstention 题** | 30（`question_id` 以 `_abs` 结尾） |
| **需回答题** | 470 |

### 2.2 问题类型（question_type）分布

| question_type | 数量 | 说明（对应论文能力） |
|---------------|------|----------------------|
| `single-session-user` | 70 | 单 session，证据在 user 发言中（Information Extraction） |
| `single-session-assistant` | 56 | 单 session，证据在 assistant 发言中 |
| `single-session-preference` | 30 | 单 session，个性化偏好类 |
| `multi-session` | 133 | 需跨多个 session 推理（Multi-Session Reasoning） |
| `temporal-reasoning` | 133 | 时间推理（Temporal Reasoning） |
| `knowledge-update` | 78 | 知识更新（Knowledge Updates） |
| **Abstention** | 30 | 正确答案为「未提及」，需模型识别不可答 |

---

## 三、单条数据格式（每个 JSON 文件相同）

每个文件是一个 **list**，长度为 500，元素结构如下：

```json
{
  "question_id": "e47becba",
  "question_type": "single-session-user",
  "question": "What degree did I graduate with?",
  "question_date": "2023/05/30 (Tue) 23:40",
  "answer": "Business Administration",
  "answer_session_ids": ["answer_280352e9"],
  "haystack_dates": ["2023/05/20 (Sat) 02:21", "..."],
  "haystack_session_ids": ["sharegpt_yywfIrx_0", "85a1be56_1", "..."],
  "haystack_sessions": [
    [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "...", "has_answer": true}
    ],
    [...]
  ]
}
```

| 字段 | 含义 |
|------|------|
| `question_id` | 唯一 ID；若以 `_abs` 结尾则为 abstention 题 |
| `question_type` | 题目类型（见上表） |
| `question` | 问题文本 |
| `answer` | 标准答案；abstention 题为解释性说明 |
| `haystack_dates` | 每个 session 的时间戳 |
| `haystack_session_ids` | 每个 session 的 ID |
| `haystack_sessions` | 对话内容，与 dates/ids 一一对应；每项为 `[{role, content}, ...]`，证据 turn 可有 `has_answer: true` |
| `answer_session_ids` | 包含证据的 session ID 列表（session 级 ground truth） |

---

## 四、三个文件的区别

| 文件 | sessions 数 | 约 tokens | 用途 |
|------|-------------|-----------|------|
| `longmemeval_oracle.json` | 仅 evidence | 短 | 测 QA 正确率上限（理想检索） |
| `longmemeval_s_cleaned.json` | ~40 | ~115k | 128k 内长上下文模型 |
| `longmemeval_m_cleaned.json` | ~500 | 超长 | RAG / 记忆系统、多步检索 |

---

## 五、如何结合 module_unit/LongMemEval 使用

### 5.1 模块目录结构

```
module_unit/LongMemEval/
├── README.md
├── README_locomo.md           # 数据格式速览
├── quick_test_gpt4omini.py    # 单题 gpt-4o-mini 小测试
├── data/                      # 官方推荐放 data/，我们改用 benchmark 路径
├── src/
│   ├── evaluation/            # QA 评估
│   │   ├── evaluate_qa.py     # 自动评估 QA 正确率
│   │   └── print_qa_metrics.py
│   ├── retrieval/             # 记忆检索
│   ├── generation/            # 生成与 RAG
│   └── index_expansion/       # 索引扩展
└── requirements-*.txt
```

### 5.2 数据路径对应

**官方期望**：`LongMemEval/data/xxx.json`  

**本地实际**：`benchmark/LongMemEval/data/xxx.json`  

使用时需显式指定路径，或做软链接：

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo/module_unit/LongMemEval
ln -sf ../../benchmark/LongMemEval/data data
```

### 5.3 标准评测流程

1. **跑自己的系统**：把 `haystack_sessions` 喂给模型，收集 `question_id` 和模型回答 `hypothesis`。
2. **保存为 jsonl**：每行 `{"question_id": "...", "hypothesis": "..."}`。
3. **调用官方评估**：

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo/module_unit/LongMemEval
source /mnt/petrelfs/leihaodong/ICML/locomo/scripts/env.sh

# 使用 benchmark 下的数据
python src/evaluation/evaluate_qa.py gpt-4o your_hypothesis.jsonl \
  /mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/data/longmemeval_oracle.json
```

### 5.4 快速单题测试

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh
python module_unit/LongMemEval/quick_test_gpt4omini.py
```

该脚本会从 `longmemeval_s_cleaned.json` 取第 0 题，拼好 history 后调用 gpt-4o-mini 回答，并打印模型答案与 gold answer。

### 5.5 网页查看器（浏览题目）

可用本地查看器逐题查看 JSON 数据：

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval
python serve_viewer.py
# 浏览器打开 http://localhost:8765/viewer.html
```

或直接打开 `viewer.html`，通过「选择本地 JSON 文件」加载任意 longmemeval 的 json。

### 5.6 与 LoCoMo 项目集成建议

- 数据统一放在 `benchmark/LongMemEval/data/`，避免重复下载。
- 评估脚本通过绝对路径引用：  
  `/mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval/data/longmemeval_*.json`
- 若需在 module_unit 内运行官方脚本，可用软链接将 `data` 指到 `benchmark/LongMemEval/data`。
