# LongMemEval Haystack → LightRAG

为 LongMemEval 单题的 haystack 构建 LightRAG 索引。

## 用法

```bash
cd /mnt/petrelfs/leihaodong/ICML/locomo
source scripts/env.sh

# 基本用法
python benchmark/LongMemEval/tool/buildrag/build_lightrag_longmemeval.py \
  -i benchmark/LongMemEval/dataset/0000_e47becba.json \
  -o /path/to/rag_storage/0000_e47becba

# 设置 chunk 大小（token）
python benchmark/LongMemEval/tool/buildrag/build_lightrag_longmemeval.py \
  -i benchmark/LongMemEval/dataset/0000_e47becba.json \
  --chunk-token-size 512
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `-i, --input` | 必填 | 输入 JSON（dataset 下单个题目） |
| `-o, --output-dir` | RAG_STORAGE_BASE/longmemeval_{stem} | 输出目录 |
| `--chunk-token-size` | 512 | 每个 chunk 的最大 token 数；长 turn 会被拆分 |
| `--chunk-overlap` | 100 | chunk 间重叠 token 数 |
| `--embedding-model` | text-embedding-3-small | 嵌入模型 |

## 数据格式

- **doc_id**：`haystack_session_ids` 中的 session_id + `_` + turn 序号（必要时加 `_c` + chunk 序号）
- **file_paths**：`haystack_dates`（时间戳）
- **speaker**：使用原始 role（`user`、`assistant`）
- **chunking**：当 turn 的 content 超过 `--chunk-token-size` 时按 token 切分

## 依赖

- `tiktoken`：用于 token 计数（可选，无则用字符近似）
- `pip install tiktoken` 或 `pip install -e module_unit/LightRAG`
