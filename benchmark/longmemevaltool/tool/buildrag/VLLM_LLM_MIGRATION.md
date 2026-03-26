# vLLM 本地 LLM 接入方案

## 背景

当前 `build_lightrag_longmemeval.py` 中 LightRAG 的两个模型来源**不同**：

| 用途 | 当前实现 | 走什么网络 |
|------|----------|------------|
| **Embedding** (`openai_embed`) | `text-embedding-3-small` via OpenAI API | 代理 → 外网 `$OPENAI_BASE_URL` |
| **LLM 实体抽取** (`gpt_4o_mini_complete`) | `gpt-4o-mini` via OpenAI API | 代理 → 外网 `$OPENAI_BASE_URL` |

**目标**：LLM 改为走内网 vLLM 部署的 Qwen（`http://10.140.37.10:8100/v1`），Embedding 保持不变。

## 核心难点

环境变量 `OPENAI_BASE_URL` / `OPENAI_API_KEY` 是给 Embedding 用的（走代理到外网），而 vLLM 是内网地址、不走代理、key 为 `EMPTY`。两者不能共用同一套环境变量。

## 方案：自定义 `llm_model_func`

LightRAG 的 `llm_model_func` 只需要是一个 `async def(prompt, system_prompt, history_messages, **kwargs) -> str` 签名的函数。底层 `openai_complete_if_cache` 支持显式传入 `base_url` 和 `api_key`，会覆盖环境变量。

### 改动点（仅 `build_lightrag_longmemeval.py`）

```python
from lightrag.llm.openai import openai_complete_if_cache

# vLLM 本地部署配置
VLLM_BASE_URL = "http://10.140.37.10:8100/v1"   # 从 vllm_connection_*.txt 获取
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "Qwen3-30B-A3B-Instruct-2507"  # vLLM --served-model-name

async def vllm_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        VLLM_MODEL_NAME,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        base_url=VLLM_BASE_URL,
        api_key=VLLM_API_KEY,
        # 不走代理（内网直连）
        openai_client_configs={"trust_env": False},
        **kwargs,
    )
```

然后把 `LightRAG(...)` 中的 `llm_model_func=gpt_4o_mini_complete` 改为 `llm_model_func=vllm_complete`。

### 为什么可行

1. `openai_complete_if_cache` 内部调用 `create_openai_async_client(base_url=..., api_key=...)`，显式参数会覆盖 `$OPENAI_BASE_URL`。
2. `client_configs={"trust_env": False}` 传给 `AsyncOpenAI`（底层 httpx），确保内网请求不走 `$http_proxy`。
3. `openai_embed` 继续读环境变量 `$OPENAI_BASE_URL` + `$OPENAI_API_KEY`，走代理到外网，互不干扰。

### 流程图

```
build_lightrag_longmemeval.py
├── openai_embed (Embedding)
│   ├── base_url = $OPENAI_BASE_URL (http://35.220.164.252:3888/v1)
│   ├── api_key  = $OPENAI_API_KEY
│   └── 走代理 → 外网 OpenAI 兼容 API
│
└── vllm_complete (LLM 实体抽取)
    ├── base_url = http://10.140.37.10:8100/v1 (硬编码或 --vllm-url 参数)
    ├── api_key  = EMPTY
    ├── trust_env = False (不走代理)
    └── 内网直连 → vLLM (Qwen3-30B-A3B-Instruct-2507)
```

## 实施步骤

1. **确认 vLLM 服务在线**：`curl http://10.140.37.10:8100/v1/models`
2. **修改 `build_lightrag_longmemeval.py`**：
   - 新增 `vllm_complete` 函数
   - 新增 CLI 参数 `--vllm-url`、`--vllm-model`（可选，带默认值）
   - `LightRAG(llm_model_func=...)` 根据参数选择 `gpt_4o_mini_complete` 或 `vllm_complete`
3. **修改 slurm 脚本**：不需要额外改动（env.sh 设置的 `OPENAI_*` 给 embedding 用，vLLM 地址在代码里）
4. **测试**：先跑单个问题验证实体抽取日志中出现 Qwen 的输出风格
5. **批量运行**：确认无误后 `--batch-size 20` 批量跑

## 注意事项

- vLLM 地址 `10.140.37.10:8100` 是 slurm job 分配的，**重启后会变**。建议从 `vllm_connection_*.txt` 读取，或通过 `--vllm-url` 参数传入。
- vLLM 的 context length 需要 >= LightRAG 的单次 prompt 长度（实体抽取 prompt 通常 2k-4k tokens）。
- Qwen 的实体抽取质量需要验证，建议先对比一个问题的 `gpt-4o-mini` 和 `Qwen` 的 KG 输出。
