"""
vLLM 本地部署的 LLM 函数，供 LightRAG 的 llm_model_func 使用。

Embedding 继续走环境变量 $OPENAI_BASE_URL（代理→外网），
LLM 走内网 vLLM（不走代理），两者互不干扰。

用法:
    from vllm_llm import create_vllm_complete

    vllm_complete = create_vllm_complete(
        base_url="http://10.140.37.10:8100/v1",
        model_name="Qwen3-30B-A3B-Instruct-2507",
    )

    rag = LightRAG(
        ...
        llm_model_func=vllm_complete,
    )
"""

import os
import sys
from pathlib import Path
import httpx
import json
import re
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
PROJECT_ROOT = BENCHMARK_DIR.parent.parent
LIGHTRAG_PATH = PROJECT_ROOT / "module_unit" / "LightRAG"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LIGHTRAG_PATH))

from lightrag.llm.openai import openai_complete_if_cache

# 默认值（可通过 create_vllm_complete 覆盖）
DEFAULT_VLLM_BASE_URL = "http://10.140.37.10:8100/v1"
DEFAULT_VLLM_MODEL = "Qwen3-30B-A3B-Instruct-2507"
DEFAULT_VLLM_API_KEY = "EMPTY"


def create_vllm_complete(
    base_url: str = DEFAULT_VLLM_BASE_URL,
    model_name: str = DEFAULT_VLLM_MODEL,
    api_key: str = DEFAULT_VLLM_API_KEY,
):
    """
    创建一个走内网 vLLM 的 LLM 函数，签名兼容 LightRAG 的 llm_model_func。

    Args:
        base_url: vLLM 的 OpenAI 兼容 API 地址（从 vllm_connection_*.txt 获取）
        model_name: vLLM 的 --served-model-name
        api_key: vLLM 通常不校验，填 "EMPTY"
    """

    async def vllm_complete(
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        return await openai_complete_if_cache(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            base_url=base_url,
            api_key=api_key,
            # 内网直连，不读环境变量中的代理
            openai_client_configs={"http_client": httpx.AsyncClient(trust_env=False)},
            **kwargs,
        )

    return vllm_complete


def _load_vllm_config(config_path: Path) -> dict[str, Any]:
    """
    兼容“非严格 JSON”的 config 文件：
    - 如果是标准 JSON：直接 json.loads
    - 如果是从代码片段复制出来的：用正则提取 DEFAULT_* 里的取值
    """
    if not config_path.exists():
        raise FileNotFoundError(f"vllm config not found: {config_path}")
    text = config_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    try:
        cfg = json.loads(text)
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass

    def _extract(pattern: str) -> str | None:
        m = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
        return m.group(1).strip() if m else None

    base_url = _extract(r'DEFAULT_VLLM_BASE_URL\s*=\s*"([^"]+)"')
    model_name = _extract(r'DEFAULT_VLLM_MODEL\s*=\s*"([^"]+)"')
    api_key = _extract(r'DEFAULT_VLLM_API_KEY\s*=\s*"([^"]+)"')

    # 允许直接写成 base_url: "..." 形式
    if base_url is None:
        base_url = _extract(r'base_url\s*[:=]\s*"([^"]+)"')
    if model_name is None:
        model_name = _extract(r'model_name\s*[:=]\s*"([^"]+)"')
    if api_key is None:
        api_key = _extract(r'api_key\s*[:=]\s*"([^"]+)"')

    out: dict[str, Any] = {}
    if base_url is not None:
        out["base_url"] = base_url
    if model_name is not None:
        out["model_name"] = model_name
    if api_key is not None:
        out["api_key"] = api_key
    return out


def _make_llm_model_func(
    vllm_config_path: Path,
    llm_kwargs: dict[str, Any] | None = None,
):
    # vllm 模式：读取 config.json -> create_vllm_complete
    from vllm_llm import create_vllm_complete

    cfg = _load_vllm_config(vllm_config_path)
    merged_kwargs = dict(cfg)
    if llm_kwargs:
        merged_kwargs.update({k: v for k, v in llm_kwargs.items() if v is not None})

    base_url = merged_kwargs.pop("base_url", None)
    model_name = merged_kwargs.pop("model_name", None)
    api_key = merged_kwargs.pop("api_key", "EMPTY")

    if not base_url or not model_name:
        raise ValueError(
            "vllm config 缺少 base_url/model_name，请检查 vllm-config 或显式传 --vllm-base-url/--vllm-model-name"
        )

    vllm_complete = create_vllm_complete(base_url=base_url, model_name=model_name, api_key=api_key)

    # 若在 config 中显式提供这些采样参数，则透传；否则保持 None，走 openai.py / API 默认行为。
    default_temperature = merged_kwargs.pop("temperature", None)
    default_top_p = merged_kwargs.pop("top_p", None)
    default_max_tokens = merged_kwargs.pop("max_tokens", None)
    default_timeout = merged_kwargs.pop("timeout", None)
    default_top_k = merged_kwargs.pop("top_k", None)

    async def llm_complete_with_defaults(
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        if default_temperature is not None:
            kwargs.setdefault("temperature", default_temperature)
        if default_top_p is not None:
            kwargs.setdefault("top_p", default_top_p)
        if default_max_tokens is not None:
            # openai/vllm openai-compatible 使用 max_tokens
            kwargs.setdefault("max_tokens", default_max_tokens)
        if default_timeout is not None:
            kwargs.setdefault("timeout", default_timeout)
        if default_top_k is not None:
            kwargs.setdefault("top_k", default_top_k)

        # 允许在 config 里透传其他 vLLM openai-compatible 参数
        for k, v in merged_kwargs.items():
            kwargs.setdefault(k, v)
        return await vllm_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            **kwargs,
        )

    return llm_complete_with_defaults
