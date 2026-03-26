#!/usr/bin/env python3
"""测试 vllm_llm.create_vllm_complete 是否能正常调用内网 vLLM。

用法:
    cd /mnt/petrelfs/leihaodong/ICML/locomo
    python benchmark/LongMemEval/tool/UT/test_qwen_vllm.py

    # 指定 vLLM 地址
    python benchmark/LongMemEval/tool/UT/test_qwen_vllm.py --url http://10.140.37.10:8100/v1
"""

import asyncio
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
BUILDRAG_DIR = PROJECT_ROOT / "benchmark" / "LongMemEval" / "tool" / "buildrag"
sys.path.insert(0, str(BUILDRAG_DIR))

from vllm_llm import create_vllm_complete


async def test_basic_completion(vllm_complete):
    """基本对话测试"""
    resp = await vllm_complete(
        prompt="用一句话介绍你自己，不超过50个token",
        system_prompt="请只输出最终答案，不要输出思考过程。",
    )
    assert resp and len(resp.strip()) > 0, "返回为空"
    print(f"[PASS] 基本对话: {resp.strip()[:100]}")


async def test_entity_extraction(vllm_complete):
    """模拟 LightRAG 实体抽取场景"""
    prompt = (
        "Extract entities and relationships from the following text.\n\n"
        'Text: user said, "I went to Tokyo last week and visited the Senso-ji temple '
        'with my friend Alice. We had amazing ramen at Ichiran."\n\n'
        "Entities and relationships:"
    )
    resp = await vllm_complete(prompt=prompt)
    assert resp and len(resp.strip()) > 0, "实体抽取返回为空"
    print(f"[PASS] 实体抽取: {resp.strip()[:200]}")


async def test_no_proxy(vllm_complete):
    """确认 trust_env=False 生效（即使环境变量有代理也不报错）"""
    import os
    old = os.environ.get("http_proxy")
    os.environ["http_proxy"] = "http://invalid-proxy:9999"
    try:
        resp = await vllm_complete(prompt="Say hello", system_prompt="Be brief.")
        assert resp and len(resp.strip()) > 0
        print(f"[PASS] 不走代理: 即使设置了无效 proxy 也能成功调用")
    finally:
        if old is not None:
            os.environ["http_proxy"] = old
        else:
            os.environ.pop("http_proxy", None)


async def main(url, model):
    vllm_complete = create_vllm_complete(base_url=url, model_name=model)
    print(f"vLLM: {url}  model: {model}\n")

    await test_basic_completion(vllm_complete)
    await test_entity_extraction(vllm_complete)
    await test_no_proxy(vllm_complete)

    print("\n全部通过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://10.140.37.10:8100/v1")
    parser.add_argument("--model", default="Qwen3-30B-A3B-Instruct-2507")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.model))
