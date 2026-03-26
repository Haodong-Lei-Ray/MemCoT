import httpx
from openai import OpenAI

# 内网 vLLM 不走代理，后续代码的全局代理不受影响
client = OpenAI(
    api_key="EMPTY",
    base_url="http://10.140.37.10:8100/v1",
    http_client=httpx.Client(trust_env=False),
)

resp = client.chat.completions.create(
    # model="Qwen3.5-27B",
    model="Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "system", "content": "请只输出最终答案，不要输出思考过程。"},
        {"role": "user", "content": "用一句话介绍你自己，不超过256个token"},
    ],
    temperature=0.0,
    max_tokens=256,
)

print(resp.choices[0].message.content)

#后面的代码要代理