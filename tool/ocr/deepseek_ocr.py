#!/usr/bin/env python3
"""
使用 OpenAI 兼容接口调用 deepseek-ai/DeepSeek-OCR 做 OCR。

环境变量（二选一即可）：
  - OPENAI_API_KEY：你的第三方 key
  - OPENAI_BASE_URL：第三方 OpenAI-compat base_url（例如 https://xxx/v1）

可选：
  - OCR_MODEL：默认模型名（默认 deepseek-ai/DeepSeek-OCR）

Usage:
  python deepseek_ocr.py -i /path/to/image.png
  python deepseek_ocr.py -i /path/to/image.png -o out.txt
  python deepseek_ocr.py -i /path/to/image.png -o out.json --format json
"""

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


DEFAULT_MODEL = "deepseek-ai/DeepSeek-OCR"


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _to_data_url(path: Path) -> str:
    mime = _guess_mime(path)
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _extract_text_from_response(resp: Any) -> str:
    # 兼容 responses / chat.completions 两种风格
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    # chat.completions
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek OCR via OpenAI-compatible API")
    parser.add_argument("-i", "--input", required=True, help="输入图片路径（png/jpg/webp 等）")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径（默认 stdout）")
    parser.add_argument("--model", default=None, help=f"模型名（默认 {DEFAULT_MODEL} 或 OCR_MODEL）")
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="输出格式：text=纯文本；json=包含 model/path/text/raw (default: text)",
    )
    parser.add_argument(
        "--prompt",
        default="请对图片做 OCR，输出识别到的全部文字。只输出文字内容，不要额外解释。",
        help="OCR 提示词",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    model = args.model or (Path().expanduser() and None)  # keep simple; overwritten below
    model = args.model or (  # env override
        __import__("os").environ.get("OCR_MODEL") or DEFAULT_MODEL
    )

    client = OpenAI()  # 使用 OPENAI_API_KEY / OPENAI_BASE_URL

    image_data_url = _to_data_url(in_path)

    # 优先走 responses API（openai>=1.0），失败则 fallback chat.completions
    text = ""
    raw: Dict[str, Any] = {}
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": args.prompt},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }
            ],
        )
        print(resp)
        text = _extract_text_from_response(resp)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
    except Exception as e1:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.prompt},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                ],
            )
        print(resp)
            text = _extract_text_from_response(resp)
            raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        except Exception as e2:
            raise RuntimeError(f"OCR call failed. responses_err={e1} chat_err={e2}") from e2

    if args.format == "text":
        out_str = text
    else:
        out_str = json.dumps(
            {
                "model": model,
                "input_path": str(in_path),
                "text": text,
                "raw": raw,
            },
            ensure_ascii=False,
            indent=2,
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_str, encoding="utf-8")
        print(str(out_path))
    else:
        print(out_str)


if __name__ == "__main__":
    main()

