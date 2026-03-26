## DeepSeek OCR (OpenAI-compat)

`deepseek_ocr.py` 使用 OpenAI Python SDK，通过 **OpenAI 兼容**的第三方接口调用 `deepseek-ai/DeepSeek-OCR` 做 OCR。

### 环境变量

- **`OPENAI_API_KEY`**：第三方 API key
- **`OPENAI_BASE_URL`**：第三方 OpenAI-compat base_url（通常以 `/v1` 结尾）

可选：
- **`OCR_MODEL`**：默认模型名（默认 `deepseek-ai/DeepSeek-OCR`）

### 用法

```bash
# 输出到 stdout
python deepseek_ocr.py -i /path/to/image.png

# 输出到文件
python deepseek_ocr.py -i /path/to/image.png -o out.txt

# 输出 json（包含 raw）
python deepseek_ocr.py -i /path/to/image.png -o out.json --format json
```

