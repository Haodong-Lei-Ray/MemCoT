<p align="center">
  <img src="asset/banner.png" alt="MemCoT Banner" width="100%">
</p>

# MemCoT 🦉: Test-Time Scaling through Memory-Driven Chain-of-Thought

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English Version"></a>
  <a href="https://arxiv.org/abs/2604.08216"><img src="https://img.shields.io/badge/Paper-arXiv%3A2604.08216-9cf?style=for-the-badge" alt="Paper"></a>
</p>

## 🚩 NEWS

- 2026-05-07: 完整的 code 完成了。
- 2026-04-09: Paper *MemCoT: Test-Time Scaling through Memory-Driven Chain-of-Thought* is published.

## 1. 环境安装

首先，克隆本仓库并进入项目目录：
```bash
git https://github.com/Haodong-Lei-Ray/MemCoT.git
cd MemCoT
```

推荐使用 Conda 创建虚拟环境：
```bash
conda create -n memcot python=3.10 -y
conda activate memcot
```

安装所需依赖（已包含 `fastapi` 等后台守护进程所需库）：
```bash
pip install -r re.txt
```

## 2. 命令行工具 (CLI)

MemCoT 提供了一个基于 C/S 架构的命令行工具 `memcot_cil.py`，支持在后台常驻运行 MemCoT 实例，并提供快速的检索问答服务。

支持的命令如下：
- `start`：在后台启动 MemCoT 守护进程 (Daemon)。日志将自动写入 `memcot_daemon.log`。
- `stop`：停止后台运行的守护进程。
- `status`：查看当前守护进程的运行状态。
- `session`：列出当前所有可用的会话 (Session) 及其 RAG 状态。
- `add --idx <N>`：为指定索引 `N` 的会话构建 RAG 索引 (Embedding) 并保存。
- `switch --idx <N>`：根据索引 `N` 切换当前会话 (Session)。
- `search -q "<query>" -o "<dir>"`：在已加载的会话中搜索指定问题，并生成回答，结果保存在指定目录。
- `logshow`：实时监控后台守护进程的日志输出（类似 `tail -f`，按 `Ctrl+C` 退出监控）。

## 3. 快速开始示例

以下是一个完整的python运行示例，展示了如何启动服务、构建索引、进行搜索并停止服务：

```bash
# 1. 启动后台守护进程
python memcot_cil.py start

# (可选) 实时查看日志
python memcot_cil.py logshow

# 2. 查看所有可用的会话列表
python memcot_cil.py session

# 3. 选择一个会话（例如 idx=0）构建 RAG 索引
python memcot_cil.py add --idx 0

# 4. 切换到构建好的会话
python memcot_cil.py switch --idx 0

# 5. 执行搜索查询
python memcot_cil.py search -q "hi" -o "./output"

# 6. 任务完成后，停止守护进程
python memcot_cil.py stop
```

## Openclaw等Agent请读这个

如果你是 OpenClaw 或其他 AI Agent，请务必阅读并遵循 `skills/openclaw-memcot-cli/SKILL.md` 中的指令。
它详细规定了你如何通过自然语言与用户交互，并在后台调用 `memcot_cil.py` 进行长文本记忆检索。

##  评测 (For Benchmark)

### LoCoMo

请先在脚本 `script/run_locomo_qwen_14b.sh` 中提前设置以下环境变量：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

然后直接运行：

```bash
bash /home/lei/Project/MemCoT/script/run_locomo_qwen_14b.sh
```

### LongMemEval

TODO

## Citation

If you use this codebase, or otherwise found our work valuable, please cite:

```bibtex
@misc{lei2026memcottesttimescalingmemorydriven,
      title={MemCoT: Test-Time Scaling through Memory-Driven Chain-of-Thought}, 
      author={Haodong Lei and Junming Liu and Yirong Chen and Ding Wang and Hongsong Wang},
      year={2026},
      eprint={2604.08216},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2604.08216}, 
}
```