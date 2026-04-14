# MemCoT: Test-Time Scaling through Memory-Driven Chain-of-Thought

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
- `start`：在后台启动 MemCoT 守护进程 (Daemon)。支持 `--log-file` 参数将日志输出到文件。
- `stop`：停止后台运行的守护进程。
- `status`：查看当前守护进程的运行状态。
- `session`：列出当前所有可用的会话 (Session) 及其 RAG 状态。
- `add --idx <N>`：为指定索引 `N` 的会话构建 RAG 索引 (Embedding) 并保存。
- `search -q "<query>" -o "<dir>"`：在已加载的会话中搜索指定问题，并生成回答，结果保存在指定目录。

## 3. 快速开始示例

以下是一个完整的运行示例，展示了如何启动服务、构建索引、进行搜索并停止服务：

```bash
# 1. 启动后台守护进程
python memcot_cil.py start

# 2. 查看所有可用的会话列表
python memcot_cil.py session

# 3. 选择一个会话（例如 idx=0）构建 RAG 索引
python memcot_cil.py add --idx 0

# 4. 执行搜索查询
python memcot_cil.py search -q "hi" -o "./output"

# 5. 任务完成后，停止守护进程
python memcot_cil.py stop
```

## 4. 评测 (For Benchmark)

TODO

## 5. 引用

如果您在研究中使用了本项目，请引用我们的论文：

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