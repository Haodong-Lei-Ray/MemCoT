<p align="center">
  <img src="asset/banner.png" alt="MemCoT Banner" width="100%">
</p>

# [MemCoT: Test-Time Scaling through Memory-Driven Chain-of-Thought](https://arxiv.org/abs/2604.08216) 🦉
[MemCoT: Test-Time Scaling through Memory-Driven Chain-of-Thought](https://arxiv.org/abs/2604.08216)

<img width="1861" height="705" alt="image" src="https://github.com/user-attachments/assets/1e0ec4c8-7b4e-4b14-acd5-92b376ace845" />


<p align="center">
  <a href="./README_zh.md"><img src="https://img.shields.io/badge/Lang-%E4%B8%AD%E6%96%87-red?style=for-the-badge" alt="Chinese Version"></a>
  <a href="https://arxiv.org/abs/2604.08216"><img src="https://img.shields.io/badge/Paper-arXiv%3A2604.08216-9cf?style=for-the-badge" alt="Paper"></a>
</p>

## 🚩 NEWS

- 2026-05-07: The full codebase is completed.
- 2026-04-09: The paper *MemCoT: Test-Time Scaling through Memory-Driven Chain-of-Thought* is published.

## For OpenClaw and Other Agents

If you are OpenClaw or another AI Agent, please read and follow the instructions in `clawhub/openclaw-memcot-cli/SKILL.md`.
It specifies how to interact with users in natural language and call `memcot_cil.py` in the background for long-context memory retrieval.

## 1. Environment Setup

First, clone this repository and enter the project directory:
```bash
git clone https://github.com/Haodong-Lei-Ray/MemCoT.git
cd MemCoT
```

It is recommended to create a virtual environment with Conda:
```bash
conda create -n memcot python=3.10 -y
conda activate memcot
```

Install required dependencies (including libraries needed for the daemon service such as `fastapi`):
```bash
pip install -r re.txt
```

## 2. Command-Line Interface (CLI)

MemCoT provides a C/S-style command-line tool, `memcot_cil.py`, which supports running a persistent MemCoT instance in the background and provides fast retrieval-based QA service.

Supported commands:
- `start`: Start the MemCoT daemon in the background. Logs are written to `memcot_daemon.log`.
- `stop`: Stop the running daemon process.
- `status`: Check current daemon status.
- `session`: List all available sessions and their RAG status.
- `add --idx <N>`: Build and save a RAG index (embedding) for session index `N`.
- `switch --idx <N>`: Switch to session index `N`.
- `search -q "<query>" -o "<dir>"`: Search the loaded session and generate an answer; outputs are saved to the target directory.
- `logshow`: Monitor daemon logs in real time (similar to `tail -f`, press `Ctrl+C` to exit).

## 3. Quick Start Example

Below is a complete Python workflow example that shows how to start the service, build index, run search, and stop the service:

```bash
# 1. Start daemon
python memcot_cil.py start

# (Optional) Monitor logs in real time
python memcot_cil.py logshow

# 2. List available sessions
python memcot_cil.py session

# 3. Build RAG index for one session (e.g., idx=0)
python memcot_cil.py add --idx 0

# 4. Switch to the indexed session
python memcot_cil.py switch --idx 0

# 5. Execute a search query
python memcot_cil.py search -q "hi" -o "./output"

# 6. Stop daemon after finishing
python memcot_cil.py stop
```

## Benchmark Evaluation

### LoCoMo

Please set the following environment variables in `script/run_locomo_qwen_14b.sh` in advance:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

Then run:

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
