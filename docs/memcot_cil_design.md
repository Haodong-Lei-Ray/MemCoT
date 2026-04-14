# MemCoT CIL Design

## 1. 架构概述 (Architecture Overview)
采用 **C/S (Client-Server) 架构**，以保证 `MemCoT` 模型和 `RagRetriever` 等重型资源在后台常驻内存，避免每次执行命令都重新加载模型和配置。

- **Daemon (Server)**: 基于 `FastAPI` 构建的后台服务。启动时加载 `/home/lei/Project/MemCoT/config/rag/openclawnaiverag.json` 和 `memcot.json`，初始化 `MemCoT` 类并常驻内存。
- **CLI (Client)**: 命令行工具 `memcot_cil.py`。通过解析用户输入的命令（如 `session`, `add`, `search`），向后台 Daemon 发送 HTTP 请求，并将结果格式化输出到终端。

## 2. 核心命令与功能 (Core Commands)

| 命令 (Command) | 描述 (Description) | 对应后端操作 (Backend Action) |
| --- | --- | --- |
| `memcot help` | 显示帮助信息。 | 纯本地执行 (argparse help) |
| `memcot start` | 在后台启动 FastAPI 守护进程。 | 启动 `uvicorn`，保存 PID 到 `.memcot.pid` |
| `memcot stop` | 停止后台守护进程。 | 读取 PID 并发送 SIGTERM 信号终止进程 |
| `memcot status` | 查看后台服务运行状态。 | `GET /status` |
| `memcot session` | 获取完整的 session 列表。 | `GET /session` -> `memcot_instance.ragretriever.get_session_list()` |
| `memcot add --idx <N>` | 为指定的 session 执行数据清洗和 embedding。 | `POST /add` -> `memcot_instance.ragretriever.build_rag(idx)` |
| `memcot switch --idx <N>` | 根据 session 列表中的索引切换当前对话。 | `POST /switch` -> `memcot_instance.switch_session(idx)` |
| `memcot search -q "<Q>" -o "<dir>"` | 在当前加载的 session 中检索相关记忆并回答。 | `POST /search` -> `memcot_instance.run(query, output_dir)` -> `finalize_memcot_exit` |

## 3. 详细设计 (Detailed Design)

### 3.1 守护进程管理 (Daemon Management)
- **启动**: `start` 命令会使用 `subprocess.Popen` 启动 FastAPI 服务，并将输出重定向到 `memcot_daemon.log`。进程 PID 会写入 `.memcot.pid` 文件。
- **停止**: `stop` 命令读取 `.memcot.pid`，使用 `os.kill` 终止进程，并清理 PID 文件。

### 3.2 FastAPI 路由设计 (API Endpoints)
- `GET /status`: 返回 `{"status": "running"}`。
- `GET /session`: 调用 `memcot_instance.ragretriever.get_session_list()`，返回 JSON 格式的 session 列表。
- `POST /add`: 接收 JSON `{ "idx": int }`，调用 `memcot_instance.ragretriever.build_rag(idx)`，返回执行结果。
- `POST /search`: 接收 JSON `{ "query": str, "output_dir": str }`。调用 `memcot_instance.run(query, output_dir)`，然后通过 `finalize_memcot_exit` 保存轨迹，并返回最终回答。

### 3.3 依赖库 (Dependencies)
- `fastapi`, `uvicorn`: 构建和运行后台服务。
- `requests`: CLI 发送 HTTP 请求。
- `argparse`: 解析命令行参数。

## 4. 实施步骤 (Implementation Steps)
1. 编写此设计文档以明确边界。
2. 在 `memcot_cil.py` 中实现 FastAPI 的基础路由和全局变量初始化。
3. 在 `memcot_cil.py` 中实现 `argparse` 命令行解析器。
4. 联调测试 `start`, `stop`, `status` 基础命令。
5. 逐步实现 `session`, `add`, `search` 业务命令。
