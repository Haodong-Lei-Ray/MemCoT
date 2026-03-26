# eval_react_lightrag 并发改造方案

## 1. 现状问题

### 1.1 当前调用链

```
eval_react_lightrag.py::main()
  ├── create_lightrag(working_dir)          # 创建唯一 rag 实例
  └── for qa in qa_list:                    # 串行遍历每个 QA
        └── _process_one_qa(i, qa)
              └── run_react_lightrag(...)    # react_allrag.py
                    ├── loop = _get_rag_event_loop()
                    ├── rag_retrieve_multi(...)
                    │     └── loop.run_until_complete(_run_all())
                    │           └── await rag.aquery_data(query, param)   # async
                    ├── rag_view_agent(...)      → run_chatgpt()          # sync HTTP
                    ├── middle_view_agent(...)   → run_chatgpt()          # sync HTTP
                    ├── observation_agent(...)   → run_chatgpt()          # sync HTTP
                    └── answer_agent(...)        → run_chatgpt()          # sync HTTP
```

每个 QA 走 ReAct 循环（最多 max_step=8 步），每步涉及：
- **1 次 LightRAG 检索**（async，通过 `loop.run_until_complete` 桥接）
- **3~5 次 LLM API 调用**（`run_chatgpt`，同步 HTTP 请求）

整个流水线的瓶颈是 **I/O 等待**（LLM API 响应），CPU 几乎不忙。

### 1.2 当前并发版本的问题

上一版用 `ThreadPoolExecutor` 实现并发，存在三个竞争风险：

| 问题 | 原因 | 后果 |
|------|------|------|
| **asyncio event loop 线程不安全** | `_get_rag_event_loop()` 在多个线程中被调用，`asyncio.get_event_loop()` 在子线程中行为不可预测；多个线程同时 `loop.run_until_complete()` 同一个 loop 会崩溃 | RuntimeError / 不确定行为 |
| **LightRAG 内部 asyncio.Lock** | `pipeline_status_lock` 是 `asyncio.Lock`，只在同一个 event loop 内生效，跨线程无效 | 共享状态竞争 |
| **`_shared_dicts` 全局字典** | `shared_storage.py` 的进程级全局字典无 `threading.Lock` 保护 | 并发读写 dict 可能 corrupt |

## 2. 方案选择

### 方案 A：ThreadPoolExecutor + 每线程独立 event loop

**思路**：保持多线程，但每个线程内 `asyncio.new_event_loop()` 创建独立 loop。

- 优点：改动小
- 缺点：
  - `rag` 实例共享 → `_shared_dicts` 仍有竞争
  - 要么共享 rag（有风险），要么每线程创建 rag（内存 ×N，初始化 ×N）
  - `run_chatgpt` 是同步阻塞调用，线程数 = 真实并发数，20 线程占 20 个 OS 线程

### 方案 B：asyncio 原生并发（推荐）

**思路**：把整个 `_process_one_qa` 改为 `async def`，用 `asyncio.Semaphore` 控制并发度，所有 QA 在同一个 event loop 里 `asyncio.gather` 并发。

- 优点：
  - 只需 **1 个 rag 实例**，内存不增加
  - 只需 **1 个 event loop**，`asyncio.Lock` 天然生效
  - `rag.aquery_data()` 本身就是 async，无需桥接
  - 协程切换开销远小于线程
- 缺点：
  - `run_chatgpt()`（agent 的 LLM 调用）目前是同步函数，需要包一层 `asyncio.to_thread()` 或改成 async
  - 改动量稍大（但都是机械替换）

### 结论：选方案 B

理由：整条链路的瓶颈全是 I/O（LLM API），asyncio 是 Python 处理 I/O 并发的标准方式，且能彻底避免线程安全问题。

## 3. 具体改动清单

### 3.1 `react_allrag.py`

#### 改动 1：删除 `_get_rag_event_loop` 桥接

**文件**：`react_allrag.py`，第 126-139 行

**原因**：async 并发后，所有 async 函数都在同一个 event loop 里直接 `await`，不再需要 `loop.run_until_complete()` 桥接。

**做法**：
- `_get_rag_event_loop()` 函数保留（串行模式仍需要），但并发模式下不再调用
- `lightrag_retrieve_multi` 新增 async 版本 `lightrag_retrieve_multi_async`
- `run_react_lightrag` 新增 async 版本 `run_react_lightrag_async`

#### 改动 2：`lightrag_retrieve_multi` → async 版

**文件**：`react_allrag.py`，第 240-271 行

**当前**：
```python
def lightrag_retrieve_multi(queries, conv_id, top_k, working_dir, rag):
    loop = _get_rag_event_loop()
    async def _run_all():
        for q in queries:
            res = await _lightrag_retrieve_async(q, ...)
    results, result_list = loop.run_until_complete(_run_all())
```

**改为**：
```python
async def lightrag_retrieve_multi_async(queries, conv_id, top_k, working_dir, rag):
    all_results = []
    result_list = []
    for q in queries:
        res = await _lightrag_retrieve_async(q, ...)
        ...
    return deduped, result_list
```

原同步版保留不删（串行模式 / CLI 入口仍需要）。

#### 改动 3：Agent 调用包装为 async

**文件**：`react_allrag.py`，第 436-577 行（`run_react_lightrag` 内部）

**当前**：各 agent 调用都是同步的：
```python
rag_view_result = rag_view_agent(...)       # 内部调 run_chatgpt() → sync HTTP
middle_view_result = middle_view_agent(...)
obs = observation_agent(...)
ans = answer_agent(...)
```

**做法**：在 `run_react_lightrag_async` 中，用 `asyncio.to_thread()` 包装这些同步 agent 调用：
```python
rag_view_result = await asyncio.to_thread(rag_view_agent, ...)
middle_view_result = await asyncio.to_thread(middle_view_agent, ...)
obs = await asyncio.to_thread(observation_agent, ...)
ans = await asyncio.to_thread(answer_agent, ...)
```

`asyncio.to_thread` 会把同步函数扔到线程池执行，但 event loop 本身不阻塞，其他协程可以继续运行。**不需要改 agent 内部代码**。

#### 改动 4：新增 `run_react_lightrag_async`

**文件**：`react_allrag.py`

**做法**：复制 `run_react_lightrag`，改名为 `run_react_lightrag_async`：
- 函数签名改为 `async def`
- 内部 `rag_retrieve_multi` → `await lightrag_retrieve_multi_async`
- 内部各 agent 调用 → `await asyncio.to_thread(agent_func, ...)`
- 删除 `loop = _get_rag_event_loop()` 那一行

原 `run_react_lightrag` 同步版保留不删。

### 3.2 `eval_react_lightrag.py`

#### 改动 5：`_process_one_qa` → `async def`

**文件**：`eval_react_lightrag.py`，第 235-313 行

**当前**：
```python
def _process_one_qa(i, qa):
    ...
    res = run_react_lightrag(...)
    ...
```

**改为**：
```python
async def _process_one_qa(i, qa):
    ...
    res = await run_react_lightrag_async(...)
    ...
```

其余逻辑（F1 计算、recall 计算）不变。

#### 改动 6：并发执行改为 `asyncio.gather` + `Semaphore`

**文件**：`eval_react_lightrag.py`，第 328-347 行

**当前**：
```python
with ThreadPoolExecutor(max_workers=concurrency) as pool:
    futures = {pool.submit(_process_one_qa, i, qa): i ...}
    for fut in as_completed(futures):
        ...
```

**改为**：
```python
sem = asyncio.Semaphore(concurrency)

async def _bounded_process(i, qa):
    async with sem:
        return await _process_one_qa(i, qa)

loop = asyncio.get_event_loop()  # 或 asyncio.run()
tasks = [_bounded_process(i, qa) for i, qa in pending]
results_list = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

for idx, result in zip([i for i, _ in pending], results_list):
    if isinstance(result, Exception):
        print(f"  [QA {idx+1}] FATAL: {result}")
    else:
        _collect_result(result)
```

`Semaphore(concurrency)` 控制同时运行的协程数量，效果等同于 `ThreadPoolExecutor(max_workers=concurrency)`。

#### 改动 7：删除 `threading` 相关代码

**文件**：`eval_react_lightrag.py`

- 删除 `import threading`
- 删除 `from concurrent.futures import ThreadPoolExecutor, as_completed`
- `_results_lock = threading.Lock()` 不再需要（单线程 event loop 内不存在竞争）
- `_collect_result` 内的 `with _results_lock:` 去掉（改为普通函数调用）

### 3.3 不需要改的文件

| 文件 | 原因 |
|------|------|
| `agent/agent.py` | 所有 agent 函数保持同步，由 `asyncio.to_thread()` 包装 |
| `agent/prompt.py` | 纯模板，无需改 |
| `global_methods.py` (`run_chatgpt`) | 同步 HTTP 调用，由 `asyncio.to_thread()` 自动扔到线程池 |
| `lightrag/lightrag.py` | LightRAG 本身不改，`aquery_data` 已经是 async |
| shell 脚本 | `--concurrency` 参数不变 |

## 4. 改动量评估

| 文件 | 改动类型 | 预估行数 |
|------|----------|----------|
| `react_allrag.py` | 新增 `lightrag_retrieve_multi_async` | +20 行 |
| `react_allrag.py` | 新增 `run_react_lightrag_async`（从同步版复制改造） | +250 行（大部分复制） |
| `eval_react_lightrag.py` | `_process_one_qa` 改 async + 执行逻辑替换 | 改 ~30 行 |
| 合计 | | ~300 行 |

## 5. 向后兼容

- `--concurrency 1`（默认）时走串行路径，调用原同步函数，行为与改造前完全一致
- `--concurrency >1` 时走 async 并发路径
- CLI 入口 `react_allrag.py __main__` 保持调用同步版本
- 所有 agent 函数无需改动

## 6. 预期效果

假设每个 QA 走 5 步 ReAct，每步 1 次 RAG + 4 次 LLM API 调用，每次 API 耗时 ~3 秒：
- 串行：5 × 5 × 3s = 75s/QA
- 并发 20：20 个 QA 同时跑，总耗时 ≈ 75s（而非 75s × 20 = 25 分钟）
- 瓶颈转移到 LLM 服务端的并发承载能力
