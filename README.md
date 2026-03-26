# F2 Qwen2.5-7B 评估脚本说明

本目录下 3 个脚本已改为**基于脚本相对路径定位项目根目录**，不再依赖历史绝对路径（如 `ICML/locomo`）。

## 脚本关系

- `command.sh`: 批量提交多个 `conv-*` 任务
- `sbatch_eval_react_lightrag.sh`: 包装器，创建输出目录并提交 sbatch
- `eval_react_lightrag_conv.sh`: 真正执行 `srun python3 eval_react_lightrag.py`

## 当前路径约定

- 项目根目录自动推断为：`SCRIPT_DIR/../../../../..`
- 评估脚本：`${PROJECT_ROOT}/eval_react_lightrag.py`
- 输出目录：`${PROJECT_ROOT}/eval_output/Qwen/2.5-7B/F2/<conv-id>/`
- LightRAG 索引路径：默认 `${PROJECT_ROOT}/tool/rag_storage`

> 若你的 LightRAG 索引不在默认位置，可通过环境变量覆盖：
>
> `export LIGHTRAG_BASE=/your/path/to/rag_storage`

## 使用方式

在本目录执行：

```bash
bash command.sh
```

或单个任务：

```bash
bash sbatch_eval_react_lightrag.sh conv-30 10
```

## 日志位置

每个 `conv` 的日志在：

- `${PROJECT_ROOT}/eval_output/Qwen/2.5-7B/F2/<conv-id>/slurm_<jobid>.out`
- `${PROJECT_ROOT}/eval_output/Qwen/2.5-7B/F2/<conv-id>/slurm_<jobid>.err`

## 运行 debug 脚本所需外部数据（`ICML/locomo/data`）

针对脚本：

- `/mnt/petrelfs/leihaodong/DMSMem/script/qwen/2.5-7B/debug/eval_react_lightrag_conv.sh`

并结合代码检查：

- `/mnt/petrelfs/leihaodong/DMSMem/eval_react_lightrag.py`
- `/mnt/petrelfs/leihaodong/DMSMem/react_allrag.py`

运行时依赖的 `data` 目录文件（位于 `/mnt/petrelfs/leihaodong/ICML/locomo/data`）至少包括：

1. `locomo10.json`
   - 用途：评测入口读取样本与 QA（`DATA_PATH = PROJECT_ROOT / "data" / "locomo10.json"`）。
2. `skip/<conv-id>.json`（例如 `skip/conv-26.json`、`skip/conv-30.json`）
   - 用途：`eval_react_lightrag.py` 中硬编码跳过清单路径：
     `/mnt/petrelfs/leihaodong/ICML/locomo/data/skip/<sample_id>.json`。

> 说明：当前 `DMSMem` 根目录下默认没有 `data/`，因此如果直接跑 `DMSMem` 版本评测，通常需要把
> `ICML/locomo/data` 挂到 `DMSMem/data`（软链）或复制对应文件。

推荐软链方式（一次即可）：

```bash
ln -s /mnt/petrelfs/leihaodong/ICML/locomo/data /mnt/petrelfs/leihaodong/DMSMem/data
```

运行前快速自检：

```bash
test -f /mnt/petrelfs/leihaodong/ICML/locomo/data/locomo10.json && echo "locomo10.json OK"
test -f /mnt/petrelfs/leihaodong/ICML/locomo/data/skip/conv-26.json && echo "skip/conv-26.json OK"
```

## `agent-flag` 与视觉依赖目录说明

`agent-flag` 为 5 位开关，顺序为：

`rag_view / middle_view / full_view / agentic_graph / visual_search`

### 重点：`agent-flag=10100`

- 第 3 位是 `1`：会启用 **视觉 OCR agent**（读取会话图片进行 OCR）
- 第 5 位是 `0`：**不会**启用视觉检索索引（`img_retriever` 不会初始化）

因此，`10100` 的视觉相关依赖是：

1. 需要图片目录（OCR 输入）
   - `${PROJECT_ROOT}/data/img_pdf_minor/<conv-id>/D*/`
   - 例如：`/mnt/petrelfs/leihaodong/DMSMem/data/img_pdf_minor/conv-26/D1/*.png`

2. 不需要图像索引目录（CLIP 检索）
   - 不要求 `${img_index_base}/<conv-id>`（默认 `img_mem/vit_rag/<conv-id>`）
   - 只有当第 5 位为 `1`（如 `10101`）才需要
