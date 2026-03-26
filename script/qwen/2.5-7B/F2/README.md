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
