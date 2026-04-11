# LLM 分数计算说明（`evaluate_qa.py`）

本文档说明 `benchmark/longmemeval/src/evaluation/evaluate_qa.py` 中 LLM 自动评测分数（Accuracy）的计算方式。

## 1. 输入与目标

脚本执行方式：

```bash
python evaluate_qa.py metric_model hyp_file ref_file
```

- `metric_model`：用于评判答案是否正确的评测模型（如 `gpt-4o-mini`、`gpt-4o` 等）。
- `hyp_file`：被评测模型的输出，核心字段为：
  - `question_id`
  - `hypothesis`（模型回答）
- `ref_file`：参考答案数据，核心字段为：
  - `question_id`
  - `question`
  - `answer`
  - `question_type`

目标：让评测 LLM 对每条样本给出二分类判定（正确/错误），再统计准确率。

## 2. 单条样本如何打分

对每个 `question_id`：

1. 在参考集里找到对应的 `question`、`answer`、`question_type`。
2. 根据 `question_type` 选择不同的评测提示词模板（prompt）：
   - 常规任务：要求判断回答是否包含正确答案；
   - `temporal-reasoning`：额外允许日期/天数等 off-by-one 误差；
   - `knowledge-update`：只要更新后的答案正确即可判对；
   - `single-session-preference`：按个性化 rubric 判断是否满足；
   - 若 `question_id` 包含 `_abs`：走“不可回答问题（abstention）”模板。
3. 调用评测 LLM（`temperature=0`，`max_tokens=10`），要求只回答 `yes` 或 `no`。
4. 解析评测模型输出：
   - 若输出文本（转小写）中包含 `"yes"`，记为 `label=True`；
   - 否则记为 `label=False`。

对应的数值分：

- `label=True` -> `1`
- `label=False` -> `0`

## 3. 总分（Accuracy）如何计算

脚本最终输出的总分是所有样本 0/1 分数的平均值：

\[
\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\text{label}_i=\text{True})
\]

其中：
- \(N\)：被成功评测并写入日志的样本数；
- \(\mathbf{1}(\cdot)\)：指示函数，成立为 1，不成立为 0。

代码中等价于：
- 先把每条样本转成 `1 if label else 0`；
- 再用 `numpy.mean` 取平均；
- 最后四舍五入到 4 位小数打印。

## 4. 分类型分数（Per-question-type）

脚本还会按 `question_type` 分组统计：

\[
\text{Acc}_{t} = \frac{1}{N_t}\sum_{i \in t}\mathbf{1}(\text{label}_i=\text{True})
\]

- \(N_t\)：类型 \(t\) 下的样本数；
- 输出格式：`type_name: acc (count)`。

## 5. 结果文件

会生成：

- `result_file = hyp_file + ".eval-results-" + metric_model_short`

每条样本会追加 `autoeval_label` 字段，例如：

```json
{
  "autoeval_label": {
    "model": "gpt-4o-mini-2024-07-18",
    "label": true
  }
}
```

这份结果可用于复核单条样本判定。

