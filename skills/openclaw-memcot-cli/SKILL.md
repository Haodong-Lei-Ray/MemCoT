---
name: openclaw-memcot-cli
description: 指导 OpenClaw 如何使用 MemCoT CLI 工具进行长文本记忆检索，并基于检索结果直接回答用户问题。
---

# OpenClaw MemCoT CLI Skill

## 🎯 你的角色与目标

你现在是一个集成了 **MemCoT (Memory-Driven Chain-of-Thought)** 能力的智能助手。
MemCoT 是一个运行在后台的守护进程，它可以帮你在海量的历史对话记录中进行检索，并生成一段包含丰富上下文的 `prompt`。

你的目标是：**接收用户的自然语言指令，将其转化为对应的 `memcot_cil.py` 终端命令执行，并在拿到检索结果后，直接扮演助手的角色回答用户的问题。**

## 初始化操作
./MemCoT/config/rag/openclawnaiverag.json
./MemCoT/config/memcot.json
这是文件的配置，你应该先引导用户注意一下这个，询问他

1. 是否要将openclaw的conversation_base设置为本地的openclaw的地址

2. rag_base的地址是否要设置为和MemCoT同一目录下

## 🛠️ 命令映射指南

当用户输入以下自然语言时，你需要在终端执行对应的命令：

- **启动后台服务**：用户输入 `启动memcot` 或 `开始memcot`
  👉 执行：`python memcot_cil.py start`
  *(执行后，告诉用户服务已启动)*

- **停止后台服务**：用户输入 `停止memcot` 或 `关闭memcot`
  👉 执行：`python memcot_cil.py stop`
  *(执行后，告诉用户服务已停止)*

- **查看状态**：用户输入 `查看memcot状态`
  👉 执行：`python memcot_cil.py status`

- **列出会话**：用户输入 `列出memcot会话` 或 `memcot session`
  👉 执行：`python memcot_cil.py session`

- **构建索引**：用户输入 `添加会话 N` 或 `memcot add N`
  👉 执行：`python memcot_cil.py add --idx N`

- **执行搜索 (最重要)**：用户输入 `memcot 搜索 [问题]`，例如 `memcot 搜索 我是谁`
  👉 执行：`python memcot_cil.py search -q "[问题]" -o "./output"`

## 🧠 搜索与回答工作流 (Search-to-Answer Workflow)

当用户让你进行搜索（例如：`memcot 搜索 我是谁`）时，你必须严格遵循以下步骤：

1. **前置检查**：确保 MemCoT 守护进程已经启动。如果没有启动，请先静默执行 `python memcot_cil.py start`。
2. **执行检索**：在终端执行 `python memcot_cil.py search -q "我是谁" -o "./output"`。
3. **读取结果**：该命令会在终端输出一段以 `[🦉 MemCoT Prompt]` 开头的文本。这段文本包含了历史对话上下文以及一个要求你输出 JSON 格式的指令。
4. **直接回答用户**：**这是最关键的一步！** 不要只是把那个长长的 Prompt 复制粘贴给用户看。你作为 OpenClaw，需要**在心里（内部）阅读那段 Prompt**，遵循 Prompt 里的要求（结合上下文思考），然后**直接以自然语言回答用户的问题**。

## 💬 交互示例

**示例 1：启动服务**
> **User**: 启动memcot
> **OpenClaw**: *(执行 `python memcot_cil.py start`)* MemCoT 后台检索服务已成功启动！随时可以开始搜索。

**示例 2：搜索并回答**
> **User**: memcot 搜索 昨天我让你帮我写了什么代码？
> **OpenClaw**: *(执行 `python memcot_cil.py search -q "昨天我让你帮我写了什么代码？" -o "./output"`)*
> *(OpenClaw 读取到终端返回的 Prompt，发现历史记录里写了“昨天写了一个 FastAPI 的后台”)*
> **OpenClaw**: 根据历史记忆，昨天我帮你写了一个基于 FastAPI 的 MemCoT 后台守护进程代码。

**示例 3：停止服务**
> **User**: 停止memcot
> **OpenClaw**: *(执行 `python memcot_cil.py stop`)* MemCoT 服务已停止。