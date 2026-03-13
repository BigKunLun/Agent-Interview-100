# 什么是 LLM Agent？与传统 LLM 应用有何区别？

> 难度：基础
> 分类：Agent 架构

## 简短回答

LLM Agent 是以大语言模型为核心推理引擎、能够自主感知环境、制定计划、调用工具并执行多步任务的智能系统。与传统 LLM 应用（单轮问答、文本生成）最大的区别在于：Agent 是有状态的、目标驱动的、能与外部世界交互的自主系统，而传统 LLM 应用是无状态的、被动响应式的文本处理器。

## 详细解析

### 传统 LLM 应用

传统 LLM 应用本质上是一个"高级文本处理器"。用户输入 Prompt，模型返回文本输出，整个交互是单轮、无状态的。典型场景包括：文案生成、文本摘要、翻译、代码补全等。

核心特征：
- **被动响应**：完全依赖用户输入，不会主动发起行为
- **无状态**：每次调用独立，不保留先前交互的记忆
- **纯文本**：输入输出都是文本，无法直接操作外部系统
- **单步完成**：一次调用生成最终结果，没有多步推理过程

### LLM Agent

LLM Agent 将 LLM 从"文本生成器"升级为"决策引擎"。Agent 接收一个高层目标，然后自主分解任务、选择工具、执行操作、根据反馈调整策略，直到目标完成。

核心特征：
- **自主性（Autonomy）**：定义目标后，Agent 自主决定"如何做"
- **有状态（Stateful）**：维护短期记忆（当前任务上下文）和长期记忆（跨会话知识）
- **工具使用（Tool Use）**：能调用 API、查询数据库、执行代码、操作文件系统
- **多步推理（Multi-step Reasoning）**：通过循环迭代完成复杂任务
- **自适应（Adaptive）**：根据中间结果和错误反馈动态调整策略

### 关键区别对照表

| 维度 | 传统 LLM 应用 | LLM Agent |
|------|-------------|-----------|
| 交互模式 | 单轮问答 | 多轮循环（Loop） |
| 状态管理 | 无状态 | 有状态（记忆系统） |
| 外部交互 | 仅文本 I/O | 工具调用、API、数据库 |
| 任务复杂度 | 单步任务 | 多步复合任务 |
| 自主程度 | 被动响应 | 目标驱动的自主执行 |
| 错误处理 | 无 | 自我纠正、重试、回退 |
| 可观测性 | Prompt → Response | Trace（思考→行动→观察循环） |

### 自主性光谱

AI 系统并非"传统 LLM"或"Agent"的二元分类，而是存在一个自主性光谱：

1. **Level 0 — 纯 LLM**：直接调用模型 API，无任何额外逻辑
2. **Level 1 — Chain/Pipeline**：多个 LLM 调用串联，但流程固定
3. **Level 2 — Router**：根据输入路由到不同的处理分支
4. **Level 3 — Tool-augmented LLM**：LLM 可以调用工具，但流程仍由人控制
5. **Level 4 — Autonomous Agent**：LLM 自主决策、规划、执行，人仅定义目标
6. **Level 5 — Multi-Agent System**：多个自主 Agent 协作完成复杂任务

大多数实际生产应用处于 Level 2-4 之间。

## 代码示例

```python
# 传统 LLM 应用：单轮文本生成
from anthropic import Anthropic

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "总结这篇文章"}]
)
print(response.content[0].text)

# LLM Agent：多步自主执行
# Agent 接收目标 → 规划 → 调用工具 → 观察结果 → 继续或结束
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "search_web",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
]

messages = [{"role": "user", "content": "调研 2026 年 AI Agent 市场规模并写一份报告"}]

# Agent Loop: 持续运行直到任务完成
while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        messages=messages
    )
    # 如果模型选择使用工具，执行工具并继续循环
    if response.stop_reason == "tool_use":
        # 执行工具调用，将结果反馈给模型
        tool_result = execute_tool(response)  # 实际执行工具
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_result})
    else:
        # 模型认为任务完成，退出循环
        print(response.content[0].text)
        break
```

## 常见误区 / 面试追问

1. **误区："有了 Function Calling 就是 Agent"** — Function Calling 只是 Agent 的一个能力组件。真正的 Agent 需要自主决策循环（Agent Loop），包括规划、执行、观察、反思的完整闭环。

2. **误区："Agent 比传统 LLM 应用总是更好"** — Agent 引入了额外的复杂性、延迟和成本。简单任务用传统 LLM 应用更高效。选择 Agent 的信号是：任务需要多步推理、外部交互、或动态决策。

3. **追问："Agent 调试比传统 LLM 应用难在哪里？"** — 传统 LLM 应用失败时只需检查 Prompt。Agent 失败可能源于规划逻辑错误、工具调用失败、记忆损坏、或无限循环，调试需要分析完整的执行 Trace。

4. **追问："如何决定一个任务是否需要 Agent？"** — 问自己：这个任务能用单次 LLM 调用完成吗？如果需要多步操作、外部数据获取、或根据中间结果做决策，就考虑用 Agent。

## 参考资料

- [LLMs vs AI Agents: What Is The Actual Difference (Medium)](https://medium.com/@speaktoharisudhan/llms-vs-ai-agents-what-is-the-actual-difference-cebd4cb789cd)
- [Understanding AI Agents vs LLMs: Key Differences Explained (EMA)](https://www.ema.ai/additional-blogs/addition-blogs/ai-agent-vs-llm-key-differences)
- [AI Agent vs LLM: Everything You Need to Know (Kanerika)](https://kanerika.com/blogs/ai-agents-vs-llm/)
- [What Are AI Agents? (IBM)](https://www.ibm.com/think/topics/ai-agents)
- [Agentic AI vs LLM: Comparing What Scales Better (Lyzr AI)](https://www.lyzr.ai/blog/agentic-ai-vs-llm/)
