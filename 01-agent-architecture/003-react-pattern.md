# ReAct 模式是什么？它如何结合推理与行动？

> 难度：基础
> 分类：Agent 架构

## 简短回答

ReAct（Reasoning + Acting）是由 Yao et al. (2022) 提出的 LLM Agent 框架，核心思想是让 LLM 交替进行推理（Thought）和行动（Action），并根据外部环境的反馈（Observation）动态调整下一步。与纯推理（Chain-of-Thought）不同，ReAct 通过工具调用获取真实信息，有效减少幻觉；与纯行动（直接调用工具）不同，ReAct 通过显式推理提升了可解释性和决策质量。

## 详细解析

### 核心循环：Thought → Action → Observation

ReAct 的运行机制是一个迭代循环：

1. **Thought（思考）**：LLM 分析当前状态，思考下一步应该做什么
2. **Action（行动）**：基于思考结果，选择并执行一个工具/操作
3. **Observation（观察）**：接收工具的执行结果作为新的上下文
4. 重复上述过程，直到 LLM 认为已有足够信息给出最终答案

```
Thought 1: 我需要查找某公司的最新财报数据
Action 1: search_web("Company X Q4 2025 earnings report")
Observation 1: Company X reported revenue of $5.2B in Q4 2025...

Thought 2: 现在我有了财报数据，需要计算同比增长率
Action 2: calculator("(5.2 - 4.8) / 4.8 * 100")
Observation 2: 8.33

Thought 3: 我现在有了足够信息来回答问题
Final Answer: Company X 的 Q4 2025 营收为 $5.2B，同比增长 8.33%。
```

### 为什么不用纯推理（CoT）？

Chain-of-Thought (CoT) 让 LLM 逐步推理，但完全依赖模型的内部知识。问题在于：

- **幻觉（Hallucination）**：模型可能"编造"看似合理但实际错误的事实
- **知识过时**：模型的训练数据有截止日期，无法获取最新信息
- **错误传播**：一步推理出错，后续步骤全部基于错误前提

ReAct 通过在推理过程中引入工具调用（Action），让模型能够从外部环境获取真实、最新的信息，形成"事实锚点"（Ground Truth Anchor），有效缓解这些问题。

### 为什么不用纯行动（直接工具调用）？

直接让 LLM 调用工具，跳过推理步骤，问题在于：

- **缺乏规划**：不知道"为什么"调用这个工具，调用顺序可能不合理
- **不可解释**：无法追踪决策逻辑
- **无法纠错**：没有反思机制，一旦选错工具就无法调整

ReAct 的 Thought 步骤提供了显式的推理 trace，使得决策过程透明、可调试、可审计。

### ReAct 的 Prompt 结构

一个典型的 ReAct Prompt 模板：

```
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation cycle can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

### Python 实现（简化版）

```python
import re
import anthropic

client = anthropic.Anthropic()

TOOLS = {
    "search": lambda q: web_search(q),
    "calculate": lambda expr: str(eval(expr)),  # ⚠️ 安全警告：生产环境不应使用 eval()，应使用安全的数学解析库（如 numexpr 或 asteval）
}

SYSTEM_PROMPT = """You are a helpful assistant. You can use these tools:
- search: Search the web. Input: search query string.
- calculate: Do math. Input: math expression.

Use this format:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input>

When you have the final answer:
Thought: I now know the final answer
Final Answer: <answer>
"""

def react_agent(question: str, max_steps: int = 10) -> str:
    """简化版 ReAct Agent"""
    prompt = f"Question: {question}\n"

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text
        prompt += text + "\n"

        # 检查是否有最终答案
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        # 解析并执行 Action
        action_match = re.search(r"Action: (\w+)\nAction Input: (.+)", text)
        if action_match:
            tool, tool_input = action_match.groups()
            observation = TOOLS[tool](tool_input.strip())
            prompt += f"Observation: {observation}\n"

    return "达到最大步数限制，未能得出结论。"
```

### ReAct 的实际效果

原始论文（Yao et al., 2022）的实验结果表明：
- 在 **HotpotQA**（多跳问答）上，ReAct 通过与 Wikipedia API 交互，显著超越纯 CoT，减少了幻觉和错误传播
- 在 **Fever**（事实验证）上，ReAct 的事实核查能力优于基线方法
- ReAct 生成的执行轨迹（Trajectory）更接近人类的问题解决方式，可解释性更强

### 主流框架中的 ReAct

ReAct 已成为 Agent 框架的默认模式：
- **LangChain/LangGraph**：`create_react_agent()` 直接创建 ReAct Agent
- **CrewAI**：Agent 默认使用 ReAct 范式交替推理和行动
- **Anthropic Claude**：通过 Tool Use API 天然支持 ReAct 模式（模型自动在思考和工具调用间交替）

## 常见误区 / 面试追问

1. **误区："ReAct 就是 Function Calling"** — Function Calling 是底层能力（让 LLM 生成结构化的工具调用请求），ReAct 是上层模式（在推理和行动之间交替的决策框架）。ReAct 可以基于 Function Calling 实现，但两者不是一回事。

2. **误区："ReAct 总是最优选择"** — ReAct 的每一步都需要完整的 LLM 调用（携带完整上下文），8 步任务可能消耗 50K-100K tokens。对于结构确定的任务，Plan-and-Execute 更高效。

3. **追问："ReAct 的主要缺陷是什么？"** — (1) 高 token 消耗和延迟；(2) 可能陷入推理循环（反复调用同一工具）；(3) 无法并行执行多个独立操作，因为每步都是顺序的。

4. **追问："如何改进 ReAct？"** — (1) 加入 Reflexion 机制让 Agent 从失败中学习；(2) 混合 Plan-and-Execute，先生成粗略计划再 ReAct 执行；(3) 设置最大步数和重复检测来防止死循环。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629)
- [ReAct Prompting (Prompt Engineering Guide)](https://www.promptingguide.ai/techniques/react)
- [What is a ReAct Agent? (IBM)](https://www.ibm.com/think/topics/react-agent)
- [A Simple Python Implementation of the ReAct Pattern (Simon Willison)](https://til.simonwillison.net/llms/python-react-pattern)
- [ReAct Pattern: Interleaving Reasoning and Action (Michael Brenndoerfer)](https://mbrenndoerfer.com/writing/react-pattern-llm-reasoning-action-agents)
