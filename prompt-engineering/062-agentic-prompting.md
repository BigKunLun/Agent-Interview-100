# Agentic Prompting：如何编写让 LLM 自主执行任务的 Prompt？

> 难度：中级
> 分类：Prompt Engineering

## 简短回答

Agentic Prompting 是为 LLM Agent 设计的专用 Prompt 工程——不同于标准 Prompt（一问一答），Agentic Prompt 需要让 LLM 理解它拥有哪些工具、如何规划多步任务、何时停止执行、以及如何处理异常。核心要素包括：(1) **角色与目标定义**——明确 Agent 的身份和任务边界；(2) **工具描述**——精确描述每个工具的功能、参数和使用场景；(3) **推理格式**——定义 Thought/Action/Observation 的交互模式；(4) **约束与护栏**——设置最大步数、禁止行为、降级策略；(5) **输出规范**——定义最终输出的格式。与标准 Prompt 的关键区别：标准 Prompt 优化单次输出质量，Agentic Prompt 优化多步决策链的整体质量。

## 详细解析

### 标准 Prompt vs Agentic Prompt

```python
# 标准 Prompt：一问一答
standard_prompt = """
请分析以下代码的 bug：
{code}
"""

# Agentic Prompt：赋予自主执行能力
agentic_prompt = """
你是一个代码调试 Agent。你的目标是找到并修复代码中的 bug。

## 可用工具
- read_file(path): 读取源文件
- run_tests(path): 运行测试套件
- edit_file(path, old_text, new_text): 修改文件
- search_code(pattern): 搜索代码库

## 工作流程
1. 先阅读错误信息和相关代码
2. 分析可能的原因
3. 修改代码
4. 运行测试验证修复
5. 如果测试仍然失败，回到步骤 1

## 推理格式
Thought: [你的分析和计划]
Action: [工具名称]
Action Input: [工具参数]
Observation: [工具返回结果]
... (重复直到问题解决)
Final Answer: [修复总结]

## 约束
- 最多执行 15 步
- 不要修改测试文件
- 每次只修改一处，验证后再继续
"""
```

### Agentic Prompt 的核心组件

```python
agentic_prompt_template = """
# 1. 角色定义（WHO）
你是 {agent_name}，一个专注于 {domain} 的 AI Agent。
你的核心能力是 {capabilities}。

# 2. 目标定义（WHAT）
你的任务是：{task_description}
成功标准：{success_criteria}

# 3. 工具描述（WITH WHAT）
你可以使用以下工具：

### {tool_name_1}
- 功能：{description}
- 参数：{parameters}
- 返回：{return_type}
- 使用场景：{when_to_use}
- 注意事项：{caveats}

### {tool_name_2}
...

# 4. 推理与行动格式（HOW）
请按以下格式思考和行动：

Thought: 分析当前状况，决定下一步
Action: 选择一个工具
Action Input: 提供工具参数（JSON 格式）
Observation: 观察工具返回的结果

# 5. 约束与护栏（BOUNDARIES）
- 最多执行 {max_steps} 步
- 不确定时问用户而非猜测
- 涉及 {dangerous_actions} 时必须确认
- 如果连续 3 次失败，报告问题并停止

# 6. 输出规范（OUTPUT）
完成后使用：
Final Answer: {output_format}
"""
```

### 工具描述的最佳实践

```python
# ❌ 差的工具描述
bad_tool_description = {
    "name": "search",
    "description": "搜索功能"
}

# ✓ 好的工具描述
good_tool_description = {
    "name": "web_search",
    "description": (
        "搜索互联网获取最新信息。适用于需要实时数据、"
        "最新新闻或模型训练数据中不包含的信息的场景。"
        "不适合搜索代码库内部信息（请用 search_code）。"
    ),
    "parameters": {
        "query": {
            "type": "string",
            "description": "搜索关键词，建议使用英文以获得更多结果"
        },
        "max_results": {
            "type": "integer",
            "description": "返回结果数量，默认 5",
            "default": 5
        }
    },
    "examples": [
        {"query": "Python FastAPI deployment best practices", "max_results": 3},
        {"query": "React 19 new features 2024", "max_results": 5}
    ]
}

# 工具描述的质量直接影响 Agent 的工具选择准确率
# 研究表明：好的工具描述可以提升工具选择准确率 20-30%
```

### ReAct 模式的 Prompt 设计

```python
react_prompt = """
请按照 ReAct 模式解决问题。

问题：{question}

你可以使用以下工具：
{tool_descriptions}

按以下格式回答：

Thought: 我需要思考下一步做什么。首先分析问题，然后决定用什么工具。
Action: tool_name
Action Input: {"param1": "value1"}

等待工具返回结果后继续：

Observation: [工具返回的结果]
Thought: 根据结果，我需要...
Action: ...

当你有足够信息时：
Thought: 我现在有了足够的信息来回答问题。
Final Answer: [最终答案]

重要规则：
- 每次只执行一个 Action
- 在 Thought 中解释你的推理过程
- 如果工具调用失败，在 Thought 中分析原因并尝试替代方案
- 不要编造工具不存在的返回结果
"""
```

### 高级 Agentic Prompt 技巧

```python
advanced_techniques = {
    "角色强化": {
        "技巧": "给 Agent 一个具体的专家身份而非通用助手",
        "示例": "你是一个有 10 年经验的 SRE 工程师"
                " vs 你是一个有帮助的助手",
        "效果": "更专业的判断和更谨慎的行动",
    },
    "思维链引导": {
        "技巧": "在 Thought 格式中引导分析结构",
        "示例": """
Thought:
  当前状态：[描述已知信息]
  目标差距：[还缺什么]
  下一步计划：[选择的行动及理由]
  风险评估：[可能的问题]
""",
    },
    "失败处理指令": {
        "技巧": "明确告诉 Agent 失败时如何应对",
        "示例": """
如果工具调用失败：
1. 分析错误原因
2. 尝试修改参数重试（最多 2 次）
3. 如果仍然失败，尝试替代工具
4. 如果无替代方案，向用户报告
""",
    },
    "自我检查": {
        "技巧": "在输出前要求 Agent 验证自己的结果",
        "示例": "在给出 Final Answer 之前，检查：\n"
                "1. 答案是否完整回答了用户的问题？\n"
                "2. 引用的数据是否来自工具返回的真实结果？\n"
                "3. 是否有遗漏的重要信息？",
    },
}
```

### 多 Agent 的 Prompt 设计

```python
# Supervisor Agent 的 Prompt
supervisor_prompt = """
你是任务调度 Agent。你的职责是将用户任务分配给合适的专家 Agent。

可用的专家 Agent：
- researcher: 擅长信息搜索和数据收集
- analyst: 擅长数据分析和可视化
- writer: 擅长报告撰写和内容创作

你的工作流程：
1. 分析用户任务
2. 将任务分解为子任务
3. 将每个子任务分配给最合适的专家
4. 收集结果并汇总

使用 delegate(agent_name, task) 工具来分配任务。
"""

# Worker Agent 的 Prompt
worker_prompt = """
你是 {role_name}，接受调度 Agent 的任务指派。
你只负责 {specialization}，其他类型的任务请回复"超出能力范围"。

收到任务后：
1. 确认任务在你的能力范围内
2. 执行任务
3. 返回结构化结果
"""
```

## 常见误区 / 面试追问

1. **误区："Agentic Prompt 越详细越好"** — 过度详细的指令可能限制 Agent 的灵活性。好的 Agentic Prompt 应该明确"做什么"和"边界在哪"，但在"怎么做"上给予适当自由度。尤其对强模型，过多的微观指令反而降低效果。

2. **误区："标准 Prompt 技巧直接适用于 Agent"** — Agent Prompt 有独特挑战：需要处理多轮交互、工具调用结果的不确定性、以及长上下文中的指令遗忘。标准 Prompt 优化的是单次生成，Agent Prompt 优化的是整个决策链。

3. **追问："如何减少 Agent 的'工具滥用'？"** — 在 Prompt 中明确工具使用条件："只在需要外部信息时使用搜索工具，能通过推理得到的答案不要搜索"。同时用 negative example 展示不该使用工具的场景。

4. **追问："Agentic Prompt 在不同模型间可移植吗？"** — 不太可移植。不同模型对推理格式、工具调用方式的偏好不同。GPT-4 偏好 JSON 格式的 Action Input，Claude 偏好 XML 标签。最佳实践是针对每个模型调优 Prompt。

## 参考资料

- [Agent Prompts vs Standard LLM Prompts (APXML)](https://apxml.com/courses/prompt-engineering-agentic-workflows/chapter-1-foundations-agentic-ai-systems/contrasting-agent-standard-prompts)
- [Agentic Prompt Engineering: A Deep Dive into LLM Roles (Clarifai)](https://www.clarifai.com/blog/agentic-prompt-engineering)
- [Zero to One: Learning Agentic Patterns (Phil Schmid)](https://www.philschmid.de/agentic-pattern)
- [How to Write Killer Prompts for Agentic AI Workflow (Medium)](https://medium.com/@vithika16k/how-to-write-killer-prompts-for-your-agentic-ai-workflow-183e37390808)
- [Introduction to AI Agents (Prompt Engineering Guide)](https://www.promptingguide.ai/agents/introduction)
