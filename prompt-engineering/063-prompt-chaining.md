# Prompt Chaining：多步骤 Prompt 的设计与编排

> 难度：中级
> 分类：Prompt Engineering

## 简短回答

Prompt Chaining（提示链）是将复杂任务分解为多个顺序执行的 LLM 调用，每个调用的输出作为下一个调用的输入。它是 Agentic AI 中最基础的设计模式，也被称为 Pipeline 模式。核心优势：(1) **可控性**——每步都有明确的输入输出，易于调试；(2) **可靠性**——每个 Prompt 专注于一件事，比要求 LLM 一次完成所有事更准确；(3) **可观测性**——可以在任意步骤插入验证、人工审核或条件分支。典型链式结构包括：**线性链**（A→B→C）、**条件链**（根据中间结果分支）、**并行链**（独立步骤并行执行后合并）。AWS 将 Prompt Chaining 列为 Agentic AI 的核心工作流模式之一。

## 详细解析

### Prompt Chaining 的基本模式

```
线性链：
  [提取实体] → [分析关系] → [生成摘要] → [格式化输出]
      ↓              ↓              ↓              ↓
  实体列表      关系图谱       分析报告       最终文档

条件链：
  [分类意图] → 意图=查询 → [搜索] → [回答]
            → 意图=操作 → [确认] → [执行]
            → 意图=闲聊 → [对话]

并行链：
  [任务分解] → [子任务A] ┐
             → [子任务B] ├→ [合并结果] → [最终输出]
             → [子任务C] ┘
```

### 线性链实现

```python
class PromptChain:
    """线性 Prompt Chain 实现"""

    def __init__(self, llm):
        self.llm = llm
        self.steps = []

    def add_step(self, name, prompt_template, output_parser=None):
        self.steps.append({
            "name": name,
            "prompt": prompt_template,
            "parser": output_parser or (lambda x: x)
        })
        return self

    async def run(self, initial_input):
        context = {"input": initial_input}

        for step in self.steps:
            # 用当前上下文渲染 Prompt
            prompt = step["prompt"].format(**context)

            # 调用 LLM
            response = await self.llm.invoke(prompt)

            # 解析输出并加入上下文
            parsed = step["parser"](response)
            context[step["name"]] = parsed

        return context

# 使用示例：文档分析链
chain = PromptChain(llm)
chain.add_step(
    "extract",
    "从以下文本中提取所有关键实体和数据点：\n{input}",
    output_parser=json.loads
)
chain.add_step(
    "analyze",
    "分析以下实体之间的关系和趋势：\n{extract}",
)
chain.add_step(
    "report",
    "基于以下分析生成一份简洁的报告：\n{analyze}\n\n原始实体：{extract}",
)

result = await chain.run("2024年Q3财报数据...")
```

### 条件链（Gate/Router 模式）

```python
class ConditionalChain:
    """根据中间结果选择不同的后续链"""

    async def run(self, user_input):
        # Step 1: 分类意图
        intent = await self.llm.invoke(f"""
        将以下用户消息分类为一种意图：
        - QUESTION: 用户在提问
        - ACTION: 用户要求执行操作
        - FEEDBACK: 用户在提供反馈

        消息：{user_input}
        意图：
        """)

        # Step 2: 根据意图选择不同的处理链
        if "QUESTION" in intent:
            # 问题回答链
            answer = await self.question_chain(user_input)
            return answer
        elif "ACTION" in intent:
            # 操作执行链（带确认）
            plan = await self.plan_action(user_input)
            confirmed = await self.confirm_with_user(plan)
            if confirmed:
                return await self.execute_action(plan)
        elif "FEEDBACK" in intent:
            return await self.process_feedback(user_input)
```

### 并行链

```python
import asyncio

class ParallelChain:
    """并行执行独立的子任务，然后合并结果"""

    async def analyze_competitors(self, company_list):
        # Step 1: 并行分析每个竞品
        tasks = [
            self.analyze_single(company)
            for company in company_list
        ]
        results = await asyncio.gather(*tasks)

        # Step 2: 合并分析结果
        merged = await self.llm.invoke(f"""
        以下是各竞品的独立分析结果：
        {self.format_results(results)}

        请综合以上信息，生成一份竞品对比报告，
        包括各公司的优劣势对比和市场定位分析。
        """)
        return merged

    async def analyze_single(self, company):
        return await self.llm.invoke(
            f"详细分析 {company} 的产品特性、市场定位和竞争优势。"
        )
```

### 带验证的链（Quality Gate）

```python
class ValidatedChain:
    """每步都带输出验证的链"""

    async def run(self, input_data):
        # Step 1: 生成
        draft = await self.llm.invoke(
            f"为以下产品写一份营销文案：{input_data}"
        )

        # Step 2: 验证（Quality Gate）
        validation = await self.llm.invoke(f"""
        审查以下营销文案的质量：
        {draft}

        检查：
        1. 是否包含虚假宣传？
        2. 语法是否正确？
        3. 是否符合品牌调性？

        输出 JSON：{{"pass": true/false, "issues": [...]}}
        """)

        if not validation["pass"]:
            # Step 3: 修正
            revised = await self.llm.invoke(f"""
            修正以下文案中的问题：
            文案：{draft}
            问题：{validation["issues"]}
            """)
            return revised

        return draft
```

### 设计原则

```python
chaining_principles = {
    "单一职责": {
        "原则": "每个 Prompt 只做一件事",
        "原因": "LLM 在聚焦的任务上表现最好",
        "示例": "提取 → 分析 → 格式化，而非一步到位",
    },
    "显式传递": {
        "原则": "明确传递上下文，不依赖隐含假设",
        "原因": "每次 LLM 调用是独立的，没有记忆",
        "示例": "将前一步的输出显式嵌入下一步的 Prompt",
    },
    "渐进精化": {
        "原则": "先粗后细，每步增加细节",
        "示例": "大纲 → 段落 → 润色 → 校对",
    },
    "错误隔离": {
        "原则": "每步独立验证，错误不传播",
        "方法": "在关键步骤后加 validation gate",
    },
    "成本意识": {
        "原则": "简单步骤用小模型，关键步骤用大模型",
        "示例": "分类用 Haiku，分析用 Opus",
    },
}
```

### Prompt Chaining vs Agent Loop

```
Prompt Chaining（确定性工作流）：
  步骤固定、顺序确定、可预测
  适合：标准化流程（审核、转换、报告生成）

Agent Loop（自主决策循环）：
  步骤动态、根据结果决定下一步
  适合：开放式任务（调试、研究、探索）

混合方案（推荐）：
  用 Chaining 定义主流程骨架
  在需要灵活性的步骤内嵌 Agent Loop
```

## 常见误区 / 面试追问

1. **误区："链越长越好，任务拆得越细越好"** — 每增加一步链就增加一次 LLM 调用的延迟和成本，也增加一次出错的机会。关键是找到合适的粒度——每步应该是 LLM 能可靠完成的最小有意义单元。

2. **误区："Prompt Chaining 就是把多个 Prompt 串联起来"** — 好的 Chaining 还包括：中间结果的解析和验证、错误处理和重试、条件分支、并行执行、以及上下文管理。简单的串联只是 Chaining 的最基础形式。

3. **追问："如何处理链中某一步失败？"** — 三种策略：(1) 重试当前步骤（带指数退避）；(2) 跳过当前步骤使用默认值；(3) 回退到上一步用不同方式重试。选择哪种取决于该步骤的关键程度。

4. **追问："Prompt Chaining 和 LangChain 的 Chain 是什么关系？"** — LangChain 的 Chain（现已被 LCEL/LangGraph 取代）是 Prompt Chaining 设计模式的框架级实现。但 Prompt Chaining 是通用设计模式，可以用任何语言/框架实现，不依赖 LangChain。

## 参考资料

- [Prompt Chaining - Agentic Design Patterns](https://agentic-design.ai/patterns/prompt-chaining)
- [Workflow for Prompt Chaining (AWS Prescriptive Guidance)](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-prompt-chaining.html)
- [Prompt Chaining for AI Agents: Modular, Reliable, and Scalable (Medium)](https://medium.com/@nivalabs.ai/prompt-chaining-for-the-ai-agents-modular-reliable-and-scalable-workflows-a22d15fd5d33)
- [Prompt Chaining for AI Engineers: A Practical Guide (Maxim)](https://www.getmaxim.ai/articles/prompt-chaining-for-ai-engineers-a-practical-guide-to-improving-llm-output-quality/)
- [Multi-Step LLM Chains: Best Practices for Complex Workflows (Deepchecks)](https://deepchecks.com/orchestrating-multi-step-llm-chains-best-practices/)
