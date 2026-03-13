# Plan-and-Solve 策略：先规划再执行

> 难度：中级
> 分类：Planning & Reasoning

## 简短回答

Plan-and-Solve (PS) 是 Wang et al. (2023, ACL) 提出的零样本 Prompting 策略，核心思想是将任务执行分为两个阶段：**先制定计划**（将任务分解为子步骤），**再逐步执行**（按计划依次完成每个子步骤）。它解决了 Zero-shot CoT 的三大问题：计算错误、遗漏步骤、语义理解错误。增强版 PS+ 加入了"提取相关变量"、"计算中间结果"、"检查计算过程"等详细指令，在多个数学推理基准上超越了 Zero-shot CoT，甚至在部分任务上接近 Few-shot CoT 的效果——而无需提供任何示例。在 Agent 系统中，Plan-and-Solve 演化为 Plan-and-Execute 架构：Planner Agent 生成全局计划，Executor Agent 逐步执行，支持动态重规划。

## 详细解析

### 从 Zero-shot CoT 到 Plan-and-Solve

```
Zero-shot CoT：
  Prompt: "问题... Let's think step by step."
  问题：
  1. 可能遗漏关键步骤
  2. 可能出现计算错误
  3. 模型自由发挥，推理质量不稳定

Plan-and-Solve (PS)：
  Prompt: "问题... Let's first understand the problem and
           devise a plan to solve it. Then, let's carry out
           the plan and solve the problem step by step."
  改进：
  1. 明确要求先"理解问题"
  2. 明确要求"制定计划"
  3. 然后按计划执行
```

### PS vs PS+ 的 Prompt 模板

```python
# 基础 PS Prompt
ps_prompt = """
{question}

Let's first understand the problem and devise a plan to solve it.
Then, let's carry out the plan and solve the problem step by step.
"""

# 增强版 PS+ Prompt（加入更详细的指令）
ps_plus_prompt = """
{question}

Let's first understand the problem, extract relevant variables
and their corresponding numerals, and devise a plan to solve it.
Then, let's carry out the plan, calculate intermediate results
(pay attention to correct numerical calculation and commonsense),
and solve the problem step by step.
"""

# PS+ 的三个关键增强：
# 1. "extract relevant variables" → 防止遗漏关键信息
# 2. "calculate intermediate results" → 强制记录中间结果
# 3. "pay attention to correct numerical calculation" → 减少计算错误
```

### 基准测试结果

```
数学推理基准（text-davinci-003）：
┌─────────────────┬──────────┬──────────┬──────────┐
│ 方法            │ GSM8K    │ SVAMP    │ MultiArith│
├─────────────────┼──────────┼──────────┼──────────┤
│ Zero-shot       │ 10.4%    │ 63.7%    │ 17.7%    │
│ Zero-shot CoT   │ 56.4%    │ 74.3%    │ 78.7%    │
│ Plan-and-Solve  │ 58.2%    │ 77.8%    │ 79.3%    │
│ PS+             │ 58.7%    │ 79.2%    │ 81.2%    │
│ Few-shot CoT    │ 58.8%    │ 79.0%    │ 83.8%    │
└─────────────────┴──────────┴──────────┴──────────┘

注意：PS+ 几乎追平 Few-shot CoT，但不需要提供任何示例！
```

### Plan-and-Execute Agent 架构

```python
class PlanAndExecuteAgent:
    """Plan-and-Solve 在 Agent 系统中的扩展"""

    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm   # 规划用的 LLM（可用更强模型）
        self.executor = executor_llm  # 执行用的 LLM（可用更便宜模型）
        self.tools = tools

    async def run(self, task: str):
        # 阶段 1：规划
        plan = await self.plan(task)

        # 阶段 2：逐步执行
        results = []
        for i, step in enumerate(plan.steps):
            result = await self.execute_step(step, results)
            results.append(result)

            # 阶段 3（可选）：检查是否需要重规划
            if result.needs_replan:
                plan = await self.replan(task, plan, results, i)

        # 阶段 4：汇总
        return await self.synthesize(task, results)

    async def plan(self, task):
        prompt = f"""
        任务：{task}

        请制定一个详细的执行计划：
        1. 分析任务目标和约束
        2. 列出完成任务所需的步骤
        3. 标注每步需要的工具
        4. 标注步骤间的依赖关系

        输出格式：
        Step 1: [描述] | 工具: [工具名] | 依赖: []
        Step 2: [描述] | 工具: [工具名] | 依赖: [Step 1]
        """
        return await self.planner.invoke(prompt)

    async def execute_step(self, step, previous_results):
        prompt = f"""
        当前步骤：{step.description}
        可用工具：{step.tool}
        前序步骤结果：{previous_results}

        请执行这一步并返回结果。
        """
        return await self.executor.invoke(prompt)
```

### 与 ReAct 的对比

```
ReAct（思考 → 行动 → 观察 循环）：
  ✓ 灵活，根据每步结果动态决策
  ✗ 没有全局视角，容易陷入局部循环
  ✗ 短视——只看下一步，不看全局

Plan-and-Execute：
  ✓ 先有全局计划，再逐步执行
  ✓ 计划明确了总步数和依赖关系
  ✓ 支持对计划的提前审核
  ✗ 初始计划可能不完美
  ✗ 需要重规划机制应对意外

混合方案（LangGraph 推荐）：
  Plan-and-Execute 负责全局计划
  + 每个子步骤用 ReAct 模式执行
  = 全局规划 + 局部灵活性
```

### LangGraph 中的 Plan-and-Execute

```python
import operator
from typing import Annotated
from langgraph.graph import StateGraph

class PlanExecuteState(TypedDict):
    task: str
    plan: list[str]
    current_step: int
    results: Annotated[list[str], operator.add]  # Reducer：自动追加而非覆盖
    final_answer: str

def planner(state):
    """生成执行计划"""
    plan = llm.invoke(f"为任务制定计划: {state['task']}")
    return {"plan": plan.steps}

def executor(state):
    """执行当前步骤"""
    step = state["plan"][state["current_step"]]
    result = react_agent.invoke(step)  # 每步用 ReAct
    return {
        "results": [result],  # Reducer 会自动追加到列表
        "current_step": state["current_step"] + 1
    }

def should_continue(state):
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "executor"

# 构建图
graph = StateGraph(PlanExecuteState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("synthesize", synthesize)
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue)
```

### Plan-and-Solve 的适用场景

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 简单问答 | 直接回答 | 规划是多余的 |
| 数学推理 | PS+ | 减少遗漏和计算错误 |
| 多步任务 | Plan-and-Execute | 需要全局视角 |
| 动态环境 | Plan + ReAct | 全局计划 + 局部灵活 |
| 高可靠性 | Plan + 人工审核 | 计划可被人类审核和修改 |

## 常见误区 / 面试追问

1. **误区："Plan-and-Solve 就是 Chain-of-Thought 的变体"** — PS 不仅是让模型"逐步思考"，而是明确将过程分为"规划"和"执行"两个独立阶段。CoT 是一次性生成推理链，PS 是先生成计划再按计划执行。

2. **误区："计划一旦制定就不应该改变"** — 好的 Plan-and-Execute 系统必须支持重规划（Replanning）。执行过程中可能遇到意外情况（工具失败、信息不符合预期），需要动态调整计划。

3. **追问："PS+ 为什么不需要 Few-shot 示例就能接近 Few-shot CoT？"** — 因为 PS+ 的详细指令（提取变量、计算中间结果、注意计算正确性）本质上将 Few-shot 示例中隐含的推理策略显式化了。指令替代了示例的作用。

4. **追问："Plan-and-Execute 架构中，Planner 和 Executor 应该用同一个模型吗？"** — 不一定。常见做法是 Planner 用更强的模型（如 Claude Opus）保证计划质量，Executor 用更快更便宜的模型（如 Claude Haiku）降低成本。这种异构模型配置是生产中的最佳实践。

## 参考资料

- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought (ACL 2023)](https://arxiv.org/abs/2305.04091)
- [Plan-and-Solve Prompting (Learn Prompting)](https://learnprompting.org/docs/advanced/decomposition/plan_and_solve)
- [Plan-and-Solve Plus (PS+) Framework (PromptEngineering.org)](https://promptengineering.org/plan-and-solve-plus-ps-a-prompting-framework-for-enhanced-llm-reasoning/)
- [Plan & Solve Agent Pattern (Agent Patterns)](https://agent-patterns.readthedocs.io/en/stable/patterns/plan-and-solve.html)
- [Planning for Agents (LangChain Blog)](https://blog.langchain.com/planning-for-agents/)
