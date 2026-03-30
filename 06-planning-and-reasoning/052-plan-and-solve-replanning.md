# Plan-and-Solve 与动态重规划

> 难度：中级
> 分类：Planning & Reasoning

## 简短回答

Plan-and-Solve (PS) 是 Wang et al. (2023, ACL) 提出的零样本 Prompting 策略，核心思想是将任务执行分为"先制定计划、再逐步执行"两个阶段，解决了 Zero-shot CoT 的计算错误、遗漏步骤和语义理解错误三大问题。增强版 PS+ 加入了"提取变量"、"计算中间结果"、"检查计算"等指令，在多个数学推理基准上接近 Few-shot CoT——而无需提供任何示例。在 Agent 系统中，PS 演化为 Plan-and-Execute 架构：Planner Agent 生成全局计划，Executor Agent 逐步执行。然而初始计划不可能完美，执行中必然遇到意外（工具失败、信息偏离、环境变化），因此需要**动态重规划（Replanning）**能力。重规划包含三大机制：**触发条件检测**（何时重规划）、**计划修正策略**（如何修改）、**上下文保持**（保留已完成进度）。前沿框架 DuSAR 结合子目标导向的全局规划与经验驱动的局部适应，ALAS 引入事务性计划执行实现步骤级回滚与重试。从静态规划到动态调整，形成了完整的 Agent 规划闭环。

## 详细解析

### 一、从 Zero-shot CoT 到 Plan-and-Solve

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

### 二、PS vs PS+ 的 Prompt 模板

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

### 三、基准测试结果

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

### 四、Plan-and-Execute Agent 架构

Plan-and-Solve 在 Agent 系统中演化为 Plan-and-Execute 架构，将规划与执行分离为独立模块：

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

            # 阶段 3：检查是否需要重规划
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

### 五、与 ReAct 的对比

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

### 六、LangGraph 中的 Plan-and-Execute

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

### 七、为什么需要动态重规划？

静态计划无法应对执行中的意外，当步骤失败或前提假设被推翻时，Agent 需要动态调整：

```
初始计划：
  Step 1: 搜索竞品A的数据 ✅ 完成
  Step 2: 搜索竞品B的数据 ❌ API 超时，未获取到
  Step 3: 对比分析 A 和 B   → 依赖 Step 2，无法执行
  Step 4: 生成报告

不重规划 → Agent 卡死或跳过关键步骤
重规划后：
  Step 2': 换用备用数据源搜索竞品B
  Step 3: 对比分析（保持不变）
  Step 4: 生成报告（保持不变）
```

### 八、重规划的触发条件

```python
class ReplanTrigger:
    """检测何时需要重规划"""

    def should_replan(self, state) -> bool:
        # 触发条件 1：步骤执行失败
        if state.last_step_failed:
            return True

        # 触发条件 2：结果与预期严重偏离
        if state.deviation_score > self.threshold:
            return True

        # 触发条件 3：发现新信息改变了前提假设
        if state.assumptions_invalidated:
            return True

        # 触发条件 4：已执行步骤超过预期，可能陷入循环
        if state.steps_executed > state.expected_steps * 1.5:
            return True

        # 触发条件 5：用户干预请求修改目标
        if state.user_intervention:
            return True

        return False
```

### 九、重规划策略

#### 策略 1：反应式重规划

步骤失败时立即触发，保留已完成进度，重新规划剩余部分：

```python
class ReactiveReplanner:
    """步骤失败时立即重规划"""

    async def execute_plan(self, plan, task, max_retries=3):
        if max_retries <= 0:
            raise RuntimeError("超过最大重规划次数，终止执行")

        results = []
        for i, step in enumerate(plan.steps):
            result = await self.execute_step(step)

            if result.failed:
                # 重规划：保留已完成步骤，重新规划剩余部分
                new_plan = await self.replan(
                    original_task=task,
                    completed_steps=results,
                    failed_step=step,
                    failure_reason=result.error,
                    remaining_steps=plan.steps[i+1:]
                )
                # 递归执行新计划，递减重试次数
                return await self.execute_plan(new_plan, task, max_retries - 1)

            results.append(result)
        return results

    async def replan(self, original_task, completed_steps,
                     failed_step, failure_reason, remaining_steps):
        prompt = f"""
        原始任务：{original_task}

        已完成的步骤和结果：
        {self.format_results(completed_steps)}

        失败的步骤：{failed_step}
        失败原因：{failure_reason}

        原剩余计划：{remaining_steps}

        请根据失败原因重新规划剩余步骤。
        要求：
        1. 不要重复已完成的步骤
        2. 尝试用不同方式完成失败的步骤
        3. 调整后续步骤以适应变化
        """
        return await self.planner_llm.invoke(prompt)
```

#### 策略 2：周期性重规划

每执行 N 步后主动评估计划有效性，无需等待失败：

```python
class PeriodicReplanner:
    """每 N 步重新评估和调整计划"""

    async def execute_with_checkpoints(self, plan, task):
        results = []
        checkpoint_interval = 3  # 每 3 步检查一次

        for i, step in enumerate(plan.steps):
            result = await self.execute_step(step)
            results.append(result)

            # 每 N 步检查是否需要调整计划
            if (i + 1) % checkpoint_interval == 0:
                assessment = await self.assess_progress(
                    task=task,
                    plan=plan,
                    completed=results,
                    remaining=plan.steps[i+1:]
                )
                if assessment.needs_adjustment:
                    plan = await self.adjust_plan(
                        task, results, assessment.suggestions
                    )

        return results

    async def assess_progress(self, task, plan, completed, remaining):
        """评估当前进度是否符合预期"""
        prompt = f"""
        任务目标：{task}
        已完成步骤及结果：{completed}
        剩余计划：{remaining}

        评估：
        1. 当前进度是否朝目标方向推进？
        2. 已获取的信息是否改变了后续步骤的必要性？
        3. 是否有更高效的方式完成剩余任务？
        """
        return await self.llm.invoke(prompt)
```

#### 策略 3：ALAS 事务性规划

每个步骤作为原子事务执行，失败可回滚和重试：

```python
class TransactionalPlanner:
    """ALAS: 每个步骤作为原子事务执行"""

    async def execute_step_transactional(self, step, context):
        """事务性执行：成功提交，失败回滚"""
        checkpoint = self.save_state()  # 保存当前状态

        try:
            result = await self.execute(step)

            if self.validate(result):
                self.commit(result)   # 提交变更
                return result
            else:
                self.rollback(checkpoint)  # 验证失败，回滚
                return await self.retry_with_alternative(step)

        except Exception as e:
            self.rollback(checkpoint)      # 异常回滚

            if self.retries_left(step) > 0:
                return await self.retry(step)  # 重试
            else:
                return await self.replan_from_here(step, e)  # 重规划
```

### 十、DuSAR 双策略框架

DuSAR 结合全局规划与局部适应，在长时域任务上表现优异：

```python
class DuSAR:
    """双策略自适应推理框架"""

    async def solve(self, task):
        # 策略 1：子目标导向（全局规划）
        subgoals = await self.decompose_to_subgoals(task)

        for subgoal in subgoals:
            # 策略 2：经验驱动（局部适应）
            # 从历史执行中学习类似子目标的最佳策略
            strategy = self.experience_bank.get_best_strategy(subgoal)

            if strategy:
                result = await self.execute_with_strategy(subgoal, strategy)
            else:
                result = await self.explore_new_strategy(subgoal)

            # 动态调整：根据结果更新后续子目标
            if result.changes_context:
                subgoals = await self.redecompose(
                    task, completed=result, remaining=subgoals
                )

            # 更新经验库
            self.experience_bank.record(subgoal, result)
```

### 十一、重规划的关键设计原则

```python
replanning_principles = {
    "最小变更": (
        "重规划应尽量保留原计划中仍然有效的部分，"
        "只修改必须改变的步骤。避免全部推翻重来"
    ),
    "上下文传递": (
        "重规划时必须传递已完成步骤的结果和失败原因，"
        "让 LLM 理解当前状态，不要从零开始"
    ),
    "防止无限循环": (
        "设置最大重规划次数（如 3 次），"
        "超过后报告失败而非无限重试"
    ),
    "失败记忆": (
        "记住之前失败的方案，避免重规划时"
        "再次生成相同的失败计划"
    ),
    "降级策略": (
        "多次重规划失败后，应有降级方案："
        "简化目标、请求人类帮助、部分完成"
    ),
}
```

### 十二、实际系统中的重规划架构

```
┌──────────────┐
│   Planner    │ ← 初始计划
└──────┬───────┘
       ▼
┌──────────────┐     ┌──────────────┐
│  Executor    │────▶│   Monitor    │
│ (执行步骤)   │     │ (监控偏差)   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
  成功 → 继续          偏差检测 → 触发重规划
                            │
                    ┌───────▼────────┐
                    │   Replanner    │
                    │ (生成新计划)   │
                    └───────┬────────┘
                            │
                    反馈到 Executor 继续执行
```

### 适用场景总结

| 场景 | 推荐方案 | 原因 |
|------|---------|------|
| 简单问答 | 直接回答 | 规划是多余的 |
| 数学推理 | PS+ | 减少遗漏和计算错误 |
| 多步任务 | Plan-and-Execute | 需要全局视角 |
| 动态环境 | Plan + ReAct + Replanning | 全局计划 + 局部灵活 + 动态调整 |
| 高可靠性 | Plan + 人工审核 | 计划可被人类审核和修改 |
| 长时域复杂任务 | DuSAR / ALAS | 需要经验积累和事务性保障 |

## 常见误区 / 面试追问

1. **误区："Plan-and-Solve 就是 Chain-of-Thought 的变体"** — PS 不仅是让模型"逐步思考"，而是明确将过程分为"规划"和"执行"两个独立阶段。CoT 是一次性生成推理链，PS 是先生成计划再按计划执行。

2. **误区："计划一旦制定就不应该改变"** — 好的 Plan-and-Execute 系统必须支持重规划（Replanning）。执行过程中可能遇到意外情况（工具失败、信息不符合预期），需要动态调整计划。

3. **误区："每次失败都应该重规划"** — 不是所有失败都需要重规划。瞬时错误（网络超时）用重试就行，只有结构性问题（方案不可行、前提假设变化）才需要重规划。过于频繁的重规划浪费计算资源且可能引入新问题。

4. **误区："重规划 = 重新从头开始规划"** — 好的重规划是增量修改——保留已完成的进度和仍然有效的步骤，只修改必须改变的部分。全部推翻重来是最后手段。

5. **追问："PS+ 为什么不需要 Few-shot 示例就能接近 Few-shot CoT？"** — 因为 PS+ 的详细指令（提取变量、计算中间结果、注意计算正确性）本质上将 Few-shot 示例中隐含的推理策略显式化了。指令替代了示例的作用。

6. **追问："Plan-and-Execute 架构中，Planner 和 Executor 应该用同一个模型吗？"** — 不一定。常见做法是 Planner 用更强的模型（如 Claude Opus）保证计划质量，Executor 用更快更便宜的模型（如 Claude Haiku）降低成本。这种异构模型配置是生产中的最佳实践。

7. **追问："如何防止重规划陷入循环？"** — 三层防御：(1) 记录失败的计划，新计划不能重复；(2) 设置最大重规划次数上限；(3) 每次重规划必须与前次不同——可以用 LLM 自评判断新计划是否实质性不同。

8. **追问："重规划的成本如何控制？"** — 只在必要时重规划（而非每步都重规划）；重规划只传递摘要而非完整历史（控制 token 用量）；Planner 可以用较小模型做快速重规划，只在关键决策点用大模型。

## 参考资料

- [Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought (ACL 2023)](https://arxiv.org/abs/2305.04091)
- [Plan-and-Solve Prompting (Learn Prompting)](https://learnprompting.org/docs/advanced/decomposition/plan_and_solve)
- [Plan-and-Solve Plus (PS+) Framework (PromptEngineering.org)](https://promptengineering.org/plan-and-solve-plus-ps-a-prompting-framework-for-enhanced-llm-reasoning/)
- [Plan & Solve Agent Pattern (Agent Patterns)](https://agent-patterns.readthedocs.io/en/stable/patterns/plan-and-solve.html)
- [Planning for Agents (LangChain Blog)](https://blog.langchain.com/planning-for-agents/)
- [DuSAR: A Co-Adaptive Dual-Strategy Framework for LLM-Based Planning (arXiv)](https://arxiv.org/html/2512.08366v1)
- [ALAS: Transactional and Dynamic Multi-Agent LLM Planning (arXiv)](https://arxiv.org/html/2511.03094v1)
- [Dynamic Planning in LLM Agents: From ReAct to Tree-of-Thoughts](https://tao-hpu.medium.com/dynamic-planning-in-llm-agents-from-react-to-tree-of-thoughts-a3464a8b114e)
- [LLM Dynamic Planner (LLM-DP) (Emergent Mind)](https://www.emergentmind.com/topics/llm-dynamic-planner-llm-dp)
- [5 Recovery Strategies for Multi-Agent LLM Failures (Newline)](https://www.newline.co/@zaoyang/5-recovery-strategies-for-multi-agent-llm-failures--673fe4c4)
