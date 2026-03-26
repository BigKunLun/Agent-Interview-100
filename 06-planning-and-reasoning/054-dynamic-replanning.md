# 如何实现动态重规划（Replanning）？

> 难度：中级
> 分类：Planning & Reasoning

## 简短回答

动态重规划（Replanning）是 Agent 在执行过程中根据反馈调整计划的能力——初始计划不可能完美，执行中必然遇到意外（工具失败、信息不符预期、环境变化）。核心机制包括：**触发条件检测**（何时重规划）、**计划修正策略**（如何修改）、**上下文保持**（保留已完成的进度）。主要模式有三种：(1) **反应式重规划**——步骤失败时立即触发重规划；(2) **周期性重规划**——每执行 N 步后重新评估计划；(3) **预测式重规划**——检测到潜在问题前主动调整。前沿研究 **DuSAR** 提出双策略框架，结合"子目标导向"的全局规划和"经验驱动"的局部适应，在长时域任务上表现优异。**ALAS** 框架引入事务性计划执行——每个步骤是原子事务，失败可回滚和重试。

## 详细解析

### 为什么需要重规划？

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

### 重规划的触发条件

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

### 模式 1：反应式重规划

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

### 模式 2：周期性重规划

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

### 模式 3：ALAS 事务性规划

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

### DuSAR 双策略框架

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

### 重规划的关键设计原则

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

### 实际系统中的重规划架构

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

## 常见误区 / 面试追问

1. **误区："每次失败都应该重规划"** — 不是所有失败都需要重规划。瞬时错误（网络超时）用重试就行，只有结构性问题（方案不可行、前提假设变化）才需要重规划。过于频繁的重规划浪费计算资源且可能引入新问题。

2. **误区："重规划 = 重新从头开始规划"** — 好的重规划是增量修改——保留已完成的进度和仍然有效的步骤，只修改必须改变的部分。全部推翻重来是最后手段。

3. **追问："如何防止重规划陷入循环？"** — 三层防御：(1) 记录失败的计划，新计划不能重复；(2) 设置最大重规划次数上限；(3) 每次重规划必须与前次不同——可以用 LLM 自评判断新计划是否实质性不同。

4. **追问："重规划的成本如何控制？"** — 只在必要时重规划（而非每步都重规划）；重规划只传递摘要而非完整历史（控制 token 用量）；Planner 可以用较小模型（Haiku）做快速重规划，只在关键决策点用大模型。

## 参考资料

- [DuSAR: A Co-Adaptive Dual-Strategy Framework for LLM-Based Planning (arXiv)](https://arxiv.org/html/2512.08366v1)
- [ALAS: Transactional and Dynamic Multi-Agent LLM Planning (arXiv)](https://arxiv.org/html/2511.03094v1)
- [Dynamic Planning in LLM Agents: From ReAct to Tree-of-Thoughts](https://tao-hpu.medium.com/dynamic-planning-in-llm-agents-from-react-to-tree-of-thoughts-a3464a8b114e)
- [LLM Dynamic Planner (LLM-DP) (Emergent Mind)](https://www.emergentmind.com/topics/llm-dynamic-planner-llm-dp)
- [5 Recovery Strategies for Multi-Agent LLM Failures (Newline)](https://www.newline.co/@zaoyang/5-recovery-strategies-for-multi-agent-llm-failures--673fe4c4)
