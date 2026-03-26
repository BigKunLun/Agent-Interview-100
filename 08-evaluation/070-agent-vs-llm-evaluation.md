# Agent 评估与 LLM 评估有何不同？

> 难度：基础
> 分类：Evaluation

## 简短回答

LLM 评估关注**单次输入输出的质量**（一个问题一个答案），而 Agent 评估需要关注**多步决策轨迹的整体表现**——不仅看最终结果是否正确，还要评估推理过程、工具使用、规划质量和错误恢复能力。核心区别：(1) **评估对象**——LLM 评估的是文本生成质量，Agent 评估的是任务完成能力；(2) **评估维度**——Agent 多了轨迹评估（trajectory evaluation）、工具使用评估和效率评估；(3) **非确定性更强**——同一任务，Agent 可能走完全不同的路径都成功，评估不能只和"标准答案"对比；(4) **端到端复杂度**——Agent 涉及 LLM + 工具 + 环境的交互链，任何环节出错都影响结果。ACL 2025 的 Agent 评估综述提出二维分类法：评估"什么能力"（推理、规划、工具使用等）× "用什么方法"（基准测试、人工评估、LLM Judge 等）。

## 详细解析

### LLM 评估 vs Agent 评估的核心区别

```
LLM 评估：
  输入 → [LLM] → 输出
  评估：输出的质量（准确、流畅、有用）

Agent 评估：
  任务 → [推理] → [工具调用] → [观察] → [推理] → [工具调用] → ... → 结果
  评估维度：
  ├── 最终结果：任务是否完成？
  ├── 推理质量：每步推理是否合理？
  ├── 工具使用：工具选择是否正确？参数是否正确？
  ├── 效率：用了多少步？花了多少 token？
  ├── 错误恢复：遇到错误是否能恢复？
  └── 安全性：是否有越权操作？
```

### Agent 评估的四层模型

```python
agent_evaluation_layers = {
    "Layer 1 - 结果评估（What）": {
        "问题": "Agent 是否完成了任务？",
        "指标": ["任务完成率", "答案准确率", "部分完成度"],
        "方法": "自动化检查最终状态",
        "示例": "SWE-bench: 代码修改后测试是否通过",
    },
    "Layer 2 - 轨迹评估（How）": {
        "问题": "Agent 的决策路径是否合理？",
        "指标": ["步骤合理性", "是否有冗余步骤", "是否走了弯路"],
        "方法": "LLM-as-Judge 或人工评估 Trace",
        "示例": "10 步完成 vs 3 步完成，效率差异巨大",
    },
    "Layer 3 - 工具评估（With What）": {
        "问题": "Agent 是否正确使用了工具？",
        "指标": ["工具选择准确率", "参数正确率", "调用次数"],
        "方法": "与最优工具使用序列对比",
        "示例": "搜索 vs 计算——应该用计算器时却去搜索",
    },
    "Layer 4 - 鲁棒性评估（What If）": {
        "问题": "Agent 面对异常情况如何表现？",
        "指标": ["错误恢复率", "幻觉率", "安全违规率"],
        "方法": "注入故障和对抗样本",
        "示例": "工具返回错误时是否能换策略重试",
    },
}
```

### 轨迹评估（Trajectory Evaluation）

```python
class TrajectoryEvaluator:
    """评估 Agent 的完整执行轨迹"""

    async def evaluate_trajectory(self, task, trajectory):
        scores = {}

        # 1. 步骤级评估：每一步是否合理
        step_scores = []
        for i, step in enumerate(trajectory.steps):
            step_score = await self.evaluate_step(
                task=task,
                step=step,
                context=trajectory.steps[:i],  # 前序上下文
            )
            step_scores.append(step_score)
        scores["step_quality"] = np.mean(step_scores)

        # 2. 轨迹效率：是否有冗余步骤
        scores["efficiency"] = self.compute_efficiency(
            actual_steps=len(trajectory.steps),
            optimal_steps=self.get_optimal_length(task),
        )

        # 3. 目标达成度
        scores["goal_achieved"] = await self.check_goal(
            task=task,
            final_state=trajectory.final_state,
        )

        # 4. 错误恢复：遇到错误后的处理
        errors = [s for s in trajectory.steps if s.is_error]
        if errors:
            recovery_rate = sum(1 for e in errors if e.was_recovered) / len(errors)
            scores["error_recovery"] = recovery_rate

        return scores

    async def evaluate_step(self, task, step, context):
        """用 LLM 评估单步决策"""
        return await self.judge_llm.invoke(f"""
        任务：{task}
        已执行步骤：{context}
        当前步骤：{step}

        评估这一步是否合理（1-5分）：
        - 是否推进了任务目标？
        - 工具选择是否正确？
        - 参数是否合理？
        """)
```

### 主要 Agent 基准测试

```python
agent_benchmarks = {
    "代码 Agent": {
        "SWE-bench": "修复真实 GitHub Issue（Resolved Rate）",
        "HumanEval": "生成函数代码（Pass@k）",
        "MBPP": "Python 编程任务（Pass@k）",
    },
    "Web Agent": {
        "WebArena": "在真实网站完成复杂任务",
        "Mind2Web": "跨网站的通用网页操作",
        "VisualWebArena": "需要视觉理解的网页任务",
    },
    "通用推理 Agent": {
        "ALFWorld": "文本版家庭环境中的任务执行",
        "WebShop": "模拟电商购物任务",
        "GAIA": "通用 AI 助手评估（需要多工具组合）",
    },
    "工具使用 Agent": {
        "ToolBench": "评估 API 工具的选择和使用",
        "API-Bank": "评估 API 调用的正确性",
        "TaskBench": "多工具组合任务",
    },
}
```

### 评估 Agent 的实用框架

```python
class ProductionAgentEvaluator:
    """生产环境中的 Agent 评估"""

    def __init__(self):
        self.metrics = {
            # 核心指标
            "task_success_rate": "任务完成率",
            "avg_steps": "平均步骤数",
            "avg_latency": "平均延迟",
            "avg_cost": "平均成本",

            # 质量指标
            "trajectory_quality": "轨迹质量（LLM Judge）",
            "tool_accuracy": "工具使用准确率",
            "hallucination_rate": "幻觉率",

            # 安全指标
            "safety_violation_rate": "安全违规率",
            "unauthorized_action_rate": "越权操作率",
        }

    async def run_eval_suite(self, agent, test_cases):
        results = []
        for case in test_cases:
            # 执行并记录完整轨迹
            trajectory = await agent.execute_with_trace(case.task)

            # 多维评估
            eval_result = {
                "task_success": self.check_success(trajectory, case.expected),
                "steps": len(trajectory.steps),
                "cost": trajectory.total_cost,
                "latency": trajectory.total_time,
                "trajectory_score": await self.judge_trajectory(trajectory),
                "tool_accuracy": self.check_tool_usage(trajectory),
                "safety": self.check_safety(trajectory),
            }
            results.append(eval_result)

        return self.aggregate(results)
```

### LLM 评估 vs Agent 评估总结

```
┌──────────────────┬──────────────────┬──────────────────┐
│ 维度             │ LLM 评估         │ Agent 评估       │
├──────────────────┼──────────────────┼──────────────────┤
│ 评估对象         │ 单次生成          │ 多步决策轨迹     │
│ 评估范围         │ 输出文本质量      │ 任务完成 + 过程  │
│ 确定性           │ 较高              │ 低（多路径可行） │
│ 关键指标         │ 准确率、BLEU      │ 任务完成率、效率 │
│ 工具使用         │ 无                │ 核心评估维度     │
│ 安全性           │ 输出安全          │ 行为安全（操作） │
│ 评估复杂度       │ 低                │ 高               │
│ 基准测试         │ MMLU、GSM8K      │ SWE-bench、GAIA │
└──────────────────┴──────────────────┴──────────────────┘
```

## 常见误区 / 面试追问

1. **误区："Agent 评估只看最终结果就够了"** — 最终结果正确但过程不合理的 Agent 同样有问题——可能走了弯路浪费资源，可能碰巧得到正确结果但推理错误（不可靠），可能使用了不安全的操作。轨迹评估和结果评估同等重要。

2. **误区："用 LLM 基准测试就能评估 Agent"** — LLM 基准（如 MMLU）测试的是知识和推理能力，无法反映 Agent 的工具使用、规划和错误恢复能力。Agent 需要专用基准（如 SWE-bench、WebArena、GAIA）。

3. **追问："如何评估 Agent 的效率？"** — 三个维度：(1) 步骤效率——完成任务用了多少步（vs 最优步数）；(2) 成本效率——消耗了多少 token/金钱；(3) 时间效率——端到端延迟。权衡是：更多步骤可能提升准确率但增加成本。

4. **追问："Agent 评估的最大难点是什么？"** — 非确定性。同一任务可能有多条正确路径，无法用固定的"标准答案"对比。解决方案：(1) 评估最终状态而非中间步骤；(2) 用 LLM Judge 评估轨迹的合理性；(3) 多次运行取统计指标。

## 参考资料

- [Agent Evaluation vs Model Evaluation: What's the Difference (Maxim)](https://www.getmaxim.ai/articles/agent-evaluation-vs-model-evaluation-whats-the-difference-and-why-it-matters/)
- [Evaluation and Benchmarking of LLM Agents: A Survey (ACL 2025)](https://arxiv.org/html/2507.21504v1)
- [LLM Agent Evaluation: Assessing Tool Use, Task Completion (Confident AI)](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)
- [The Complete Guide to LLM & AI Agent Evaluation in 2026 (Adaline)](https://www.adaline.ai/blog/complete-guide-llm-ai-agent-evaluation-2026)
- [Understanding How AI Agent Trajectories Guide Agent Evaluation (Objectways)](https://objectways.com/blog/understanding-how-ai-agent-trajectories-guide-agent-evaluation/)
