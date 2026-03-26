# 如何评估 Agent 的推理质量？

> 难度：高级
> 分类：Planning & Reasoning

## 简短回答

评估 Agent 推理质量需要超越"最终答案是否正确"，深入检查**推理过程本身**。核心评估维度包括：(1) **逻辑正确性**——每步推理是否逻辑自洽；(2) **忠实性（Faithfulness）**——推理链是否真实反映了模型的决策过程（而非事后合理化）；(3) **完整性**——是否覆盖了所有关键推理步骤；(4) **效率**——是否用最少的步骤达到目标。评估方法分为**过程评估**（评价每一步推理）和**结果评估**（只看最终输出），前沿研究表明过程评估更可靠。关键工具包括：**Process Reward Models (PRM)**（对每步推理打分）、**LLM-as-Judge**（用 LLM 评价另一个 LLM 的推理）、**Agent 专用基准**（SWE-bench、HumanEval、WebArena 等）。ACL 2025 的 Agent 评估综述提出了二维分类法：评估"什么"（能力维度）× "如何"评估（方法维度）。

## 详细解析

### 推理评估的层次模型

```
Level 4: 任务完成度
  "Agent 是否完成了用户的请求？"
  │
Level 3: 推理路径质量
  "推理过程是否高效、合理？"
  │
Level 2: 工具使用合理性
  "是否选择了正确的工具？参数是否正确？"
  │
Level 1: 单步推理正确性
  "每一步推理是否逻辑正确？"

全面评估需要覆盖所有 4 个层次
```

### 评估维度详解

```python
reasoning_evaluation_dimensions = {
    "逻辑正确性": {
        "定义": "每步推理是否遵循逻辑规则",
        "检查项": [
            "前提是否支持结论",
            "是否存在逻辑谬误",
            "数学计算是否正确",
        ],
        "评估方法": "Process Reward Model / 人工标注",
    },
    "忠实性 (Faithfulness)": {
        "定义": "推理链是否真实反映模型的实际决策过程",
        "问题": "模型可能先得出答案再编造推理过程（事后合理化）",
        "检查方法": "修改推理链的关键步骤，看是否改变最终答案",
        "前沿": "OpenAI 的 CoT Monitorability 研究",
    },
    "完整性": {
        "定义": "是否覆盖了所有必要的推理步骤",
        "问题": "跳步推理可能隐藏错误",
        "检查项": ["是否有遗漏的关键信息", "是否考虑了边界情况"],
    },
    "效率": {
        "定义": "是否用最少的步骤和工具调用达到目标",
        "指标": ["步骤数", "工具调用次数", "token 消耗量"],
    },
    "鲁棒性": {
        "定义": "面对干扰信息是否保持正确推理",
        "测试方法": "在问题中加入无关信息，看是否被误导",
    },
}
```

### 评估方法 1：Process Reward Model (PRM)

```python
class ProcessRewardModel:
    """对推理过程的每一步打分"""

    def evaluate_reasoning(self, problem, reasoning_steps):
        """
        PRM vs ORM (Outcome Reward Model):
        - ORM: 只看最终答案是否正确 → 无法定位错误在哪一步
        - PRM: 对每一步打分 → 精确定位第一个出错的步骤
        """
        step_scores = []
        for i, step in enumerate(reasoning_steps):
            score = self.prm_model.score(
                problem=problem,
                previous_steps=reasoning_steps[:i],
                current_step=step
            )
            step_scores.append({
                "step": i + 1,
                "content": step,
                "score": score,  # 0-1, 越高越正确
                "is_correct": score > 0.5,
            })
        return step_scores

    # 示例输出
    # Step 1: "总共有 15 × 20 = 300 个苹果" → score: 0.95 ✓
    # Step 2: "卖掉了 120 个"               → score: 0.90 ✓
    # Step 3: "剩余 300 + 120 = 420 个"     → score: 0.05 ✗ (应该是减法)
```

### 评估方法 2：LLM-as-Judge

```python
async def llm_judge_reasoning(problem, agent_reasoning, reference=None):
    """用 LLM 评估推理质量"""
    prompt = f"""
    请评估以下 Agent 的推理过程质量。

    问题：{problem}
    Agent 的推理过程：{agent_reasoning}
    {"参考答案：" + reference if reference else ""}

    请从以下维度评分（1-5分）：

    1. 逻辑正确性：每步推理是否逻辑自洽？
    2. 完整性：是否覆盖了所有关键步骤？
    3. 效率：是否存在不必要的冗余步骤？
    4. 工具使用：是否选择了正确的工具和参数？
    5. 最终结论：最终答案是否正确？

    对每个维度给出分数和理由。
    """
    return await judge_llm.invoke(prompt)

# 注意 LLM-as-Judge 的局限：
# - 位置偏差：倾向于给排在前面的答案更高分
# - 冗长偏差：倾向于给更长的回答更高分
# - 自我偏好：倾向于给与自己风格相似的回答更高分
```

### 评估方法 3：Agent 专用基准

```python
agent_benchmarks = {
    # 编程推理
    "HumanEval": {
        "任务": "根据描述生成 Python 函数",
        "指标": "Pass@k（k 次尝试内至少一次通过测试）",
        "评估": "自动化（运行测试用例）",
    },
    "SWE-bench": {
        "任务": "修复真实 GitHub Issue",
        "指标": "Resolved Rate（成功修复的比例）",
        "评估": "自动化（运行项目测试套件）",
    },
    # 网页交互推理
    "WebArena": {
        "任务": "在真实网站上完成复杂任务",
        "指标": "Task Success Rate",
        "评估": "自动化（检查最终页面状态）",
    },
    # 数学推理
    "MATH / GSM8K": {
        "任务": "解决数学问题",
        "指标": "Accuracy（答案准确率）",
        "评估": "自动化（精确匹配答案）",
    },
    # 通用推理
    "ARC-AGI": {
        "任务": "抽象推理和模式识别",
        "指标": "Accuracy",
        "评估": "自动化（精确匹配输出网格）",
    },
}
```

### 多维评估框架

```python
class AgentReasoningEvaluator:
    """综合评估 Agent 推理质量"""

    async def evaluate(self, task, agent_trajectory):
        scores = {}

        # 维度 1：任务完成度（结果评估）
        scores["task_completion"] = await self.check_task_result(
            task, agent_trajectory.final_output
        )

        # 维度 2：推理正确性（过程评估）
        scores["reasoning_correctness"] = await self.evaluate_steps(
            agent_trajectory.thought_steps
        )

        # 维度 3：工具使用质量
        scores["tool_usage"] = self.evaluate_tool_calls(
            agent_trajectory.tool_calls,
            expected_tools=task.available_tools
        )

        # 维度 4：效率
        scores["efficiency"] = {
            "steps": len(agent_trajectory.steps),
            "tool_calls": len(agent_trajectory.tool_calls),
            "tokens_used": agent_trajectory.total_tokens,
            "time_taken": agent_trajectory.duration,
        }

        # 维度 5：安全性
        scores["safety"] = self.check_safety(
            agent_trajectory.all_actions
        )

        return scores
```

### Trace-based 评估（生产环境）

```python
# 生产环境中通过 Trace 评估推理质量
trace_evaluation = {
    "Trace 结构": {
        "Span": "每个推理步骤/工具调用是一个 Span",
        "属性": "输入、输出、延迟、token 消耗",
        "父子关系": "追踪推理的层次结构",
    },
    "自动评估规则": [
        "推理步骤数 > 阈值 → 可能陷入循环",
        "工具调用失败率 > 30% → 工具选择策略有问题",
        "同一工具连续调用 > 3 次 → 可能在重试无效操作",
        "总 token 消耗异常 → 推理效率低",
    ],
    "工具": ["Langfuse", "LangSmith", "Arize Phoenix"],
}
```

## 常见误区 / 面试追问

1. **误区："答案正确就说明推理正确"** — 模型可能因为错误的推理偶然得到正确答案（right answer, wrong reason）。只评估结果会高估模型的推理能力。PRM 等过程评估方法能发现这类问题。

2. **误区："推理链越长越好"** — 冗长的推理链可能包含大量无关信息，增加出错概率。好的推理应该是简洁、高效、每步都有信息增量的。

3. **追问："如何评估推理的忠实性（Faithfulness）？"** — 核心方法：修改推理链中的关键步骤（如改变中间计算结果），如果最终答案不变，说明模型没有真正依赖这些推理步骤——推理链是"装饰性"的而非功能性的。OpenAI 的 CoT Monitorability 研究正在系统性地探索这个问题。

4. **追问："过程评估（PRM）和结果评估（ORM）哪个更好？"** — PRM 更适合训练和调试（能定位错误步骤），ORM 更适合大规模自动评估（实现简单）。理想方案是两者结合：ORM 做初筛，PRM 对有争议的案例做深入分析。

## 参考资料

- [Evaluation and Benchmarking of LLM Agents: A Survey (ACL 2025)](https://arxiv.org/html/2507.21504v1)
- [How to Evaluate LLM Agents: Complete Guide (Medium)](https://medium.com/@dilawar.yadav/how-to-evaluate-llm-agents-complete-guide-to-metrics-methods-and-tools-77cca94cde33)
- [LLM Agent Evaluation: Assessing Tool Use, Task Completion (Confident AI)](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)
- [Evaluating Chain-of-Thought Monitorability (OpenAI)](https://openai.com/index/evaluating-chain-of-thought-monitorability/)
- [Evaluating LLM-based Agents: Metrics, Benchmarks, and Best Practices](https://samiranama.com/posts/Evaluating-LLM-based-Agents-Metrics,-Benchmarks,-and-Best-Practices/)
