# 如何实现 Agent 的自我反思（Self-Reflection）和自我纠正？

> 难度：高级
> 分类：Agent 架构

## 简短回答

Agent 自我反思的核心框架是 Reflexion（Shinn et al., 2023），它将环境反馈转化为语言化的自我反思，存入长期记忆，供下一轮迭代参考——本质上是一种"语言化的强化学习"。除 Reflexion 外，还有 Self-Refine（迭代自我改进）、Self-Debug（自动调试代码）、Self-RAG（自我评估检索质量）等变体。关键争议在于：LLM 能否在没有外部反馈的情况下真正自我纠正推理？

## 详细解析

### 为什么需要自我反思？

传统 Agent 执行失败后通常只是简单重试，不会从失败中学习。自我反思赋予 Agent "元认知"能力——它不仅执行任务，还能回顾自己的表现、识别错误原因、生成改进建议，并在后续尝试中应用这些经验。

类比人类学习：
- **无反思**：做错题 → 重做一遍 → 可能还是错
- **有反思**：做错题 → 分析为什么错 → 总结规律 → 下次避免同类错误

### Reflexion 框架详解

Reflexion 是最有影响力的 Agent 自我反思框架，核心思想是用**语言反馈代替梯度更新**。

#### 三个核心组件

```
┌─────────────┐
│    Actor     │ ←── 执行任务，生成行动轨迹（Trajectory）
└──────┬──────┘
       │ 轨迹 + 结果
       ▼
┌─────────────┐
│  Evaluator   │ ←── 评估执行结果（成功/失败/部分成功）
└──────┬──────┘
       │ 评估信号
       ▼
┌─────────────┐
│ Self-Reflect │ ←── 基于评估生成语言化的反思
└──────┬──────┘     "我在第 3 步选错了工具，应该用 X 而非 Y"
       │ 反思文本
       ▼
┌─────────────┐
│ Long-term   │ ←── 存储反思，供下次迭代参考
│   Memory    │
└─────────────┘
```

#### 执行流程

```python
class ReflexionAgent:
    def __init__(self, actor_llm, evaluator, reflector_llm):
        self.actor = actor_llm
        self.evaluator = evaluator
        self.reflector = reflector_llm
        self.memory = []  # 长期反思记忆

    def run(self, task: str, max_trials: int = 3) -> str:
        for trial in range(max_trials):
            # 1. Actor 执行任务（带上历史反思作为上下文）
            trajectory = self.actor.execute(
                task=task,
                past_reflections=self.memory
            )

            # 2. Evaluator 评估结果
            score, feedback = self.evaluator.evaluate(task, trajectory)

            if score >= THRESHOLD:
                return trajectory.final_answer

            # 3. Self-Reflect 生成反思
            reflection = self.reflector.generate(
                f"任务: {task}\n"
                f"你的执行轨迹: {trajectory}\n"
                f"评估反馈: {feedback}\n"
                f"分析你的错误原因，总结改进策略。"
            )

            # 4. 存入记忆
            self.memory.append(reflection)

        return trajectory.final_answer  # 返回最后一次尝试的结果
```

#### 实验结果

- **AlfWorld**（顺序决策任务）：ReAct + Reflexion 完成 130/134 任务，显著优于纯 ReAct
- **HumanEval / MBPP**（代码生成）：Reflexion 超越先前 SOTA
- **LeetCode Hard**：Reflexion 在困难编程题上展现出从失败中学习的能力

### 实现自我反思的多种模式

#### 模式 1：内置反思循环

在单次任务内加入反思步骤：

```python
# Agent 完成初稿后自我审查
draft = agent.generate(task)
critique = agent.reflect(
    f"审查你的输出：\n{draft}\n"
    f"列出可能的问题、遗漏或改进点。"
)
final = agent.revise(draft, critique)
```

#### 模式 2：双 Agent 反思（Andrew Ng 推荐）

用两个 Agent 实现：一个生成，一个批评：

```python
# Generator Agent + Critic Agent
generator = Agent(system_prompt="生成高质量的代码实现")
critic = Agent(system_prompt="你是严格的代码审查者，找出所有问题")

output = generator.run(task)
for round in range(max_rounds):
    criticism = critic.review(output)
    if criticism.no_issues:
        break
    output = generator.revise(output, criticism)
```

#### 模式 3：基于测试的自我纠正（Self-Debug）

特别适合代码生成场景：

```python
def self_debug_loop(task: str, max_attempts: int = 3) -> str:
    code = llm.generate_code(task)
    for attempt in range(max_attempts):
        test_result = run_tests(code)
        if test_result.all_passed:
            return code
        # 将错误信息反馈给 LLM，让它修复
        code = llm.debug(
            f"代码:\n{code}\n"
            f"测试失败:\n{test_result.errors}\n"
            f"修复这些问题。"
        )
    return code
```

### 关键争议：LLM 能真正自我纠正吗？

Huang et al. (2023) 的研究《Large Language Models Cannot Self-Correct Reasoning Yet》指出：

- **没有外部反馈时**，LLM 的"自我纠正"可能反而把正确答案改错
- **有外部反馈时**（如测试结果、搜索结果），自我纠正才真正有效
- 关键区别："intrinsic self-correction"（纯内省）vs "extrinsic self-correction"（基于外部信号）

**实践建议**：不要依赖 LLM 凭空反思。确保反思环节有外部信号输入——测试结果、评估分数、工具返回值、人工反馈。

### Self-Reflection vs Self-Correction 的区别

| 概念 | 定义 | 实现 |
|------|------|------|
| **Self-Reflection** | Agent 回顾执行过程，生成语言化的经验总结 | Reflexion 框架 |
| **Self-Correction** | Agent 检测并修复输出中的错误 | Self-Debug、Self-Refine |
| **Self-Evaluation** | Agent 评估自己输出的质量 | Self-RAG、LLM-as-Judge |

三者互补：先 Self-Evaluation（发现问题）→ Self-Reflection（分析原因）→ Self-Correction（修复问题）。

## 常见误区 / 面试追问

1. **误区："让 LLM 反思就能提升效果"** — 无外部信号的纯内省可能适得其反。Reflexion 的效果来源于环境反馈（Evaluator），不是 LLM 凭空反思。

2. **误区："反思越多轮越好"** — 过多反思轮次增加成本且可能引入过度修正。通常 2-3 轮反思就够，设置反思轮次上限。

3. **追问："Reflexion 的记忆如何管理？"** — 论文使用滑动窗口保留最近的反思。生产中可用向量数据库存储，按相关性检索历史反思。

4. **追问："如何区分'值得反思的失败'和'不可恢复的失败'？"** — 工具不存在、权限不足等结构性问题不需要反思，直接报错。逻辑错误、策略选择错误才适合反思。结合错误分类（第 008 题）来决定。

## 参考资料

- [Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366)
- [Self-Reflection in LLM Agents: Effects on Problem-Solving Performance (arXiv:2405.06682)](https://arxiv.org/abs/2405.06682)
- [Reflexion (Prompt Engineering Guide)](https://www.promptingguide.ai/techniques/reflexion)
- [Agentic Design Patterns Part 2: Reflection (Andrew Ng / DeepLearning.AI)](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/)
- [How Do Agents Learn from Their Own Mistakes? (HuggingFace Blog)](https://huggingface.co/blog/Kseniase/reflection)
