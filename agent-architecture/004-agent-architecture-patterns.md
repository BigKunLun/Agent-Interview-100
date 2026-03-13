# 比较不同的 Agent 架构模式：ReAct、Plan-and-Execute、LATS

> 难度：中级
> 分类：Agent 架构

## 简短回答

ReAct 交替推理与行动，灵活但 token 消耗高；Plan-and-Execute 先规划后执行，高效但适应性低；LATS 用树搜索探索多条路径，质量最高但成本是 ReAct 的 3-5 倍。实际生产中，大多数系统采用混合模式：先生成粗略计划，再以 ReAct 方式逐步执行，保留偏离计划的自由度。

## 详细解析

### 1. ReAct（Reason + Act）

**核心机制：** 在一个循环中交替执行 Thought（思考）→ Action（行动）→ Observation（观察），每一步都由 LLM 动态决策。

```
循环：
  Thought: 我需要查找 X 的信息
  Action: search("X")
  Observation: X 是...
  Thought: 现在我需要计算 Y
  Action: calculate("Y")
  Observation: 结果是 Z
  Thought: 我可以给出最终答案了
  Final Answer: ...
```

**优势：**
- 高度灵活，能根据中间结果动态调整策略
- 可解释性强，每步都有显式的推理 trace
- 适合探索性、不确定性高的任务

**劣势：**
- 每步都需要完整的 LLM 调用（携带全部上下文），token 消耗高
- 8 步任务可能消耗 50K-100K tokens
- 无法并行执行，所有步骤严格顺序
- 可能陷入推理循环

**适用场景：** 需要动态探索、中间结果不可预测的任务，如开放域问答、研究调查、交互式调试。

### 2. Plan-and-Execute

**核心机制：** 将任务分为两个阶段——Planner（规划器）生成完整的行动计划，Executor（执行器）逐步执行计划中的每一步。

```
阶段 1 — 规划:
  Plan:
    Step 1: 搜索 X 的最新数据
    Step 2: 从结果中提取关键指标
    Step 3: 计算同比增长率
    Step 4: 生成分析报告

阶段 2 — 执行:
  Execute Step 1 → Result 1
  Execute Step 2 → Result 2
  Execute Step 3 → Result 3
  Execute Step 4 → Final Output
```

**优势：**
- LLM 调用次数少（规划一次 + 每步执行一次，执行可用更小的模型）
- 成本显著低于 ReAct
- 强制 LLM 预先想清楚完整步骤，减少遗漏
- 每步执行可以并行化（如果步骤间无依赖）

**劣势：**
- 初始计划的质量是瓶颈——计划错了，后续全错
- 适应性差，面对意外情况难以偏离原计划
- 需要额外的 Replanning 机制来应对执行中的变化
- 不适合高度动态、不确定的任务

**适用场景：** 结构明确的多步任务，如数据处理流水线、报告生成、自动化测试。

### 3. LATS（Language Agent Tree Search）

**核心机制：** 借鉴蒙特卡洛树搜索（MCTS），将 Agent 的行动空间建模为一棵树，同时探索多条路径，评估每条路径的质量，在死胡同时回溯尝试其他分支。

```
                     根节点（初始状态）
                    /        |         \
              Action A    Action B    Action C
              /    \         |         /    \
           A1      A2      B1       C1      C2
           ✗      ✓        ✓        ✗       ✓
                   ↓        ↓                ↓
                  展开     展开             展开
                   ↓
                最优路径
```

**优势：**
- 通过并行探索多条路径，找到更高质量的解
- 具备回溯能力，不会被单一错误路径困死
- Zhou et al. (2023) 的论文表明 LATS 在多步推理任务上超越 ReAct
- 特别适合有多种可行方案需要比较的场景

**劣势：**
- 成本极高：通常是 ReAct 的 3-5 倍
- 延迟更大：需要并行生成和评估多个分支
- 实现复杂度高
- 对简单任务过度设计

**适用场景：** 复杂推理、代码生成（需要探索多种实现方案）、数学证明、需要高可靠性的关键决策。

### 对比总结

| 维度 | ReAct | Plan-and-Execute | LATS |
|------|-------|-------------------|------|
| **核心思路** | 边想边做 | 想好再做 | 多条路同时探索 |
| **灵活性** | 高 | 低（无 replanning 时） | 极高 |
| **成本** | 中高 | 低 | 极高（3-5x ReAct） |
| **延迟** | 中 | 低 | 高 |
| **结果质量** | 良好 | 结构化任务优秀 | 最高 |
| **可解释性** | 强（每步有 Thought） | 中（有计划但执行不透明） | 中（有树结构但复杂） |
| **并行能力** | 无 | 部分（独立步骤） | 强（多分支并行） |
| **错误恢复** | 动态调整 | 需要显式 replanning | 自动回溯 |
| **实现难度** | 低 | 中 | 高 |

### 混合模式：生产实践中的最佳选择

实际生产中，大多数团队不会只用一种模式，而是组合使用：

```python
# 混合模式：Plan-and-Execute + ReAct
class HybridAgent:
    def run(self, task: str):
        # 阶段 1：用强模型生成粗略计划
        plan = self.planner.generate_plan(task)

        # 阶段 2：用 ReAct 方式逐步执行
        # 每步都可以根据实际情况偏离计划
        results = []
        for step in plan.steps:
            result = self.react_executor.execute_with_reasoning(
                step=step,
                context=results,
                allow_deviation=True  # 允许偏离计划
            )
            results.append(result)

            # 如果偏离过大，触发 replanning
            if result.deviated_significantly:
                plan = self.planner.replan(task, results)

        return self.synthesizer.combine(results)
```

**常见混合策略：**
1. **ReAct + Reflexion**：在 ReAct 失败后加入反思，从失败中学习
2. **Plan-and-Execute + ReAct**：先计划，再用 ReAct 执行每一步，允许动态调整
3. **LATS + Plan-and-Execute**：用 LATS 探索多种计划方案，选最优计划后执行
4. **分层混合**：高层用 Plan-and-Execute 做战略规划，低层用 ReAct 做战术执行

### 选择决策树

```
任务是否结构明确？
├── 是 → 步骤间是否有依赖关系？
│        ├── 大量依赖 → Plan-and-Execute
│        └── 独立步骤 → Plan-and-Execute（并行执行）
└── 否 → 是否需要高可靠性？
         ├── 是 → 预算允许高成本？
         │        ├── 是 → LATS
         │        └── 否 → ReAct + Reflexion
         └── 否 → ReAct
```

## 常见误区 / 面试追问

1. **误区："ReAct 是最先进的，应该总是使用"** — ReAct 适合探索性任务，但对结构明确的任务来说，Plan-and-Execute 更高效、更便宜。没有通用最优架构。

2. **误区："Plan-and-Execute 无法处理变化"** — 加入 Replanning 机制后，Plan-and-Execute 也能应对执行中的意外。关键是设计好触发 replan 的条件。

3. **追问："如何在成本和质量间取舍？"** — 从 ReAct 开始建立 baseline，如果质量不够再考虑 LATS。用 Model Routing 对简单任务用 ReAct + 小模型，复杂任务用 LATS + 强模型。

4. **追问："Gartner 预测 40% 的企业应用将包含 Agent，主流模式是什么？"** — 混合模式（Planning Preamble + ReAct Execution），因为它在灵活性和成本间取得了最佳平衡。

## 参考资料

- [ReAct vs Plan-and-Execute: A Practical Comparison (DEV Community)](https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9)
- [Navigating Modern LLM Agent Architectures (Wollen Labs)](https://www.wollenlabs.com/blog-posts/navigating-modern-llm-agent-architectures-multi-agents-plan-and-execute-rewoo-tree-of-thoughts-and-react)
- [Agent Architectures: ReAct, Self-Ask, Plan-and-Execute (APXML)](https://apxml.com/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools/agent-architectures)
- [How to Build a Plan-and-Execute AI Agent (EMA)](https://www.ema.ai/additional-blogs/addition-blogs/build-plan-execute-agents)
- [LATS: Language Agent Tree Search (Zhou et al., 2023)](https://arxiv.org/abs/2310.04406)
