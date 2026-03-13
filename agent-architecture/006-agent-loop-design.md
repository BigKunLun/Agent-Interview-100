# Agent Loop 的设计：如何处理循环终止、最大步数限制？

> 难度：中级
> 分类：Agent 架构

## 简短回答

Agent Loop 是 Agent 与普通 LLM 调用的核心区别——Agent 通过"推理→行动→观察"的循环迭代完成任务。循环终止设计的核心原则是**外部强制（External Enforcement）**：由系统而非 Agent 自身保证终止。常用策略包括最大迭代次数限制、语义完成检测、重复输出检测、资源预算监控，以及达到上限后的优雅降级。

## 详细解析

### Agent Loop 的本质

Agent 的定义特征就是循环（Loop）。它使用 LLM 推理、执行工具、观察结果，然后重复，直到目标达成。这与单次 LLM 调用有本质区别：

```python
# 单次 LLM 调用
response = llm.generate(prompt)

# Agent Loop
while not done:
    thought = llm.reason(state)        # 推理
    action_result = execute(thought)    # 行动
    state = observe(action_result)      # 观察
    done = check_termination(state)     # 终止检查
```

### 为什么需要终止控制？

与传统代码中显而易见的死循环不同，Agent 的循环陷阱更加微妙：
- LLM 的概率性本质可能导致它**误解终止信号**
- 模糊的目标（如"研究 X"）没有明确的完成定义，会引发"无限好奇心"
- 工具调用失败后反复重试同一操作
- 工具过多或描述模糊导致选择混乱

### 五种终止策略

#### 1. 最大迭代次数（最基础也最关键）

```python
class AgentLoop:
    def __init__(self, max_iterations: int = 15):
        self.max_iterations = max_iterations

    def run(self, task: str) -> str:
        for i in range(self.max_iterations):
            result = self.step(task)
            if result.is_final:
                return result.answer
        # 达到上限：优雅降级
        return self.graceful_degradation()
```

不同框架的默认值：LangChain `AgentExecutor` 默认 15 次，Google ADK `LoopAgent` 可自定义上限。

#### 2. 语义完成检测

让 Agent 输出明确的终止标记：

```
"当你确信已回答问题时，以 'FINAL_ANSWER: ...' 结尾"
```

或通过结构化输出：
```python
class AgentResponse(BaseModel):
    thought: str
    action: Optional[ToolCall]
    final_answer: Optional[str]  # 非 None 时表示完成
```

#### 3. 重复输出检测

```python
def detect_loop(history: list[str], window: int = 3) -> bool:
    """检测 Agent 是否在重复相同的操作"""
    recent = history[-window:]
    return len(set(recent)) < len(recent)  # 有重复则判定为循环
```

#### 4. 资源预算监控

```python
class ResourceBudget:
    def __init__(self, max_tokens: int = 100_000, max_cost: float = 1.0):
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.used_tokens = 0
        self.used_cost = 0.0

    def check(self) -> bool:
        return self.used_tokens < self.max_tokens and self.used_cost < self.max_cost
```

#### 5. 子 Agent 信号终止

在多 Agent 系统中，子 Agent 可以通过事件、标志或特定返回值向 Orchestrator 发出终止信号。

### 达到上限后的优雅降级

当 Agent 达到迭代上限时，不应简单地崩溃或返回空结果：

```python
def graceful_degradation(self):
    """优雅降级：总结已有进度，而非直接失败"""
    return self.llm.generate(
        f"你已达到最大步数限制。基于以下已完成的工作，"
        f"给出你目前能提供的最佳答案：\n{self.state.summary()}"
    )
```

关键策略：
- **进度总结**：汇报已完成和未完成的部分
- **状态检查点**：保存当前状态以便后续恢复
- **模型降级**：切换到更便宜的模型完成剩余工作
- **人工升级**：通知人类介入处理

### 不同架构的终止机制对比

| 架构 | 终止方式 | 特点 |
|------|---------|------|
| **ReAct** | LLM 在 response 中生成终止 token（如 "Final Answer:"） | 默认继续，主动终止 |
| **MemGPT** | `request_heartbeat` 默认为 False | 默认终止，主动继续 |
| **Plan-and-Execute** | 计划中的所有步骤执行完毕 | 确定性终止 |
| **现代 Agent** | 多步工具调用 + 自导向终止条件 | 混合策略 |

"默认终止 vs 默认继续"是一个关键的设计决策。MemGPT 的设计哲学是默认终止更安全——Agent 必须显式请求继续执行，而非依赖 LLM 判断何时停止。

### 根因分析：为什么反复达到上限？

迭代上限是**症状检测器，不是疾病本身**。如果 Agent 反复触及上限，真正的问题通常是：

1. **目标模糊**：缺少明确的完成标准
2. **工具签名不清**：工具描述有歧义，LLM 不知道选哪个
3. **停止条件弱**：没有教会 LLM 识别"已完成"
4. **Prompt 设计差**：没有在 System Prompt 中定义终止规则

## 常见误区 / 面试追问

1. **误区："设大一点的 max_iterations 就行"** — 盲目增大上限只会增加成本和延迟。应该先分析为什么达到上限，修复根因（目标不清、工具不当、Prompt 不佳）。

2. **误区："让 LLM 自己决定什么时候停"** — LLM 可能无法可靠判断任务完成。必须在系统层面设置硬性终止条件作为兜底。

3. **追问："如何设定合理的 max_iterations？"** — 根据任务类型估算：简单问答 3-5 步，多步研究 10-15 步，复杂分析 20-30 步。建议从小值开始，通过观测实际使用情况逐步调整。

4. **追问："Agent 的两个循环（Inner Loop / Outer Loop）是什么？"** — Inner Loop 是单次任务内的"思考-行动-观察"循环。Outer Loop 是跨任务的"执行-评估-改进"循环（如 Reflexion 的自我反思机制）。

## 参考资料

- [The Two Agentic Loops: How to Design and Scale Agentic Apps (Plano)](https://planoai.dev/blog/the-two-agentic-loops-how-to-design-and-scale-agentic-apps)
- [Agent Loop Definition: How AI Agents Use Iterative Processes (Glean)](https://www.glean.com/ai-glossary/agent-loop)
- [Cap the Max Number of Iterations (LangChain)](https://python.langchain.com/v0.1/docs/modules/agents/how_to/max_iterations/)
- [Rearchitecting Letta's Agent Loop: Lessons from ReAct, MemGPT, & Claude Code (Letta)](https://www.letta.com/blog/letta-v1-agent)
- [Loop Agents (Google ADK)](https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/)
