# 如何实现 Agent 间的 Handoff（任务交接）？

> 难度：中级
> 分类：Multi-Agent

## 简短回答

Agent Handoff 是多 Agent 系统中一个 Agent 将控制权、任务和对话上下文转交给另一个 Agent 的过程。最常用的实现是 OpenAI 提出的 **`transfer_to_XXX` 模式**——将 Handoff 包装为工具调用，当 Agent 判断任务超出自身能力时，调用 `transfer_to_specialist_agent` 函数触发交接。关键挑战不在于触发 Handoff，而在于**上下文传递的可靠性**——大多数"Agent 失败"实际上是 Handoff 时的上下文丢失或格式错误。最佳实践是使用结构化的 Handoff 数据（JSON Schema），而非自由文本。

## 详细解析

### 为什么需要 Handoff？

单个 Agent 配备太多工具或过大的上下文时，决策质量会下降。Handoff 允许将任务路由给专精的 Agent：

```
用户: "我想退货并了解退款进度"
         │
    ┌────┴────┐
    │ 路由 Agent│  判断需要退货处理
    └────┬────┘
         │ Handoff
    ┌────┴────┐
    │ 退货 Agent│  专精退货流程
    └────┬────┘
         │ Handoff（退货完成，用户追问退款）
    ┌────┴────┐
    │ 财务 Agent│  专精退款查询
    └─────────┘
```

### 实现方式 1：Tool-Based Handoff（OpenAI 模式）

将 Handoff 作为工具暴露给 LLM：

```python
# OpenAI Agents SDK 风格
from agents import Agent, handoff

# 定义专家 Agent
refund_agent = Agent(
    name="Refund Agent",
    instructions="你负责处理退款请求...",
    tools=[execute_refund, check_refund_status],
)

shipping_agent = Agent(
    name="Shipping Agent",
    instructions="你负责处理物流问题...",
    tools=[track_package, update_address],
)

# 路由 Agent 有 Handoff 能力
triage_agent = Agent(
    name="Triage Agent",
    instructions="根据用户需求将请求路由给合适的专家",
    handoffs=[refund_agent, shipping_agent],
    # 内部自动生成工具：transfer_to_refund_agent, transfer_to_shipping_agent
)

# LLM 看到的工具列表：
# - transfer_to_refund_agent: "将对话转交给退款专家"
# - transfer_to_shipping_agent: "将对话转交给物流专家"
```

**工作原理：** LLM 足够智能，会在合适的时机调用 `transfer_to_XXX`。当 Handoff 发生时，新 Agent 接管对话并获得完整的对话历史。

### 实现方式 2：Supervisor 模式（LangGraph）

```python
from langgraph.graph import StateGraph

class SupervisorState(TypedDict):
    messages: list
    current_agent: str

def supervisor_node(state):
    """Supervisor 决定下一个处理的 Agent"""
    response = supervisor_llm.invoke(
        f"当前对话：{state['messages']}\n"
        f"可用专家：researcher, coder, writer\n"
        f"谁应该处理下一步？回复 agent 名称或 'FINISH'"
    )
    return {"current_agent": response.agent_name}

def route(state):
    if state["current_agent"] == "FINISH":
        return END
    return state["current_agent"]

graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("coder", coder_node)
graph.add_conditional_edges("supervisor", route)
# 每个 worker 完成后回到 supervisor
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
```

### 实现方式 3：函数返回 Agent 对象

```python
# Swarm 风格：工具函数返回 Agent 对象触发 Handoff
def handle_refund(order_id: str):
    """处理退款请求"""
    status = check_order(order_id)
    if status == "delivered":
        # 返回 Agent 对象 = 触发 Handoff
        return refund_agent
    else:
        return f"订单 {order_id} 尚未送达，无法退款"

triage_agent = Agent(
    name="Triage",
    tools=[handle_refund],  # 函数可能返回 Agent
)
```

### 上下文传递策略

Handoff 最大的挑战是上下文传递。三种策略：

```python
# 策略 1：完整历史传递（简单但可能超出上下文窗口）
def handoff_full_history(target_agent, conversation_history):
    return target_agent.invoke(messages=conversation_history)

# 策略 2：摘要传递（节省 token，但可能丢失细节）
def handoff_with_summary(target_agent, conversation_history):
    summary = summarize(conversation_history)
    return target_agent.invoke(messages=[
        {"role": "system", "content": f"前序对话摘要：{summary}"},
        conversation_history[-1]  # 最新一条用户消息
    ])

# 策略 3：结构化上下文传递（最佳实践）
def handoff_structured(target_agent, context):
    handoff_payload = {
        "user_intent": "退款查询",
        "order_id": "12345",
        "previous_actions": ["已验证用户身份", "已确认订单已送达"],
        "pending_issues": ["退款金额待确认"],
        "user_messages": conversation_history[-3:]
    }
    return target_agent.invoke(
        messages=[{"role": "system", "content": json.dumps(handoff_payload)}]
    )
```

### Input Filter（上下文过滤）

```python
# OpenAI Agents SDK 的 input_filter
from agents import handoff

def filter_for_billing(handoff_input):
    """只传递与计费相关的上下文给 Billing Agent"""
    filtered = []
    for msg in handoff_input.messages:
        if is_billing_related(msg) or msg == handoff_input.messages[-1]:
            filtered.append(msg)
    handoff_input.messages = filtered
    return handoff_input

billing_handoff = handoff(
    agent=billing_agent,
    input_filter=filter_for_billing  # 自定义过滤逻辑
)
```

### Handoff 的可靠性设计

```python
class ReliableHandoff:
    async def execute(self, from_agent, to_agent, context):
        # 1. 验证目标 Agent 可用
        if not to_agent.is_healthy():
            return await self.fallback(from_agent, context)

        # 2. 结构化上下文打包
        payload = self.pack_context(context)

        # 3. 执行 Handoff
        try:
            result = await to_agent.invoke(payload)
        except Exception as e:
            # 4. 失败回退
            return await from_agent.handle_handoff_failure(e, context)

        # 5. 记录 Handoff 日志
        self.log_handoff(from_agent.id, to_agent.id, payload, result)
        return result
```

### 编排模式与 Handoff 的关系

| 编排模式 | Handoff 方式 | 特点 |
|---------|-------------|------|
| Sequential | 固定顺序传递 | 可预测，适合简单流程 |
| Supervisor | Supervisor 决定路由 | 集中控制，灵活但有瓶颈 |
| Decentralized | Agent 自主 Handoff | 去中心化，灵活但难追踪 |
| Hierarchical | 层级间逐级传递 | 结构化，适合大型系统 |

## 常见误区 / 面试追问

1. **误区："Handoff 只是把对话历史复制过去"** — 自由文本 Handoff 是上下文丢失的主要原因。应该用结构化数据（JSON Schema）传递上下文，像对待公开 API 一样对待 Agent 间接口。

2. **误区："任何 Agent 都可以 Handoff 给任何 Agent"** — 应该限制 Handoff 的路径。每个 Agent 只能 Handoff 给预定义的目标 Agent，防止意外的循环 Handoff 或权限升级。

3. **追问："如何防止 Handoff 循环？"** — 追踪 Handoff 链路，设置最大 Handoff 次数。如果 Agent A → B → A → B 循环出现，触发断路器并升级给人工处理。

4. **追问："Handoff 和工具调用有什么区别？"** — 工具调用后控制权回到原 Agent；Handoff 后控制权完全转移到新 Agent。工具是"用完即还"，Handoff 是"交接班"。

## 参考资料

- [How Agent Handoffs Work in Multi-Agent Systems (Towards Data Science)](https://towardsdatascience.com/how-agent-handoffs-work-in-multi-agent-systems/)
- [Handoffs (OpenAI Agents SDK)](https://openai.github.io/openai-agents-python/handoffs/)
- [Handoffs (LangChain Docs)](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs)
- [Best Practices for Multi-Agent Orchestration and Reliable Handoffs (Skywork AI)](https://skywork.ai/blog/ai-agent-orchestration-best-practices-handoffs/)
- [Orchestrating Agents: Routines and Handoffs (OpenAI Cookbook)](https://developers.openai.com/cookbook/examples/orchestrating_agents/)
