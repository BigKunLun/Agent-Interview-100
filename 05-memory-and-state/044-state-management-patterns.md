# 状态管理在 Agent 系统中的设计模式

> 难度：中级
> 分类：Memory & State

## 简短回答

Agent 系统的状态管理决定了"Agent 在任意时刻知道什么、做过什么、下一步该做什么"。核心设计模式包括：**共享状态图（LangGraph 模式）**——用 TypedDict 定义全局状态，通过 Reducer 函数合并并发更新，内置 Checkpointing 实现持久化和恢复；**事件溯源（Event Sourcing）**——记录所有状态变更事件而非最终状态，支持完整回放和审计；**有限状态机（FSM）**——用明确的状态和转移规则管理流程；**黑板模式**——共享空间让多 Agent 自主读写。LangGraph 的 State + Reducer + Checkpoint 模式已成为业界主流，被 Klarna、Replit 等企业用于生产。

## 详细解析

### 为什么 Agent 需要状态管理？

```
无状态 Agent：
  每次 LLM 调用都是独立的 → 无法做多步任务
  工具调用后结果丢失 → 无法基于前序结果决策
  中断后无法恢复 → 长任务不可靠

有状态 Agent：
  状态跨步骤持久化 → 支持复杂多步工作流
  可以暂停和恢复 → 支持 Human-in-the-Loop
  状态可检查 → 支持调试和审计
```

### 模式 1：共享状态图（LangGraph 模式）

LangGraph 的核心——状态是流经图中每个节点的共享内存对象：

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator

# 1. 定义状态 Schema
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]     # 追加模式
    research_data: str                          # 替换模式
    draft: str
    revision_count: int
    status: str

# 2. 节点读取状态、处理、返回更新
def research_node(state: AgentState) -> dict:
    query = state["messages"][-1]["content"]
    data = search(query)
    return {
        "research_data": data,
        "status": "research_complete"
    }
    # LangGraph 自动用 Reducer 合并更新到全局状态

def write_node(state: AgentState) -> dict:
    draft = generate(state["research_data"])
    return {
        "draft": draft,
        "revision_count": state["revision_count"] + 1,
        "status": "draft_complete"
    }

# 3. 条件路由基于状态
def should_revise(state: AgentState) -> str:
    if state["revision_count"] >= 3:
        return "end"
    if quality_score(state["draft"]) < 0.8:
        return "revise"
    return "end"

# 4. 构建图
graph = StateGraph(AgentState)
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_edge(START, "research")
graph.add_edge("research", "write")
graph.add_conditional_edges("write", should_revise)
```

**Reducer 的关键作用：**
```python
# Reducer 决定状态如何更新
class State(TypedDict):
    messages: Annotated[list, operator.add]  # 追加：新消息加到列表末尾
    count: Annotated[int, operator.add]      # 累加：数字相加
    result: str                              # 替换：新值直接覆盖旧值
```

### 模式 2：Checkpointing（持久化与恢复）

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 编译图时附加 Checkpointer
checkpointer = SqliteSaver.from_conn_string("./agent_state.db")
app = graph.compile(checkpointer=checkpointer)

# 每个执行步骤自动保存状态快照
config = {"configurable": {"thread_id": "user_123_session_1"}}
result = app.invoke({"messages": [user_msg]}, config)

# 可以从任意检查点恢复
# 场景：服务重启、Human-in-the-Loop 审批后继续
result = app.invoke(None, config)  # 从上次暂停处继续
```

**Checkpointing 使得以下场景成为可能：**
- 2 小时的 Agent 任务中途 Pod 重启 → 从检查点恢复
- Human-in-the-Loop：暂停等待审批 → 审批后从暂停点继续
- 时间旅行调试：回到第 N 步查看当时的完整状态

### 模式 3：事件溯源（Event Sourcing）

```python
class EventSourcedState:
    """记录所有状态变更事件，支持回放"""

    def __init__(self):
        self.events = []      # 事件日志
        self.current_state = {}  # 当前状态（由事件推导）

    def apply_event(self, event: dict):
        self.events.append({
            **event,
            "timestamp": datetime.now(),
            "sequence": len(self.events)
        })
        self.current_state = self.rebuild()

    def rebuild(self, up_to: int = None):
        """从事件日志重建状态"""
        state = {}
        events = self.events[:up_to] if up_to else self.events
        for event in events:
            if event["type"] == "tool_result":
                state[event["tool"]] = event["result"]
            elif event["type"] == "decision":
                state["last_decision"] = event["decision"]
            elif event["type"] == "handoff":
                state["current_agent"] = event["to_agent"]
        return state

    def replay_to(self, step: int):
        """回放到特定步骤（时间旅行调试）"""
        return self.rebuild(up_to=step)
```

### 模式 4：有限状态机（FSM）

```python
class AgentFSM:
    """用 FSM 管理 Agent 流程状态"""

    TRANSITIONS = {
        "idle":       {"receive_task": "planning"},
        "planning":   {"plan_ready": "executing", "plan_failed": "error"},
        "executing":  {"step_done": "executing", "all_done": "reviewing",
                      "error": "error"},
        "reviewing":  {"approved": "complete", "rejected": "executing"},
        "error":      {"retry": "planning", "abort": "idle"},
        "complete":   {"reset": "idle"},
    }

    def __init__(self):
        self.state = "idle"
        self.history = []

    def transition(self, event: str):
        allowed = self.TRANSITIONS.get(self.state, {})
        if event not in allowed:
            raise InvalidTransition(f"Cannot {event} from {self.state}")
        old_state = self.state
        self.state = allowed[event]
        self.history.append((old_state, event, self.state))
```

### 模式 5：Human-in-the-Loop 状态管理

```python
# LangGraph 的中断与恢复
from langgraph.types import interrupt

def sensitive_action_node(state):
    """需要人工审批的敏感操作"""
    action = plan_action(state)

    # 暂停执行，等待人工审批
    approval = interrupt({
        "action": action,
        "reason": "此操作将修改生产数据库",
        "options": ["approve", "reject", "modify"]
    })

    if approval["decision"] == "approve":
        return execute(action)
    elif approval["decision"] == "modify":
        return execute(approval["modified_action"])
    else:
        return {"status": "rejected"}
```

### 设计原则

| 原则 | 说明 |
|------|------|
| 状态可序列化 | 所有状态必须可以 JSON 序列化（用于持久化和传输） |
| 幂等性 | 从同一检查点重放应产生相同结果 |
| 最小化状态 | 只在状态中保留必要信息，避免膨胀 |
| 版本管理 | Schema 变更时需要迁移策略 |
| 显式 Reducer | 明确定义状态更新规则（追加 vs 替换） |

### 模式选择指南

| 场景 | 推荐模式 |
|------|---------|
| 简单线性流程 | FSM |
| 复杂分支 + 循环 | LangGraph 状态图 |
| 需要审计追踪 | 事件溯源 |
| 多 Agent 协作 | 共享状态 + Reducer |
| 需要暂停/恢复 | Checkpointing |
| 高风险操作 | Human-in-the-Loop + Checkpoint |

## 常见误区 / 面试追问

1. **误区："全局变量就是状态管理"** — Agent 的状态需要持久化、可恢复、可审计。用全局变量管理状态在 Pod 重启后丢失、无法调试、不支持并发。需要 Checkpointer 或事件溯源等正式机制。

2. **误区："状态越详细越好"** — 状态越大，序列化/反序列化越慢，Checkpoint 占用越多存储。只保留决策所需的最小信息集。中间推理过程应该在 Trace 中记录而非状态中保存。

3. **追问："LangGraph 的 Reducer 和 Redux 的 Reducer 有什么关系？"** — 概念相同——都是纯函数，接收当前状态和更新，返回新状态。LangGraph 用 Python 的 `Annotated` 类型标注来声明 Reducer 规则（如 `operator.add` 表示追加）。

4. **追问："Checkpointing 对性能的影响？"** — 每个 super-step 后保存检查点会增加延迟。优化方案：(1) 使用异步写入；(2) 只在关键步骤做 Checkpoint；(3) 用 Redis 替代 SQLite 做高性能 Checkpoint。

## 参考资料

- [LangGraph State Machines: Managing Complex Agent Task Flows (DEV)](https://dev.to/jamesli/langgraph-state-machines-managing-complex-agent-task-flows-in-production-36f4)
- [LangGraph State Management Best Practices (Medium)](https://medium.com/@bharatraj1918/langgraph-state-management-part-1-how-langgraph-manages-state-for-multi-agent-workflows-da64d352c43b)
- [Production Multi-Agent System with LangGraph: Checkpointing & Error Recovery](https://markaicode.com/langgraph-production-agent/)
- [Agentic Design Patterns: The 2026 Guide (SitePoint)](https://www.sitepoint.com/the-definitive-guide-to-agentic-design-patterns-in-2026/)
- [Mastering LangGraph State Management in 2025 (SparkCo)](https://sparkco.ai/blog/mastering-langgraph-state-management-in-2025)
