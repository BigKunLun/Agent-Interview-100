# LangGraph 的核心概念：节点、边、状态

> 难度：基础
> 分类：Frameworks

## 简短回答

LangGraph 是 LangChain 生态中专门用于构建**有状态、多步骤 Agent 工作流**的框架，于 2025 年 10 月发布 1.0 稳定版，被称为"Agent 编排的 React"。其核心思想是将 Agent 的执行流程建模为一个**有向图（Graph）**，包含三个基本构建块：(1) **State（状态）**——图的共享记忆，是一个 TypedDict，所有节点通过读写 State 来传递信息，支持 Reducer 函数自定义状态合并逻辑（如消息列表追加而非覆盖）；(2) **Node（节点）**——图中的"工作单元"，每个节点是一个 Python 函数，接收当前 State 并返回 State 更新（可以是 LLM 调用、工具执行、条件判断等任何操作）；(3) **Edge（边）**——连接节点的"路由器"，分为普通边（固定路由）和条件边（基于 State 动态路由，实现分支逻辑）。LangGraph 还提供 **Checkpointer（检查点）** 实现状态持久化，支持 Human-in-the-Loop（`interrupt()` 暂停等待人工审批）、错误恢复、长时间任务。执行模型是"超步骤（Super-step）"——每个超步骤中并行执行所有就绪节点，然后同步 State，直到到达 END 节点。

## 详细解析

### 核心架构图

```
LangGraph 核心概念：

┌─────────────────────────────────────────┐
│              StateGraph                  │
│                                          │
│  State (共享状态 - TypedDict)            │
│  ┌─────────────────────────────────┐    │
│  │ messages: list[BaseMessage]     │    │
│  │ current_step: str               │    │
│  │ tool_results: dict              │    │
│  └─────────────────────────────────┘    │
│                                          │
│  Nodes (节点 - Python 函数)              │
│  ┌──────┐    ┌──────┐    ┌──────┐      │
│  │ plan │───→│ act  │───→│ check│      │
│  └──────┘    └──────┘    └──┬───┘      │
│                              │          │
│  Edges (边 - 路由逻辑)       ▼          │
│              ┌──────────────────┐       │
│              │ 条件边：         │       │
│              │ done? → END      │       │
│              │ retry? → act     │       │
│              └──────────────────┘       │
│                                          │
│  Checkpointer (持久化)                   │
│  ├── MemorySaver (开发)                  │
│  ├── PostgresSaver (生产)                │
│  └── 支持 interrupt() + resume           │
└─────────────────────────────────────────┘
```

### State：图的共享记忆

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# 定义 State：所有节点共享的数据结构
class AgentState(TypedDict):
    # Annotated + add_messages = Reducer 模式
    # 新消息追加到列表，而非覆盖
    messages: Annotated[list[BaseMessage], add_messages]

    # 普通字段：新值直接覆盖旧值
    current_task: str
    step_count: int
    final_answer: str

# State 设计原则：
# 1. 只放需要跨节点共享的数据
# 2. 用 Reducer 处理需要"累加"的字段（如消息历史）
# 3. 默认行为是覆盖（最后写入的值生效）
# 4. State 是不可变的——节点返回更新字典，由框架合并

# Reducer 示例：自定义合并逻辑
def merge_tool_results(existing: dict, new: dict) -> dict:
    """工具结果合并：新结果追加到已有结果"""
    merged = {**existing, **new}
    return merged

class AdvancedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    tool_results: Annotated[dict, merge_tool_results]
```

### Node：图中的工作单元

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o")

# 节点 1：调用 LLM
def call_llm(state: AgentState) -> dict:
    """每个节点接收 State，返回 State 更新"""
    response = llm.invoke(state["messages"])
    # 返回的字典会与当前 State 合并
    return {"messages": [response]}

# 节点 2：执行工具
def execute_tools(state: AgentState) -> dict:
    """执行 LLM 请求的工具调用"""
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        result = run_tool(tool_call)
        results.append(result)
    return {"messages": results}

# 节点 3：检查是否完成
def should_continue(state: AgentState) -> str:
    """条件边的路由函数——返回下一个节点的名称"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"      # 有工具调用 → 去执行工具
    else:
        return END           # 无工具调用 → 结束

# 节点的本质：
# - 就是普通的 Python 函数
# - 输入：当前 State
# - 输出：State 更新字典（不是完整 State）
# - 可以包含任何逻辑：LLM 调用、API 请求、数据处理
```

### Edge：连接节点的路由器

```python
# 构建完整的 Graph
graph = StateGraph(AgentState)

# 添加节点
graph.add_node("llm", call_llm)
graph.add_node("tools", execute_tools)

# 普通边：固定路由（A → B）
graph.add_edge(START, "llm")        # 入口 → LLM
graph.add_edge("tools", "llm")      # 工具执行后 → 回到 LLM

# 条件边：动态路由（基于 State 决定去哪）
graph.add_conditional_edges(
    "llm",                           # 从 LLM 节点出发
    should_continue,                  # 路由函数
    {
        "tools": "tools",            # 返回 "tools" → 去 tools 节点
        END: END,                     # 返回 END → 结束
    }
)

# 编译图
app = graph.compile()

# 边的类型：
# 1. 普通边 add_edge(A, B)：A 执行后一定去 B
# 2. 条件边 add_conditional_edges(A, func, mapping)：
#    A 执行后，由 func(state) 返回值决定去哪
# 3. START：虚拟入口节点
# 4. END：虚拟终止节点
```

### Checkpointer：状态持久化与 Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# 开发环境：内存检查点
checkpointer = MemorySaver()

# 生产环境：PostgreSQL 检查点
# checkpointer = PostgresSaver(conn_string="postgresql://...")

# 编译时注入 Checkpointer
app = graph.compile(checkpointer=checkpointer)

# 使用 thread_id 管理对话
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(
    {"messages": [HumanMessage(content="帮我查一下订单")]},
    config=config,
)

# 同一个 thread_id 的后续调用会自动加载历史 State
result2 = app.invoke(
    {"messages": [HumanMessage(content="退款怎么操作？")]},
    config=config,  # 自动携带之前的对话历史
)

# Human-in-the-Loop：interrupt() 暂停等待人工
from langgraph.types import interrupt

def sensitive_action(state: AgentState) -> dict:
    """执行敏感操作前暂停，等待人工确认"""
    approval = interrupt(
        {"question": "是否批准执行此操作？", "details": state["current_task"]}
    )
    if approval == "yes":
        return {"messages": [AIMessage(content="操作已执行")]}
    else:
        return {"messages": [AIMessage(content="操作已取消")]}

# Checkpointer 使能的能力：
# 1. 对话持久化——跨请求保持状态
# 2. 错误恢复——从最后一个检查点重试
# 3. Human-in-the-Loop——暂停/恢复执行
# 4. 时间旅行——回滚到任意历史状态
```

### 完整 ReAct Agent 示例

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴天，25°C"

@tool
def search_flights(origin: str, destination: str) -> str:
    """搜索航班"""
    return f"{origin}→{destination}：找到 3 个航班"

# 一行创建完整的 ReAct Agent
agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_weather, search_flights],
    checkpointer=MemorySaver(),
)

# 执行
result = agent.invoke(
    {"messages": [("human", "我想去北京，先查天气再搜航班")]},
    config={"configurable": {"thread_id": "trip-1"}},
)
```

## 常见误区 / 面试追问

1. **误区："LangGraph 就是 LangChain 的升级版"** — LangGraph 不是 LangChain 的替代品，而是补充。LangChain 处理单步链式调用（Prompt → LLM → Parser），LangGraph 处理多步有状态工作流（循环、分支、人工审核）。简单任务用 LangChain，复杂 Agent 用 LangGraph。

2. **误区："State 就是全局变量"** — State 不是可变的全局变量。节点不能直接修改 State，只能返回更新字典，由框架通过 Reducer 合并。这种设计保证了状态变更的可追踪性和一致性。

3. **追问："LangGraph 的执行模型是什么？"** — "超步骤（Super-step）"模型：每个超步骤中，所有没有未满足依赖的节点并行执行，执行完毕后同步 State，然后进入下一个超步骤。这类似于 Pregel 图计算模型（Google 的大规模图处理框架）。

4. **追问："LangGraph 如何处理错误？"** — 三层错误处理：(1) 节点内 try/catch 处理预期错误；(2) Checkpointer 支持从失败点重试（不丢失已完成步骤）；(3) Graph 级别的 `retry_policy` 配置自动重试策略。结合 `interrupt()` 还可以在错误时暂停并请求人工介入。

## 参考资料

- [LangGraph Overview (Official Docs)](https://docs.langchain.com/oss/python/langgraph/overview)
- [A Beginner's Guide to LangGraph: Core Concepts (Medium)](https://medium.com/@ajaykumargajula7/a-beginners-guide-to-langgraph-understanding-the-core-concepts-bc2b1011d675)
- [Mastering AI Agent Systems with LangGraph in 2025 (Towards AI)](https://pub.towardsai.net/from-single-brains-to-team-intelligence-mastering-ai-agent-systems-with-langgraph-in-2025-3520af4fc758)
- [Understanding Core Concepts of LangGraph Deep Dive (Dev.to)](https://dev.to/raunaklallala/understanding-core-concepts-of-langgraph-deep-dive-1d7h)
- [LangGraph Basics: Understanding State, Schema, Nodes, and Edges (Medium)](https://medium.com/@vivekvjnk/langgraph-basics-understanding-state-schema-nodes-and-edges-77f2fd17cae5)
