# 多 Agent 通信模式：消息传递、共享状态、黑板模式

> 难度：基础
> 分类：Multi-Agent

## 简短回答

多 Agent 通信有三种基本模式：**消息传递（Message Passing）**——Agent 间通过点对点或广播方式直接交换结构化消息，适合动态、针对性的信息共享；**共享状态（Shared State）**——所有 Agent 读写同一个状态存储（如数据库、Key-Value Store），提供一致的全局视图，但可能成为瓶颈；**黑板模式（Blackboard）**——Agent 围绕一个共享的"黑板"协作，各自观察黑板上的信息，基于自身专长决定是否贡献输出。研究表明黑板模式在端到端任务成功率上比传统模式提升 13%-57%，同时具有更好的 token 效率。

## 详细解析

### 模式 1：消息传递（Message Passing）

Agent 之间通过直接发送消息通信，类似人类之间发消息：

```python
# 点对点消息传递
class MessageBus:
    def __init__(self):
        self.queues = {}  # agent_id → message_queue

    async def send(self, from_agent: str, to_agent: str, message: dict):
        """发送消息给特定 Agent"""
        msg = {
            "from": from_agent,
            "to": to_agent,
            "content": message,
            "timestamp": datetime.now()
        }
        await self.queues[to_agent].put(msg)

    async def broadcast(self, from_agent: str, message: dict):
        """广播消息给所有 Agent"""
        for agent_id, queue in self.queues.items():
            if agent_id != from_agent:
                await queue.put({"from": from_agent, "content": message})

# 使用示例
bus = MessageBus()
await bus.send("researcher", "writer",
    {"type": "research_complete", "data": findings})
```

**优势：** 针对性强、延迟低、Agent 间解耦
**劣势：** 可能形成信息孤岛、路由逻辑复杂、难以保证全局一致性

**生产实现：** 通常使用消息队列（RabbitMQ、Redis Streams）实现可靠的异步消息传递。

### 模式 2：共享状态（Shared State）

所有 Agent 读写同一个全局状态，类似多人编辑同一个文档：

```python
# 共享状态模式（LangGraph 风格）
from typing import TypedDict, Annotated

class SharedState(TypedDict):
    messages: list[str]
    research_data: dict
    draft: str
    review_comments: list[str]
    status: str

# 每个 Agent 读取状态、处理、更新状态
def researcher_node(state: SharedState) -> SharedState:
    # 读取当前状态
    query = state["messages"][-1]
    # 执行研究
    data = search(query)
    # 更新共享状态
    return {"research_data": data, "status": "research_complete"}

def writer_node(state: SharedState) -> SharedState:
    # 读取研究数据（由 researcher 写入）
    data = state["research_data"]
    # 生成草稿
    draft = generate_draft(data)
    return {"draft": draft, "status": "draft_complete"}
```

**优势：** 全局一致视图、状态可持久化、便于调试（检查状态快照）
**劣势：** 可能成为吞吐瓶颈、需要锁机制防止竞态条件、单点故障风险

**并发控制：**
```python
# 防止竞态条件
import asyncio

class SafeSharedState:
    def __init__(self):
        self.state = {}
        self.lock = asyncio.Lock()

    async def update(self, agent_id: str, updates: dict):
        async with self.lock:  # 确保原子更新
            self.state.update(updates)
            self.state["last_updated_by"] = agent_id
```

### 模式 3：黑板模式（Blackboard）

经典 AI 架构模式，Agent 围绕共享"黑板"自主协作：

```python
class Blackboard:
    def __init__(self):
        self.public_space = []   # 公共区域：所有 Agent 可见
        self.private_spaces = {} # 私有区域：各 Agent 独立空间

    def post(self, agent_id: str, content: dict, visibility="public"):
        entry = {
            "author": agent_id,
            "content": content,
            "timestamp": datetime.now()
        }
        if visibility == "public":
            self.public_space.append(entry)
        else:
            self.private_spaces.setdefault(agent_id, []).append(entry)

    def read(self, agent_id: str) -> list:
        """Agent 读取黑板上与自己相关的内容"""
        return self.public_space + self.private_spaces.get(agent_id, [])

class BlackboardSystem:
    def __init__(self, blackboard, agents, controller):
        self.blackboard = blackboard
        self.agents = agents      # 专家 Agent 列表
        self.controller = controller  # 控制器（决定谁行动）

    async def run(self, task):
        self.blackboard.post("system", {"task": task})

        while not self.is_solved():
            # 控制器选择下一个行动的 Agent
            active_agent = self.controller.select(
                self.blackboard, self.agents
            )
            # Agent 读取黑板、处理、写回结果
            result = await active_agent.process(self.blackboard.read(active_agent.id))
            self.blackboard.post(active_agent.id, result)
```

**黑板模式的关键特性：**
- **自愿参与**：Agent 基于自身专长决定是否响应黑板上的请求
- **迭代精化**：多轮读取-处理-写回，逐步完善解决方案
- **去中心化**：不需要协调器预先知道所有 Agent 的能力
- **Token 高效**：共享公共记忆替代个体记忆，减少重复

**研究结果：** 黑板架构在端到端任务成功率上比 RAG 和 Master-Slave 模式提升 13%-57%。

### 三种模式对比

| 维度 | 消息传递 | 共享状态 | 黑板模式 |
|------|---------|---------|---------|
| 通信方式 | 点对点/广播 | 读写全局存储 | 读写共享空间 |
| 耦合度 | 低 | 中 | 低 |
| 扩展性 | 高 | 中（可能瓶颈） | 高 |
| 一致性 | 最终一致 | 强一致 | 最终一致 |
| Token 效率 | 低（重复传递） | 中 | 高（共享上下文） |
| 调试难度 | 高（追踪消息流） | 低（检查状态） | 中 |
| 典型框架 | AutoGen | LangGraph | LbMAS、Terrarium |

### 实际选择指南

```python
def choose_pattern(scenario):
    if scenario == "少量 Agent + 简单流程":
        return "共享状态（LangGraph State）"
    elif scenario == "Agent 数量多 + 动态组合":
        return "黑板模式"
    elif scenario == "异步处理 + 微服务架构":
        return "消息传递（消息队列）"
    elif scenario == "需要最大灵活性":
        return "混合模式（消息传递 + 共享状态）"
```

## 常见误区 / 面试追问

1. **误区："消息传递就是聊天"** — 多 Agent 通信中的消息传递是结构化的、带类型的数据交换，不是自然语言对话。消息应包含类型、发送者、接收者、数据负载和时间戳。

2. **误区："共享状态很简单，用全局变量就行"** — 生产环境中共享状态需要锁机制、版本控制和持久化。多 Agent 并发读写同一状态时，必须防止竞态条件。

3. **追问："黑板模式和共享状态有什么本质区别？"** — 共享状态是被动的数据存储，Agent 按预定流程读写；黑板模式是主动的协作平台，Agent 自主观察黑板内容并决定是否参与。黑板有控制器决定执行顺序，Agent 有自愿参与的机制。

4. **追问："如何处理 Agent 间的信息冗余？"** — 研究显示 token 冗余率可达 53-86%。解决方案包括：消息摘要（传递摘要而非原文）、黑板的公共记忆（替代个体记忆）、去重机制。

## 参考资料

- [LLM-based Multi-Agent Blackboard System (arXiv)](https://arxiv.org/html/2510.01285v1)
- [Shared Awareness & Coordination in Multi-Agent Systems (APXML)](https://apxml.com/courses/agentic-llm-memory-architectures/chapter-5-multi-agent-systems/shared-awareness-coordination)
- [Terrarium: Revisiting the Blackboard for Multi-Agent (arXiv)](https://www.arxiv.org/pdf/2510.14312)
- [Implementing Multi-Agent Systems: Architecture Patterns (21medien)](https://www.21medien.de/en/blog/implementing-multi-agent-systems)
- [Multi-Agent Collaboration Mechanisms: A Survey (arXiv)](https://arxiv.org/html/2501.06322v1)
