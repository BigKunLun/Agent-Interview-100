# A2A（Agent-to-Agent）协议是什么？它与 MCP 有何区别？

> 难度：中级
> 分类：Multi-Agent

## 简短回答

**A2A（Agent-to-Agent）** 是 Google 于 2025 年 4 月发布的开放协议，旨在让不同厂商、不同框架构建的 AI Agent 之间实现标准化通信与协作。协议基于 **HTTP + JSON-RPC 2.0 + SSE** 构建，核心概念包括 **Agent Card**（服务发现）、**Task**（任务生命周期管理）、**Message** 和 **Artifact**（交互载体）。与 Anthropic 主导的 **MCP（Model Context Protocol）** 不同，MCP 解决的是 Agent 与 Tool 之间的纵向能力扩展（"Agent 如何调用工具"），A2A 解决的是 Agent 与 Agent 之间的横向协作（"Agent 如何委托另一个 Agent"）。两者定位互补——一个 Agent 可以同时用 MCP 连接工具、用 A2A 与其他 Agent 协作。

## 详细解析

### A2A 的诞生背景

随着 AI Agent 在企业中大规模部署，不同团队、不同厂商构建的 Agent 之间无法直接对话，形成了"Agent 孤岛"。

```
┌─────────────┐     ╳     ┌─────────────┐     ╳     ┌─────────────┐
│  LangChain  │  不兼容   │  CrewAI     │  不兼容   │  AutoGen    │
│  Agent      │◄────────►│  Agent      │◄────────►│  Agent      │
└─────────────┘           └─────────────┘           └─────────────┘
                          引入 A2A 后 ↓
┌─────────────┐   A2A    ┌─────────────┐   A2A    ┌─────────────┐
│  LangChain  │◄────────►│  CrewAI     │◄────────►│  AutoGen    │
│  Agent      │  JSON-RPC │  Agent      │  JSON-RPC │  Agent      │
└─────────────┘           └─────────────┘           └─────────────┘
```

Google 联合 Atlassian、Salesforce、SAP 等 50+ 企业伙伴推出 A2A，目标是成为 Agent 间通信的"HTTP"——与框架无关、与模型无关的开放标准。

### 协议层次架构

```
┌───────────┬──────────────────────────────────────┐
│  应用层    │  Agent Card / Task / Message / Artifact │
├───────────┼──────────────────────────────────────┤
│  消息格式  │  JSON-RPC 2.0                         │
├───────────┼──────────────────────────────────────┤
│  实时推送  │  SSE（Server-Sent Events）             │
├───────────┼──────────────────────────────────────┤
│  传输层    │  HTTP / HTTPS                         │
├───────────┼──────────────────────────────────────┤
│  安全层    │  OAuth 2.0 / API Key / JWT            │
└───────────┴──────────────────────────────────────┘
```

### 核心概念详解

#### 1. Agent Card（服务发现）

每个 A2A Agent 在 `/.well-known/agent.json` 发布 Agent Card，声明能力和认证方式，类似 OpenAPI spec。

#### 2. Task 生命周期

Task 是协议核心工作单元，有明确的状态机：

```
  ┌───────────┐    ┌───────────┐    ┌───────────┐
  │ submitted │───►│  working  │───►│ completed │
  │  (已提交)  │    │  (执行中)  │    │  (已完成)  │
  └───────────┘    └─────┬─────┘    └───────────┘
                         │
                         ▼
                  ┌──────────────┐    ┌───────────┐
                  │input-required│───►│  failed   │
                  │ (需要输入)    │    │  (失败)   │
                  └──────┬───────┘    └───────────┘
                         │ 补充输入后
                         └──────────► working ──► completed / failed
```

#### 3. Message 和 Artifact

**Message** 是 Agent 间的对话载体（含 text、file 等多模态 parts）；**Artifact** 是任务产出的结构化成果物，与 Message 分离，便于下游消费。

### A2A 与 MCP 的定位差异

```
┌──────────────────────────────────────────────────────┐
│   Agent A                          Agent B           │
│  ┌─────────┐      A2A 协议       ┌─────────┐        │
│  │ Client  │◄═══════════════════►│ Server  │        │
│  │ Agent   │  Agent↔Agent 横向    │ Agent   │        │
│  └────┬────┘                     └────┬────┘        │
│       │ MCP                           │ MCP          │
│       ▼                               ▼             │
│  ┌─────────┐                     ┌─────────┐        │
│  │ 数据库   │                     │ 搜索引擎 │        │
│  │ API 工具 │                     │ 代码执行 │        │
│  └─────────┘                     └─────────┘        │
└──────────────────────────────────────────────────────┘
```

| 维度 | A2A | MCP |
|------|-----|-----|
| **发起者** | Google（2025.04） | Anthropic（2024.11） |
| **解决问题** | Agent 之间如何协作 | Agent 如何连接工具/数据源 |
| **关系模型** | 对等/委托（Agent↔Agent） | 主从（Agent→Tool） |
| **通信方式** | HTTP + JSON-RPC 2.0 + SSE | JSON-RPC 2.0 + stdio/HTTP+SSE |
| **服务发现** | Agent Card（/.well-known/agent.json） | 能力协商（initialize 握手） |
| **状态管理** | 有状态（Task 生命周期） | 无状态（单次调用） |
| **类比** | 公司之间的合作协议 | 员工使用办公软件 |

一句话概括：**MCP 让 Agent 变得更强（纵向），A2A 让 Agent 之间能合作（横向）**。

### Python 代码示例：A2A 交互模拟

```python
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum


class TaskState(Enum):
    """A2A Task 状态枚举"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentCard:
    """Agent Card — 声明 Agent 的能力，发布于 /.well-known/agent.json"""
    name: str
    description: str
    url: str
    skills: list[dict] = field(default_factory=list)
    capabilities: dict = field(default_factory=lambda: {
        "streaming": True, "pushNotifications": False,
    })

    def to_json(self) -> str:
        return json.dumps(vars(self), ensure_ascii=False, indent=2)


@dataclass
class Task:
    """A2A Task — 协议核心工作单元，带状态机校验"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TaskState = TaskState.SUBMITTED
    messages: list[dict] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)

    # 合法状态转换表
    _transitions = {
        TaskState.SUBMITTED: {TaskState.WORKING, TaskState.FAILED},
        TaskState.WORKING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.INPUT_REQUIRED},
        TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.FAILED},
    }

    def transition(self, new_state: TaskState):
        allowed = self._transitions.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(f"非法状态转换: {self.state.value} → {new_state.value}")
        self.state = new_state


class A2AServer:
    """模拟 A2A 服务端——接收 JSON-RPC 请求、管理 Task 生命周期"""

    def __init__(self, card: AgentCard):
        self.card = card
        self.tasks: dict[str, Task] = {}

    def handle_request(self, request: dict) -> dict:
        """路由 JSON-RPC 2.0 请求"""
        method = request["method"]
        params = request.get("params", {})
        req_id = request["id"]

        if method == "tasks/send":
            result = self._send(params)
        elif method == "tasks/get":
            task = self.tasks[params["id"]]
            result = {"id": task.id, "state": task.state.value}
        else:
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32601, "message": "方法不存在"}}

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _send(self, params: dict) -> dict:
        task_id = params.get("id", str(uuid.uuid4()))
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.messages.append(params["message"])
            task.transition(TaskState.WORKING)  # 从 input-required 恢复
        else:
            task = Task(id=task_id, messages=[params["message"]])
            self.tasks[task_id] = task
            task.transition(TaskState.WORKING)

        # 模拟业务逻辑：缺信息则请求补充，否则完成
        text = task.messages[-1]["parts"][0]["text"]
        if "分析" in text and "时间" not in text:
            task.transition(TaskState.INPUT_REQUIRED)
            return {"id": task.id, "state": task.state.value,
                    "messages": [{"role": "agent",
                                  "parts": [{"type": "text", "text": "请指定分析时间范围"}]}]}

        task.transition(TaskState.COMPLETED)
        task.artifacts.append({
            "name": "report",
            "parts": [{"type": "text", "text": f"基于 [{text}] 的分析报告..."}]
        })
        return {"id": task.id, "state": task.state.value, "artifacts": task.artifacts}


# ─── 演示：两轮交互（submitted → input-required → completed）───
if __name__ == "__main__":
    server = A2AServer(AgentCard(
        name="财务分析 Agent",
        description="专精于财务报表分析",
        url="https://finance-agent.example.com",
        skills=[{"id": "financial_analysis", "name": "财务报表分析"}],
    ))
    print("=== Agent Card ===")
    print(server.card.to_json())

    # 第一轮：缺少时间范围 → input-required
    r1 = server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tasks/send",
        "params": {"id": "task-001",
                   "message": {"role": "user",
                               "parts": [{"type": "text", "text": "分析特斯拉营收趋势"}]}}
    })
    print(f"\n第一轮状态: {r1['result']['state']}")  # input-required

    # 第二轮：补充时间 → completed
    r2 = server.handle_request({
        "jsonrpc": "2.0", "id": 2, "method": "tasks/send",
        "params": {"id": "task-001",
                   "message": {"role": "user",
                               "parts": [{"type": "text", "text": "时间范围：2023-2024"}]}}
    })
    print(f"第二轮状态: {r2['result']['state']}")  # completed
    print(f"产出: {r2['result']['artifacts'][0]['parts'][0]['text']}")
```

### A2A 的关键设计原则

| 原则 | 说明 |
|------|------|
| **不透明执行** | 客户端不需要了解服务端内部实现（模型、框架均可不同） |
| **框架无关** | LangChain、CrewAI、AutoGen 等均可实现 A2A |
| **能力协商** | 通过 Agent Card 动态发现和选择合适的 Agent |
| **安全优先** | 内置企业级认证授权（OAuth 2.0, API Key, mTLS） |

## 常见误区 / 面试追问

1. **误区："A2A 会取代 MCP"** — A2A 和 MCP 解决的是两个正交问题。MCP 是 Agent 连接工具和数据源的标准（纵向扩展能力），A2A 是 Agent 之间协作通信的标准（横向建立合作）。一个成熟的 Agent 系统往往同时需要两者：用 MCP 获取能力，用 A2A 实现分工协作。它们更像是 TCP 和 HTTP 的关系——不同层次、互相配合。

2. **误区："A2A 只适用于同构 Agent"** — 恰恰相反，A2A 的核心设计目标就是让异构 Agent 互通。无论 Agent 基于 Claude、GPT 还是 Gemini，用 LangChain 还是自研框架，只要实现 A2A 协议就能互相协作。Agent Card 中的 skills 是语义级描述，而非实现级绑定。

3. **追问："A2A 如何处理长时间运行的任务？"** — 三种机制：(1) **SSE 流式推送**——通过 `tasks/sendSubscribe` 实时推送中间状态和增量结果；(2) **Push Notification**——客户端注册 webhook 回调，服务端在状态变更时主动通知，适合数小时级任务；(3) **Task 状态轮询**——通过 `tasks/get` 随时查询进度。三种机制覆盖秒级到天级的各种任务时长。

4. **追问："如何在 A2A 中实现身份验证和授权？"** — Agent Card 的 `authentication` 字段声明支持的认证方案。协议复用成熟的 Web 安全标准：**OAuth 2.0**（企业间协作首选）、**API Key**（内部系统）、**JWT Bearer Token**（无状态场景）。同时 Agent Card 的 `skills` 可设置权限级别，实现细粒度能力授权——如允许"查询"但禁止"修改"。

## 参考资料

- [A2A Protocol Specification (Google)](https://google.github.io/A2A/)
- [Announcing the Agent2Agent Protocol (Google Cloud Blog)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [A2A vs MCP: Understanding AI Agent Protocols (Composio)](https://composio.dev/blog/a2a-vs-mcp/)
- [A2A GitHub Repository (google/A2A)](https://github.com/google/A2A)
- [Model Context Protocol Specification (Anthropic)](https://modelcontextprotocol.io/)
