# 如何设计可测试、可扩展的 Agent 框架抽象层？

> 难度：高级
> 分类：Frameworks

## 简短回答

设计可测试、可扩展的 Agent 框架抽象层，核心是将 Agent 系统分解为**职责清晰、边界明确、可独立替换**的模块，使每个模块可以被单独测试和扩展。关键架构原则：(1) **端口-适配器架构（Hexagonal Architecture）**——业务逻辑（Agent Loop、推理策略）在核心层，外部依赖（LLM API、工具、存储）通过接口（Port）隔离，具体实现（Adapter）可插拔替换；(2) **依赖注入（DI）**——LLM Client、Tool Registry、State Store 等通过构造函数注入，测试时可注入 Mock；(3) **中间件模式**——日志、追踪、安全检查等横切关注点通过可组合的中间件实现，不污染核心逻辑；(4) **策略模式**——推理策略（ReAct/Plan-Execute/CoT）、路由策略（模型选择）、重试策略等可插拔；(5) **事件驱动**——Agent 执行过程发出事件（on_llm_call、on_tool_use、on_step_complete），观测和扩展通过事件监听实现。测试策略：单元测试 Mock LLM 响应 → 集成测试用真实 API → 端到端测试验证完整工作流。OpenAI Agents SDK 的"最小抽象"和 LangChain 1.0 的"中间件架构"都是这些原则的实际体现。

## 详细解析

### 分层架构设计

```
可测试 Agent 框架的分层架构：

┌─────────────────────────────────────────────┐
│            应用层 (Application)              │
│  具体的业务 Agent（客服、代码助手等）        │
├─────────────────────────────────────────────┤
│            编排层 (Orchestration)            │
│  Agent Loop / 多 Agent 协调 / 工作流引擎    │
├─────────────────────────────────────────────┤
│            核心层 (Core)                     │
│  接口定义 / 数据模型 / 事件系统             │
├─────────────────────────────────────────────┤
│            适配器层 (Adapters)               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ LLM     │ │ Tools   │ │ Storage │      │
│  │ Adapter │ │ Adapter │ │ Adapter │      │
│  └─────────┘ └─────────┘ └─────────┘      │
├─────────────────────────────────────────────┤
│            中间件层 (Middleware)             │
│  Logging / Tracing / Guardrails / Caching  │
└─────────────────────────────────────────────┘

关键原则：
- 依赖方向：外层依赖内层，内层不依赖外层
- 核心层只定义接口（Protocol），不包含实现
- 适配器可独立替换，不影响其他层
```

### 核心接口定义（Port）

```python
from abc import ABC, abstractmethod
from typing import Protocol, AsyncIterator
from dataclasses import dataclass

# ===== 数据模型 =====
@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list["ToolCall"] | None = None

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    success: bool

# ===== 核心接口（Port）=====

class LLMPort(Protocol):
    """LLM 接口——所有 LLM 实现必须满足此协议"""
    async def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> Message: ...

    async def stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[str]: ...

class ToolPort(Protocol):
    """工具接口"""
    @property
    def name(self) -> str: ...
    @property
    def schema(self) -> dict: ...
    async def execute(self, arguments: dict) -> ToolResult: ...

class StatePort(Protocol):
    """状态存储接口"""
    async def load(self, session_id: str) -> dict | None: ...
    async def save(self, session_id: str, state: dict) -> None: ...
    async def delete(self, session_id: str) -> None: ...

class MiddlewarePort(Protocol):
    """中间件接口"""
    async def before_llm_call(self, messages: list[Message]) -> list[Message]: ...
    async def after_llm_call(self, response: Message) -> Message: ...
    async def before_tool_call(self, tool_call: ToolCall) -> ToolCall: ...
    async def after_tool_call(self, result: ToolResult) -> ToolResult: ...
```

### 可插拔的适配器实现

```python
# ===== LLM 适配器 =====

class OpenAIAdapter:
    """OpenAI LLM 适配器"""
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(self, messages, tools=None):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.to_openai_format() for m in messages],
            tools=tools,
        )
        return Message.from_openai(response.choices[0].message)


class AnthropicAdapter:
    """Anthropic LLM 适配器"""
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self.model = model

    async def complete(self, messages, tools=None):
        response = await self.client.messages.create(
            model=self.model,
            messages=[m.to_anthropic_format() for m in messages],
            tools=tools,
        )
        return Message.from_anthropic(response)


class MockLLMAdapter:
    """Mock LLM 适配器——用于测试"""
    def __init__(self, responses: list[Message]):
        self.responses = iter(responses)

    async def complete(self, messages, tools=None):
        return next(self.responses)

# ===== 状态存储适配器 =====

class InMemoryStateStore:
    """内存状态存储——用于测试和开发"""
    def __init__(self):
        self._store = {}

    async def load(self, session_id):
        return self._store.get(session_id)

    async def save(self, session_id, state):
        self._store[session_id] = state

class RedisStateStore:
    """Redis 状态存储——用于生产"""
    def __init__(self, redis_url: str):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url)

    async def load(self, session_id):
        data = await self.redis.get(f"agent:state:{session_id}")
        return json.loads(data) if data else None

    async def save(self, session_id, state):
        await self.redis.set(f"agent:state:{session_id}", json.dumps(state))
```

### Agent Loop 与依赖注入

```python
class AgentLoop:
    """核心 Agent 执行循环——通过依赖注入实现可测试性"""

    def __init__(
        self,
        llm: LLMPort,                          # 注入 LLM
        tools: list[ToolPort] = None,           # 注入工具
        state_store: StatePort = None,          # 注入状态存储
        middlewares: list[MiddlewarePort] = None,# 注入中间件
        max_steps: int = 10,
        strategy: "ReasoningStrategy" = None,   # 注入推理策略
    ):
        self.llm = llm
        self.tool_registry = {t.name: t for t in (tools or [])}
        self.state_store = state_store or InMemoryStateStore()
        self.middlewares = middlewares or []
        self.max_steps = max_steps
        self.strategy = strategy or ReactStrategy()
        self.event_bus = EventBus()  # 事件系统

    async def run(self, user_input: str, session_id: str = None) -> str:
        """执行 Agent"""
        # 加载或初始化状态
        state = await self.state_store.load(session_id) if session_id else None
        messages = state["messages"] if state else []
        messages.append(Message(role="user", content=user_input))

        for step in range(self.max_steps):
            # 中间件：前处理
            for mw in self.middlewares:
                messages = await mw.before_llm_call(messages)

            # 发出事件
            self.event_bus.emit("before_llm_call", {"step": step, "messages": messages})

            # 调用 LLM
            tool_schemas = [t.schema for t in self.tool_registry.values()]
            response = await self.llm.complete(messages, tools=tool_schemas or None)

            # 中间件：后处理
            for mw in self.middlewares:
                response = await mw.after_llm_call(response)

            messages.append(response)
            self.event_bus.emit("after_llm_call", {"step": step, "response": response})

            # 检查是否需要工具调用
            if not response.tool_calls:
                # 保存状态
                if session_id:
                    await self.state_store.save(session_id, {"messages": messages})
                return response.content

            # 执行工具
            for tc in response.tool_calls:
                tool = self.tool_registry[tc.name]
                result = await tool.execute(tc.arguments)
                messages.append(Message(role="tool", content=result.content))
                self.event_bus.emit("tool_executed", {"tool": tc.name, "result": result})

        return "达到最大步数限制"
```

### 中间件模式

```python
# 中间件：横切关注点的可组合实现

class LoggingMiddleware:
    """日志中间件"""
    async def before_llm_call(self, messages):
        logger.info(f"LLM 调用：{len(messages)} 条消息")
        return messages

    async def after_llm_call(self, response):
        logger.info(f"LLM 响应：{response.content[:100]}...")
        return response

    async def before_tool_call(self, tool_call):
        logger.info(f"工具调用：{tool_call.name}")
        return tool_call

    async def after_tool_call(self, result):
        logger.info(f"工具结果：{result.success}")
        return result


class GuardrailMiddleware:
    """安全护栏中间件"""
    async def before_llm_call(self, messages):
        last_msg = messages[-1].content
        if self.detect_injection(last_msg):
            raise SecurityError("检测到 Prompt 注入")
        return messages

    async def after_llm_call(self, response):
        if self.contains_pii(response.content):
            response.content = self.redact_pii(response.content)
        return response


class CachingMiddleware:
    """缓存中间件"""
    def __init__(self, cache: dict = None):
        self.cache = cache or {}

    async def before_llm_call(self, messages):
        key = self._hash(messages)
        if key in self.cache:
            raise CacheHit(self.cache[key])  # 跳过 LLM 调用
        return messages

    async def after_llm_call(self, response):
        # 缓存响应
        return response


# 中间件组合：按顺序执行
agent = AgentLoop(
    llm=OpenAIAdapter(),
    middlewares=[
        GuardrailMiddleware(),   # 1. 先检查安全
        CachingMiddleware(),     # 2. 再检查缓存
        LoggingMiddleware(),     # 3. 最后记录日志
    ],
)
```

### 测试策略

```python
import pytest

class TestAgentLoop:
    """Agent Loop 的单元测试——使用 Mock"""

    @pytest.mark.asyncio
    async def test_simple_response(self):
        """测试：LLM 直接返回答案（无工具调用）"""
        mock_llm = MockLLMAdapter(responses=[
            Message(role="assistant", content="答案是 42"),
        ])
        agent = AgentLoop(llm=mock_llm)
        result = await agent.run("生命的意义是什么？")
        assert result == "答案是 42"

    @pytest.mark.asyncio
    async def test_tool_call_flow(self):
        """测试：LLM 调用工具后给出答案"""
        mock_llm = MockLLMAdapter(responses=[
            # 第一轮：请求调用工具
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id="1", name="search", arguments={"q": "test"})]),
            # 第二轮：基于工具结果给出答案
            Message(role="assistant", content="搜索结果是..."),
        ])
        mock_tool = MockTool(name="search", result="找到 3 条结果")

        agent = AgentLoop(llm=mock_llm, tools=[mock_tool])
        result = await agent.run("搜索 test")
        assert "搜索结果" in result

    @pytest.mark.asyncio
    async def test_max_steps_limit(self):
        """测试：达到最大步数时停止"""
        # 模拟 LLM 无限循环调用工具
        mock_llm = MockLLMAdapter(responses=[
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id=str(i), name="search", arguments={})])
            for i in range(20)
        ])
        agent = AgentLoop(llm=mock_llm, tools=[MockTool("search")], max_steps=3)
        result = await agent.run("test")
        assert "最大步数" in result

    @pytest.mark.asyncio
    async def test_state_persistence(self):
        """测试：状态在会话间持久化"""
        store = InMemoryStateStore()
        mock_llm = MockLLMAdapter(responses=[
            Message(role="assistant", content="你好！"),
        ])
        agent = AgentLoop(llm=mock_llm, state_store=store)
        await agent.run("hi", session_id="s1")

        state = await store.load("s1")
        assert len(state["messages"]) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_middleware_execution_order(self):
        """测试：中间件按正确顺序执行"""
        order = []

        class TrackingMiddleware:
            def __init__(self, name):
                self.name = name
            async def before_llm_call(self, msgs):
                order.append(f"{self.name}:before")
                return msgs
            async def after_llm_call(self, resp):
                order.append(f"{self.name}:after")
                return resp

        agent = AgentLoop(
            llm=MockLLMAdapter([Message(role="assistant", content="ok")]),
            middlewares=[TrackingMiddleware("A"), TrackingMiddleware("B")],
        )
        await agent.run("test")
        assert order == ["A:before", "B:before", "A:after", "B:after"]
```

## 常见误区 / 面试追问

1. **误区："可测试性意味着 100% 覆盖率"** — Agent 系统中，LLM 输出的非确定性使得 100% 确定性测试不现实。正确的测试策略是分层的：单元测试 Mock LLM（验证逻辑）→ 集成测试用真实 API（验证集成）→ 评估测试用 LLM-as-Judge（验证质量）。每层覆盖不同的风险。

2. **误区："接口越多越好"** — 过度抽象和不够抽象一样有害。只为真正需要替换的组件定义接口。如果你的项目永远不会换 LLM 提供商，一个简单的封装函数就够了，不需要 Port/Adapter 全套架构。**YAGNI 原则同样适用于框架设计**。

3. **追问："如何处理 LLM 响应的非确定性测试？"** — 三种方法：(1) **Mock**——固定 LLM 输出，测试围绕它的逻辑；(2) **快照测试**——记录真实 LLM 输出作为基线，后续对比偏差；(3) **属性测试**——不测具体输出内容，测输出的属性（如"返回的 JSON 必须包含 name 字段"、"不包含 PII"）。

4. **追问："事件驱动 vs 回调 vs 中间件，怎么选？"** — (1) **回调（Callback）**适合简单的钩子（on_start/on_end）；(2) **中间件（Middleware）**适合可组合的请求/响应处理管线；(3) **事件驱动（Event Bus）**适合松耦合的观测和扩展，多个监听器互不影响。生产框架通常三者结合：中间件处理核心管线，事件驱动处理观测，回调提供简单的用户扩展点。

## 参考资料

- [Building Extensible AI Agents: Middleware Patterns (FlowHunt)](https://www.flowhunt.io/blog/building-extensible-ai-agents-with-langchain-1-0/)
- [Dependency Injection Patterns for Agent Architectures (Moltbook)](https://www.moltbook.com/post/d5b5d112-35b4-4317-bc98-cc97682b3962)
- [A Practical Guide for Designing, Developing, and Deploying Agentic AI (arXiv)](https://arxiv.org/html/2512.08769v1)
- [Architectures for Building Agentic AI (arXiv)](https://arxiv.org/html/2512.09458v1)
- [Architecting Multi-Agent Systems: Evolving Proven Patterns (Medium)](https://medium.com/@chris.p.hughes10/architecting-multi-agent-systems-evolving-proven-patterns-to-agentic-systems-01b2b74e1fa5df)
