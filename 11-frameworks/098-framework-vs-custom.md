# 框架 vs 自研：什么时候应该自己构建 Agent 框架？

> 难度：中级
> 分类：Frameworks

## 简短回答

框架 vs 自研是 Agent 开发中最关键的架构决策之一。**用框架**的场景：快速原型验证、团队 LLM 经验不足、需求与框架能力高度匹配、项目生命周期短。**自研**的场景：对性能/延迟有极致要求、需要深度定制控制流、框架抽象层导致调试困难（"框架深度"问题）、团队已积累足够的 LLM 工程经验。核心判断原则：**"框架的价值 = 节省的开发时间 − 绕过框架限制的时间"**。当后者开始超过前者时，就是考虑自研的信号。实际经验表明：LangChain 非常适合原型阶段（开发速度快 3x），但一旦扩展到复杂生产场景，其多层抽象会成为维护噩梦。Anthropic、OpenAI 等公司的官方建议是"从最简单的方案开始"——先用原生 API + 少量工具代码，只在确实需要时才引入框架。最佳实践是**渐进式方法**：原型用框架 → 验证需求 → 核心路径自研 → 非核心保留框架组件。

## 详细解析

### 决策矩阵

```
用框架 vs 自研的决策维度：

┌────────────────┬──────────────────┬──────────────────┐
│ 维度           │ 选择框架         │ 选择自研         │
├────────────────┼──────────────────┼──────────────────┤
│ 项目阶段       │ PoC / MVP        │ 成熟产品         │
│ 团队经验       │ LLM 新手         │ LLM 老手         │
│ 定制需求       │ 标准流程         │ 深度定制         │
│ 性能要求       │ 延迟不敏感       │ 毫秒级优化       │
│ 可调试性       │ 黑箱可接受       │ 需要完全透明     │
│ 开发速度       │ 急需上线         │ 可以慢一点       │
│ 维护成本       │ 能接受框架升级   │ 想控制依赖       │
│ 生态需求       │ 需要大量集成     │ 集成点有限       │
│ 团队规模       │ 小团队（1-3人）  │ 有专人维护基建   │
└────────────────┴──────────────────┴──────────────────┘

决策流程图：
需要多少 LLM 调用？
├── 单次调用 → 直接用 API（不需要框架）
├── 2-5 步链式 → 轻量封装或框架
└── 复杂多步 Agent →
    ├── 标准 ReAct/RAG 模式 → 用框架
    └── 高度定制流程 → 自研
```

### 框架的价值与代价

```python
# 框架带来的价值
framework_benefits = {
    "快速启动": "几行代码实现 RAG/Agent，开发速度 3-5x",
    "最佳实践": "内置 ReAct、Plan-and-Execute 等成熟模式",
    "生态集成": "几十种 LLM、向量数据库、工具的预集成",
    "可观测性": "LangSmith/Langfuse 等追踪工具的原生支持",
    "社区支持": "问题容易找到答案，示例丰富",
}

# 框架带来的代价
framework_costs = {
    "抽象税": "多层封装导致调试困难，错误信息不直观",
    "性能损耗": "额外的序列化/反序列化、中间层开销",
    "升级风险": "API 频繁变更，升级可能破坏现有代码",
    "灵活性限制": "框架假设的流程可能不符合业务需求",
    "依赖锁定": "深度使用后难以迁移到其他方案",
    "过度工程": "简单需求也要遵循框架的复杂模式",
}

# 关键信号：何时从框架迁移到自研
migration_signals = [
    "频繁绕过框架限制（monkey patch / 自定义 callback）",
    "调试时间 > 开发时间",
    "框架升级导致回归问题",
    "需要的功能框架不支持，但又不能脱离框架实现",
    "性能瓶颈来自框架抽象层",
]
```

### 自研 Agent 框架的核心组件

```python
# 从第一性原理构建 Agent——只需要几个核心组件

import json
from openai import OpenAI

class MinimalAgent:
    """最小可用的自研 Agent 框架"""

    def __init__(self, model="gpt-4o", tools=None, system_prompt=""):
        self.client = OpenAI()
        self.model = model
        self.tools = {t["function"]["name"]: t for t in (tools or [])}
        self.tool_functions = {}
        self.system_prompt = system_prompt

    def register_tool(self, name, func, schema):
        """注册工具：函数 + JSON Schema"""
        self.tool_functions[name] = func
        self.tools[name] = schema

    def run(self, user_message, max_steps=10):
        """Agent 执行循环——这就是 Agent 的本质"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for step in range(max_steps):
            # 1. 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=list(self.tools.values()) if self.tools else None,
            )
            msg = response.choices[0].message
            messages.append(msg)

            # 2. 检查是否需要调用工具
            if not msg.tool_calls:
                return msg.content  # 无工具调用 → 返回最终答案

            # 3. 执行工具
            for tool_call in msg.tool_calls:
                func = self.tool_functions[tool_call.function.name]
                args = json.loads(tool_call.function.arguments)
                result = func(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

        return "达到最大步数限制"

# 核心洞察：Agent 本质上就是一个 while 循环
# LLM 调用 → 检查工具调用 → 执行工具 → 循环
# 不需要框架也能实现
```

### 渐进式策略：最佳实践

```python
# 推荐的渐进式方法：

stages = {
    "阶段 1：原型验证（1-2 周）": {
        "方法": "使用框架（LangChain/LangGraph）",
        "目标": "验证 Agent 方案可行性",
        "产出": "可演示的原型",
        "代码": "大量使用框架预构建组件",
    },
    "阶段 2：需求明确（2-4 周）": {
        "方法": "框架 + 自定义组件",
        "目标": "明确生产需求和性能瓶颈",
        "产出": "性能基线、需求文档",
        "代码": "开始替换框架中的瓶颈组件",
    },
    "阶段 3：核心自研（1-3 月）": {
        "方法": "核心路径自研 + 非核心保留框架",
        "目标": "优化性能、提高可控性",
        "产出": "自研 Agent 核心 + 框架辅助模块",
        "代码": "Agent Loop 自研，RAG/工具可保留框架",
    },
    "阶段 4：完全自主（可选）": {
        "方法": "完全自研",
        "条件": "只有当框架的存在确实造成显著问题时",
        "注意": "大多数项目在阶段 3 就够了",
    },
}

# 关键原则：
# 1. 不要一开始就自研——先用框架验证方向
# 2. 不要永远依赖框架——根据需求逐步自主化
# 3. 可以混用——核心路径自研，辅助功能用框架
# 4. 抽象边界——即使用框架，也要隔离框架依赖
```

### 自研时的架构建议

```python
# 自研 Agent 框架的分层架构

architecture = {
    "Layer 1: LLM 接口层": {
        "职责": "封装 LLM API 调用，支持多提供商",
        "设计": "适配器模式——统一接口，底层可切换",
        "示例": "LLMClient(provider='openai'|'anthropic'|'google')",
    },
    "Layer 2: 工具层": {
        "职责": "工具注册、Schema 生成、执行",
        "设计": "装饰器模式——@tool 自动生成 JSON Schema",
        "示例": "@tool def search(query: str) -> str: ...",
    },
    "Layer 3: Agent Loop": {
        "职责": "核心执行循环——推理、行动、观察",
        "设计": "策略模式——可插拔的推理策略",
        "示例": "AgentLoop(strategy='react'|'plan_execute')",
    },
    "Layer 4: 状态管理": {
        "职责": "会话状态、检查点、记忆",
        "设计": "仓库模式——可插拔的存储后端",
        "示例": "StateStore(backend='memory'|'redis'|'postgres')",
    },
    "Layer 5: 可观测性": {
        "职责": "日志、追踪、指标",
        "设计": "中间件模式——非侵入式",
        "示例": "agent.use(TracingMiddleware())",
    },
}
```

## 常见误区 / 面试追问

1. **误区："自研总是更好"** — 自研意味着你要维护 Agent Loop、错误处理、状态管理、工具集成、可观测性等所有组件。小团队可能无法承受这个维护成本。框架把这些"无差别劳动"打包好了，让你专注于业务逻辑。

2. **误区："框架性能太差"** — 框架的额外开销通常在毫秒级，而 LLM API 调用在秒级。在大多数场景下，框架不是性能瓶颈。只有在极高并发或极低延迟的场景下，框架开销才值得关注。

3. **追问："如何隔离框架依赖？"** — 使用**端口-适配器模式（Hexagonal Architecture）**：定义自己的接口（Port），用框架实现这些接口（Adapter）。业务代码只依赖自己的接口，切换框架只需替换 Adapter。这是"用框架但不被框架绑定"的关键。

4. **追问："Anthropic/OpenAI 自己推荐怎么做？"** — Anthropic 的建议："Start with the simplest solution"——先用原生 API，只在确实需要时引入框架。OpenAI Agents SDK 的设计哲学是"最小抽象"——只封装 Agent Loop 和工具调用，其他交给开发者。这反映了行业趋势：框架在变轻，而非变重。

## 参考资料

- [Building AI Agents Without Frameworks: What LangChain Won't Teach You (Medium)](https://medium.com/@candemir13/building-ai-agents-without-frameworks-what-langchain-wont-teach-you-035a11d9d80c)
- [LangChain, LangGraph, or Custom? Choosing the Right Agentic Framework (Turgon AI)](https://www.turgon.ai/post/langchain-langgraph-or-custom-choosing-the-right-agentic-framework)
- [The Complete Guide to Choosing an AI Agent Framework in 2025 (Langflow)](https://www.langflow.org/blog/the-complete-guide-to-choosing-an-ai-agent-framework-in-2025)
- [AI Agent Frameworks Compared: Architecture Patterns and Trade-offs (Arsum)](https://arsum.com/blog/posts/ai-agent-frameworks/)
- [Stop Using Frameworks: Build an AI Agent From First Principles (YouTube)](https://www.youtube.com/watch?v=TdcAslNDCGc)
