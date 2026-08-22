# OpenAI Responses API / Agents SDK vs Anthropic Claude Agent SDK 对比

> 难度：中级
> 分类：Frameworks

## 简短回答

OpenAI 和 Anthropic 分别推出了官方 Agent 开发方案，代表了两种不同的设计哲学。**OpenAI Agents SDK**（2025-03 发布，**底层基于新的 Responses API**——Responses API 才是 2025-08 deprecated 的 Assistants API 的直接替代者，Agents SDK 是其上层的轻量级编排框架）——核心概念仅三个：Agent（带指令和工具的 LLM）、Handoff（Agent 间任务交接）、Guardrails（输入/输出验证），强调"最小抽象、最大控制"，内置追踪但不强制托管状态；同时提供 **Python `openai-agents`** 与 **TypeScript `@openai/agents`** 双语言 SDK。**Anthropic Claude Agent SDK**（**2025-09** 从 `claude-code-sdk` 重命名而来，与 Claude Sonnet 4.5 同步发布）——基于 Claude Code 实战经验构建，核心特性：原生 MCP（Model Context Protocol）支持、Tool Permissions 细粒度授权、Subagent 派生、Hooks 机制，强调"工具优先"的 Agent 设计，与 MCP 生态深度绑定。关键差异：OpenAI SDK 是"模型无关的理想"但实际优化 OpenAI 模型；Claude SDK 明确绑定 Claude 模型但 MCP 是开放标准。OpenAI 走"SDK 轻+云重"路线（Responses API 在云端管理状态）；Anthropic 走"MCP 协议开放+SDK 原生集成"路线（MCP 连接万物）。截至 2026.7，两条路线均已生产成熟：Claude Agent SDK 是 Claude Code 的运行时底座，配合 2026-06-30 发布的 Claude Sonnet 5（定位"最 agentic 的 Sonnet"，入门定价 $2/MTok 输入 / $10/MTok 输出）强化自主任务执行能力；MCP 生态同期爆发至 5500+ Server、约 80% Fortune 500 企业参与，使 Claude SDK 在工具密集型场景获得结构性优势。OpenAI Agents SDK 则依托 Responses API 的云端状态托管持续完善，但这也意味着更强的平台锁定。选择建议：已在 OpenAI 生态 → Agents SDK；需要 MCP 工具生态 → Claude SDK；需要模型无关 → Vercel AI SDK 或自研。

## 详细解析

### 核心对比

| 维度 | OpenAI Agents SDK | Claude Agent SDK |
|------|-------------------|------------------|
| 发布时间 | 2025-03 | 2025-09（重命名） |
| 包名 | openai-agents / @openai/agents | claude-agent-sdk |
| 设计哲学 | 最小抽象 | 工具优先+MCP |
| 核心概念 | Agent/Handoff/Guardrails | ClaudeAgentOptions/query/Subagent |
| 底层 API | Responses API | Messages API+MCP |
| 模型支持 | OpenAI 优先 | Claude 专属 |
| 工具协议 | Function Calling | MCP（开放标准） |
| 状态管理 | Responses API（云端托管） | Hooks + 文件检查点 |
| 多 Agent | Handoff 模式；2026-08 起 Responses API 提供**原生多 Agent 编排** | Subagent 派生 |
| Human-in-the-Loop | 需自行实现 | permission_mode |
| 追踪/可观测 | 内置 Tracing | Hooks 钩子 |
| 语言支持 | Python+TypeScript | Python+TypeScript |
| 开源 | 是 | 是 |
| 适用场景 | 通用 Agent | Coding/工具密集 |

### OpenAI Agents SDK

```python
# OpenAI Agents SDK：最小抽象，三个核心概念
from agents import Agent, Runner, handoff, GuardrailFunctionOutput
from agents import input_guardrail

# 核心概念 1：Agent — 带指令和工具的 LLM
triage_agent = Agent(
    name="Triage Agent",
    instructions="根据用户问题类型，转接到合适的专家。",
    handoffs=[sales_agent, support_agent],  # 可以交接给谁
)

sales_agent = Agent(
    name="Sales Agent",
    instructions="处理销售相关问题。",
    tools=[lookup_pricing, create_quote],
)

support_agent = Agent(
    name="Support Agent",
    instructions="处理技术支持问题。",
    tools=[search_kb, create_ticket],
)

# 核心概念 2：Handoff — Agent 间的任务交接
# LLM 自动决定何时交接，交接时携带上下文

# 核心概念 3：Guardrails — 输入/输出安全检查
@input_guardrail
async def check_injection(ctx, agent, input):
    """检查 Prompt 注入"""
    result = await Runner.run(
        injection_detector,
        input,
        context=ctx.context,
    )
    return GuardrailFunctionOutput(
        output_info=result,
        tripwire_triggered=result.is_injection,
    )

# 运行 Agent
result = await Runner.run(triage_agent, "我想了解企业版定价")

# Agents SDK 特点：
# 1. 极简 API——三个概念覆盖大部分场景
# 2. 内置追踪——自动记录每步执行
# 3. 类型安全——Pydantic 模型验证输出
# 4. Agent Loop 内置——自动处理工具调用循环
```

### Anthropic Claude Agent SDK

```python
# Claude Agent SDK（pip install claude-agent-sdk，2025-09 从 claude-code-sdk 重命名）
# 核心 API：ClaudeAgentOptions + 异步 query() 函数
# 详见：https://docs.anthropic.com/en/api/agent-sdk

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

# 1. 最简用法：直接 query()
async def simple_example():
    options = ClaudeAgentOptions(
        model="claude-sonnet-5",
        system_prompt="你是一个数据分析助手。",
    )
    async for msg in query(prompt="分析 sales.csv 并给出 Top 3 趋势", options=options):
        print(msg)

# 2. 配置 MCP 工具服务器（真实 dict 结构，非 URL 列表）
options = ClaudeAgentOptions(
    model="claude-sonnet-5",
    mcp_servers={
        # stdio 子进程式（最常见）
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
        },
        # HTTP 远程
        "github": {
            "type": "http",
            "url": "https://mcp.github.com/v1",
            "headers": {"Authorization": "Bearer $GITHUB_TOKEN"},
        },
    },
    # 允许哪些工具，可精确到工具名
    allowed_tools=["Read", "Write", "Bash", "mcp__filesystem__read_file"],
)

# 3. Permission Mode（细粒度授权控制——这是真正的 HITL 机制）
options = ClaudeAgentOptions(
    model="claude-sonnet-5",
    permission_mode="acceptEdits",  # 自动接受文件编辑
    # 可选值: "default" | "acceptEdits" | "bypassPermissions" | "plan"
    # plan 模式下 Agent 只规划不执行，必须人工 confirm 后才能 run
)

# 4. Hooks 机制：在工具调用前后注入逻辑（这是真实可观测/HITL 的方式）
async def pre_tool_hook(tool_name: str, tool_input: dict):
    if tool_name == "Bash" and "rm" in tool_input.get("command", ""):
        return {"deny": True, "reason": "Dangerous command blocked"}
    return {"deny": False}

options = ClaudeAgentOptions(
    model="claude-sonnet-5",
    hooks={"PreToolUse": pre_tool_hook},
)

# 5. Subagent 模式（多 Agent 编排）：通过 Task 工具派生子 Agent
# Claude SDK 没有静态的"sub_agents 列表"参数，而是运行时由主 Agent 决定派生
# 主 Agent 调用 Task 工具时，会启动一个独立上下文的子 Claude 实例
options = ClaudeAgentOptions(
    model="claude-sonnet-5",
    system_prompt="你是项目经理，可通过 Task 工具委派 researcher/coder/reviewer 子 Agent",
    allowed_tools=["Task", "Read", "Write"],
)

# Claude SDK 真实特性：
# 1. MCP 原生集成——配置即工具发现
# 2. Permission Mode + Hooks——细粒度的 HITL 和安全护栏
# 3. Subagent via Task 工具——主 Agent 运行时派生子 Agent，非静态声明
# 4. 文件系统检查点——基于 .claude/ 目录的状态持久化
```

### 关键差异深度分析

```python
key_differences = {
    "工具生态策略": {
        "OpenAI": "Function Calling — 工具定义在 Agent 代码中，静态",
        "Anthropic": "MCP — 工具由独立服务提供，动态发现",
        "影响": "MCP 更灵活但增加了基础设施复杂度",
        "2026-08": "OpenAI 在 Responses API 增加程序化工具调用；"
                   "Anthropic 的 Python SDK v0.2.140 把进程内 MCP server 换成 mcp 官方 in-memory transport",
    },
    "状态管理": {
        "OpenAI": "Responses API 在云端托管对话状态和文件",
        "Anthropic": "SDK 内置检查点，可用本地或云端存储",
        "影响": "OpenAI 更省心但锁定其云服务；Anthropic 更灵活",
    },
    "多 Agent 模式": {
        "OpenAI": "Handoff — 扁平化交接，Agent A 把控制权给 Agent B",
        "Anthropic": "Orchestrator — 层级化，主 Agent 调度子 Agent",
        "影响": "Handoff 更简单直观；Orchestrator 更适合复杂任务",
        "2026-08": "编排从 SDK 层下沉到 API 层——Responses API 已提供原生多 Agent 编排，"
                   "OpenAI 不再只把编排留给应用侧的 Agents SDK",
    },
    "模型绑定": {
        "OpenAI": "SDK 设计为模型无关（但实际优化 OpenAI 模型）",
        "Anthropic": "明确绑定 Claude 模型",
        "影响": "如果需要多模型支持，OpenAI SDK 更容易适配",
    },
    "开放性": {
        "OpenAI": "SDK 开源，但生态围绕 OpenAI API",
        "Anthropic": "SDK 绑定 Claude，但 MCP 是完全开放的标准",
        "影响": "MCP 的长期价值可能超过任何单一 SDK",
    },
}
```

### 双侧最新动向（2026-08）：编排下沉 vs 协议做深

2026 年 8 月两侧各出了一批动作，正好落在本文核心论断的延长线上——**「云侧收拢 vs 开放协议」的分野没变，但各自都往前走了一步**。

| 维度 | OpenAI（GPT-5.6 时期） | Anthropic（Claude Agent SDK） |
|------|------------------------|-------------------------------|
| 本期主线 | 把编排能力**收进 API 层** | 把 MCP 集成**做深到协议原生** |
| 代表变更 | Responses API 新增保留推理态 / 原生多 Agent 编排 / 程序化工具调用 | Python SDK v0.2.140 支持 MCP 2.x，进程内 server 改走官方 in-memory transport |
| 模型策略 | 不默认用旗舰模型，按任务下沉到更省的 Luna / Terra | 按任务选 Opus / Sonnet / Fable（SDK 侧一直如此） |
| 方向含义 | 云端承担更多 Agent 运行时职责 → 平台锁定进一步加深 | 依赖开放协议的官方实现 → 减少自研代码、提升互操作保真度 |

**OpenAI 侧：编排从 SDK 层下沉到 API 层。** 官方文章 The builder's guide to GPT-5.6（2026-08-13）给出的核心建议是不要默认用旗舰模型、按任务下沉到成本更低的小模型（Luna / Terra），并推介 Responses API 的三项新能力：

- **保留推理态（retained reasoning）**——跨轮次保留模型的推理过程，不必每轮重新推理，多步任务的连贯性与成本都受益；
- **原生多 Agent 编排**——多 Agent 的调度由 API 直接提供，不再必须由应用层的 Agents SDK 承担；
- **程序化工具调用（programmatic tool calling）**——工具调用可由代码逻辑驱动组织，而不是每一步都走模型决策。

配套的定价文章（2026-07-30）显示 Luna 降价约 80%、Terra 约 20%，Sol 增加最高 2.5 倍速的 Fast mode——「按任务下沉到小模型」这条建议背后有实打实的价格结构支撑。

其中**「原生多 Agent 编排」进 Responses API 是本文核心论断的直接演进**：此前 OpenAI 的分层是「Responses API 管状态、Agents SDK 管编排」，编排属于应用层的事；现在编排被收进 API。这让「SDK 轻 + 云重」的重心进一步向云侧偏移——好处是应用代码更薄，代价是原本可以自己换掉的编排逻辑也变成了平台能力，锁定加深。

**Anthropic 侧：MCP 集成从自研分发换成协议原生。** `claude-agent-sdk-python` v0.2.140（2026-08-18）落地 MCP 2.x 支持：

```python
# v0.2.140 的变化：依赖放宽到 mcp>=1.23.0,<3.0.0
# 进程内 SDK MCP server 不再手写 JSON-RPC 分发，改走 mcp 官方 in-memory transport
from mcp.server import Server

server = Server("my-tools")   # 手工构造的 mcp.server.Server 现已全保真可用
# resources / prompts / 全部 result content type 原样透传，不再被 SDK 中间层削减
# mcp 2.x 下还支持中断时的工具取消
```

意义在于**保真度和维护成本**：过去 SDK 自己实现 JSON-RPC（一种基于 JSON 的远程调用协议）分发，能力边界由 SDK 的实现决定，MCP 协议里的 resources、prompts 等特性会在中间层被削掉；现在直接用协议官方的进程内传输实现，手工构造的 `mcp.server.Server` 可以原样接入，全部 result content type 原样透传。这与「MCP 是开放标准」的定位是自洽的——把实现让给协议官方，而不是长期维护一份平行实现。

两边合起来看：OpenAI 把 Agent 运行时的能力往自家 API 里收，Anthropic 把自家 SDK 里的自研部分往开放协议上让。分野依旧，只是各自更彻底了。

> 注：OpenAI 两篇文章的发布日期取自官方列表页的单一来源（正文页经代理读取时不渲染日期），未做正文交叉验证，日期以官方页面为准。

### 第三方替代方案

除了官方 SDK，还有模型无关或厂商生态的选择：

| 方案 | 特点 |
|------|------|
| Vercel AI SDK | 真正模型无关，支持 OpenAI/Claude/Gemini，TypeScript 优先 |
| LangGraph | 1.0 GA（2025-10），durable agent 首个稳定版，月 PyPI 下载超 3800 万 |
| Microsoft Agent Framework | 2026-04 发布 1.0 GA，Azure 生态，企业级多 Agent 编排 |
| Google ADK | Agent Development Kit，原生集成 A2A 协议 + Vertex AI Agent Engine |
| 自研 | 完全控制，无依赖，但需要自己处理所有细节 |

```
选择决策：
├── 绑定 OpenAI → OpenAI Agents SDK
├── 绑定 Claude + 需要 MCP → Claude Agent SDK
├── 需要多模型 + TypeScript → Vercel AI SDK
├── 需要复杂工作流/durable → LangGraph
├── 绑定 Azure/微软生态 → Microsoft Agent Framework
├── 绑定 Google Cloud/Gemini → Google ADK
└── 需要完全控制 → 自研
```

### 实际选型建议

```
项目类型 → 推荐方案：

1. 客服 Bot（单模型）
   └── OpenAI Agents SDK（Handoff 模式天然适合客服分流）

2. 知识密集型 Agent（RAG + 多数据源）
   └── Claude Agent SDK + MCP（工具自动发现 + 上下文注入）

3. 代码助手（需要文件系统/终端访问）
   └── Claude Agent SDK + MCP（Claude Code 就是这么构建的）

4. 多模型策略（成本优化）
   └── Vercel AI SDK 或 LiteLLM + 自研 Agent Loop

5. 企业级部署（Azure 基础设施）
   └── Microsoft Agent Framework + Azure OpenAI
```

## 常见误区 / 面试追问

1. **误区："Assistants API 和 Agents SDK 是一回事"，或"Agents SDK 是 Assistants API 的直接替代者"** — 实际三者关系是：(1) **Assistants API**（2023）是云端托管的有状态 API（Thread/Message/Run 模型），OpenAI 于 **2025-08-26 公告 deprecated**，**计划于 2026-08-26 关闭**（以 OpenAI 官方公告为准）；(2) **Responses API**（2025-03）才是 Assistants API 的**直接替代者**——同样云端管理状态，但 API 设计更现代；(3) **Agents SDK**（2025-03）是**基于 Responses API 的上层轻量级编排框架**，提供 Agent/Handoff/Guardrails 抽象。三者是分层关系：Assistants → Responses（API 层替代）→ Agents SDK（SDK 层编排）。

2. **误区："选了一个 SDK 就不能用其他模型"** — OpenAI Agents SDK 的 `model_settings` 可以配置兼容 OpenAI API 格式的任何提供商。Claude Agent SDK 绑定 Claude，但 MCP 服务器可以被任何客户端使用。关键是区分"SDK 绑定"和"协议绑定"。

3. **追问："MCP 会成为行业标准吗？"** — 截至 2026 年初，MCP 已从"Anthropic 提议的标准"成长为事实标准：PulseMCP 收录超 **5500 个 MCP Server**，约 **80% Fortune 500 企业**开始参与生态，OpenAI、Google、微软均已宣布支持或兼容 MCP。它解决了"每个 Agent 框架都要自己实现工具集成"的 N×M 问题。但 ACM 2026 学术研究也揭示了开发者痛点：文件系统操作问题占比最高（21.32%），其次是数据验证和类型问题——说明协议走向成熟的同时，工程实践仍在追赶。

4. **追问："如果需要同时用 GPT 和 Claude 怎么办？"** — 推荐架构：(1) 使用 LiteLLM 或 Portkey 作为 Model Gateway，统一 API 接口；(2) 自研轻量 Agent Loop，不绑定任何特定 SDK；(3) MCP 作为工具层标准，与模型层解耦。核心思想是**模型层和工具层分离**。

## 参考资料

- [Claude Agent SDK vs OpenAI Agent SDK vs Vercel AI SDK (Reddit)](https://www.reddit.com/r/vercel/comments/1r5meu6/claude_agent_sdk_vs_openai_agent_sdk_vs_vercel_ai/)
- [OpenAI's Agents SDK and Anthropic's MCP (PromptHub)](https://www.prompthub.us/blog/openais-agents-sdk-and-anthropics-model-context-protocol-mcp)
- [Compare Claude API with OpenAI Agent API (Top AI Product)](https://topaiproduct.com/2025/03/12/compare-claude-api-with-openai-agent-api-for-building-an-ai-agent/)
- [Winning in the Autonomous AI Agents Race: Anthropic vs OpenAI (Medium)](https://rabot.medium.com/winning-in-the-autonomous-ai-agents-race-a0c03d52acad)
- [The builder's guide to GPT-5.6 (OpenAI，2026-08-13)](https://openai.com/index/builders-guide-to-gpt-5-6/) —— 模型下沉建议 + Responses API 保留推理态 / 原生多 Agent 编排 / 程序化工具调用
- [Advancing the price-performance frontier with GPT-5.6 (OpenAI，2026-07-30)](https://openai.com/index/advancing-the-price-performance-frontier-with-gpt-5-6/) —— Luna 降价约 80%、Terra 约 20%，Sol 新增 Fast mode
- [claude-agent-sdk-python v0.2.140 —— MCP 2.x 支持、官方 in-memory transport（2026-08-18）](https://github.com/anthropics/claude-agent-sdk-python/releases/tag/v0.2.140)
