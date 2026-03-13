# Agent 系统的基本部署架构

> 难度：基础
> 分类：Production & Deployment

## 简短回答

生产级 Agent 系统的部署架构包含五个核心层：(1) **接入层**——API Gateway 负责认证、限流、路由，支持 WebSocket/SSE 流式响应；(2) **Agent 编排层**——Agent 运行时（LangGraph、自研框架）管理推理循环、工具调用、状态管理；(3) **模型网关层**——LLM Gateway（LiteLLM/Portkey）统一多模型 API、故障转移、Prompt 缓存；(4) **数据与工具层**——向量数据库（RAG）、工具服务（MCP）、持久化存储；(5) **可观测性层**——Trace 追踪（Langfuse）、指标监控（Prometheus）、告警（PagerDuty）。2025 年的关键趋势：**Context Engineering 成为核心学科**——将上下文视为有独立架构和生命周期的一等系统；**Plan-then-Execute 架构**优于 ReAct 的逐步推理——将规划和执行解耦，支持 DAG 并行执行；容器化部署（Docker + K8s/Cloud Run）是标配；**MCP 协议**正在成为工具集成的行业标准。部署建议：先用 Serverless（Cloud Run/Lambda）快速上线，后期按需迁移到 K8s 集群。

## 详细解析

### 五层架构全景

```
┌──────────────────────────────────────────────────────┐
│                    接入层                             │
│  API Gateway / Load Balancer / WebSocket             │
│  认证 → 限流 → 路由 → 流式响应                     │
├──────────────────────────────────────────────────────┤
│                Agent 编排层                           │
│  Agent Runtime（LangGraph / 自研框架）               │
│  推理循环 → 工具选择 → 状态管理 → 检查点            │
├──────────────────────────────────────────────────────┤
│               模型网关层                              │
│  LLM Gateway（LiteLLM / Portkey）                    │
│  模型路由 → 故障转移 → 缓存 → 成本追踪             │
├──────────┬───────────┬───────────┬───────────────────┤
│ 向量数据库│ 工具服务  │ 状态存储  │ 安全护栏         │
│ (RAG)    │ (MCP)     │ (Redis/PG)│ (Guardrails)     │
├──────────┴───────────┴───────────┴───────────────────┤
│                 可观测性层                            │
│  Traces(Langfuse) + Metrics(Prometheus) + Alerts     │
└──────────────────────────────────────────────────────┘
```

### 各层详解

```python
# 1. 接入层
api_layer = {
    "API Gateway": {
        "职责": "认证、限流、路由、CORS",
        "选择": "Kong / AWS API Gateway / Nginx",
    },
    "流式响应": {
        "协议": "SSE（Server-Sent Events）用于单向流",
        "场景": "Token 流式输出，用户无需等待完整回答",
        "实现": "FastAPI StreamingResponse / WebSocket",
    },
    "健康检查": {
        "端点": "/health 检查服务状态",
        "内容": "服务可用性 + LLM API 连通性 + DB 连通性",
    },
}

# 2. Agent 编排层
orchestration_layer = {
    "Agent Runtime": {
        "职责": "管理 Agent 的推理-行动循环",
        "框架选择": {
            "LangGraph": "最成熟，适合复杂有状态工作流",
            "CrewAI": "多 Agent 协作场景",
            "自研": "需要完全控制时",
        },
    },
    "状态管理": {
        "检查点": "每步保存状态，支持恢复和回放",
        "会话管理": "跨请求维护对话上下文",
        "存储": "Redis（短期）+ PostgreSQL（长期）",
    },
    "部署模式": {
        "Plan-then-Execute": {
            "优势": "规划和执行解耦，支持并行",
            "适用": "复杂多步任务",
        },
        "ReAct": {
            "优势": "简单灵活，逐步推理",
            "适用": "简单任务、对话式交互",
        },
    },
}

# 3. 模型网关层
model_gateway = {
    "统一 API": "一个接口调用 OpenAI/Anthropic/Google 等",
    "故障转移": "主模型不可用时自动切换到备用模型",
    "Prompt 缓存": "相同前缀的请求复用缓存，降低成本 90%",
    "模型路由": "按任务复杂度选择模型（简单→小模型，复杂→大模型）",
    "成本追踪": "实时记录每个请求的 Token 用量和费用",
    "工具": "LiteLLM, Portkey, OpenRouter",
}

# 4. 数据与工具层
data_tool_layer = {
    "向量数据库": {
        "用途": "RAG 检索",
        "选择": "Pinecone（托管）/ Weaviate（自部署）/ pgvector（嵌入PG）",
    },
    "工具服务": {
        "协议": "MCP（Model Context Protocol）",
        "说明": "标准化 LLM 与外部工具/数据源的连接",
    },
    "持久化存储": {
        "对话历史": "PostgreSQL",
        "会话缓存": "Redis",
        "文件存储": "S3 / GCS",
    },
}
```

### 容器化部署方案

```yaml
# docker-compose.yml — 基础部署配置
version: '3.8'
services:
  agent-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://pg:5432/agent
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        limits:
          memory: 2G  # Agent 需要较多内存
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_DB=agent
    volumes:
      - pgdata:/var/lib/postgresql/data
```

### 部署策略选择

```
你的场景是什么？
│
├── 早期/MVP（< 1000 用户）
│   → Serverless（Cloud Run / Lambda）
│   → 按用量付费，无需管理服务器
│   → 注意：冷启动延迟 + 执行时间限制
│
├── 中期/增长（1K-100K 用户）
│   → 容器化（ECS / Cloud Run 持续运行）
│   → 自动扩缩容 + 健康检查
│   → 引入 Redis 缓存和 CDN
│
├── 大规模（100K+ 用户）
│   → Kubernetes 集群
│   → 多区域部署 + 全球负载均衡
│   → 自建模型服务（vLLM）降低成本
│
└── 共同关注：
    ├── 密钥管理（Vault / AWS Secrets Manager）
    ├── 网络隔离（Agent 不直接访问公网）
    ├── CI/CD（代码 + Prompt 都走管道）
    └── 可观测性（从 Day 1 接入 Trace）
```

## 常见误区 / 面试追问

1. **误区："直接在应用代码里调 LLM API 就行"** — 生产系统需要模型网关层来处理故障转移、限流、成本控制和缓存。直接调 API 会导致：单点故障、无法切换模型、成本失控、无法追踪。LiteLLM 或 Portkey 可以一行代码解决这些问题。

2. **误区："Agent 系统和普通 Web 服务部署一样"** — Agent 有独特的部署挑战：(1) 长时间运行（一次任务可能执行数分钟）；(2) 状态管理（多步执行需要检查点）；(3) 高内存需求（上下文窗口占用大量内存）；(4) 不确定的成本（每次请求的 Token 消耗不同）。

3. **追问："如何处理 Agent 的长时间运行任务？"** — (1) 异步执行：接收请求后立即返回任务 ID，客户端轮询或 WebSocket 推送结果；(2) 检查点机制：每步保存状态，支持断点恢复；(3) 超时保护：设置最大执行时间和最大步数；(4) 流式输出：中间结果实时推送给用户。

4. **追问："选择框架还是自研？"** — 参考原则：如果需求匹配框架能力的 80%+，用框架（LangGraph）；如果需要深度定制或框架是瓶颈，自研核心 Agent 循环但复用社区工具（LiteLLM、Langfuse）。框架选择比模型选择更重要。

## 参考资料

- [LLM Agents in Production: Architectures, Challenges, and Best Practices (ZenML)](https://www.zenml.io/blog/llm-agents-in-production-architectures-challenges-and-best-practices)
- [Deploying AI Agents to Production: Architecture and Implementation Roadmap (MLM)](https://machinelearningmastery.com/deploying-ai-agents-to-production-architecture-infrastructure-and-implementation-roadmap/)
- [Architecting Efficient Context-Aware Multi-Agent Framework for Production (Google)](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)
- [What 1,200 Production Deployments Reveal About LLMOps in 2025 (ZenML)](https://www.zenml.io/blog/what-1200-production-deployments-reveal-about-llmops-in-2025)
- [LLM Agents: The Enterprise Technical Guide 2025 (Aisera)](https://aisera.com/blog/llm-agents/)
