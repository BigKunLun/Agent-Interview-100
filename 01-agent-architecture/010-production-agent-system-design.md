# 生产级 Agent 系统设计（含智能客服实战案例）

> 难度：高级
> 分类：Agent 架构

## 简短回答

生产级 Agent 系统远不止一个 LLM + 工具的 Demo。它需要在架构层面解决五大挑战：可靠的 Agent Loop 设计、上下文工程（Context Engineering）、成本与延迟控制、全链路可观测性、以及安全护栏。核心原则是"right-size your architecture"——用满足需求的最简架构，避免过度设计。本文先阐述通用的生产级 Agent 系统设计原则和架构，然后以**日均 10 万对话的智能客服系统**为实战案例，完整演练从需求澄清到成本估算的系统设计全流程。

## 详细解析

---

### 第一部分：通用生产级 Agent 系统设计原则

---

#### 从原型到生产的鸿沟

> "构建一个 LLM 原型极其简单，但构建一个在生产中可靠运行的 LLM 系统是完全不同的挑战——它需要严格的工程、谨慎的系统设计和运维成熟度。"

原型和生产的关键差异：

| 维度 | 原型 | 生产系统 |
|------|------|---------|
| 可靠性 | "大多数时候能用" | 99.9% 可用性 |
| 成本 | 无预算约束 | 每次调用有成本上限 |
| 延迟 | 等几秒没关系 | P95 < 目标延迟 |
| 安全 | 信任 LLM 输出 | 假设 LLM 会犯错 |
| 可观测性 | print 调试 | 全链路 Trace |
| 扩展性 | 单用户 | 高并发多用户 |

#### 生产级架构全景

```
┌───────────────────────────────────────────────────────┐
│                     API Gateway                       │
│              (认证、限流、负载均衡)                      │
├───────────────────────────────────────────────────────┤
│                   Agent Orchestrator                  │
│         (状态机 + 路由 + 循环控制 + 重规划)              │
├──────────┬──────────┬──────────┬──────────────────────┤
│ Tool     │ RAG      │ Memory   │ Safety/Policy        │
│ Gateway  │ Service  │ Service  │ Engine               │
│ (工具权限 │ (检索+   │ (短期+   │ (Guardrails +        │
│  + 执行)  │  重排序)  │  长期)   │  Human-in-the-Loop)  │
├──────────┴──────────┴──────────┴──────────────────────┤
│                  Observability Layer                   │
│          (Tracing + Logging + Metrics + Alerts)        │
└───────────────────────────────────────────────────────┘
```

#### 挑战 1：上下文工程（Context Engineering）

随着 Agent 运行时间增长，需要追踪的信息爆炸式增长——对话历史、工具输出、检索文档、中间推理。简单地把所有内容塞进上下文窗口不可持续。

Google ADK 的三个设计原则：

```python
# 原则 1：存储与展示分离
class ContextManager:
    def __init__(self):
        self.session_store = SessionStore()    # 持久化状态
        self.working_context = WorkingContext() # 每次调用的视图

    def build_context(self, session_id: str, current_task: str) -> str:
        """从持久化状态构建当前调用的上下文"""
        session = self.session_store.get(session_id)

        return self.working_context.compile(
            system_prompt=session.system_prompt,
            relevant_memory=session.memory.search(current_task, top_k=5),
            recent_history=session.history[-10:],  # 最近 10 轮
            current_task=current_task
        )

# 原则 2：显式转换（不是 ad-hoc 字符串拼接）
class ContextPipeline:
    processors = [
        SystemPromptProcessor(),
        MemoryRetrievalProcessor(),
        HistorySummarizationProcessor(),  # 旧历史自动摘要
        ToolResultProcessor(),
        CurrentTaskProcessor(),
    ]

    def compile(self, raw_state: dict) -> str:
        context = ""
        for processor in self.processors:
            context = processor.process(context, raw_state)
        return context

# 原则 3：最小上下文原则
# 每次模型调用和子 Agent 只看到必要的最小上下文
```

上下文工程的四个策略：
1. **写出窗口外**：便签簿（Scratchpad）、长期记忆
2. **选择相关上下文**：RAG、记忆检索
3. **压缩上下文**：摘要、修剪
4. **隔离上下文**：多 Agent 系统、沙箱

#### 挑战 2：成本与延迟控制

Agent 的多步性质导致级联 LLM 调用，成本和延迟可能失控。

##### 模型路由（Model Routing）

```python
class ModelRouter:
    def route(self, task: str, complexity: float) -> str:
        """根据任务复杂度选择最经济的模型"""
        if complexity < 0.3:
            return "claude-haiku-4-5-20251001"   # 简单任务用小模型
        elif complexity < 0.7:
            return "claude-sonnet-4-20250514"     # 中等任务用中等模型
        else:
            return "claude-opus-4-20250514"       # 复杂任务用强模型
```

##### 语义缓存

根据 GPTCache 等项目的实测数据，语义缓存可以显著减少 LLM API 调用（部分场景可达 60-70%）：

```python
class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.cache = VectorStore()
        self.threshold = similarity_threshold

    def get_or_compute(self, query: str, compute_fn):
        # 检查是否有语义相似的缓存结果
        cached = self.cache.search(query, top_k=1)
        if cached and cached[0].similarity > self.threshold:
            return cached[0].result  # 缓存命中：15x 更快，70% 成本节省
        # 缓存未命中：计算并缓存
        result = compute_fn(query)
        self.cache.store(query, result)
        return result
```

#### 挑战 3：可观测性

```python
# 使用 OpenTelemetry 风格的 Trace/Span
class AgentTracer:
    def trace_step(self, step_name: str):
        span = self.tracer.start_span(step_name)
        span.set_attributes({
            "agent.step": step_name,
            "agent.model": self.current_model,
            "agent.tokens_used": 0,
            "agent.tool_calls": [],
        })
        return span

# 生产中的典型 Trace 结构：
# Trace: "用户请求 → Agent 执行"
#   ├── Span: "规划" (model=opus, tokens=2000, latency=3s)
#   ├── Span: "工具调用: search" (latency=1.2s)
#   ├── Span: "工具调用: database" (latency=0.3s)
#   ├── Span: "推理 + 生成" (model=sonnet, tokens=1500, latency=2s)
#   └── Span: "安全检查" (latency=0.1s)
```

关键监控指标：
- **成功率**：任务完成率、工具调用成功率
- **成本**：每任务 token 消耗、API 费用
- **延迟**：端到端延迟、各步骤延迟分布
- **质量**：幻觉率、用户反馈（点赞/点踩）

#### 挑战 4：安全与护栏

```python
class SafetyLayer:
    def __init__(self):
        self.input_guardrail = InputGuardrail()   # 输入过滤
        self.output_guardrail = OutputGuardrail()  # 输出过滤
        self.action_guardrail = ActionGuardrail()  # 行动审批

    def check_action(self, action: ToolCall) -> bool:
        """高风险操作需要人工审批"""
        if action.risk_level == "high":
            return self.human_approval(action)  # Human-in-the-Loop
        if action.risk_level == "medium":
            return self.automated_review(action) # 自动审查
        return True  # 低风险直接放行
```

#### 挑战 5：选择正确的架构级别

> "用满足需求的最简架构。"

```
简单单步任务              → 不需要 Agent，用 LLM + Prompt
结构明确的多步任务        → Plan-and-Execute
动态探索性任务            → ReAct
复杂多领域问题            → 多 Agent 系统（但协调开销必须物有所值）
```

#### 生产检查清单

- [ ] Agent Loop 有最大步数限制和超时
- [ ] 错误处理：重试 + 降级 + 错误分类 + 检查点
- [ ] 上下文管理：摘要 + 检索 + 最小上下文原则
- [ ] 成本控制：模型路由 + 语义缓存 + 预算上限
- [ ] 可观测性：Trace + 日志 + 指标 + 告警
- [ ] 安全：输入/输出过滤 + 行动审批 + 权限最小化
- [ ] 评估：上线前基准测试 + 上线后持续评估
- [ ] 降级方案：Agent 失败时的回退策略

---

### 第二部分：实战案例——设计日均 10 万对话的智能客服 Agent 系统

---

> **面试场景**：请设计一个日均 10 万对话的智能客服 Agent 系统，支持多渠道接入、FAQ 自动回答、工单处理和人工转接，峰值流量为日均的 5 倍。

**系统概述**：核心是构建**四层架构**——用户接入层、**Agent 编排层**、LLM 推理层和数据层。Agent 编排层是系统的大脑，通过**意图路由器**（轻量分类模型 + LLM fallback）分发请求，结合**对话状态机**管理多轮交互。知识问答走 **RAG 管道**（Embedding → 向量检索 → Re-ranking），工单和投诉类走结构化流程。系统必须内置 **Human Handoff** 机制，在置信度低、情绪激动或多轮失败时自动转人工。扩展性上，通过**无状态服务 + 消息队列**实现水平扩展，用**语义缓存**降低 LLM 调用成本，并设计**降级方案**应对模型服务不可用的场景。峰值 5 倍流量意味着系统需要支撑约 **58 QPS** 的并发处理能力，同时将端到端延迟控制在 **3 秒以内**。

#### Step 1: 需求澄清

在面试中，第一步永远是明确需求边界，而不是直接画架构图。

**功能需求（Functional Requirements）：**

| 需求项 | 说明 |
|--------|------|
| 多渠道接入 | Web、App、微信公众号 / 小程序，统一消息协议 |
| 意图识别 | 识别用户意图：FAQ 咨询、工单提交、投诉、闲聊等 |
| FAQ 回答 | 基于知识库的检索增强生成（RAG） |
| 工单处理 | 结构化信息收集 → 自动创建工单 → 状态追踪 |
| 人工转接 | 低置信度 / 敏感话题 / 用户主动要求时平滑切换 |
| 多轮对话 | 支持上下文连续对话，最长 50 轮 |

**非功能需求（Non-Functional Requirements）：**

| 指标 | 目标值 | 推导 |
|------|--------|------|
| 日均对话量 | 100,000 | 业务给定 |
| 峰值倍数 | 5x | 促销 / 故障期间的流量峰值 |
| 平均 QPS | ~1.16（100K / 86400） | 日均换算 |
| 峰值 QPS | ~58（5x × 100K / 86400） | 需要扩容的目标 |
| 端到端延迟 | < 3 秒（P95） | 用户体验要求 |
| 可用性 | 99.9%（年停机 < 8.76h） | SLA 承诺 |
| 每次对话轮数 | 平均 8 轮 | 行业经验值 |

#### Step 2: 高层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户接入层 (Gateway)                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │   Web     │    │   App    │    │  微信     │    │  API     │     │
│   │ WebSocket │    │  SDK     │    │ 公众号    │    │ 开放接口  │     │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘     │
│        └───────────────┼───────────────┼───────────────┘           │
│                        ↓               ↓                           │
│              ┌─────────────────────────────┐                       │
│              │   统一消息协议 (Protobuf)     │                       │
│              └──────────┬──────────────────┘                       │
└─────────────────────────┼───────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent 编排层 (Orchestrator)                       │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │  意图路由器   │───→│  对话状态机   │───→│  工具调度器   │         │
│   │ IntentRouter │    │ StateMachine │    │ ToolDispatch │         │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘         │
│          │                   │                   │                  │
│   ┌──────┴───────┐    ┌──────┴───────┐    ┌──────┴───────┐         │
│   │ Human Handoff│    │  Session Mgr │    │  Guard Rails │         │
│   │   判断器      │    │  会话管理     │    │  安全护栏     │         │
│   └──────────────┘    └──────────────┘    └──────────────┘         │
└─────────────────────────┼───────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM 推理层 (Inference)                          │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │  模型路由器   │    │  语义缓存     │    │  Rate Limiter│         │
│   │ ModelRouter  │    │ SemanticCache│    │  限流器       │         │
│   └──────┬───────┘    └──────────────┘    └──────────────┘         │
│          ↓                                                          │
│   ┌──────────────┐    ┌──────────────┐                              │
│   │  GPT-4 级别  │    │  轻量模型     │                              │
│   │  复杂推理     │    │  简单 FAQ     │                              │
│   └──────────────┘    └──────────────┘                              │
└─────────────────────────┼───────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       数据层 (Data Store)                           │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │  向量知识库   │    │  工单系统     │    │  用户画像     │         │
│   │  Milvus/PG   │    │  MySQL/PG    │    │  Redis/ES    │         │
│   └──────────────┘    └──────────────┘    └──────────────┘         │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐                              │
│   │  对话日志     │    │  分析数据仓库 │                              │
│   │  MongoDB     │    │  ClickHouse  │                              │
│   └──────────────┘    └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

#### Step 3: 核心组件设计

##### 3.1 意图路由器（Intent Router）

意图路由器采用**两级架构**：第一级是轻量分类模型（如 BERT-base fine-tuned），延迟 < 20ms，处理 80% 的常见意图；第二级是 LLM fallback，处理分类模型无法识别的长尾意图。

```
┌────────────────────────────────────────────┐
│              用户输入                        │
└─────────────────┬──────────────────────────┘
                  ↓
┌─────────────────────────────────────────────┐
│  第一级：轻量分类模型 (BERT fine-tuned)       │
│  延迟 < 20ms | 覆盖 80% 常见意图              │
└─────────┬────────────────┬──────────────────┘
          ↓                ↓
    confidence ≥ 0.85   confidence < 0.85
          ↓                ↓
    直接路由         ┌──────────────────┐
                     │ 第二级：LLM 分类  │
                     │ 延迟 ~500ms       │
                     └────────┬─────────┘
                              ↓
                         路由到对应处理流程
```

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import numpy as np


class Intent(Enum):
    """客服系统支持的意图类型"""
    FAQ = "faq"                    # 常见问题咨询
    ORDER_QUERY = "order_query"    # 订单查询
    TICKET_CREATE = "ticket_create"  # 工单创建
    COMPLAINT = "complaint"        # 投诉
    CHITCHAT = "chitchat"          # 闲聊
    UNKNOWN = "unknown"            # 未知意图


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: Intent
    confidence: float
    source: str  # "classifier" 或 "llm_fallback"


class IntentRouter:
    """两级意图路由器：轻量分类模型 + LLM fallback"""

    CONFIDENCE_THRESHOLD = 0.85  # 分类模型置信度阈值

    def __init__(self, classifier_model, llm_client):
        self.classifier = classifier_model   # 轻量 BERT 分类模型
        self.llm_client = llm_client         # LLM 客户端
        self._intent_cache = {}              # 意图缓存

    async def route(self, user_input: str, session_context: dict) -> IntentResult:
        """
        两级意图识别：
        1. 先用轻量分类模型（< 20ms）
        2. 置信度不足时 fallback 到 LLM（~500ms）
        """
        # 第一级：轻量分类模型
        intent, confidence = await self.classifier.predict(user_input)

        if confidence >= self.CONFIDENCE_THRESHOLD:
            return IntentResult(
                intent=intent,
                confidence=confidence,
                source="classifier"
            )

        # 第二级：LLM fallback — 带上下文的精细分类
        llm_intent = await self._llm_classify(user_input, session_context)
        return IntentResult(
            intent=llm_intent,
            confidence=0.90,  # LLM 分类默认置信度
            source="llm_fallback"
        )

    async def _llm_classify(self, user_input: str, context: dict) -> Intent:
        """使用 LLM 进行意图分类，处理长尾场景"""
        prompt = f"""你是一个意图分类器。根据用户输入和对话上下文，判断用户意图。
只返回以下类别之一：faq, order_query, ticket_create, complaint, chitchat

对话历史：{context.get('history', '无')}
用户输入：{user_input}

意图类别："""
        response = await self.llm_client.complete(prompt, max_tokens=10)
        return Intent(response.strip())
```

##### 3.2 知识库 RAG 管道

RAG（Retrieval-Augmented Generation）管道负责从企业知识库中检索相关文档并生成回答。

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ 用户问题  │───→│ Query 改写    │───→│ Embedding    │───→│ 向量检索  │
└──────────┘    │ + 扩展       │    │ text-embed-3 │    │ Top-K=20 │
                └──────────────┘    └──────────────┘    └─────┬────┘
                                                              ↓
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ 流式响应  │←───│ LLM 生成      │←───│ Prompt 组装  │←───│ Re-rank  │
│ 返回用户  │    │ with Citation │    │ Top-5 文档   │    │ Top-5    │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘
```

```python
from dataclasses import dataclass, field


@dataclass
class RetrievedDoc:
    """检索到的知识库文档"""
    doc_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """知识库 RAG 管道：Embedding → 向量检索 → Re-ranking → 生成"""

    def __init__(self, embedder, vector_store, reranker, llm_client):
        self.embedder = embedder        # Embedding 模型
        self.vector_store = vector_store  # 向量数据库 (Milvus / pgvector)
        self.reranker = reranker         # Re-ranking 模型 (cross-encoder)
        self.llm_client = llm_client     # LLM 生成模型

    async def answer(self, query: str, session_context: dict) -> str:
        """端到端 RAG 回答流程"""
        # 1. Query 改写：利用对话上下文补全指代
        rewritten_query = await self._rewrite_query(query, session_context)

        # 2. Embedding + 向量检索：召回 Top-20
        query_embedding = await self.embedder.encode(rewritten_query)
        candidates = await self.vector_store.search(
            vector=query_embedding,
            top_k=20,
            filter={"status": "published"}  # 只检索已发布文档
        )

        # 3. Re-ranking：精排到 Top-5
        reranked = await self.reranker.rerank(
            query=rewritten_query,
            documents=[c.content for c in candidates],
            top_k=5
        )

        # 4. 生成回答（带引用来源）
        context_text = "\n\n".join(
            f"[文档{i+1}] {doc.content}" for i, doc in enumerate(reranked)
        )
        prompt = f"""基于以下知识库文档回答用户问题。如果文档中没有相关信息，请诚实说明。
回答末尾请标注引用来源编号。

知识库文档：
{context_text}

用户问题：{rewritten_query}

回答："""
        return await self.llm_client.complete(prompt, max_tokens=500)

    async def _rewrite_query(self, query: str, context: dict) -> str:
        """基于对话上下文改写查询，解决指代消歧"""
        history = context.get("history", [])
        if not history:
            return query
        prompt = f"""将用户最新的问题改写为独立完整的查询语句。

对话历史：
{chr(10).join(history[-4:])}

最新问题：{query}

改写后的查询："""
        return await self.llm_client.complete(prompt, max_tokens=100)
```

##### 3.3 Human Handoff 判断逻辑

Human Handoff 是智能客服系统的关键安全机制。判断维度包含四个方面：

| 触发条件 | 阈值 | 说明 |
|---------|------|------|
| 置信度低 | < 0.6 | 意图识别或回答生成的置信度过低 |
| 敏感话题 | 关键词命中 | 法律纠纷、人身安全、资金损失等 |
| 用户情绪 | 负面 ≥ 0.8 | 用户明显愤怒或不满 |
| 多轮失败 | ≥ 3 次 | 连续多轮未能解决用户问题 |

```python
@dataclass
class HandoffDecision:
    """Human Handoff 决策结果"""
    should_handoff: bool
    reason: str
    priority: str  # "high", "medium", "low"
    suggested_skill_group: str  # 建议转接的客服技能组


class HumanHandoffJudge:
    """Human Handoff 判断器：多维度评估是否需要转人工"""

    # 敏感话题关键词（实际生产中应使用分类模型）
    SENSITIVE_TOPICS = {"法律", "律师", "报警", "投诉到工信部", "人身安全", "资金被盗"}

    def __init__(self, sentiment_model):
        self.sentiment_model = sentiment_model

    async def evaluate(
        self,
        user_input: str,
        intent_result: IntentResult,
        session_context: dict
    ) -> HandoffDecision:
        """
        多维度评估是否需要转人工：
        1. 意图识别置信度
        2. 敏感话题检测
        3. 用户情绪分析
        4. 多轮失败计数
        """
        reasons = []

        # 维度 1：置信度检查
        if intent_result.confidence < 0.6:
            reasons.append(f"意图置信度过低: {intent_result.confidence:.2f}")

        # 维度 2：敏感话题检测
        if any(kw in user_input for kw in self.SENSITIVE_TOPICS):
            reasons.append("触发敏感话题关键词")

        # 维度 3：用户情绪分析
        sentiment_score = await self.sentiment_model.analyze(user_input)
        if sentiment_score.negative >= 0.8:
            reasons.append(f"用户情绪负面: {sentiment_score.negative:.2f}")

        # 维度 4：多轮失败计数
        fail_count = session_context.get("consecutive_failures", 0)
        if fail_count >= 3:
            reasons.append(f"连续 {fail_count} 轮未解决")

        # 用户主动要求转人工
        if self._user_requests_human(user_input):
            reasons.append("用户主动要求转人工")

        if not reasons:
            return HandoffDecision(
                should_handoff=False, reason="", priority="low",
                suggested_skill_group=""
            )

        # 根据触发原因确定优先级和技能组
        priority = "high" if "敏感话题" in str(reasons) else "medium"
        skill_group = self._select_skill_group(intent_result.intent, reasons)

        return HandoffDecision(
            should_handoff=True,
            reason=" | ".join(reasons),
            priority=priority,
            suggested_skill_group=skill_group
        )

    def _user_requests_human(self, text: str) -> bool:
        """检测用户是否主动要求转人工"""
        keywords = {"转人工", "人工客服", "真人", "找人", "不想跟机器人"}
        return any(kw in text for kw in keywords)

    def _select_skill_group(self, intent: Intent, reasons: list) -> str:
        """根据意图和原因选择客服技能组"""
        skill_map = {
            Intent.COMPLAINT: "投诉处理组",
            Intent.ORDER_QUERY: "订单服务组",
            Intent.TICKET_CREATE: "技术支持组",
        }
        return skill_map.get(intent, "综合服务组")
```

##### 3.4 多轮对话状态管理

使用 Redis 存储 Session 状态，通过对话摘要控制上下文长度：

```python
import json
import time
from typing import Optional


class SessionManager:
    """多轮对话 Session 管理器：Redis 存储 + 对话摘要"""

    SESSION_TTL = 3600  # Session 过期时间：1 小时
    MAX_HISTORY_TURNS = 20  # 保留的最大对话轮数
    SUMMARIZE_THRESHOLD = 10  # 触发摘要的轮数阈值

    def __init__(self, redis_client, llm_client):
        self.redis = redis_client
        self.llm_client = llm_client

    async def get_session(self, session_id: str) -> dict:
        """获取 Session，不存在则创建"""
        data = await self.redis.get(f"session:{session_id}")
        if data:
            return json.loads(data)
        return {
            "session_id": session_id,
            "history": [],
            "summary": "",
            "state": "active",
            "created_at": time.time(),
            "consecutive_failures": 0,
            "metadata": {}
        }

    async def update_session(self, session_id: str, user_msg: str, bot_msg: str):
        """更新 Session：追加对话 + 按需摘要"""
        session = await self.get_session(session_id)
        session["history"].append({"role": "user", "content": user_msg})
        session["history"].append({"role": "assistant", "content": bot_msg})

        # 对话轮数超过阈值时触发摘要压缩
        if len(session["history"]) > self.SUMMARIZE_THRESHOLD * 2:
            session = await self._compress_history(session)

        await self.redis.set(
            f"session:{session_id}",
            json.dumps(session, ensure_ascii=False),
            ex=self.SESSION_TTL
        )

    async def _compress_history(self, session: dict) -> dict:
        """压缩对话历史：前半部分生成摘要，保留最近对话"""
        history = session["history"]
        # 保留最近 6 轮（12 条消息），其余生成摘要
        to_summarize = history[:-12]
        to_keep = history[-12:]

        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in to_summarize
        )
        prompt = f"""请将以下客服对话摘要为 3-5 句话，保留关键信息（用户问题、已提供的解决方案、待处理事项）。

对话内容：
{history_text}

摘要："""
        new_summary = await self.llm_client.complete(prompt, max_tokens=200)
        session["summary"] = (session.get("summary", "") + "\n" + new_summary).strip()
        session["history"] = to_keep
        return session
```

#### Step 4: 数据流设计

一次完整请求从用户发出到收到回复的全链路：

```
用户发送消息
     │
     ↓
┌─────────────────────────┐
│ 1. Gateway 接收          │  统一协议转换，鉴权，限流
│    延迟 ~5ms             │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ 2. 加载 Session          │  从 Redis 读取对话上下文
│    延迟 ~3ms             │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ 3. 意图路由              │  BERT 分类 (~15ms)
│    延迟 15-500ms         │  或 LLM fallback (~500ms)
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ 4. Handoff 判断          │  多维度评估是否转人工
│    延迟 ~20ms            │
└────────┬────────────────┘
         ↓ (不转人工)
┌─────────────────────────┐
│ 5. 语义缓存查询          │  相似问题命中率约 30%
│    延迟 ~10ms            │
└────────┬────────────────┘
    命中 ↓         未命中 ↓
  直接返回   ┌─────────────────────────┐
             │ 6. RAG / LLM 生成       │
             │    延迟 800-2000ms       │
             └────────┬────────────────┘
                      ↓
         ┌─────────────────────────┐
         │ 7. Guard Rails 检查      │  安全过滤 + 质量校验
         │    延迟 ~50ms            │
         └────────┬────────────────┘
                  ↓
         ┌─────────────────────────┐
         │ 8. 更新 Session + 异步   │  写 Redis + 异步写日志
         │    延迟 ~5ms             │
         └────────┬────────────────┘
                  ↓
            返回响应给用户
        总延迟 P95 < 2500ms
```

#### Step 5: 扩展性设计

##### 5.1 水平扩展

```
                    ┌──────────────┐
                    │  负载均衡器   │
                    │   (Nginx)    │
                    └──────┬───────┘
            ┌──────────────┼──────────────┐
            ↓              ↓              ↓
    ┌──────────────┐┌──────────────┐┌──────────────┐
    │ Orchestrator ││ Orchestrator ││ Orchestrator │
    │   Pod #1     ││   Pod #2     ││   Pod #N     │
    │  (无状态)     ││  (无状态)     ││  (无状态)     │
    └──────┬───────┘└──────┬───────┘└──────┬───────┘
           └───────────────┼───────────────┘
                           ↓
                ┌──────────────────────┐
                │   消息队列 (Kafka)    │  ← 削峰填谷
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │  LLM Worker Pool     │  ← 独立扩缩容
                │  (GPU / API 调用)     │
                └──────────────────────┘
```

关键设计点：

- **无状态编排服务**：Session 存储在 Redis，编排器本身无状态，可水平扩缩容
- **消息队列削峰**：Kafka 缓冲请求，LLM Worker 按能力消费，避免模型过载
- **独立扩缩容**：接入层、编排层、推理层分别根据各自指标自动扩缩

##### 5.2 语义缓存策略（生产实现）

```python
class CustomerServiceSemanticCache:
    """语义缓存：相似问题复用已有回答，降低 LLM 调用成本"""

    SIMILARITY_THRESHOLD = 0.92  # 语义相似度阈值

    def __init__(self, embedder, vector_store, redis_client):
        self.embedder = embedder
        self.vector_store = vector_store  # 缓存专用向量库
        self.redis = redis_client

    async def get(self, query: str) -> Optional[str]:
        """查询语义缓存，命中则返回缓存回答"""
        query_vec = await self.embedder.encode(query)
        results = await self.vector_store.search(vector=query_vec, top_k=1)

        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            cached = await self.redis.get(f"cache:{results[0].doc_id}")
            if cached:
                return json.loads(cached)["answer"]
        return None

    async def put(self, query: str, answer: str, ttl: int = 86400):
        """写入语义缓存，默认保留 24 小时"""
        query_vec = await self.embedder.encode(query)
        doc_id = f"q_{hash(query) % (10**12)}"
        await self.vector_store.upsert(doc_id=doc_id, vector=query_vec, content=query)
        await self.redis.set(
            f"cache:{doc_id}",
            json.dumps({"query": query, "answer": answer}, ensure_ascii=False),
            ex=ttl
        )
```

##### 5.3 降级方案

当 LLM 服务不可用时，系统不能完全瘫痪：

| 降级级别 | 触发条件 | 策略 |
|---------|---------|------|
| Level 1 | LLM 延迟 > 5s | 切换到轻量模型（如 GPT-3.5 级别） |
| Level 2 | LLM 完全不可用 | 切换到 FAQ 关键词匹配模式 |
| Level 3 | FAQ 也不可用 | 自动转人工 + 排队提示 |

#### Step 6: 成本估算

以使用 GPT-4o 级别模型为例：

| 项目 | 计算 | 每日成本 |
|------|------|---------|
| 日均对话数 | 100,000 | — |
| 每次对话平均轮数 | 8 轮 | — |
| 每轮平均 Input Token | ~800 tokens（含上下文） | — |
| 每轮平均 Output Token | ~200 tokens | — |
| 总 Input Token / 日 | 100K × 8 × 800 = 6.4 亿 | — |
| 总 Output Token / 日 | 100K × 8 × 200 = 1.6 亿 | — |
| Input 成本（$2.5/1M tokens） | 640M × $2.5 / 1M | $1,600 |
| Output 成本（$10/1M tokens） | 160M × $10 / 1M | $1,600 |
| **LLM 日成本合计** | — | **$3,200** |
| 语义缓存命中率 30% | 节省 30% | -$960 |
| 轻量模型分流 40% FAQ | FAQ 用 $0.15/1M 模型 | -$1,200（约） |
| **优化后日成本** | — | **~$1,040** |
| **月成本** | × 30 | **~$31,200** |

加上基础设施（Redis、向量数据库、Kafka、K8s 集群等），月总成本约 **$40,000 - $50,000**。

#### 核心编排器：整合所有组件

```python
class CustomerServiceOrchestrator:
    """智能客服 Agent 编排器：串联所有核心组件"""

    def __init__(
        self,
        intent_router: IntentRouter,
        rag_pipeline: RAGPipeline,
        handoff_judge: HumanHandoffJudge,
        session_manager: SessionManager,
        semantic_cache: CustomerServiceSemanticCache,
    ):
        self.intent_router = intent_router
        self.rag_pipeline = rag_pipeline
        self.handoff_judge = handoff_judge
        self.session_manager = session_manager
        self.semantic_cache = semantic_cache

    async def handle_message(self, session_id: str, user_input: str) -> dict:
        """
        处理一条用户消息的完整流程：
        Session 加载 → 意图识别 → Handoff 判断 → 缓存查询 → 生成回答 → Session 更新
        """
        # 1. 加载 Session
        session = await self.session_manager.get_session(session_id)

        # 2. 意图识别
        intent_result = await self.intent_router.route(user_input, session)

        # 3. Human Handoff 判断
        handoff = await self.handoff_judge.evaluate(user_input, intent_result, session)
        if handoff.should_handoff:
            return {
                "type": "handoff",
                "message": "正在为您转接人工客服，请稍候...",
                "reason": handoff.reason,
                "priority": handoff.priority,
                "skill_group": handoff.suggested_skill_group,
            }

        # 4. 查询语义缓存
        cached_answer = await self.semantic_cache.get(user_input)
        if cached_answer:
            await self.session_manager.update_session(
                session_id, user_input, cached_answer
            )
            return {"type": "answer", "message": cached_answer, "source": "cache"}

        # 5. 根据意图分发处理
        if intent_result.intent == Intent.FAQ:
            answer = await self.rag_pipeline.answer(user_input, session)
        elif intent_result.intent == Intent.ORDER_QUERY:
            answer = await self._handle_order_query(user_input, session)
        elif intent_result.intent == Intent.TICKET_CREATE:
            answer = await self._handle_ticket_create(user_input, session)
        else:
            answer = await self.rag_pipeline.answer(user_input, session)

        # 6. 写入缓存 + 更新 Session
        await self.semantic_cache.put(user_input, answer)
        await self.session_manager.update_session(session_id, user_input, answer)

        return {"type": "answer", "message": answer, "source": "generated"}

    async def _handle_order_query(self, user_input: str, session: dict) -> str:
        """订单查询：提取订单号 → 调用订单 API → 生成回答"""
        # 实际实现中会调用订单系统 API
        return "正在为您查询订单信息..."

    async def _handle_ticket_create(self, user_input: str, session: dict) -> str:
        """工单创建：收集必要信息 → 创建工单 → 返回工单号"""
        # 实际实现中会调用工单系统 API
        return "正在为您创建工单..."
```

## 常见误区 / 面试追问

1. **误区："先做 Demo 再考虑生产化"** — 生产化不是在 Demo 上打补丁。延迟、成本、可观测性应该在设计初期就考虑（"design for latency upfront"）。

2. **误区："静态 Benchmark 高分 = 生产好用"** — 95% 的静态 Benchmark 准确率在生产中可能"完全崩溃"。原因是静态 Benchmark 基于预录制的轨迹评估，但生产环境是动态的。

3. **误区："直接用一个大模型处理所有请求就行了"** — 日均 10 万对话如果全部走 GPT-4 级别模型，月成本将超过 $10 万，且延迟无法保证。正确做法是分层处理：80% 的简单 FAQ 走轻量模型或语义缓存，只有复杂场景才调用强模型。单一模型也无法处理工单创建等需要调用外部 API 的结构化任务，必须结合 Tool Use 和状态机设计。

4. **误区："智能客服不需要 Human Handoff，AI 可以处理一切"** — 在生产环境中，Human Handoff 是不可或缺的安全兜底机制。无论模型多强大，总有它无法可靠处理的场景：涉及法律责任的投诉、情绪极度激动的用户、需要越权操作的请求等。缺少 Human Handoff 的系统会导致用户满意度急剧下降，甚至引发舆情危机。业界经验是 AI 自主处理率达到 70-85% 即为优秀，剩余的必须有人工兜底。

5. **追问："市场数据如何？"** — AI Agent 市场从 2024 年的 $54 亿增长到 2025 年的 $76 亿，预计 2030 年达 $503 亿（CAGR 45.8%）。McKinsey 数据显示 23% 的组织已在规模化部署 Agent 系统。

6. **追问："Context Engineering 是什么新概念？"** — 它是 2025-2026 年出现的新学科，将上下文视为一等公民——有自己的架构、生命周期和约束，而非简单的字符串拼接。

7. **追问："如何处理高峰期 5 倍流量？"** — 三层防护：第一层是 Gateway 限流 + 排队机制，保护下游服务不被压垮；第二层是 Kafka 消息队列削峰填谷，LLM Worker 按自身能力消费；第三层是弹性扩缩容（K8s HPA），根据 QPS 和队列积压自动扩容编排器 Pod。同时，高峰期应自动提升语义缓存的复用阈值（如从 0.92 降到 0.88），并将更多请求分流到轻量模型，在回答质量和系统可用性之间做动态平衡。

8. **追问："如何衡量这套系统的效果？"** — 建立多层指标体系。**业务指标**：自主解决率（目标 > 75%）、用户满意度 CSAT（目标 > 4.2/5）、平均对话轮数（越少越好）、人工转接率（目标 < 25%）。**技术指标**：P95 延迟（< 3s）、系统可用性（> 99.9%）、RAG 检索准确率（> 85%）、意图识别 F1-score（> 0.90）。**成本指标**：单次对话成本（目标 < $0.05）、缓存命中率（目标 > 30%）。建议搭建 A/B 实验平台，持续迭代 Prompt、检索策略和模型选型。

## 参考资料

- [LLM Agents in Production: Architectures, Challenges, and Best Practices (ZenML)](https://www.zenml.io/blog/llm-agents-in-production-architectures-challenges-and-best-practices)
- [Architecting Efficient Context-Aware Multi-Agent Framework for Production (Google Developers Blog)](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)
- [AI Agent Architecture: Build Systems That Work in 2026 (Redis)](https://redis.io/blog/ai-agent-architecture/)
- [Building Production-Ready AI Agents (Diagrid)](https://www.diagrid.io/blog/building-production-ready-ai-agents-what-your-framework-needs)
- [Engineering Production-Ready LLM Systems (Medium)](https://medium.com/@eng.fadishaar/building-large-language-model-llm-systems-that-work-in-production-7292d675b80b)
- [Building LLM-Powered Customer Service Agents (LangChain Blog)](https://blog.langchain.dev/building-llm-powered-customer-service-agents/)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Designing AI-First Customer Service Architectures (AWS Architecture Blog)](https://aws.amazon.com/blogs/architecture/designing-ai-first-customer-service-architectures/)
- [Large Language Model based Long-tail Query Rewriting in Taobao Search (Alibaba, 2023)](https://arxiv.org/abs/2311.03758)
- [Semantic Caching for LLM Applications (GPTCache Documentation)](https://gptcache.readthedocs.io/en/latest/)
