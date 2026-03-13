# 设计一个生产级 Agent 系统：从单轮对话到多步任务执行

> 难度：高级
> 分类：Agent 架构

## 简短回答

生产级 Agent 系统远不止一个 LLM + 工具的 Demo。它需要在架构层面解决五大挑战：可靠的 Agent Loop 设计、上下文工程（Context Engineering）、成本与延迟控制、全链路可观测性、以及安全护栏。核心原则是"right-size your architecture"——用满足需求的最简架构，避免过度设计。

## 详细解析

### 从原型到生产的鸿沟

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

### 生产级架构全景

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

### 挑战 1：上下文工程（Context Engineering）

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

### 挑战 2：成本与延迟控制

Agent 的多步性质导致级联 LLM 调用，成本和延迟可能失控。

#### 模型路由（Model Routing）

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

#### 语义缓存

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

### 挑战 3：可观测性

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

### 挑战 4：安全与护栏

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

### 挑战 5：选择正确的架构级别

> "用满足需求的最简架构。"

```
简单单步任务              → 不需要 Agent，用 LLM + Prompt
结构明确的多步任务        → Plan-and-Execute
动态探索性任务            → ReAct
复杂多领域问题            → 多 Agent 系统（但协调开销必须物有所值）
```

### 生产检查清单

- [ ] Agent Loop 有最大步数限制和超时
- [ ] 错误处理：重试 + 降级 + 错误分类 + 检查点
- [ ] 上下文管理：摘要 + 检索 + 最小上下文原则
- [ ] 成本控制：模型路由 + 语义缓存 + 预算上限
- [ ] 可观测性：Trace + 日志 + 指标 + 告警
- [ ] 安全：输入/输出过滤 + 行动审批 + 权限最小化
- [ ] 评估：上线前基准测试 + 上线后持续评估
- [ ] 降级方案：Agent 失败时的回退策略

## 常见误区 / 面试追问

1. **误区："先做 Demo 再考虑生产化"** — 生产化不是在 Demo 上打补丁。延迟、成本、可观测性应该在设计初期就考虑（"design for latency upfront"）。

2. **误区："静态 Benchmark 高分 = 生产好用"** — 95% 的静态 Benchmark 准确率在生产中可能"完全崩溃"。原因是静态 Benchmark 基于预录制的轨迹评估，但生产环境是动态的。

3. **追问："市场数据如何？"** — AI Agent 市场从 2024 年的 $54 亿增长到 2025 年的 $76 亿，预计 2030 年达 $503 亿（CAGR 45.8%）。McKinsey 数据显示 23% 的组织已在规模化部署 Agent 系统。

4. **追问："Context Engineering 是什么新概念？"** — 它是 2025-2026 年出现的新学科，将上下文视为一等公民——有自己的架构、生命周期和约束，而非简单的字符串拼接。

## 参考资料

- [LLM Agents in Production: Architectures, Challenges, and Best Practices (ZenML)](https://www.zenml.io/blog/llm-agents-in-production-architectures-challenges-and-best-practices)
- [Architecting Efficient Context-Aware Multi-Agent Framework for Production (Google Developers Blog)](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)
- [AI Agent Architecture: Build Systems That Work in 2026 (Redis)](https://redis.io/blog/ai-agent-architecture/)
- [Building Production-Ready AI Agents (Diagrid)](https://www.diagrid.io/blog/building-production-ready-ai-agents-what-your-framework-needs)
- [Engineering Production-Ready LLM Systems (Medium)](https://medium.com/@eng.fadishaar/building-large-language-model-llm-systems-that-work-in-production-7292d675b80b)
