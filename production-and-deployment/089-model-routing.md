# 模型路由（Model Routing）：如何根据任务复杂度选择模型？

> 难度：中级
> 分类：Production & Deployment

## 简短回答

模型路由（Model Routing）是在运行时根据每个请求的特征（复杂度、类型、延迟要求、成本约束）**动态选择最合适的 LLM** 的技术。核心思想：不用一个模型处理所有任务——简单问答用 GPT-4o-mini（$0.15/M token），复杂推理用 GPT-4o（$2.50/M），极难问题用 o3（$10/M）。ICLR 2025 发表的 **RouteLLM** 证明：通过智能路由可以仅使用 26% 的 GPT-4 调用就达到 95% 的 GPT-4 质量，成本削减约 **85%**。主要路由策略：(1) **基于规则**——关键词/长度/类别匹配（最简单）；(2) **分类器路由**——训练小模型判断复杂度（RouteLLM 方法）；(3) **语义路由**——基于查询的向量相似度匹配最佳模型（Red Hat LLM Semantic Router）；(4) **级联路由**——先用小模型尝试，不确定时升级到大模型。2026 年调查显示 37% 的企业在生产中使用 5+ 个模型，智能路由已从"优化手段"变为"必备基础设施"。主流工具：RouteLLM、LiteLLM、Portkey、Azure Model Router、OpenRouter。

## 详细解析

### 为什么需要模型路由

```
问题：所有请求都用同一个模型
├── 用最强模型（如 GPT-4o）→ 成本高，简单任务浪费
├── 用最便宜模型（如 GPT-4o-mini）→ 复杂任务质量差
└── 手动选择 → 不可扩展，需要领域知识

解决：智能路由
├── 自动判断每个请求的复杂度
├── 路由到"刚好够用"的模型
├── 平衡质量、成本和延迟
└── 结果：85% 成本削减 + 95% 质量保持
```

### 四种路由策略

```python
# 策略 1：基于规则的路由（最简单，立即可用）
class RuleBasedRouter:
    """基于预定义规则路由"""

    def route(self, query):
        # 按任务类型
        if query.task_type == "translation":
            return "gpt-4o-mini"      # 翻译用小模型即可
        if query.task_type == "code_generation":
            return "claude-sonnet-4-6"  # 代码生成用中端模型
        if query.task_type == "complex_reasoning":
            return "o3"               # 复杂推理用最强模型

        # 按输入长度
        if len(query.text) < 100:
            return "gpt-4o-mini"      # 短查询用小模型
        elif len(query.text) < 2000:
            return "gpt-4o"
        else:
            return "claude-sonnet-4-6"  # 长文本用大上下文模型

    # 优势：简单、快速、可预测
    # 劣势：规则维护成本高，无法处理边界情况


# 策略 2：分类器路由（RouteLLM 方法）
class ClassifierRouter:
    """用训练好的分类器判断路由"""

    def __init__(self):
        # RouteLLM 的矩阵分解路由器
        self.router = MatrixFactorizationRouter(
            strong_model="gpt-4o",
            weak_model="gpt-4o-mini",
            threshold=0.7,  # 置信度阈值
        )

    async def route(self, query):
        # 分类器预测：这个查询需要强模型吗？
        score = self.router.predict(query)

        if score > self.threshold:
            return "gpt-4o"       # 需要强模型
        else:
            return "gpt-4o-mini"  # 弱模型即可

    # RouteLLM 成果（ICLR 2025）：
    # - 95% GPT-4 质量，仅 26% GPT-4 调用
    # - 数据增强后：95% 质量，仅 14% 强模型调用
    # - 成本削减 75-85%


# 策略 3：语义路由（基于向量相似度）
class SemanticRouter:
    """基于查询语义匹配最佳模型"""

    def __init__(self):
        # 每个模型的擅长领域用向量表示
        self.model_profiles = {
            "code_model": embed("代码生成、调试、重构..."),
            "creative_model": embed("写作、创意、文案..."),
            "reasoning_model": embed("数学、逻辑、分析..."),
            "general_model": embed("通用问答、对话..."),
        }

    async def route(self, query):
        query_vector = embed(query)
        # 找到语义最匹配的模型
        best_match = max(
            self.model_profiles.items(),
            key=lambda x: cosine_similarity(query_vector, x[1]),
        )
        return best_match[0]


# 策略 4：级联路由（先小后大）
class CascadeRouter:
    """先用小模型尝试，不确定时升级"""

    async def route_with_cascade(self, query):
        # Step 1: 先用最便宜的模型
        response = await self.call("gpt-4o-mini", query)

        # Step 2: 检查置信度
        if response.confidence > 0.8:
            return response  # 小模型够用

        # Step 3: 升级到强模型
        response = await self.call("gpt-4o", query)
        return response

    # 优势：确保质量下限
    # 劣势：不确定时成本翻倍（两次调用）
    # 优化：只对 ~20% 的请求需要升级
```

### 多维度路由决策

```python
class MultiDimensionalRouter:
    """综合考虑多个维度的路由决策"""

    async def route(self, query, constraints):
        candidates = self.get_available_models()

        scored_candidates = []
        for model in candidates:
            score = self.score_model(
                model=model,
                query=query,
                weights={
                    "quality":  constraints.get("quality_weight", 0.4),
                    "cost":     constraints.get("cost_weight", 0.3),
                    "latency":  constraints.get("latency_weight", 0.2),
                    "privacy":  constraints.get("privacy_weight", 0.1),
                },
            )
            scored_candidates.append((model, score))

        return max(scored_candidates, key=lambda x: x[1])

    # 路由维度：
    # - 质量：模型在该类型任务上的预期表现
    # - 成本：每个 token 的价格
    # - 延迟：模型的响应时间
    # - 隐私：是否需要本地部署（敏感数据）
    # - 可用性：模型的当前健康状态
```

### 路由工具生态

```
┌────────────────┬────────────────────────────────────┐
│ 工具           │ 特色                               │
├────────────────┼────────────────────────────────────┤
│ RouteLLM       │ ICLR 2025，学术界验证的路由算法   │
│                │ 矩阵分解路由器，开源               │
├────────────────┼────────────────────────────────────┤
│ LiteLLM        │ 统一 100+ 模型 API                 │
│                │ 内置路由、故障转移、负载均衡       │
├────────────────┼────────────────────────────────────┤
│ Portkey        │ AI Gateway，企业级路由             │
│                │ 条件路由 + 自动故障转移             │
├────────────────┼────────────────────────────────────┤
│ OpenRouter     │ 统一接口访问所有主流模型           │
│                │ 自动选择性价比最优的提供商         │
├────────────────┼────────────────────────────────────┤
│ Azure Model    │ 微软企业级路由                     │
│ Router         │ 实时评估复杂度、成本、性能         │
├────────────────┼────────────────────────────────────┤
│ AnyLLM         │ 强化学习驱动的动态路由             │
│                │ 自动学习最优路由策略               │
└────────────────┴────────────────────────────────────┘
```

## 常见误区 / 面试追问

1. **误区："路由器本身的开销会抵消收益"** — 分类器路由（如 RouteLLM 的矩阵分解模型）推理时间 < 10ms，成本几乎为零。相比于将简单请求从 $2.50/M 路由到 $0.15/M 的节省，路由器开销微不足道。

2. **误区："两个模型就够了（强+弱）"** — 实际生产系统通常需要 3-5 个模型：最轻量（FAQ/简单任务）→ 中端（一般任务）→ 高端（复杂推理）→ 特化模型（代码/数学/创意）。37% 的企业在 2026 年使用 5+ 个模型。

3. **追问："如何评估路由器的效果？"** — 两个核心指标：(1) **质量保持率**——路由后的整体质量 vs 全部使用强模型的质量（目标 > 95%）；(2) **成本削减率**——路由后的总成本 vs 全部使用强模型的成本（目标 > 50%）。同时监控各模型的实际调用比例和各自的质量指标。

4. **追问："路由器会不会过时？更强的模型会不会让路由不再需要？"** — 不会。即使最强的模型变得很便宜，不同模型在不同任务上仍有各自优势（速度、专业性、上下文长度）。而且新的强模型出来后，旧的模型变成"弱模型"，路由的价值持续存在。这是一个结构性需求。

## 参考资料

- [RouteLLM: Learning to Route LLMs with Preference Data (ICLR 2025)](https://www.marktechpost.com/2025/12/30/meet-llmrouter-an-intelligent-routing-system-designed-to-optimize-llm-inference-by-dynamically-selecting-the-most-suitable-model-for-each-query/)
- [LLM Semantic Router: Intelligent Request Routing (Red Hat)](https://developers.redhat.com/articles/2025/05/20/llm-semantic-router-intelligent-request-routing)
- [Task-Based LLM Routing: Optimizing LLM Performance (Portkey)](https://portkey.ai/blog/task-based-llm-routing/)
- [Intelligent LLM Routing: How Multi-Model AI Cuts Costs by 85% (Swfte AI)](https://www.swfte.com/blog/intelligent-llm-routing-multi-model-ai)
- [What is LLM Router? (TrueFoundry)](https://www.truefoundry.com/blog/what-is-llm-router)
