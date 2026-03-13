# LLM API 的成本优化策略

> 难度：基础
> 分类：Production & Deployment

## 简短回答

LLM API 成本优化可以通过组合策略实现最高 **80-90% 的成本削减**。六大核心策略：(1) **Prompt 优化**——精简 Prompt、去除冗余词、使用结构化输出（减少 15-40%）；(2) **Prompt 缓存**——复用静态 Prompt 前缀的处理结果，Anthropic 缓存可降低 90% 输入成本（$0.30/M vs $3.00/M）；(3) **语义缓存**——对语义相似的请求返回缓存结果，Redis LangCache 在高重复场景下减少 73% 成本；(4) **模型路由**——按任务复杂度选择模型，简单任务用小模型（RouteLLM 实现 85% 成本削减且保持 95% 质量）；(5) **RAG 替代长上下文**——用检索替代将整个文档塞入 Prompt（减少 70%+ 上下文 Token）；(6) **批处理**——非实时任务批量处理，OpenAI Batch API 50% 折扣。关键认知：**输出 Token 成本是输入 Token 的 2-5 倍**，限制输出长度有显著的成本影响。当月调用量超过 100 万次时，应考虑自建模型服务（vLLM），6-12 个月内可回收硬件投入。

## 详细解析

### 成本优化策略全景

```
成本优化策略（按投入产出排序）：

立即见效（0 额外成本）：
├── Prompt 精简：去除冗余词、缩短指令 → -15-40%
├── 限制输出长度：max_tokens 控制 → -20-30%
├── 降低温度：temperature=0 减少重试 → -5-10%
└── 结构化输出：JSON 比自然语言更省 token → -10-20%

短期投入（1-2 周）：
├── Prompt 缓存：静态内容前置 → -50-90% 缓存部分
├── 模型路由：简单任务用小模型 → -40-85%
├── 语义缓存：相似请求复用结果 → -30-73%
└── RAG 替代长上下文 → -70% 上下文 token

中期投入（1-3 个月）：
├── 批处理 API → -50% (OpenAI Batch)
├── 微调小模型替代大模型 → -60-80%
└── 自建推理服务（vLLM）→ 大规模时最经济
```

### 具体实施方案

```python
# 策略 1：Prompt 优化
class PromptOptimizer:
    """Prompt 级别的成本优化"""

    def optimize(self, prompt):
        """精简 Prompt 减少 token"""
        # 1. 去除冗余指令
        # Bad: "请你仔细地、认真地、全面地分析以下文本..."
        # Good: "分析以下文本："

        # 2. 结构化输出代替自然语言
        # Bad: "请用自然语言详细描述你的分析结果"
        # Good: "输出JSON: {result, confidence, key_points[]}"

        # 3. 输出限制
        # 输出 token 成本 = 输入的 2-5 倍！
        # 限制 max_tokens 是最直接的成本控制

        # 4. Prompt 压缩工具
        # LLMLingua: 可压缩 Prompt 最多 20x 且保持语义
        pass


# 策略 2：Prompt 缓存
class PromptCaching:
    """利用提供商的 Prompt 缓存功能"""

    # 核心原理：将静态内容放在 Prompt 开头
    # 提供商会缓存已处理的前缀，后续请求复用

    pricing_comparison = {
        "Anthropic Claude": {
            "正常输入": "$3.00/M tokens",
            "缓存命中": "$0.30/M tokens",  # 90% 折扣
            "缓存写入": "$3.75/M tokens",  # 首次写入稍贵
        },
        "OpenAI": {
            "正常输入": "标准价格",
            "缓存命中": "50% 折扣",  # 自动缓存
        },
    }

    def structure_for_caching(self, system_prompt, user_query):
        """将 Prompt 结构化以最大化缓存命中"""
        return [
            # 静态部分（会被缓存）——放在最前面
            {"role": "system", "content": system_prompt},       # 系统指令
            {"role": "system", "content": knowledge_base},      # 知识库
            {"role": "system", "content": few_shot_examples},   # 示例

            # 动态部分（每次不同）——放在最后
            {"role": "user", "content": user_query},
        ]


# 策略 3：语义缓存
class SemanticCache:
    """对语义相似的请求返回缓存结果"""

    def __init__(self):
        self.cache = RedisSemanticCache(
            similarity_threshold=0.95,  # 相似度阈值
            ttl=3600,                    # 缓存 1 小时
        )

    async def query(self, prompt):
        # 1. 检查语义缓存
        cached = await self.cache.search(prompt)
        if cached and cached.similarity > 0.95:
            return cached.response  # 毫秒级返回，零 API 成本

        # 2. 缓存未命中，调用 LLM
        response = await self.llm.invoke(prompt)

        # 3. 存入缓存供后续复用
        await self.cache.store(prompt, response)
        return response

    # 适用场景：FAQ、客服、重复性查询
    # Redis LangCache: 高重复场景下减少 73% 成本


# 策略 4：模型路由
class CostAwareRouter:
    """按任务复杂度选择模型"""

    model_tiers = {
        "tier_1": {"model": "gpt-4o-mini", "cost": "$0.15/M in"},
        "tier_2": {"model": "gpt-4o",      "cost": "$2.50/M in"},
        "tier_3": {"model": "o3",          "cost": "$10.00/M in"},
    }

    async def route(self, query):
        # 用轻量分类器判断复杂度
        complexity = await self.classify_complexity(query)

        if complexity == "simple":
            return self.model_tiers["tier_1"]   # 简单问答
        elif complexity == "moderate":
            return self.model_tiers["tier_2"]   # 分析推理
        else:
            return self.model_tiers["tier_3"]   # 复杂推理

    # RouteLLM 研究：95% GPT-4 质量，仅用 26% GPT-4 调用
    # 成本削减约 85%
```

### 成本监控与预算控制

```python
class CostMonitor:
    """成本监控和预算控制"""

    def __init__(self, daily_budget=100.0):
        self.daily_budget = daily_budget

    def track_request(self, request_id, model, input_tokens, output_tokens):
        """追踪每个请求的成本"""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        self.store.record(request_id, cost)

        # 预算告警
        daily_total = self.store.get_daily_total()
        if daily_total > self.daily_budget * 0.8:
            self.alert("日预算已使用 80%")
        if daily_total > self.daily_budget:
            self.alert("日预算已超出！考虑降级模型")

    def generate_report(self):
        """成本分析报告"""
        return {
            "按模型": self.cost_by_model(),
            "按功能": self.cost_by_feature(),
            "按用户": self.cost_by_user(),
            "Top 10 最贵请求": self.most_expensive_requests(),
            "优化建议": self.optimization_suggestions(),
        }
```

### 成本优化决策树

```
你的月调用量？
│
├── < 10K 次/月
│   → Prompt 优化 + Prompt 缓存（免费/低成本）
│   → 预计月成本 $10-100
│
├── 10K-100K 次/月
│   → + 模型路由 + 语义缓存
│   → + 批处理（非实时任务）
│   → 预计月成本 $100-1000
│
├── 100K-1M 次/月
│   → + RAG 替代长上下文
│   → + 微调小模型替代部分大模型调用
│   → 预计月成本 $500-5000
│
└── > 1M 次/月
    → 考虑自建推理服务（vLLM + 开源模型）
    → 硬件投资 $10K-50K，6-12 个月回本
    → 预计月成本取决于硬件折旧
```

## 常见误区 / 面试追问

1. **误区："用最便宜的模型就是最省钱"** — 便宜模型如果质量差导致用户重试，实际成本可能更高。正确做法是模型路由——让合适的模型处理合适的任务。RouteLLM 证明了这种方法可以在保持 95% 质量的同时削减 85% 成本。

2. **误区："成本优化会牺牲质量"** — Prompt 缓存和语义缓存不会影响质量（返回的是同样的结果）。模型路由也不是降质量——简单任务本来就不需要最强模型。真正的风险在于过度压缩 Prompt 导致指令不清晰。

3. **追问："如何预估 LLM 项目的成本？"** — (1) 估算每个用户每天的平均请求数；(2) 估算每个请求的平均 Token 数（输入+输出）；(3) 乘以模型单价；(4) 加上缓存命中率的折扣。公式：月成本 ≈ DAU × 日均请求 × 平均 Token × 单价 × (1 - 缓存率) × 30。

4. **追问："什么时候应该考虑自建？"** — 月调用量 > 100 万次、对延迟有严格要求（< 200ms）、数据隐私要求不能发送到第三方、或使用开源模型已满足质量需求时。自建的关键工具：vLLM（推理引擎）+ 开源模型（Llama/Qwen）。

## 参考资料

- [LLM Cost Optimization: Complete Guide to Reducing AI Expenses by 80% (Koombea)](https://ai.koombea.com/blog/llm-cost-optimization)
- [Reduce LLM Costs: Token Optimization Strategies (Glukhov)](https://www.glukhov.org/post/2025/11/cost-effective-llm-applications)
- [Optimize LLM Response Costs and Latency with Effective Caching (AWS)](https://aws.amazon.com/blogs/database/optimize-llm-response-costs-and-latency-with-effective-caching/)
- [LLM Token Optimization: Cut Costs & Latency in 2026 (Redis)](https://redis.io/blog/llm-token-optimization-speed-up-apps/)
- [How to Save 90% on LLM API Costs Without Losing Performance (Prem AI)](https://blog.premai.io/how-to-save-90-on-llm-api-costs-without-losing-performance/)
