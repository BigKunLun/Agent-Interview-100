# 对话上下文窗口管理与压缩策略

> 难度：基础
> 分类：Memory & State

## 简短回答

上下文窗口是 LLM 的"工作记忆"——所有输入（System Prompt、对话历史、检索内容、工具结果）和输出都必须装进这个固定大小的 token 空间（4K 到百万级 tokens，如 Claude 支持 200K，Gemini 支持 1M+）。随着对话增长，上下文窗口的管理与压缩成为核心挑战。管理策略从简到繁包括：**截断（Truncation）**——直接裁剪最早的消息；**滑动窗口（Sliding Window）**——只保留最近 N 轮对话；**观察掩码（Observation Masking）**——用占位符替代旧内容；**LLM 摘要（Summarization）**——用模型将旧对话压缩为摘要；**分层摘要（Hierarchical）**——越旧的信息压缩越激进；**结构化提取**——提取实体和事实到结构化格式。研究表明，简单策略（掩码、截断）在总效率上往往不逊于复杂的 LLM 摘要。生产最佳实践是将上下文控制在窗口的 75% 以内，用 ACON 框架的分层阈值方法（40%/60%/95%）做自适应压缩，并注意"中间丢失"效应。

## 详细解析

### 一、为什么需要管理上下文窗口？

```
上下文窗口 = LLM 能"看到"的全部信息

┌─────────────────────────────────────┐
│           上下文窗口 (128K tokens)    │
│                                     │
│  System Prompt        ~2K tokens    │
│  对话历史 (50轮)      ~30K tokens   │
│  RAG 检索结果         ~5K tokens    │
│  工具调用结果         ~3K tokens    │
│  ─────────────────────────────      │
│  已使用: ~40K tokens                │
│  剩余给输出: ~88K tokens            │
└─────────────────────────────────────┘

对话增长曲线：
轮次  1: 500 tokens    ✅ 轻松
轮次 10: 5,000 tokens  ✅ 正常
轮次 50: 25,000 tokens ⚠️ 开始吃力
轮次 100: 50,000 tokens ❌ 超窗口 / 成本过高 / 注意力下降

核心问题：
1. 对话持续增长 → 总有一天超出窗口限制
2. 上下文越长 → 成本越高（按 token 计费）
3. 上下文越长 → 注意力分散（"中间丢失"效应）
4. 上下文越长 → 延迟越大
```

### 二、基础策略：截断与滑动窗口

#### 策略 1：截断（最简单）

```python
def truncate_messages(messages, max_tokens):
    """保留最新消息，截断最旧的"""
    total = 0
    kept = []
    # 从后往前保留
    for msg in reversed(messages):
        msg_tokens = count_tokens(msg)
        if total + msg_tokens > max_tokens:
            break
        kept.insert(0, msg)
        total += msg_tokens
    return kept
```

**改进版——优先级截断：**
```python
def smart_truncate(messages, max_tokens):
    """区分必须保留和可选内容"""
    must_keep = [m for m in messages if m["priority"] == "high"]
    # 系统消息 + 最新用户消息 = 必须保留
    optional = [m for m in messages if m["priority"] == "normal"]

    remaining = max_tokens - count_tokens(must_keep)
    middle = []

    for msg in reversed(optional):
        if count_tokens(msg) <= remaining:
            middle.insert(0, msg)
            remaining -= count_tokens(msg)

    return middle + must_keep
```

#### 策略 2：滑动窗口

```python
def sliding_window(messages, window_size=10):
    """只保留最近 N 轮对话"""
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    return system_messages + conversation[-window_size * 2:]  # 每轮 = user + assistant
```

**优势：** 零额外成本、可预测的 token 用量
**劣势：** 丢失所有旧上下文，Agent 会"遗忘"早期对话

#### 策略 3：观察掩码（Observation Masking）

```python
def observation_masking(messages, window=5):
    """用占位符替代旧的工具结果和长回复"""
    result = []
    for i, msg in enumerate(messages):
        if i < len(messages) - window:
            if msg["role"] == "tool" or len(msg["content"]) > 500:
                result.append({
                    "role": msg["role"],
                    "content": "[内容已省略——如需详情请重新查询]"
                })
            else:
                result.append(msg)
        else:
            result.append(msg)  # 最近 N 条保持原样
    return result
```

研究表明，掩码策略在**总效率（性能/成本比）**上与 LLM 摘要相当甚至更优。

### 三、进阶策略：LLM 摘要、分层摘要与结构化提取

#### 策略 4：LLM 摘要

```python
class IncrementalSummarizer:
    """增量式对话摘要"""

    def __init__(self):
        self.running_summary = ""

    def update(self, new_messages):
        prompt = f"""
        已有摘要：
        {self.running_summary}

        新的对话内容：
        {self.format_messages(new_messages)}

        请更新摘要。保留以下信息：
        - 用户的核心意图和需求
        - 已做出的重要决定
        - 关键数据和数字
        - 待解决的问题
        删除：
        - 寒暄和重复内容
        - 已解决的中间步骤细节
        """
        self.running_summary = llm.invoke(prompt)
        return self.running_summary
```

**优势：** 保留语义完整性，不丢失关键信息
**劣势：** 每次压缩需要一次 LLM 调用（成本 + 延迟），可能丢失细节

#### 策略 5：分层压缩（Hierarchical）

> 以下为概念示意代码，实际实现需处理边界情况（如消息不足、批次重叠等）。

```python
class HierarchicalCompression:
    """越旧的信息压缩越激进"""

    def compress(self, messages):
        total = len(messages)
        result = []

        for i, msg in enumerate(messages):
            age = total - i  # 消息的"年龄"

            if age <= 5:
                # 最近 5 轮：保留原文
                result.append(msg)
            elif age <= 20:
                # 5-20 轮前：压缩为摘要
                if i % 4 == 0:  # 每 4 条做一次摘要
                    batch = messages[i:i+4]
                    result.append(self.summarize_batch(batch))
            else:
                # 20 轮以前：只保留关键事实
                pass  # 已在更早的压缩周期中处理

        return result
```

#### 策略 6：结构化提取

```python
class StructuredExtractor:
    """从对话中提取结构化信息"""

    def extract(self, conversation):
        prompt = f"""
        从以下对话中提取关键信息，以结构化格式输出：

        {conversation}

        输出格式：
        CONTEXT: 当前在做什么
        ENTITIES: 提到的关键实体（人名、产品、订单号等）
        DECISIONS: 已做出的决定
        PENDING: 待解决的问题
        USER_PREFERENCES: 用户表达的偏好
        """
        return llm.invoke(prompt)
```

**优势：** 信息密度最高，不丢失关键事实
**劣势：** 提取质量依赖 LLM，可能遗漏隐式信息

### 四、生产实践方案

#### ACON 框架：自适应分层压缩

```python
class ACONCompressor:
    """ACON: Agent Context Optimization"""

    THRESHOLDS = {
        "normal": 0.40,     # < 40% 上下文使用：不压缩
        "moderate": 0.60,   # 40-60%：温和压缩（摘要旧内容）
        "aggressive": 0.95, # 60-95%：激进压缩（只保留关键信息）
        "emergency": 1.00,  # > 95%：紧急截断
    }

    def compress(self, messages, current_usage_ratio):
        if current_usage_ratio < self.THRESHOLDS["normal"]:
            return messages  # 不压缩

        elif current_usage_ratio < self.THRESHOLDS["moderate"]:
            # 温和压缩：摘要 20 轮前的内容
            return self.summarize_old(messages, keep_recent=20)

        elif current_usage_ratio < self.THRESHOLDS["aggressive"]:
            # 激进压缩：只保留最近 5 轮 + 结构化摘要
            return self.structural_compress(messages, keep_recent=5)

        else:
            # 紧急截断
            return self.emergency_truncate(messages, keep_recent=3)
```

ACON 研究结果：减少 26-54% 峰值 token 使用量，同时保持任务成功率。对于小模型，性能还能提升 20-46%。

#### 混合策略（生产推荐）

```python
class ProductionContextManager:
    """组合多种策略的生产级上下文管理"""

    def __init__(self, config):
        self.max_context = config.max_tokens * 0.75  # 预留 25% 给输出
        self.buffer_size = config.buffer_turns        # 保留最近 N 轮原文
        self.vector_store = config.vector_store        # 语义检索历史

    def build_context(self, current_query, full_history):
        context = []
        budget = self.max_context

        # 1. System Prompt（必须保留）
        context.append(self.system_prompt)
        budget -= count_tokens(self.system_prompt)

        # 2. 最近 N 轮原文（高保真）
        recent = full_history[-self.buffer_size * 2:]
        context.extend(recent)
        budget -= count_tokens(recent)

        # 3. 语义检索相关历史（按需补充）
        if budget > 1000:
            relevant = self.vector_store.search(current_query, top_k=3)
            for doc in relevant:
                if count_tokens(doc) < budget:
                    context.insert(1, {"role": "system",
                                      "content": f"相关历史：{doc}"})
                    budget -= count_tokens(doc)

        # 4. 旧对话摘要（兜底）
        if len(full_history) > self.buffer_size * 2:
            old = full_history[:-self.buffer_size * 2]
            summary = self.summarize(old)
            context.insert(1, {"role": "system",
                              "content": f"历史摘要：{summary}"})

        return context
```

#### 注意"中间丢失"效应

```
LLM 对上下文中不同位置的注意力：

高 ██████░░░░░░░░░░░░░░░░░██████  高
   开头                        结尾
         ░░░░░░░░░░░░░░░░
              中间（容易被忽略）

实践建议：
- 最重要的指令放在开头（System Prompt）
- 当前用户消息放在结尾
- 中间放参考资料和历史上下文
```

#### 策略选择指南

| 策略 | 成本 | 信息保留 | 实现复杂度 | 适用场景 |
|------|------|---------|-----------|---------|
| 截断 | 零 | 低 | 极简 | 短对话、快速原型 |
| 滑动窗口 | 零 | 低 | 极简 | 简单聊天 |
| 观察掩码 | 零 | 中 | 简单 | 工具密集型 Agent |
| LLM 摘要 | 高（+7%） | 高 | 中 | 长对话助手 |
| 分层压缩 | 中 | 高 | 中 | 长时运行的 Agent |
| 结构化提取 | 中 | 最高 | 高 | 客服/销售场景 |
| ACON 自适应 | 自适应 | 高 | 高 | 生产级 Agent |

#### 关键洞察

```python
insights = {
    "简单未必差": "掩码和截断将成本减半，且可靠性往往优于摘要",
    "摘要≠万能": "通用摘要捕获'发生了什么'但丢失'进展到哪了'",
    "场景决定策略": "工具密集型用掩码，对话密集型用摘要",
    "组合优于单一": "最佳实践是组合多种策略：最近原文 + 中期摘要 + 远期结构化提取",
}
```

## 常见误区 / 面试追问

1. **误区："上下文窗口越大越好 / 越大就不需要压缩"** — 更大的窗口意味着更高的成本（按 token 计费）、更大的延迟、以及更严重的"中间丢失"效应。即使 200K 窗口，长对话仍会导致成本线性增长和注意力质量下降。压缩不仅省钱，还能提高模型注意力的质量。最佳实践是精心管理上下文内容，而非依赖大窗口。

2. **误区："只要不超过上下文限制就没问题"** — 即使不超限，上下文中的噪声信息也会降低模型的准确率（"上下文腐化"）。精简的、高相关性的上下文优于冗长的完整历史。

3. **误区："LLM 摘要是最佳策略"** — 研究表明简单的观察掩码在总效率上不逊于 LLM 摘要。摘要额外消耗约 7%+ 成本（经验估算值，实际因场景而异），且不总是能保留正确信息。应该根据场景选择，而非默认使用最复杂的方案。

4. **追问："摘要压缩的成本问题怎么解决？"** — 每次摘要需要一次额外 LLM 调用，可能占总成本的 7%+。优化方案：(1) 只在 token 超阈值时才触发摘要；(2) 用小模型做摘要；(3) 用观察掩码（Observation Masking）替代 LLM 摘要。

5. **追问："如何评估压缩策略的质量？"** — 两个维度：(1) 信息保留率——压缩后 Agent 能否正确回答依赖历史信息的问题；(2) 成本效率——总 token 消耗（含压缩调用本身的开销）。LoCoBench-Agent 是专门评估长上下文压缩的基准测试。

6. **追问："增量摘要 vs 全量重新摘要？"** — 增量摘要（每次只处理新消息）更快更便宜，但可能累积误差。全量重新摘要更准确但成本高。实践中用增量摘要 + 定期全量校正的混合方式。

7. **追问："Claude 的 Server-Side Compaction 是什么？"** — Claude API 支持服务端自动压缩——当对话过长时，系统自动压缩之前的消息，保持对话可以无限延续。还支持清除工具结果和思考过程来节省空间。

## 参考资料

- [Context Window Management Strategies (APXML)](https://apxml.com/courses/langchain-production-llm/chapter-3-advanced-memory-management/context-window-management)
- [Top Techniques to Manage Context Length in LLMs (Agenta)](https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms)
- [Context Window Management for AI Agents (Maxim AI)](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/)
- [Context Windows (Claude API Docs)](https://platform.claude.com/docs/en/build-with-claude/context-windows)
- [LLM Context Windows: What They Are & How They Work (Redis)](https://redis.io/blog/llm-context-windows/)
- [LLM Chat History Summarization Guide (Mem0)](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025)
- [Smarter Context Management for LLM-Powered Agents (JetBrains Research)](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [Evaluating Context Compression for AI Agents (Factory.ai)](https://factory.ai/news/evaluating-compression)
- [ACON: Optimizing Context Compression for Long-Horizon LLM Agents (arXiv)](https://arxiv.org/html/2510.00615v1)
