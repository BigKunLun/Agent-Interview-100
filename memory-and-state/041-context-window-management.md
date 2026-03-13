# 如何管理对话上下文窗口？

> 难度：基础
> 分类：Memory & State

## 简短回答

上下文窗口是 LLM 的"工作记忆"——所有输入（System Prompt、对话历史、检索内容、工具结果）和输出都必须装进这个固定大小的 token 空间（4K 到百万级 tokens，如 Claude 支持 200K，Gemini 支持 1M+）。管理策略从简到繁包括：**截断（Truncation）**——直接裁剪最早的消息；**滑动窗口（Sliding Window）**——只保留最近 N 轮对话；**摘要压缩（Summarization）**——用 LLM 将旧对话压缩为摘要；**混合策略**——最近几轮保持原文 + 更早内容压缩为摘要 + 向量检索补充相关历史。生产最佳实践是将上下文控制在窗口的 75% 以内，并注意"中间丢失"效应。

## 详细解析

### 为什么上下文窗口管理重要？

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

问题：
1. 对话持续增长 → 总有一天超出窗口限制
2. 上下文越长 → 成本越高（按 token 计费）
3. 上下文越长 → 注意力分散（"中间丢失"效应）
4. 上下文越长 → 延迟越大
```

### 策略 1：截断（最简单）

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

### 策略 2：滑动窗口

```python
def sliding_window(messages, window_size=10):
    """只保留最近 N 轮对话"""
    system_messages = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]
    return system_messages + conversation[-window_size * 2:]  # 每轮 = user + assistant
```

### 策略 3：摘要压缩

```python
class SummaryBuffer:
    def __init__(self, max_token_limit=4000):
        self.summary = ""           # 历史摘要
        self.buffer = []            # 最近几轮原文
        self.max_tokens = max_token_limit

    def add_message(self, message):
        self.buffer.append(message)

        # 当 buffer 超出限制时，压缩最早的消息到摘要中
        while count_tokens(self.buffer) > self.max_tokens:
            oldest = self.buffer[:2]  # 取最早的一轮
            self.buffer = self.buffer[2:]
            # 用 LLM 将旧消息合并到摘要
            self.summary = self.compress(self.summary, oldest)

    def compress(self, existing_summary, old_messages):
        prompt = f"""
        已有摘要: {existing_summary}
        新的对话内容: {format(old_messages)}

        请更新摘要，保留所有关键信息（用户意图、已做的决定、
        重要数据点）。摘要应简洁但信息完整。
        """
        return llm.invoke(prompt)

    def get_context(self):
        """返回：摘要 + 最近几轮原文"""
        context = []
        if self.summary:
            context.append({"role": "system",
                          "content": f"之前对话摘要：{self.summary}"})
        context.extend(self.buffer)
        return context
```

### 策略 4：混合策略（生产推荐）

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

### 注意"中间丢失"效应

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

### 优雅降级

```python
class GracefulDegradation:
    """上下文溢出时的优雅降级"""

    THRESHOLDS = {
        0.4: "正常",        # < 40% 使用量
        0.6: "警告",        # 开始压缩旧对话
        0.95: "紧急",       # 激进截断
    }

    def manage(self, usage_ratio):
        if usage_ratio < 0.4:
            return "keep_all"
        elif usage_ratio < 0.6:
            return "summarize_old"
        elif usage_ratio < 0.95:
            return "aggressive_compression"
        else:
            return "emergency_truncate"
```

## 常见误区 / 面试追问

1. **误区："上下文窗口越大越好"** — 更大的窗口意味着更高的成本（按 token 计费）、更大的延迟、以及更严重的"中间丢失"效应。最佳实践是精心管理上下文内容，而非依赖大窗口。

2. **误区："只要不超过上下文限制就没问题"** — 即使不超限，上下文中的噪声信息也会降低模型的准确率（"上下文腐化"）。精简的、高相关性的上下文优于冗长的完整历史。

3. **追问："摘要压缩的成本问题怎么解决？"** — 每次摘要需要一次额外 LLM 调用，可能占总成本的 7%+。优化方案：(1) 只在 token 超阈值时才触发摘要；(2) 用小模型做摘要；(3) 用观察掩码（Observation Masking）替代 LLM 摘要。

4. **追问："Claude 的 Server-Side Compaction 是什么？"** — Claude API 支持服务端自动压缩——当对话过长时，系统自动压缩之前的消息，保持对话可以无限延续。还支持清除工具结果和思考过程来节省空间。

## 参考资料

- [Context Window Management Strategies (APXML)](https://apxml.com/courses/langchain-production-llm/chapter-3-advanced-memory-management/context-window-management)
- [Top Techniques to Manage Context Length in LLMs (Agenta)](https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms)
- [Context Window Management for AI Agents (Maxim AI)](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/)
- [Context Windows (Claude API Docs)](https://platform.claude.com/docs/en/build-with-claude/context-windows)
- [LLM Context Windows: What They Are & How They Work (Redis)](https://redis.io/blog/llm-context-windows/)
