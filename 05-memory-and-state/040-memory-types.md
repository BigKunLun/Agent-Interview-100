# Agent 记忆的类型：短期记忆、长期记忆、工作记忆

> 难度：基础
> 分类：Memory & State

## 简短回答

LLM Agent 的记忆系统借鉴认知科学，分为三种核心类型：**短期记忆（Short-Term Memory）**——当前会话的对话历史，存在于 LLM 的上下文窗口中，会话结束即丢失；**长期记忆（Long-Term Memory）**——持久化存储在外部系统（向量数据库、关系数据库）中的知识，跨会话保留，需要检索才能使用；**工作记忆（Working Memory）**——Agent 在当前任务中主动维护的关键信息子集，本质上就是上下文窗口中当前可用的信息。关键理解：LLM 本身是无状态的，所有"记忆"都是工程层面的外部实现。

## 详细解析

### LLM 的记忆本质

```
核心事实：LLM 本身是无状态的
每次 API 调用都是独立的——模型不会"记住"之前的调用。

产品层面的"记忆"（如 ChatGPT 记住你的名字）是在 LLM 之上
工程化实现的记忆层，不是模型的固有能力。

构建 Agent 时，你需要自己实现这个记忆层。
```

### 短期记忆（Short-Term Memory / STM）

```python
# 短期记忆 = 上下文窗口中的对话历史
conversation_history = [
    {"role": "user", "content": "帮我查一下订单 #12345"},
    {"role": "assistant", "content": "订单 #12345 已发货，预计明天到达"},
    {"role": "user", "content": "物流单号是多少？"},
    # Agent 可以回答，因为"订单 #12345"在短期记忆中
]

# LLM 调用时，完整对话历史作为输入
response = llm.invoke(messages=conversation_history)
```

**特点：**
- 存储位置：LLM 的上下文窗口（Context Window）
- 生命周期：当前会话
- 容量限制：受上下文窗口大小限制（4K-200K tokens）
- 访问方式：直接可用，无需检索
- 丢失时机：会话结束、上下文窗口溢出

### 长期记忆（Long-Term Memory / LTM）

```python
# 长期记忆 = 外部存储 + 检索
class LongTermMemory:
    def __init__(self, vector_db, relational_db):
        self.vector_db = vector_db      # 语义检索
        self.relational_db = relational_db  # 结构化查询

    def store(self, memory: dict):
        """存储记忆到外部系统"""
        # 向量化存储（用于语义检索）
        embedding = embed(memory["content"])
        self.vector_db.upsert(
            id=memory["id"],
            vector=embedding,
            metadata=memory["metadata"]
        )
        # 结构化存储（用于精确查询）
        self.relational_db.insert(memory)

    def retrieve(self, query: str, top_k: int = 5) -> list:
        """基于语义相似度检索相关记忆"""
        query_embedding = embed(query)
        results = self.vector_db.search(query_embedding, top_k=top_k)
        return results
```

长期记忆的三个认知子类型（LangMem 框架）：

```python
memory_subtypes = {
    "语义记忆 (Semantic)": {
        "定义": "事实和知识——Agent 知道的东西",
        "示例": "用户偏好、产品知识、规则",
        "存储": "向量数据库 Collection 或结构化 Profile",
        "检索": "按语义相似度搜索",
    },
    "情景记忆 (Episodic)": {
        "定义": "经历和事件——Agent 做过的事",
        "示例": "成功解决问题的完整交互记录",
        "存储": "带上下文的完整交互日志",
        "检索": "按情境相似度检索（类似 few-shot）",
    },
    "程序记忆 (Procedural)": {
        "定义": "技能和流程——Agent 知道怎么做",
        "示例": "System Prompt 中的规则和步骤",
        "存储": "Prompt 模板、规则引擎",
        "检索": "通常固定加载，不动态检索",
    },
}
```

### 工作记忆（Working Memory）

```python
# 工作记忆 = 当前上下文窗口中的所有信息
# 包括：System Prompt + 对话历史 + 检索到的长期记忆 + 工具结果

working_memory = {
    "system_prompt": "你是客服助手...",           # 程序记忆
    "user_profile": "用户偏好：简洁回答",          # 从长期记忆检索
    "relevant_docs": ["退款政策: ...", "退货流程: ..."],  # RAG 检索
    "conversation": [...],                        # 短期记忆
    "tool_results": {"order_status": "shipped"},  # 工具调用结果
}

# 所有这些组合起来 = Agent 当前"能想到"的全部信息
context_window = format_for_llm(working_memory)
response = llm.invoke(context_window)
```

**类比操作系统：**
```
上下文窗口（工作记忆）≈ RAM（当前运行的程序可直接访问，容量有限）
长期记忆（外部存储）≈ 硬盘（大容量但需要读取操作）
```

MemGPT 正是基于这个类比，实现了类似操作系统虚拟内存的机制——自动在"内存"（上下文窗口）和"磁盘"（外部存储）之间移动数据。

### 记忆管理策略

```python
class MemoryManager:
    """记忆在 STM 和 LTM 之间的流转"""

    def consolidate(self, short_term_memory):
        """将重要的短期记忆转化为长期记忆"""
        for memory in short_term_memory:
            importance = self.score_importance(memory)
            if importance > self.threshold:
                self.long_term.store(memory)

    def forget(self, long_term_memory):
        """遗忘机制：清除低价值的长期记忆"""
        for memory in long_term_memory:
            # 时间衰减：越老的记忆权重越低
            memory.relevance *= 0.95
            # 频率加权：经常被访问的记忆保留
            if memory.access_count < 2 and memory.relevance < 0.1:
                self.long_term.delete(memory.id)

    def load_to_working(self, query):
        """将相关长期记忆加载到工作记忆"""
        relevant = self.long_term.retrieve(query, top_k=5)
        self.working_memory.update(relevant)
```

### 最新研究：AgeMem（统一记忆管理）

2026 年 1 月提出的 AgeMem 框架将记忆操作暴露为工具调用，让 Agent 自主决定何时存储、检索、更新、摘要或丢弃信息：

```python
# AgeMem：记忆操作即工具
memory_tools = [
    {"name": "memory_store", "description": "存储重要信息到长期记忆"},
    {"name": "memory_retrieve", "description": "从长期记忆中检索相关信息"},
    {"name": "memory_update", "description": "更新已有的记忆条目"},
    {"name": "memory_summarize", "description": "将多条记忆合并为摘要"},
    {"name": "memory_forget", "description": "删除不再需要的记忆"},
]
# Agent 在任务执行中自主调用这些工具管理记忆
```

### 三种记忆的关系

| 维度 | 短期记忆 | 工作记忆 | 长期记忆 |
|------|---------|---------|---------|
| 位置 | 上下文窗口 | 上下文窗口 | 外部存储 |
| 容量 | 受窗口限制 | 受窗口限制 | 近乎无限 |
| 持久性 | 会话级 | 请求级 | 永久 |
| 访问速度 | 即时 | 即时 | 需检索 |
| 内容 | 对话历史 | STM + 检索的 LTM | 所有历史知识 |

## 常见误区 / 面试追问

1. **误区："上下文窗口越大，就不需要长期记忆了"** — 即使 200K token 的窗口，也无法存储数月的用户交互历史。更重要的是，窗口越大成本越高，且 LLM 在超长上下文中的注意力会分散（"lost in the middle" 问题）。

2. **误区："短期记忆和工作记忆是一回事"** — 短期记忆是对话历史的累积；工作记忆是当前步骤可用的全部信息（包括从长期记忆检索的内容）。工作记忆 = 短期记忆 + 检索的长期记忆 + System Prompt + 工具结果。

3. **追问："如何决定什么信息值得存入长期记忆？"** — 用重要性评分：(1) 用户明确要求记住的信息（高优先级）；(2) 影响未来决策的事实（如用户偏好）；(3) 成功解决问题的范例（情景记忆）。避免存储临时性信息和中间推理步骤。

4. **追问："MemGPT 的核心思想是什么？"** — 将上下文窗口视为有限的"内存"资源，实现类似操作系统虚拟内存的层级管理。Agent 可以在核心记忆（RAM/上下文窗口）和归档记忆（磁盘/外部存储）之间主动移动数据，创造"无限记忆"的体验。

## 参考资料

- [Agent Memory: What, Why and How (Mem0)](https://mem0.ai/blog/memory-in-agents-what-why-and-how/)
- [LangMem: Long-term Memory Concepts (LangChain)](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [AgeMem: Unified Long-Term and Short-Term Memory (arXiv)](https://arxiv.org/abs/2601.01885)
- [Agent Memory: How to Build Agents that Learn and Remember (Letta)](https://www.letta.com/blog/agent-memory)
- [Memory Overview (LangChain Docs)](https://docs.langchain.com/oss/python/concepts/memory)
