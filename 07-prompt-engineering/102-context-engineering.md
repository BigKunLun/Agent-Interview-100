# 什么是 Context Engineering？它与 Prompt Engineering 有何本质区别？

> 难度：高级
> 分类：Prompt Engineering

## 简短回答

**Context Engineering** 是一种从"如何写好单条 Prompt"到"如何为 LLM 构建完整、精准上下文"的范式转变。它关注的核心问题不再是措辞技巧，而是**上下文的选择、组装与管理**。一个 LLM 调用的效果，70% 取决于喂给它的上下文质量，而非 Prompt 本身的遣词造句。Context Engineering 将上下文视为四大来源的动态组合：**System Prompt**、**Tool Results**、**Conversation History** 和 **External Knowledge**（RAG / API）。其核心挑战在于**上下文窗口是有限的"房地产"**——必须在有限的 token 预算内，为当前任务挑选信息密度最高的上下文片段。在 **Agentic 场景**中，这一挑战尤为突出：多轮 tool call 会持续积累上下文，若不加管理，关键信息会被淹没在噪声中，导致 Agent 性能急剧下降。

## 详细解析

### 从 Prompt Engineering 到 Context Engineering

Prompt Engineering 聚焦于"怎么问"，Context Engineering 聚焦于"用什么信息去问"。这是一个维度的跃升：

```
┌─────────────────────────────────────────────────────────┐
│                  Prompt Engineering                      │
│   "如何措辞、格式化一条指令以获得更好的输出"                  │
│   ┌─────────────────────────────┐                       │
│   │  System Prompt 写作技巧     │                       │
│   │  Few-shot 示例设计          │                       │
│   │  CoT / ReAct 格式           │                       │
│   └─────────────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼ 范式升级
┌─────────────────────────────────────────────────────────┐
│                 Context Engineering                      │
│   "如何为 LLM 构建正确的、完整的决策上下文"                  │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │ System   │ │ Tool     │ │ History  │ │ External │  │
│   │ Prompt   │ │ Results  │ │ 对话历史  │ │ Knowledge│  │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│        │            │            │            │          │
│        └────────────┴─────┬──────┴────────────┘          │
│                           ▼                              │
│               ┌───────────────────┐                      │
│               │  动态上下文组装器  │                      │
│               │ (Context Builder) │                      │
│               └─────────┬─────────┘                      │
│                         ▼                                │
│               ┌───────────────────┐                      │
│               │   LLM 调用入口    │                      │
│               └───────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### 四大上下文来源

| 来源 | 内容 | 特征 | 管理策略 |
|------|------|------|----------|
| **System Prompt** | 角色定义、工具描述、输出格式、约束规则 | 静态、每次调用都携带 | 精简压缩，避免冗余 |
| **Tool Results** | 搜索结果、API 返回、代码执行输出 | 动态生成、体积不可控 | 截断 / 摘要 / 选择性保留 |
| **Conversation History** | 用户多轮对话、之前的 Agent 推理过程 | 线性增长、含大量噪声 | 滑动窗口 / 摘要压缩 |
| **External Knowledge** | RAG 检索文档、知识库、用户画像 | 按需注入、相关性参差 | 相关性排序 + top-k 截断 |

### 上下文窗口的"房地产"管理

```
┌────────────────── Context Window (128K tokens) ──────────────────┐
│                                                                  │
│  ┌──────────────┐  固定区域：~10%                                │
│  │ System Prompt│  角色 + 工具描述 + 格式要求                      │
│  └──────────────┘                                                │
│  ┌──────────────┐  弹性区域：~30%                                │
│  │ RAG / 知识库  │  根据查询动态检索                               │
│  └──────────────┘                                                │
│  ┌──────────────┐  增长区域：~40%                                │
│  │ 对话历史      │  多轮交互 + tool call 结果                      │
│  └──────────────┘                                                │
│  ┌──────────────┐  预留区域：~20%                                │
│  │ 输出空间      │  模型生成 token 的预算                          │
│  └──────────────┘                                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

关键原则：上下文不是越多越好，而是信噪比越高越好。
每加入一段上下文都要问：它对当前决策的边际贡献 > 它消耗的 token 成本吗？
```

### Prompt Engineering vs Context Engineering 对比

| 维度 | Prompt Engineering | Context Engineering |
|------|-------------------|-------------------|
| 核心关注 | 单条指令的措辞与格式 | 整体上下文的选择与组装 |
| 优化对象 | Prompt 模板 | 上下文管道 (Context Pipeline) |
| 技能要求 | 语言直觉 + 试错调优 | 系统设计 + 信息架构 |
| 适用场景 | 单轮问答、简单任务 | Agentic 系统、复杂工作流 |
| 评估指标 | 输出质量 | 上下文信噪比 + 输出质量 |
| 可扩展性 | 低（人工逐条调优） | 高（程序化、可自动化） |
| 类比 | 写好一封邮件 | 管理整个项目的信息流 |

### 动态上下文组装器：Python 实现

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ContextSource(Enum):
    """上下文来源类型"""
    SYSTEM = "system"
    TOOL_RESULT = "tool_result"
    HISTORY = "history"
    KNOWLEDGE = "knowledge"


@dataclass
class ContextBlock:
    """单个上下文块"""
    source: ContextSource
    content: str
    priority: int = 5          # 1-10，越高越重要
    token_count: int = 0       # 预估 token 数
    metadata: dict = field(default_factory=dict)


class ContextBuilder:
    """动态上下文组装器——Context Engineering 的核心实现"""

    def __init__(self, max_tokens: int = 128_000, reserve_output: float = 0.2):
        self.max_tokens = max_tokens
        # 为模型输出预留空间
        self.available_tokens = int(max_tokens * (1 - reserve_output))
        self.blocks: list[ContextBlock] = []

        # 各来源的 token 预算上限（百分比）
        self.budget = {
            ContextSource.SYSTEM: 0.12,      # System Prompt 最多占 12%
            ContextSource.KNOWLEDGE: 0.35,   # RAG 知识最多占 35%
            ContextSource.HISTORY: 0.35,     # 对话历史最多占 35%
            ContextSource.TOOL_RESULT: 0.18, # 工具结果最多占 18%
        }

    def add(self, source: ContextSource, content: str,
            priority: int = 5, metadata: Optional[dict] = None) -> "ContextBuilder":
        """添加一个上下文块"""
        token_count = self._estimate_tokens(content)
        self.blocks.append(ContextBlock(
            source=source,
            content=content,
            priority=priority,
            token_count=token_count,
            metadata=metadata or {},
        ))
        return self  # 链式调用

    def build(self) -> list[dict]:
        """
        核心方法：根据优先级和预算，组装最终上下文。
        返回 OpenAI 兼容的 messages 格式。
        """
        # 第一步：按来源分组
        grouped: dict[ContextSource, list[ContextBlock]] = {}
        for block in self.blocks:
            grouped.setdefault(block.source, []).append(block)

        # 第二步：每组内按优先级降序排列
        for source in grouped:
            grouped[source].sort(key=lambda b: b.priority, reverse=True)

        # 第三步：按预算分配，优先级高的先占位
        selected: list[ContextBlock] = []
        for source, blocks in grouped.items():
            budget_tokens = int(self.available_tokens * self.budget.get(source, 0.1))
            used = 0
            for block in blocks:
                if used + block.token_count <= budget_tokens:
                    selected.append(block)
                    used += block.token_count
                else:
                    # 尝试截断以填满预算
                    remaining = budget_tokens - used
                    if remaining > 100:  # 至少保留 100 token 才值得截断
                        truncated = self._truncate(block.content, remaining)
                        selected.append(ContextBlock(
                            source=block.source,
                            content=truncated,
                            priority=block.priority,
                            token_count=remaining,
                        ))
                    break

        # 第四步：组装为 messages 格式
        return self._to_messages(selected)

    def _to_messages(self, blocks: list[ContextBlock]) -> list[dict]:
        """将上下文块转为 LLM messages 格式"""
        messages = []
        # System Prompt 放最前面
        sys = [b for b in blocks if b.source == ContextSource.SYSTEM]
        if sys:
            messages.append({"role": "system",
                             "content": "\n\n".join(b.content for b in sys)})
        # RAG 知识注入
        know = [b for b in blocks if b.source == ContextSource.KNOWLEDGE]
        if know:
            messages.append({"role": "system",
                             "content": "[相关知识]\n" + "\n---\n".join(b.content for b in know)})
        # 对话历史按原始顺序
        hist = sorted([b for b in blocks if b.source == ContextSource.HISTORY],
                       key=lambda b: b.metadata.get("turn_index", 0))
        for b in hist:
            messages.append({"role": b.metadata.get("role", "user"), "content": b.content})
        return messages

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（中文约 1.5 字符/token，英文约 4 字符/token）"""
        cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        en_chars = len(text) - cn_chars
        return int(cn_chars / 1.5 + en_chars / 4)

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        """按 token 预算截断文本，保留前部内容"""
        # 简化实现：按字符比例截断
        ratio = max_tokens / max(ContextBuilder._estimate_tokens(text), 1)
        cut_point = int(len(text) * min(ratio, 1.0))
        return text[:cut_point] + "\n...[已截断]"


# ── 使用示例 ──────────────────────────────────────────────
if __name__ == "__main__":
    ctx = ContextBuilder(max_tokens=128_000)
    ctx.add(ContextSource.SYSTEM, "你是资深数据分析师 Agent...", priority=10)
    ctx.add(ContextSource.KNOWLEDGE, "[文档] Q3 营收同比增长 23%...", priority=8)
    ctx.add(ContextSource.HISTORY, "帮我分析上季度销售数据", priority=7,
            metadata={"role": "user", "turn_index": 0})
    ctx.add(ContextSource.TOOL_RESULT, "SQL 结果：| 7月 | 520万 | +15% |...", priority=9)
    messages = ctx.build()  # 自动按预算裁剪、按优先级排序
    print(f"组装完成，共 {len(messages)} 条 messages")
```

### Agentic 场景的特殊挑战

在 Agent 多轮执行过程中，上下文管理面临独特挑战：

```
Agent 执行循环中的上下文膨胀问题：

Turn 1:  System(2K) + User(0.5K)                          = 2.5K tokens
Turn 3:  System(2K) + History(3K) + ToolResult(5K)         = 10K tokens
Turn 7:  System(2K) + History(12K) + ToolResults(25K)      = 39K tokens
Turn 12: System(2K) + History(28K) + ToolResults(60K)      = 90K tokens  ⚠️
Turn 15: System(2K) + History(40K) + ToolResults(85K)      = 127K tokens 💥
                                                             ↑ 接近窗口上限

常见应对策略：
┌────────────────────────────────────────────┐
│ 策略 1：滑动窗口（Sliding Window）           │
│   只保留最近 N 轮对话，丢弃早期历史           │
│                                            │
│ 策略 2：摘要压缩（Summarization）            │
│   用 LLM 将早期对话压缩为摘要                │
│                                            │
│ 策略 3：工具结果裁剪（Result Pruning）        │
│   只保留工具结果的关键字段，丢弃原始数据       │
│                                            │
│ 策略 4：分层记忆（Hierarchical Memory）       │
│   短期记忆 → 工作记忆 → 长期记忆分级存储      │
└────────────────────────────────────────────┘
```

在 Multi-Agent 系统中，挑战进一步放大：每个 Agent 有独立的上下文窗口，Agent 之间的信息传递需要精心设计——传太多导致 token 浪费，传太少导致信息丢失。Context Engineering 在此扮演的是**信息路由器**的角色，决定哪些信息传递给哪个 Agent、以什么粒度传递。

## 常见误区 / 面试追问

1. **误区："Context Engineering 就是写更好的 Prompt"** — 这是最常见的混淆。Prompt Engineering 是 Context Engineering 的一个子集。Context Engineering 不仅关注 Prompt 本身，更关注 Prompt 之外的所有信息——工具返回值如何裁剪、对话历史如何压缩、外部知识如何检索与排序。类比来说，Prompt Engineering 是写好一封邮件，Context Engineering 是管理整个项目的信息流。

2. **误区："上下文越多越好，反正模型支持 128K"** — 研究表明 LLM 存在"Lost in the Middle"现象：当上下文过长时，模型对中间位置信息的注意力显著下降。此外，无关上下文会稀释模型对关键信息的注意力，降低输出质量。正确的做法是追求**高信噪比**而非高信息量——每一段上下文都应该对当前任务有明确的边际贡献。

3. **追问："如何衡量上下文质量？"** — 可以从三个维度评估：(1) **相关性**——上下文中每段信息与当前任务的相关度（可用 embedding 相似度量化）；(2) **充分性**——是否包含完成任务所需的全部关键信息（通过 ablation test 验证）；(3) **效率**——信息密度，即有效 token 占总 token 的比例。实践中常用 A/B 测试对比不同上下文策略对最终输出质量的影响。

4. **追问："Context Engineering 在 Multi-Agent 中如何应用？"** — 在 Multi-Agent 架构中，Context Engineering 承担三重角色：(1) **上下文隔离**——每个 Agent 只接收与其职责相关的上下文，避免信息过载；(2) **上下文传递**——设计 Agent 之间的信息传递协议，决定传递摘要还是原始数据；(3) **共享上下文管理**——维护全局状态（如 Blackboard 模式），让多个 Agent 读写共享的结构化上下文，而非互相透传完整历史。

## 参考资料

- [Context Engineering for AI Agents — Lessons from Building Real Systems (Philipp Schmid)](https://www.philschmid.de/context-engineering)
- [Prompt Design != Context Engineering (Torantulino / LangChain Blog)](https://blog.langchain.dev/context-engineering/)
- [Lost in the Middle: How Language Models Use Long Contexts (Nelson Liu et al., 2023)](https://arxiv.org/abs/2307.03172)
- [Building Effective Agents (Anthropic)](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [The Shift from Prompt Engineering to Context Engineering (Andrej Karpathy)](https://x.com/karpathy/status/1937902205765607918)
