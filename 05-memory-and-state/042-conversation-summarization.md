# 对话历史的摘要与压缩策略

> 难度：基础
> 分类：Memory & State

## 简短回答

对话历史的摘要与压缩是解决上下文窗口有限的核心手段。主要策略从简到繁包括：**滑动窗口**（只保留最后 N 条消息）、**观察掩码**（用占位符替代旧内容）、**LLM 摘要**（用模型将旧对话压缩为摘要）、**分层摘要**（越旧的信息压缩越激进）、**结构化提取**（提取实体和事实到结构化格式）。研究表明，简单策略（掩码、截断）在总效率上往往不逊于复杂的 LLM 摘要——后者虽然"听起来更聪明"，但额外的 LLM 调用成本可占总花费的 7%+，且不一定能可靠地超越简单方法。推荐用 ACON 框架的分层阈值方法（40%/60%/95%）做自适应压缩。

## 详细解析

### 为什么需要压缩？

```
对话增长曲线：
轮次  1: 500 tokens    ✅ 轻松
轮次 10: 5,000 tokens  ✅ 正常
轮次 50: 25,000 tokens ⚠️ 开始吃力
轮次 100: 50,000 tokens ❌ 超窗口 / 成本过高 / 注意力下降
```

### 策略 1：滑动窗口（最简单）

```python
def sliding_window(messages, keep_last_n=10):
    """只保留最后 N 条消息"""
    system = [m for m in messages if m["role"] == "system"]
    history = [m for m in messages if m["role"] != "system"]
    return system + history[-keep_last_n:]
```

**优势：** 零额外成本、可预测的 token 用量
**劣势：** 丢失所有旧上下文，Agent 会"遗忘"早期对话

### 策略 2：观察掩码（Observation Masking）

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

### 策略 3：LLM 摘要

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

### 策略 4：分层压缩（Hierarchical）

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

### 策略 5：结构化提取

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

### ACON 框架：自适应分层压缩

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

### 策略选择指南

| 策略 | 成本 | 信息保留 | 实现复杂度 | 适用场景 |
|------|------|---------|-----------|---------|
| 滑动窗口 | 零 | 低 | 极简 | 简单聊天 |
| 观察掩码 | 零 | 中 | 简单 | 工具密集型 Agent |
| LLM 摘要 | 高（+7%） | 高 | 中 | 长对话助手 |
| 分层压缩 | 中 | 高 | 中 | 长时运行的 Agent |
| 结构化提取 | 中 | 最高 | 高 | 客服/销售场景 |
| ACON 自适应 | 自适应 | 高 | 高 | 生产级 Agent |

### 关键洞察

```python
insights = {
    "简单未必差": "掩码和截断将成本减半，且可靠性往往优于摘要",
    "摘要≠万能": "通用摘要捕获'发生了什么'但丢失'进展到哪了'",
    "场景决定策略": "工具密集型用掩码，对话密集型用摘要",
    "组合优于单一": "最佳实践是组合多种策略：最近原文 + 中期摘要 + 远期结构化提取",
}
```

## 常见误区 / 面试追问

1. **误区："LLM 摘要是最佳策略"** — 研究表明简单的观察掩码在总效率上不逊于 LLM 摘要。摘要额外消耗约 7%+ 成本（经验估算值，实际因场景而异），且不总是能保留正确信息。应该根据场景选择，而非默认使用最复杂的方案。

2. **误区："上下文窗口越大就不需要压缩"** — 即使 200K 窗口，长对话仍会导致：成本线性增长、延迟增加、"中间丢失"效应加剧。压缩不仅省钱，还能提高模型注意力的质量。

3. **追问："如何评估压缩策略的质量？"** — 两个维度：(1) 信息保留率——压缩后 Agent 能否正确回答依赖历史信息的问题；(2) 成本效率——总 token 消耗（含压缩调用本身的开销）。LoCoBench-Agent 是专门评估长上下文压缩的基准测试。

4. **追问："增量摘要 vs 全量重新摘要？"** — 增量摘要（每次只处理新消息）更快更便宜，但可能累积误差。全量重新摘要更准确但成本高。实践中用增量摘要 + 定期全量校正的混合方式。

## 参考资料

- [LLM Chat History Summarization Guide (Mem0)](https://mem0.ai/blog/llm-chat-history-summarization-guide-2025)
- [Smarter Context Management for LLM-Powered Agents (JetBrains Research)](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)
- [Evaluating Context Compression for AI Agents (Factory.ai)](https://factory.ai/news/evaluating-compression)
- [ACON: Optimizing Context Compression for Long-Horizon LLM Agents (arXiv)](https://arxiv.org/html/2510.00615v1)
- [Context Window Management for AI Agents (Maxim AI)](https://www.getmaxim.ai/articles/context-window-management-strategies-for-long-context-ai-agents-and-chatbots/)
