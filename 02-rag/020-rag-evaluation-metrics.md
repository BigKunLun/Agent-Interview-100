# 如何评估 RAG 系统的性能？关键指标与评估框架

> 难度：高级
> 分类：RAG

## 简短回答

RAG 评估需要分别评估检索器和生成器两个环节。检索器用 **Context Precision**（排序质量）和 **Context Recall**（召回完整度）衡量；生成器用 **Faithfulness**（忠实度，是否基于检索内容）和 **Answer Relevancy**（答案相关性）衡量。RAGAS 是最流行的开源评估框架，结合 LLM-as-Judge 实现自动化评估。其与人工标注的一致率在忠实度上达 95%。

## 详细解析

### 为什么 RAG 评估特殊？

RAG 系统由两个独立模块（检索器 + 生成器）组成，评估不能只看最终输出：
- 最终答案正确，可能是 LLM 碰巧用自身知识答对（检索实际没用）
- 最终答案错误，可能是检索结果好但 LLM 没用好（不是检索的问题）

因此需要**分模块评估**，定位问题出在哪个环节。

### RAGAS 框架的四大核心指标

```
                     ┌─────────────┐
      检索器指标      │             │     生成器指标
                     │   RAG 系统   │
  Context Precision ←│  检索 → 生成  │→ Faithfulness
  Context Recall   ←│             │→ Answer Relevancy
                     └─────────────┘
```

#### 1. Context Precision（上下文精确率）

**衡量什么：** 检索到的文档中，相关文档是否排在前面？

```
检索结果排序：[相关, 不相关, 相关, 不相关, 不相关]
Precision@1 = 1/1 = 1.0
Precision@2 = 1/2 = 0.5
Precision@3 = 2/3 = 0.67
Context Precision = mean(precision@k for relevant items) = (1.0 + 0.67) / 2 = 0.835
```

高 Context Precision 意味着最相关的文档排在前面，LLM 更容易利用它们。

#### Context Relevancy vs Context Precision

| 指标 | 关注点 | 适用场景 | 计算方式 |
|------|--------|---------|---------|
| **Context Precision** | 排序质量 | 相关文档是否排在前面 | 基于 precision@k 计算 |
| **Context Relevancy** | 整体相关性 | 检索内容有多大比例是相关的 | LLM 评估检索内容与查询的相关性 |

**Context Relevancy 计算：**
1. 从检索到的上下文中提取独立陈述
2. 让 LLM 评估每个陈述与查询的相关性
3. 计算相关陈述的比例：`Relevancy = 相关陈述数 / 总陈述数`

```python
# 示例
query = "RAG 的核心组件是什么？"
retrieved_context = """
RAG 是检索增强生成技术，它结合了大语言模型和外部知识库。
这种方法可以提高回答的准确性。今天天气不错。
"""

# LLM 分析：
# "RAG 是检索增强生成技术" → 相关 ✓
# "它结合了大语言模型和外部知识库" → 相关 ✓
# "这种方法可以提高回答的准确性" → 相关 ✓
# "今天天气不错" → 不相关 ✗

# Context Relevancy = 3/4 = 0.75
```

#### 2. Context Recall（上下文召回率）

**衡量什么：** 回答问题所需的所有关键信息是否都被检索到了？

这是 RAGAS 框架中唯一需要人工标注 ground truth 的指标。方法是将 ground truth 答案拆分为多个关键点（claim），然后检查检索到的文档是否覆盖了每个关键点。

```python
# 示例
ground_truth = "RAG 有三个组件：索引、检索和生成"
# 关键点：[索引, 检索, 生成]
# 检索文档覆盖了：[索引, 检索]（缺少"生成"的描述）
# Context Recall = 2/3 = 0.67
```

#### 3. Faithfulness（忠实度）

**衡量什么：** 生成的回答是否忠实于检索到的文档？（即幻觉率的反面）

```python
# 步骤 1：将回答拆分为独立的事实陈述
answer = "RAG 可以减少幻觉，成本比微调低 80%"
claims = ["RAG 可以减少幻觉", "RAG 成本比微调低 80%"]

# 步骤 2：检查每个陈述是否被检索文档支持
# "RAG 可以减少幻觉" → 文档中有支持 ✓
# "成本比微调低 80%" → 文档中没有提到具体数字 ✗

# Faithfulness = 1/2 = 0.5（LLM 编造了 80% 这个数字）
```

Faithfulness 本质上是衡量幻觉率：`幻觉率 = 1 - Faithfulness`。

#### 4. Answer Relevancy（答案相关性）

**衡量什么：** 生成的回答是否真正回答了用户的问题？

避免以下情况被评为高分：
- 回答虽然正确但偏题
- 回答包含过多无关信息
- 回答过于笼统，没有针对性

### 传统 IR 指标在 RAG 中的应用

#### MRR（Mean Reciprocal Rank，平均倒数排名）

**衡量什么：** 第一个相关文档出现在检索结果中的排名位置。

**公式：**
```
MRR = (1/Q) × Σ(1/rank_i)
```
其中：Q = 总查询数，rank_i = 查询 i 中第一个相关文档的排名

**示例：**
```python
# 查询 1: "什么是 RAG?"
# 检索结果: [不相关, 不相关, 相关, 相关, 相关]
# 第一个相关文档在第 3 位 → RR = 1/3 ≈ 0.333

# 查询 2: "如何优化 RAG?"
# 检索结果: [相关, 不相关, 不相关, 不相关, 不相关]
# 第一个相关文档在第 1 位 → RR = 1/1 = 1.0

# MRR = (0.333 + 1.0) / 2 = 0.667
```

**适用场景：**
- 只关心第一个相关结果（如问答系统）
- 用户只会看第一个答案的场景
- 想快速衡量检索效率

#### MAP（Mean Average Precision，平均平均精确率）

**衡量什么：** 所有相关文档在检索结果中的整体排序质量。

**计算步骤：**
1. 对每个查询计算 AP（Average Precision）
2. 对所有查询的 AP 取平均得到 MAP

**AP 计算：**
```
AP = (1/相关文档数) × Σ(Precision@k) for each relevant doc at position k
```

**示例：**
```python
# 查询: "RAG 的组件有哪些?"
# 检索结果: [相关, 不相关, 相关, 不相关, 相关]
#             k=1    k=2    k=3    k=4    k=5

# P@1 = 1/1 = 1.0  (第一个是相关)
# P@3 = 2/3 = 0.67 (前3个中有2个相关)
# P@5 = 3/5 = 0.6  (前5个中有3个相关)

# AP = (1.0 + 0.67 + 0.6) / 3 = 0.757

# 对多个查询的 AP 取平均即为 MAP
```

**适用场景：**
- 需要评估多个相关文档的排序质量
- 用户会浏览多个结果（如搜索引擎）
- 关心整体检索性能

#### 指标选择标准

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 问答系统，用户只看第一个答案 | **MRR** | 关注第一个相关结果的位置 |
| 搜索引擎，用户浏览多个结果 | **MAP** | 评估整体排序质量 |
| RAG 系统，上下文有多块文档 | **Context Precision** | 关注排序，LLM 会按顺序阅读 |
| RAG 系统，需要完整回答 | **Context Recall** | 确保检索到所有必要信息 |
| 多级相关性（高/中/低） | **nDCG** | 考虑相关性等级，高相关排前面更优 |

**2025 研究发现：**
- 传统 IR 指标（MRR、MAP、nDCG）在某些 RAG 场景下可能失效
- LLM 可以从部分上下文中合成正确答案，即使传统指标得分较低
- 建议结合 LLM-as-Judge 指标（如 Faithfulness、Answer Relevancy）使用

### 实现评估流水线

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

# 准备评估数据
eval_data = Dataset.from_dict({
    "question": ["什么是 RAG？", "如何优化 RAG 延迟？"],
    "answer": [rag_system.answer(q) for q in questions],       # 系统生成的答案
    "contexts": [rag_system.retrieve(q) for q in questions],   # 检索到的文档
    "ground_truth": ["RAG 是检索增强生成...", "可以通过缓存..."],  # 人工标注
})

# 运行评估
results = evaluate(
    dataset=eval_data,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

print(results)
# {'context_precision': 0.85, 'context_recall': 0.78,
#  'faithfulness': 0.92, 'answer_relevancy': 0.88}
```

### 评估指标的诊断价值

| 症状 | 可能的问题 | 优化方向 |
|------|-----------|---------|
| Precision 低, Recall 高 | 检索到了相关文档，但排序不好 | 加入 Reranking |
| Precision 高, Recall 低 | 检索到的都相关，但遗漏了关键信息 | 增加 top_k、改进查询 |
| MRR 低 | 第一个相关结果出现得太晚 | 调整排序算法、使用混合检索 |
| MAP 低 | 整体排序质量差 | 优化 embedding 模型、改进 chunk 策略 |
| Context Relevancy 低 | 检索了太多无关内容 | 调整相似度阈值、优化查询扩展 |
| Faithfulness 低 | LLM 在编造内容，没有使用检索结果 | 改进 Prompt、加入引用约束 |
| Relevancy 低 | 回答偏题或过于笼统 | 优化 Prompt、改进检索质量 |
| 全部低 | 系统性问题 | 检查分块策略和 Embedding 模型 |

### RAGAS 的可靠性

初始论文中的对比实验显示 RAGAS 与人工标注的一致率：
- **Faithfulness**：95% 一致
- **Answer Relevancy**：78% 一致
- **Context Relevance**：70% 一致

RAGAS 是可靠的第一步自动化评估方案，但不能完全替代人工评估，特别是在高风险场景中。

### 其他评估工具

| 工具 | 特色 |
|------|------|
| **RAGAS** | 开源，最流行的 RAG 专用评估框架 |
| **LangSmith** | LangChain 生态，支持 Tracing + 评估 + 监控 |
| **Braintrust** | 端到端 LLM 可观测性 |
| **DeepEval** | 开源，支持多种 RAG 和 Agent 指标 |
| **Patronus AI** | 商业化，企业级评估 |

### 评估最佳实践

1. **先关注用户满意度维度**——虽然有很多指标，但优先选择反映核心用户价值的几个
2. **优先选择客观性强的指标**——团队成员独立标注后一致率 >= 80% 说明指标足够客观
3. **持续评估**——不是上线前评一次就完事，要建立持续评估流水线监控回归
4. **分模块优化**——根据指标定位是检索还是生成的问题，针对性优化

## 常见误区 / 面试追问

1. **误区："只看最终答案对不对就行"** — 无法定位问题环节。答案正确可能是 LLM 用自身知识答对（检索没起作用），答案错误可能是检索好但 LLM 没利用好。

2. **误区："自动评估可以替代人工评估"** — RAGAS 与人工一致率 70-95%，在高风险场景（医疗、法律、金融）仍需要人工评估兜底。

3. **追问："没有 ground truth 能评估吗？"** — Faithfulness、Answer Relevancy 和 Context Relevancy 不需要 ground truth（基于 LLM-as-Judge）。但 Context Recall 需要。可以先从不需要 ground truth 的指标开始，逐步建立标注数据集。

4. **追问："MRR 和 MAP 有什么区别？"** — MRR 只关注第一个相关文档的位置，适用于问答系统等用户只看第一个结果的场景。MAP 评估所有相关文档的排序质量，适用于用户会浏览多个结果的搜索场景。选择哪个取决于用户行为模式。

5. **追问："Context Relevancy 和 Context Precision 的区别？"** — Context Relevancy 衡量检索内容与查询的整体相关性比例；Context Precision 关注排序质量，即相关文档是否排在前面。前者是"有多少相关"，后者是"相关的是否排在前面"。

6. **追问："如何评估 Agentic RAG？"** — 除了上述指标，还需要评估：(1) 工具选择正确率；(2) 检索轮次效率（几轮找到答案）；(3) 端到端延迟和成本。

7. **场景追问："你的 RAG 系统在生产环境中 Context Recall 突然从 0.8 降到 0.3，Faithfulness 也随之下降。如何快速定位根因并修复？"** — 这是检索系统退化的典型信号。根因分析路径：(1) 检查最近是否有知识库更新 → 新增内容可能索引失败或 embedding 质量不佳；(2) 检查向量数据库健康状态 → 索引损坏、存储空间不足或查询延迟异常；(3) 对比失败查询和成功查询的模式 → 可能是某类查询格式变化导致检索失效；(4) 检查 embedding 模型版本是否变更 → 模型更新可能导致向量空间漂移；(5) 修复：重建索引、回滚 embedding 模型、或针对失败模式调整查询策略。

8. **场景追问："你的 RAG 系统评估显示 Answer Relevancy 很高但 Faithfulness 很低，用户反馈回答很流畅但经常编造内容。如何优化？"** — 这是典型的"高相关性高幻觉"问题。优化路径：(1) 检查 LLM 的 System Prompt → 加入明确约束"只能基于提供的上下文回答，不能编造"；(2) 调整温度参数 → 降低温度减少随机性；(3) 加入引用要求 → LLM 必须标注每个事实的来源；(4) 实施后处理验证 → 用 LLM-as-Judge 检查输出中的每个声明是否在上下文中出现；(5) 对高幻觉问题启用人工审核通道。

9. **场景追问："你的 RAG 系统检索到的文档都很相关（高 Context Precision），但用户抱怨回答不够深入。如何改进？"** — 这是检索广度不足的问题。改进路径：(1) 增加 top_k → 检索更多文档；(2) 优化查询扩展 → 用 LLM 生成多个变体查询并合并结果；(3) 改进分块策略 → 当前分块可能太小，无法包含完整的上下文；(4) 加入 Reranking → 扩大检索范围后用 Reranker 重新排序，确保质量同时增加覆盖；(5) 实施 Agentic RAG → 让 Agent 根据需要主动发起多轮检索。

## 参考资料

- [Metrics (RAGAS Official Docs)](https://docs.ragas.io/en/stable/concepts/metrics/overview/)
- [Contextual Relevancy Metric (Confident AI)](https://www.confident-ai.com/docs/metrics/single-turn/contextual-relevancy-metric)
- [Mean Reciprocal Rank Explained (Evidently AI)](https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr)
- [MRR vs MAP vs NDCG: When to Use Them (Medium)](https://medium.com/swlh/mrr-vs-map-vs-ndcg-rank-aware-evaluation-metrics-and-when-to-use-them-5191bba16832)
- [RAG Evaluation Metrics Guide (Future AGI 2025)](https://futureagi.com/blogs/rag-evaluation-metrics-2025)
- [A Complete Guide to RAG Evaluation (Evidently AI)](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Redefining Retrieval Evaluation in the Era of LLMs (arXiv 2025)](https://arxiv.org/html/2510.21440v1)
- [RAG Evaluation Metrics (Confident AI)](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
- [RAG Evaluation Metrics: Best Practices (Patronus AI)](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)
- [Best Practices in RAG Evaluation (Qdrant)](https://qdrant.tech/blog/rag-evaluation-guide/)
- [Get Better RAG Responses with RAGAS (Redis)](https://redis.io/blog/get-better-rag-responses-with-ragas/)
