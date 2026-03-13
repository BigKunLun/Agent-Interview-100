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

3. **追问："没有 ground truth 能评估吗？"** — Faithfulness 和 Answer Relevancy 不需要 ground truth（基于 LLM-as-Judge）。但 Context Recall 需要。可以先从不需要 ground truth 的指标开始，逐步建立标注数据集。

4. **追问："如何评估 Agentic RAG？"** — 除了上述指标，还需要评估：(1) 工具选择正确率；(2) 检索轮次效率（几轮找到答案）；(3) 端到端延迟和成本。

## 参考资料

- [Metrics (RAGAS Official Docs)](https://docs.ragas.io/en/stable/concepts/metrics/overview/)
- [RAG Evaluation Metrics (Confident AI)](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)
- [RAG Evaluation Metrics: Best Practices (Patronus AI)](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)
- [Best Practices in RAG Evaluation (Qdrant)](https://qdrant.tech/blog/rag-evaluation-guide/)
- [Get Better RAG Responses with RAGAS (Redis)](https://redis.io/blog/get-better-rag-responses-with-ragas/)
