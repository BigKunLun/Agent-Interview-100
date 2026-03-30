# RAG 评估指标体系：原理、计算与实战

> **难度**：高级
> **标签**：RAG、评估指标、Retrieval、Generation、RAGAS

---

## 1. 简短回答

RAG 系统的评估需要从**检索质量**和**生成质量**两个维度同时度量。检索侧使用 Precision@K、Recall@K、MRR、MAP、NDCG 等经典信息检索指标衡量"找得准不准"；生成侧使用 Faithfulness（忠实度）和 Answer Relevancy（答案相关性）衡量"答得好不好"。RAGAS 框架将这些指标整合为一条自动化评估流水线，可在离线和 CI/CD 场景中持续监控 RAG 系统的端到端质量。

---

## 2. 详细解析

### 一、RAG 评估的整体框架

一个完整的 RAG 系统可以拆解为两个阶段，评估也相应分为两个维度：

```
Query → [Retriever] → Contexts → [Generator] → Answer
           ↓                          ↓
      检索质量评估               生成质量评估
```

| 维度 | 核心问题 | 代表指标 |
|------|---------|---------|
| **检索质量** | 检索到的文档是否相关？排序是否合理？ | Precision@K, Recall@K, Context Precision, Context Recall, MRR, MAP, NDCG |
| **生成质量** | 答案是否忠于检索到的上下文？是否切题？ | Faithfulness, Answer Relevancy |

> **关键认知**：检索质量是生成质量的上界——如果检索阶段丢失了关键信息，生成阶段不可能凭空补回来。因此实践中应**先优化检索，再调优生成**。

---

### 二、检索质量指标

#### 2.1 Precision@K 与 Recall@K

**Precision@K（精确率@K）**

- **定义**：在检索返回的前 K 个文档中，相关文档所占的比例。
- **公式**：

$$
\text{Precision@K} = \frac{|\{\text{前 K 个文档}\} \cap \{\text{相关文档}\}|}{K}
$$

**Recall@K（召回率@K）**

- **定义**：在所有相关文档中，被检索返回的前 K 个文档覆盖了多少比例。
- **公式**：

$$
\text{Recall@K} = \frac{|\{\text{前 K 个文档}\} \cap \{\text{相关文档}\}|}{|\{\text{全部相关文档}\}|}
$$

**手算示例**：

假设对于一个 query，共有 **4 篇相关文档**（ground truth），检索器返回了 Top-5 结果：

| 排名 | 文档 | 是否相关 |
|------|------|---------|
| 1 | Doc A | ✅ |
| 2 | Doc B | ❌ |
| 3 | Doc C | ✅ |
| 4 | Doc D | ✅ |
| 5 | Doc E | ❌ |

```
Precision@5 = 3 / 5 = 0.6
Recall@5    = 3 / 4 = 0.75
```

**Python 验证代码**：

```python
def precision_at_k(retrieved: list[bool], k: int) -> float:
    """retrieved: 按排名顺序的相关性列表，True 表示相关"""
    return sum(retrieved[:k]) / k

def recall_at_k(retrieved: list[bool], k: int, total_relevant: int) -> float:
    return sum(retrieved[:k]) / total_relevant

# 示例数据
retrieved = [True, False, True, True, False]
total_relevant = 4

print(f"Precision@5 = {precision_at_k(retrieved, 5)}")   # 0.6
print(f"Recall@5    = {recall_at_k(retrieved, 5, total_relevant)}")  # 0.75
```

---

#### 2.2 Context Precision 与 Context Recall

这两个指标是 RAGAS 框架对传统 Precision/Recall 的 RAG 场景适配版本，核心区别在于**用 LLM 判断相关性**而非人工标注。

**Context Precision**

- **定义**：检索到的上下文中，每个相关上下文的排名加权精确率。排名越靠前的位置出现相关上下文，得分越高。
- **公式**：

$$
\text{Context Precision} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times \mathbf{1}[d_k \text{ 相关}] \right)}{|\{\text{相关上下文}\}|}
$$

其中 $\mathbf{1}[d_k \text{ 相关}]$ 是指示函数，当第 k 个文档相关时为 1，否则为 0。

**Context Recall**

- **定义**：ground truth 答案中的每个关键陈述（claim）是否都能在检索到的上下文中找到支撑。
- **公式**：

$$
\text{Context Recall} = \frac{|\{\text{被上下文支撑的 claims}\}|}{|\{\text{ground truth 中的全部 claims}\}|}
$$

**手算示例**：

假设检索返回 4 个上下文，相关性判断为 `[1, 0, 1, 0]`（1=相关，0=不相关）：

```
Precision@1 = 1/1 = 1.0    → d₁ 相关，计入：1.0 × 1 = 1.0
Precision@2 = 1/2 = 0.5    → d₂ 不相关，计入：0.5 × 0 = 0
Precision@3 = 2/3 ≈ 0.667  → d₃ 相关，计入：0.667 × 1 = 0.667
Precision@4 = 2/4 = 0.5    → d₄ 不相关，计入：0.5 × 0 = 0

Context Precision = (1.0 + 0 + 0.667 + 0) / 2 = 0.833
```

假设 ground truth 包含 5 个关键陈述，其中 4 个能在检索上下文中找到支撑：

```
Context Recall = 4 / 5 = 0.8
```

**Python 验证代码**：

```python
def context_precision(relevance: list[int]) -> float:
    """relevance: 按排名顺序的相关性列表，1=相关，0=不相关"""
    num_relevant = sum(relevance)
    if num_relevant == 0:
        return 0.0
    score = 0.0
    cumulative_relevant = 0
    for k, rel in enumerate(relevance, 1):
        cumulative_relevant += rel
        precision_at_k = cumulative_relevant / k
        score += precision_at_k * rel
    return score / num_relevant

def context_recall(claims_supported: int, total_claims: int) -> float:
    return claims_supported / total_claims

# 示例数据
relevance = [1, 0, 1, 0]
print(f"Context Precision = {context_precision(relevance):.3f}")  # 0.833

print(f"Context Recall    = {context_recall(4, 5)}")  # 0.8
```

---

#### 2.3 MRR（Mean Reciprocal Rank）

- **定义**：对于一组 query，每个 query 取第一个相关文档的排名倒数，再求平均。反映"用户最快多久能找到想要的结果"。
- **公式**：

$$
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

其中 $\text{rank}_i$ 是第 $i$ 个 query 的第一个相关文档的排名位置。

**手算示例**：

3 个 query 的检索结果（✅ 表示第一个相关文档）：

| Query | 检索结果排名 | 第一个相关文档排名 | Reciprocal Rank |
|-------|-------------|-------------------|-----------------|
| Q1 | ❌ ✅ ❌ ❌ ❌ | 2 | 1/2 = 0.500 |
| Q2 | ✅ ❌ ❌ ❌ ❌ | 1 | 1/1 = 1.000 |
| Q3 | ❌ ❌ ❌ ✅ ❌ | 4 | 1/4 = 0.250 |

```
MRR = (0.500 + 1.000 + 0.250) / 3 = 0.583
```

**Python 验证代码**：

```python
def reciprocal_rank(retrieved: list[bool]) -> float:
    """返回第一个相关文档的排名倒数"""
    for i, is_relevant in enumerate(retrieved, 1):
        if is_relevant:
            return 1.0 / i
    return 0.0

def mrr(queries: list[list[bool]]) -> float:
    return sum(reciprocal_rank(q) for q in queries) / len(queries)

# 示例数据
queries = [
    [False, True, False, False, False],   # Q1: 第一个相关在位置 2
    [True, False, False, False, False],    # Q2: 第一个相关在位置 1
    [False, False, False, True, False],    # Q3: 第一个相关在位置 4
]

print(f"MRR = {mrr(queries):.3f}")  # 0.583
```

---

#### 2.4 MAP（Mean Average Precision）

- **定义**：对每个 query 计算 Average Precision（AP），再对所有 query 取平均。AP 考虑了**每个相关文档出现位置的精确率**，比 Precision@K 更全面地衡量排序质量。
- **公式**：

$$
\text{AP} = \frac{1}{|\{\text{相关文档}\}|} \sum_{k=1}^{N} \text{Precision@k} \times \mathbf{1}[d_k \text{ 相关}]
$$

$$
\text{MAP} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{AP}_i
$$

**手算示例**：

Query 的检索结果（Top-6），共 3 个相关文档：

| 排名 k | 是否相关 | Precision@k | 相关时计入 |
|--------|---------|-------------|-----------|
| 1 | ✅ | 1/1 = 1.000 | 1.000 |
| 2 | ❌ | 1/2 = 0.500 | — |
| 3 | ✅ | 2/3 = 0.667 | 0.667 |
| 4 | ❌ | 2/4 = 0.500 | — |
| 5 | ❌ | 2/5 = 0.400 | — |
| 6 | ✅ | 3/6 = 0.500 | 0.500 |

```
AP = (1.000 + 0.667 + 0.500) / 3 = 0.722
```

如果有 2 个 query，AP 分别为 0.722 和 0.500：

```
MAP = (0.722 + 0.500) / 2 = 0.611
```

**Python 验证代码**：

```python
def average_precision(retrieved: list[bool]) -> float:
    num_relevant = sum(retrieved)
    if num_relevant == 0:
        return 0.0
    score = 0.0
    cumulative_relevant = 0
    for k, is_relevant in enumerate(retrieved, 1):
        if is_relevant:
            cumulative_relevant += 1
            score += cumulative_relevant / k
    return score / num_relevant

def mean_average_precision(queries: list[list[bool]]) -> float:
    return sum(average_precision(q) for q in queries) / len(queries)

# 示例数据
q1 = [True, False, True, False, False, True]   # AP = 0.722
q2 = [False, True, False, False, True, False]   # AP = 0.500

print(f"AP(Q1) = {average_precision(q1):.3f}")  # 0.722
print(f"AP(Q2) = {average_precision(q2):.3f}")  # 0.500
print(f"MAP    = {mean_average_precision([q1, q2]):.3f}")  # 0.611
```

---

#### 2.5 NDCG（Normalized Discounted Cumulative Gain）

- **定义**：在二元相关性（相关/不相关）之上，NDCG 支持**多级相关性评分**（如 0/1/2/3），并通过对数折扣惩罚排名靠后的结果。它衡量的是"排序结果与理想排序的接近程度"。
- **公式**：

$$
\text{DCG@K} = \sum_{k=1}^{K} \frac{2^{rel_k} - 1}{\log_2(k + 1)}
$$

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

其中 IDCG 是将所有文档按相关性降序排列后的理想 DCG 值。

**手算示例**：

检索返回 Top-5，相关性评分（0~3）：

| 排名 k | 相关性 rel | 2^rel - 1 | log₂(k+1) | Gain |
|--------|-----------|-----------|------------|------|
| 1 | 3 | 7 | 1.000 | 7.000 |
| 2 | 0 | 0 | 1.585 | 0.000 |
| 3 | 2 | 3 | 2.000 | 1.500 |
| 4 | 1 | 1 | 2.322 | 0.431 |
| 5 | 2 | 3 | 2.585 | 1.161 |

```
DCG@5 = 7.000 + 0.000 + 1.500 + 0.431 + 1.161 = 10.092
```

理想排序（降序）：`[3, 2, 2, 1, 0]`

| 排名 k | 相关性 rel | 2^rel - 1 | log₂(k+1) | Gain |
|--------|-----------|-----------|------------|------|
| 1 | 3 | 7 | 1.000 | 7.000 |
| 2 | 2 | 3 | 1.585 | 1.893 |
| 3 | 2 | 3 | 2.000 | 1.500 |
| 4 | 1 | 1 | 2.322 | 0.431 |
| 5 | 0 | 0 | 2.585 | 0.000 |

```
IDCG@5 = 7.000 + 1.893 + 1.500 + 0.431 + 0.000 = 10.824

NDCG@5 = 10.092 / 10.824 = 0.932
```

**Python 验证代码**：

```python
import math

def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)  # i+2 因为 enumerate 从 0 开始
        for i, rel in enumerate(relevances[:k])
    )

def ndcg_at_k(relevances: list[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

# 示例数据
relevances = [3, 0, 2, 1, 2]

print(f"DCG@5  = {dcg_at_k(relevances, 5):.3f}")   # 10.092
print(f"NDCG@5 = {ndcg_at_k(relevances, 5):.3f}")  # 0.932
```

---

#### 检索指标诊断价值对比表

| 指标 | 关注维度 | 是否考虑排序 | 是否支持多级相关性 | 诊断价值 |
|------|---------|-------------|-------------------|---------|
| Precision@K | 准确性 | ❌ | ❌ | 检索噪声有多大 |
| Recall@K | 覆盖度 | ❌ | ❌ | 是否遗漏关键文档 |
| Context Precision | 准确性（加权） | ✅ | ❌ | 相关文档是否排在前面 |
| Context Recall | 覆盖度（语义） | ❌ | ❌ | 关键信息是否被覆盖 |
| MRR | 首个相关排名 | ✅ | ❌ | 用户最快找到答案的速度 |
| MAP | 综合排序质量 | ✅ | ❌ | 所有相关文档的排序全貌 |
| NDCG | 加权排序质量 | ✅ | ✅ | 最全面的排序质量评估 |

#### 适用场景速查表

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 问答系统（只需 1 个答案） | MRR | 只关心第一个命中的位置 |
| 多文档摘要 | Recall@K + MAP | 需要尽可能多地召回相关文档 |
| RAG 对话系统 | Context Precision + Faithfulness | 需要精准上下文 + 忠实生成 |
| 搜索引擎排序 | NDCG | 支持多级相关性，最贴近用户体验 |
| 端到端 RAG 评估 | RAGAS 全套指标 | 检索 + 生成联合评估 |

---

### 三、生成质量指标

#### 3.1 Faithfulness（忠实度）

- **定义**：生成的答案中，有多少比例的陈述（claim）可以从检索到的上下文中找到依据。衡量"幻觉"程度。
- **公式**：

$$
\text{Faithfulness} = \frac{|\{\text{被上下文支撑的 claims}\}|}{|\{\text{答案中的全部 claims}\}|}
$$

**评估流程**（LLM-as-Judge）：

```
Step 1: 用 LLM 将生成答案拆解为独立的事实性陈述（claims）
Step 2: 逐条判断每个 claim 是否能在检索上下文中找到支撑
Step 3: 计算被支撑的 claims 占总 claims 的比例
```

**手算示例**：

```
检索上下文：
"Python 是 Guido van Rossum 在 1991 年发布的编程语言。
 Python 3.0 于 2008 年发布。"

生成答案：
"Python 由 Guido van Rossum 创建，于 1991 年首次发布，
 是目前最流行的编程语言之一。"

Claim 拆解：
1. "Python 由 Guido van Rossum 创建"     → ✅ 上下文支撑
2. "Python 于 1991 年首次发布"            → ✅ 上下文支撑
3. "Python 是目前最流行的编程语言之一"    → ❌ 上下文无此信息

Faithfulness = 2 / 3 = 0.667
```

**Python 验证代码**：

```python
def faithfulness(claims_supported: int, total_claims: int) -> float:
    """手动计算忠实度（实际场景中 claim 拆解和判断由 LLM 完成）"""
    if total_claims == 0:
        return 1.0  # 无 claim 视为完全忠实
    return claims_supported / total_claims

print(f"Faithfulness = {faithfulness(2, 3):.3f}")  # 0.667
```

---

#### 3.2 Answer Relevancy（答案相关性）

- **定义**：生成的答案与原始问题的语义相关程度。通过反向生成问题来衡量——如果从答案中能还原出与原始问题语义一致的问题，说明答案是切题的。
- **公式**：

$$
\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(\mathbf{e}_{q_i}, \mathbf{e}_{q_{\text{orig}}})
$$

其中 $q_i$ 是从答案反向生成的第 $i$ 个问题，$q_{\text{orig}}$ 是原始问题，$\mathbf{e}$ 表示 embedding 向量。

**评估流程**：

```
Step 1: 用 LLM 基于生成答案反向生成 N 个问题
Step 2: 将反向生成的问题和原始问题分别做 embedding
Step 3: 计算每个反向问题与原始问题的余弦相似度
Step 4: 取平均值作为最终得分
```

**手算示例**：

```
原始问题："什么是 RAG？"

生成答案："RAG 是检索增强生成，通过检索外部知识来增强 LLM 的回答质量。"

反向生成的 3 个问题：
  Q1: "RAG 是什么技术？"         → cos(e_Q1, e_orig) = 0.95
  Q2: "如何通过检索增强 LLM？"    → cos(e_Q2, e_orig) = 0.82
  Q3: "什么是检索增强生成？"       → cos(e_Q3, e_orig) = 0.97

Answer Relevancy = (0.95 + 0.82 + 0.97) / 3 = 0.913
```

**Python 验证代码**：

```python
import numpy as np

def answer_relevancy(similarities: list[float]) -> float:
    """给定反向问题与原始问题的余弦相似度列表，计算答案相关性"""
    return np.mean(similarities)

# 示例数据
similarities = [0.95, 0.82, 0.97]
print(f"Answer Relevancy = {answer_relevancy(similarities):.3f}")  # 0.913
```

---

### 四、RAGAS 评估框架介绍与代码示例

[RAGAS](https://github.com/explodinggradients/ragas)（Retrieval-Augmented Generation Assessment）是目前最流行的 RAG 评估框架，集成了上述核心指标，并使用 LLM-as-Judge 实现自动化评估。

#### RAGAS 核心指标一览

| 指标 | 所需输入 | 评估维度 |
|------|---------|---------|
| Context Precision | question, contexts, ground_truth | 检索排序质量 |
| Context Recall | question, contexts, ground_truth | 检索覆盖度 |
| Faithfulness | question, contexts, answer | 生成忠实度 |
| Answer Relevancy | question, answer | 生成相关性 |

#### 完整代码示例

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

# 准备评估数据集
eval_data = {
    "question": [
        "什么是 RAG？",
        "向量数据库有哪些选择？",
    ],
    "answer": [
        "RAG 是检索增强生成技术，通过在生成前检索相关文档来提升 LLM 回答的准确性和时效性。",
        "常见的向量数据库包括 Pinecone、Milvus、Weaviate、Qdrant 和 Chroma。",
    ],
    "contexts": [
        [
            "RAG（Retrieval-Augmented Generation）是一种将检索模块与生成模型结合的技术。",
            "RAG 通过在推理时检索外部知识库来增强 LLM 的回答能力。",
        ],
        [
            "主流向量数据库包括 Pinecone、Milvus、Weaviate 等。",
            "Chroma 和 Qdrant 是轻量级的开源向量数据库。",
        ],
    ],
    "ground_truth": [
        "RAG 是检索增强生成，结合检索和生成两个阶段，先从知识库检索相关文档，再基于检索结果生成答案。",
        "常见向量数据库有 Pinecone、Milvus、Weaviate、Qdrant、Chroma、pgvector 等。",
    ],
}

dataset = Dataset.from_dict(eval_data)

# 执行评估
results = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

# 输出结果
print(results)
# 输出示例：
# {'context_precision': 0.92, 'context_recall': 0.85,
#  'faithfulness': 0.90, 'answer_relevancy': 0.88}

# 转为 DataFrame 查看每条数据的详细得分
df = results.to_pandas()
print(df)
```

#### 自定义评估指标

RAGAS 也支持自定义指标，适用于特定业务场景：

```python
from ragas.metrics.base import MetricWithLLM
from dataclasses import dataclass

@dataclass
class DomainAccuracy(MetricWithLLM):
    name: str = "domain_accuracy"

    def _score(self, row: dict) -> float:
        """自定义打分逻辑：检查答案是否包含领域关键术语"""
        answer = row["answer"]
        domain_terms = ["检索", "向量", "embedding", "chunk"]
        matches = sum(1 for term in domain_terms if term in answer)
        return matches / len(domain_terms)
```

---

### 五、评估最佳实践和流水线设计

#### 评估流水线架构

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│ 评估数据集    │───▶│ RAG Pipeline │───▶│ RAGAS 评估    │───▶│ 结果仪表盘  │
│ (Q/A/Context)│    │ (推理)       │    │ (自动打分)    │    │ (可视化)    │
└─────────────┘    └──────────────┘    └──────────────┘    └────────────┘
       ▲                                      │
       │                                      ▼
       │                              ┌──────────────┐
       └──────────────────────────────│ 反馈优化循环   │
                                      └──────────────┘
```

#### 最佳实践

**1. 构建高质量评估数据集**

```python
# 评估数据集结构（黄金标准）
eval_sample = {
    "question": "...",          # 真实用户问题或合成问题
    "ground_truth": "...",      # 人工标注的标准答案
    "ground_truth_contexts": [] # 人工标注的相关文档
}
```

- 数据集规模建议：至少 50~100 条，覆盖高频 query 类型
- 包含**简单 / 中等 / 困难**三个梯度的问题
- 包含**边界用例**：多跳推理、否定性问题、模糊查询

**2. 建立评估基线与持续监控**

```python
# CI/CD 中集成评估
import json

THRESHOLDS = {
    "context_precision": 0.80,
    "context_recall": 0.75,
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
}

def check_quality_gate(results: dict) -> bool:
    """质量门禁：所有指标必须超过阈值"""
    for metric, threshold in THRESHOLDS.items():
        score = results[metric]
        if score < threshold:
            print(f"❌ {metric}: {score:.3f} < {threshold}")
            return False
        print(f"✅ {metric}: {score:.3f} >= {threshold}")
    return True

# 在 CI 流水线中调用
results = {"context_precision": 0.92, "context_recall": 0.85,
           "faithfulness": 0.90, "answer_relevancy": 0.88}

if not check_quality_gate(results):
    raise SystemExit("Quality gate failed!")
```

**3. 指标异常的诊断思路**

| 异常现象 | 可能原因 | 优化方向 |
|---------|---------|---------|
| Recall 低 | Chunk 粒度太粗 / Embedding 模型弱 | 调整分块策略 / 更换 Embedding 模型 |
| Precision 低 | 检索 K 值过大 / 相似度阈值过低 | 减小 K 值 / 加入 Reranker |
| MRR 低 | 排序模型质量差 | 引入 Cross-Encoder Reranker |
| Faithfulness 低 | LLM 幻觉严重 | 加强 prompt 约束 / 降低 temperature |
| Answer Relevancy 低 | Query 理解偏差 / 上下文干扰 | 优化 Query Rewriting / 减少噪声文档 |

**4. A/B 评估对比模板**

```python
def ab_evaluate(pipeline_a, pipeline_b, eval_dataset):
    """对比两个 RAG pipeline 的评估结果"""
    results_a = evaluate(pipeline_a(eval_dataset))
    results_b = evaluate(pipeline_b(eval_dataset))

    print("=" * 50)
    print(f"{'指标':<25} {'Pipeline A':>10} {'Pipeline B':>10} {'Δ':>8}")
    print("=" * 50)
    for metric in THRESHOLDS:
        a = results_a[metric]
        b = results_b[metric]
        delta = b - a
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"{metric:<25} {a:>10.3f} {b:>10.3f} {delta:>+7.3f}{arrow}")
```

---

## 3. 常见误区 / 面试追问

### 常见误区

| 误区 | 正解 |
|------|------|
| "Precision 和 Recall 选一个优化就行" | 两者存在 trade-off，需结合业务场景权衡。RAG 场景通常**优先保证 Recall**（不漏关键信息），再用 Reranker 提升 Precision |
| "NDCG 和 MAP 效果差不多" | MAP 只支持二元相关性，NDCG 支持多级相关性。当文档有不同重要程度时（如"非常相关 / 一般相关 / 不相关"），必须使用 NDCG |
| "Faithfulness = 1.0 就说明答案完美" | Faithfulness 只衡量"不编造"，不衡量"是否完整"。一个只复述上下文原文的答案 Faithfulness 为 1.0，但可能完全没回答问题 |
| "评估指标高就可以上线" | 评估数据集的分布是否代表真实流量？是否覆盖了 corner case？离线指标和线上体验之间还有鸿沟 |
| "用一个综合分数就能评价 RAG" | 单一分数会掩盖问题。必须分别看检索和生成的指标，才能定位瓶颈在哪个阶段 |

### 面试追问

**Q1：MRR、MAP、NDCG 三者如何选择？**

- **MRR**：只关心第一个相关结果，适合"只需一个答案"的场景（如问答）
- **MAP**：关心所有相关结果的排名分布，适合"需要多个相关文档"的场景（如多文档摘要）
- **NDCG**：在 MAP 基础上支持多级相关性，适合"文档重要程度有差异"的场景（如搜索排序）

**Q2：RAGAS 使用 LLM-as-Judge 有什么局限？**

- LLM 本身可能存在偏见（如偏好更长的答案）
- 不同 LLM 的评估结果可能不一致，需固定评估模型版本
- Claim 拆解的粒度影响 Faithfulness 得分的稳定性
- 对于需要领域专业知识的场景，通用 LLM 的判断可能不准确

**Q3：如何构建 RAG 评估数据集？**

- **人工标注**：质量最高，但成本高。适合核心场景
- **LLM 合成**：用 LLM 基于文档自动生成 QA 对，再人工校验
- **线上日志挖掘**：从真实用户 query 中采样，标注相关文档和参考答案
- **对抗样本**：故意构造 LLM 容易出错的问题（多跳推理、否定性问题、需要计算的问题）

**Q4：检索质量和生成质量之间的关系是什么？**

- 检索质量是生成质量的上界：`生成质量 ≤ f(检索质量)`
- 当 Recall 低时，优化生成端收益有限——信息都没检索到，LLM 再强也无法凭空生成
- 当 Faithfulness 低时，可能是检索 Precision 低（噪声文档干扰）或 LLM 自身幻觉
- 建议**从左到右**排查：先确保检索质量达标，再优化生成质量

---

## 4. 参考资料

- [RAGAS 官方文档](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- Manning, Raghavan, Schütze. *Introduction to Information Retrieval*. Cambridge University Press.
- [Hugging Face - RAG 评估指南](https://huggingface.co/docs/transformers/model_doc/rag)
- [LlamaIndex - Evaluation Module](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)
- Järvelin, K., & Kekäläinen, J. (2002). *Cumulated gain-based evaluation of IR techniques*. ACM TOIS.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation*. arXiv:2309.15217.
