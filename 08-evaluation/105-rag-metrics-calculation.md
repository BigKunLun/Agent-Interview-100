# 量化计算题：手算 RAG 评估指标（Precision@K、MRR、NDCG、Context Recall）

> 难度：中级
> 分类：Evaluation

## 简短回答

RAG 系统的评估需要从**检索质量**和**生成质量**两个维度进行量化衡量。检索侧核心指标包括 **Precision@K**（Top-K 结果中相关文档占比）、**Recall@K**（Top-K 覆盖了多少真正相关文档）、**MRR**（Mean Reciprocal Rank，首个相关结果排名倒数的均值）和 **NDCG**（Normalized Discounted Cumulative Gain，考虑位置折扣的归一化累积增益）。生成侧常用 **Context Recall**（上下文对 Ground Truth 关键信息的覆盖率）和 **Faithfulness**（回答中可由上下文支撑的陈述比例）。这些指标各有侧重：Precision 关注精确性，Recall 关注完整性，MRR 关注首次命中速度，NDCG 关注排序质量，Context Recall 和 Faithfulness 则衡量 LLM 对检索内容的利用和忠实程度。面试中常要求**手算**这些指标，核心在于牢记公式并注意分母的选取。

## 详细解析

### 评估指标体系总览

```
┌─────────────────────────────────────────────────────────┐
│                   RAG 评估指标体系                        │
├────────────────────────┬────────────────────────────────┤
│      检索质量指标       │         生成质量指标            │
│  (Retrieval Quality)   │   (Generation Quality)         │
├────────────────────────┼────────────────────────────────┤
│  Precision@K           │  Context Recall                │
│  Recall@K              │  Faithfulness                  │
│  MRR                   │  Answer Relevancy              │
│  NDCG@K                │  Answer Correctness            │
│  MAP                   │  Hallucination Rate            │
└────────────┬───────────┴───────────────┬────────────────┘
             │                           │
             ↓                           ↓
     ┌───────────────┐          ┌────────────────┐
     │ 检索器返回文档 │    →     │ LLM 生成最终回答│
     └───────────────┘          └────────────────┘
```

### 场景 1: Precision@K 和 Recall@K

**题目：** 系统检索返回 5 个文档 `[D1, D2, D3, D4, D5]`，其中真正相关的是 `[D1, D3, D5]`，Ground Truth 中共有 4 个相关文档 `[D1, D3, D5, D7]`。求 Precision@5 和 Recall@5。

**公式回顾：**

```
Precision@K = |检索结果 ∩ 相关文档| / K
Recall@K    = |检索结果 ∩ 相关文档| / |全部相关文档|
```

**解题步骤：**

1. 检索结果集合：`{D1, D2, D3, D4, D5}`，K = 5
2. 相关文档集合（Ground Truth）：`{D1, D3, D5, D7}`，共 4 个
3. 交集：`{D1, D3, D5}`，共 3 个
4. Precision@5 = 3 / 5 = **0.60**
5. Recall@5 = 3 / 4 = **0.75**

**Python 验证代码：**

```python
# 场景 1: Precision@K 和 Recall@K 计算
retrieved = ["D1", "D2", "D3", "D4", "D5"]  # 检索返回的文档
ground_truth = ["D1", "D3", "D5", "D7"]      # 全部相关文档

k = len(retrieved)
relevant_in_retrieved = set(retrieved) & set(ground_truth)  # 交集

precision_at_k = len(relevant_in_retrieved) / k
recall_at_k = len(relevant_in_retrieved) / len(ground_truth)

print(f"交集: {relevant_in_retrieved}")
print(f"Precision@{k} = {len(relevant_in_retrieved)}/{k} = {precision_at_k:.2f}")
print(f"Recall@{k} = {len(relevant_in_retrieved)}/{len(ground_truth)} = {recall_at_k:.2f}")
# 输出: Precision@5 = 0.60, Recall@5 = 0.75
```

### 场景 2: MRR（Mean Reciprocal Rank）

**题目：** 有 3 个查询，各自检索结果中第一个相关文档分别出现在位置 3、1、2。求 MRR。

**公式回顾：**

```
MRR = (1/|Q|) × Σ (1 / rank_i)

其中 rank_i 是第 i 个查询中首个相关文档的排名位置
```

**解题步骤：**

1. Query 1: 首个相关文档在位置 3 → Reciprocal Rank = 1/3 ≈ 0.3333
2. Query 2: 首个相关文档在位置 1 → Reciprocal Rank = 1/1 = 1.0000
3. Query 3: 首个相关文档在位置 2 → Reciprocal Rank = 1/2 = 0.5000
4. MRR = (0.3333 + 1.0000 + 0.5000) / 3 = 1.8333 / 3 = **0.6111**

**Python 验证代码：**

```python
# 场景 2: MRR 计算
first_relevant_positions = [3, 1, 2]  # 每个查询中首个相关文档的位置

reciprocal_ranks = [1.0 / pos for pos in first_relevant_positions]
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

for i, (pos, rr) in enumerate(zip(first_relevant_positions, reciprocal_ranks)):
    print(f"Query {i+1}: 位置={pos}, Reciprocal Rank=1/{pos}={rr:.4f}")

print(f"\nMRR = ({' + '.join(f'{rr:.4f}' for rr in reciprocal_ranks)}) / {len(reciprocal_ranks)}")
print(f"MRR = {sum(reciprocal_ranks):.4f} / {len(reciprocal_ranks)} = {mrr:.4f}")
# 输出: MRR = 0.6111
```

### 场景 3: NDCG@5

**题目：** 检索返回 5 个文档，相关性评分分别为 `[3, 2, 0, 1, 3]`（0=无关, 1=部分相关, 2=相关, 3=高度相关）。求 NDCG@5。

**公式回顾：**

```
DCG@K  = Σ(i=1 to K) rel_i / log₂(i + 1)
IDCG@K = DCG@K（将相关性按降序排列后计算）
NDCG@K = DCG@K / IDCG@K
```

**解题步骤：**

**Step 1 — 计算 DCG@5：**

| 位置 i | rel_i | log₂(i+1) | rel_i / log₂(i+1) |
|--------|-------|------------|-------------------|
| 1      | 3     | log₂(2) = 1.000 | 3.0000     |
| 2      | 2     | log₂(3) = 1.585 | 1.2619     |
| 3      | 0     | log₂(4) = 2.000 | 0.0000     |
| 4      | 1     | log₂(5) = 2.322 | 0.4307     |
| 5      | 3     | log₂(6) = 2.585 | 1.1606     |

DCG@5 = 3.0000 + 1.2619 + 0.0000 + 0.4307 + 1.1606 = **5.8531**

**Step 2 — 计算 IDCG@5（理想排序 `[3, 3, 2, 1, 0]`）：**

| 位置 i | rel_i | log₂(i+1) | rel_i / log₂(i+1) |
|--------|-------|------------|-------------------|
| 1      | 3     | 1.000      | 3.0000            |
| 2      | 3     | 1.585      | 1.8928            |
| 3      | 2     | 2.000      | 1.0000            |
| 4      | 1     | 2.322      | 0.4307            |
| 5      | 0     | 2.585      | 0.0000            |

IDCG@5 = 3.0000 + 1.8928 + 1.0000 + 0.4307 + 0.0000 = **6.3235**

**Step 3 — 计算 NDCG@5：**

NDCG@5 = DCG@5 / IDCG@5 = 5.8531 / 6.3235 = **0.9256**

**Python 验证代码：**

```python
import math

# 场景 3: NDCG@5 计算
relevance_scores = [3, 2, 0, 1, 3]  # 实际检索结果的相关性评分

def dcg_at_k(rels, k):
    """计算 DCG@K"""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

k = 5

# 计算 DCG
dcg = dcg_at_k(relevance_scores, k)
print(f"DCG@{k} 逐项计算:")
for i, rel in enumerate(relevance_scores):
    val = rel / math.log2(i + 2)
    print(f"  位置 {i+1}: {rel} / log₂({i+2}) = {rel} / {math.log2(i+2):.3f} = {val:.4f}")
print(f"DCG@{k} = {dcg:.4f}\n")

# 计算 IDCG（理想排序）
ideal_scores = sorted(relevance_scores, reverse=True)
idcg = dcg_at_k(ideal_scores, k)
print(f"理想排序: {ideal_scores}")
print(f"IDCG@{k} = {idcg:.4f}\n")

# 计算 NDCG
ndcg = dcg / idcg
print(f"NDCG@{k} = {dcg:.4f} / {idcg:.4f} = {ndcg:.4f}")
# 输出: NDCG@5 = 0.9256
```

### 场景 4: Context Recall 和 Faithfulness

**题目：** Ground Truth 包含 4 个关键信息点，检索到的 Context 覆盖了其中 3 个。Agent 回答中有 5 个陈述，其中 4 个可以从 Context 中推导。求 Context Recall 和 Faithfulness。

**公式回顾：**

```
Context Recall = |Context 覆盖的 GT 信息点| / |GT 全部信息点|
Faithfulness   = |可由 Context 支撑的陈述| / |回答中全部陈述|
```

```
                    Ground Truth (4 个信息点)
                    ┌──────────────────────┐
                    │ ✓ 信息点 1           │
                    │ ✓ 信息点 2           │  Context Recall
                    │ ✓ 信息点 3           │  = 3/4 = 0.75
                    │ ✗ 信息点 4（未覆盖）  │
                    └──────────────────────┘
                              ↕
                    Retrieved Context
                              ↓
                    Agent 回答 (5 个陈述)
                    ┌──────────────────────┐
                    │ ✓ 陈述 1（有依据）    │
                    │ ✓ 陈述 2（有依据）    │  Faithfulness
                    │ ✓ 陈述 3（有依据）    │  = 4/5 = 0.80
                    │ ✓ 陈述 4（有依据）    │
                    │ ✗ 陈述 5（无依据）    │
                    └──────────────────────┘
```

**解题步骤：**

1. Ground Truth 信息点总数 = 4，Context 覆盖了 3 个
2. Context Recall = 3 / 4 = **0.75**
3. Agent 回答中陈述总数 = 5，可由 Context 支撑的 = 4
4. Faithfulness = 4 / 5 = **0.80**

**Python 验证代码：**

```python
# 场景 4: Context Recall 和 Faithfulness 计算
gt_total_points = 4         # Ground Truth 中关键信息点总数
context_covered = 3         # Context 覆盖的信息点数量
answer_total_claims = 5     # Agent 回答中的陈述总数
supported_claims = 4        # 可由 Context 支撑的陈述数量

context_recall = context_covered / gt_total_points
faithfulness = supported_claims / answer_total_claims

print(f"Context Recall = {context_covered}/{gt_total_points} = {context_recall:.2f}")
print(f"Faithfulness   = {supported_claims}/{answer_total_claims} = {faithfulness:.2f}")

# 综合评估
print(f"\n综合解读:")
print(f"  - 检索上下文覆盖了 {context_recall:.0%} 的关键信息 → {'良好' if context_recall >= 0.7 else '需改进'}")
print(f"  - 回答忠实度为 {faithfulness:.0%} → {'良好' if faithfulness >= 0.8 else '需改进'}")
# 输出: Context Recall = 0.75, Faithfulness = 0.80
```

### 指标适用场景总结

| 指标 | 衡量目标 | 适用场景 | 取值范围 | 对位置敏感 |
|------|---------|---------|---------|-----------|
| Precision@K | 检索精确度 | Top-K 结果质量要求高（如问答系统） | [0, 1] | 否 |
| Recall@K | 检索完整度 | 不能遗漏相关文档（如法律/医疗检索） | [0, 1] | 否 |
| MRR | 首次命中速度 | 用户只看第一个结果（如搜索引擎） | [0, 1] | 是（仅首个） |
| NDCG@K | 排序质量 | 多级相关性、排序很重要（如推荐系统） | [0, 1] | 是（全部位置） |
| Context Recall | 上下文覆盖度 | 评估检索对 Ground Truth 的覆盖 | [0, 1] | 否 |
| Faithfulness | 回答忠实度 | 检测幻觉、确保回答有据可依 | [0, 1] | 否 |

## 常见误区 / 面试追问

1. **误区："Precision 在 RAG 中和传统 IR 完全一样"** — 传统 IR 的 Precision 基于二元相关性判断（相关/不相关），而 RAG 场景下更强调检索到的文档是否能为 LLM 生成答案提供有效支撑。一个文档可能与 query 相关但不包含回答所需的具体信息，此时传统 Precision 会高估检索质量。RAG 评估中通常需结合 Context Relevancy 等指标进行更细粒度的衡量。

2. **误区："NDCG 不考虑文档位置"** — 恰恰相反，NDCG 的核心设计就是通过 log₂(i+1) 的位置折扣因子来惩罚排名靠后的相关文档。排在第 1 位的相关文档贡献最大（除以 log₂2=1），排在第 5 位的同样评分文档贡献会被折扣为原来的约 38.7%（除以 log₂6≈2.585）。这正是 NDCG 优于 Precision 的关键所在——它同时考虑了相关性程度和排名位置。

3. **追问："这些指标分别适用什么场景？"** — 不同业务场景对指标的侧重不同：**Precision@K** 适合展示位有限、用户期望高精度的场景（如客服问答只取 Top-3）；**Recall@K** 适合不能遗漏的场景（如法律合规检索）；**MRR** 适合用户只关心第一条结果的场景（如搜索建议）；**NDCG** 适合多级相关性且排序重要的场景（如文档推荐）；**Context Recall** 和 **Faithfulness** 是 RAG 特有指标，前者评估"检索够不够全"，后者评估"回答够不够忠实"，两者结合才能全面评估端到端质量。

4. **追问："如何用 Ragas 框架自动计算？"** — Ragas 是专为 RAG 评估设计的开源框架，它封装了 Context Recall、Faithfulness、Answer Relevancy 等指标的自动计算。核心用法是构造 `EvaluationDataset`（包含 question、answer、contexts、ground_truth 四个字段），然后调用 `evaluate()` 函数并传入所需指标列表。Ragas 内部利用 LLM-as-Judge 机制：对于 Faithfulness，它会先将回答拆分为原子陈述，再逐一判断每条陈述是否可由 context 推导；对于 Context Recall，它会将 Ground Truth 拆分为信息点，判断每个信息点是否被 context 覆盖。使用时需注意配置合适的 Judge LLM（如 GPT-4）以保证评估质量。

## 参考资料

- [Ragas: Automated Evaluation of Retrieval Augmented Generation (Ragas Docs)](https://docs.ragas.io/)
- [Evaluation Measures in Information Retrieval (Stanford NLP)](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
- [NDCG - Normalized Discounted Cumulative Gain Explained (evidentlyai.com)](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)
- [RAG Evaluation: A Systematic Framework (LlamaIndex Blog)](https://www.llamaindex.ai/blog/rag-evaluation)
- [Metrics for Evaluating RAG Applications (Hugging Face)](https://huggingface.co/learn/cookbook/en/rag_evaluation)
