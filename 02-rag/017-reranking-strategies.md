# Re-ranking 的原理与实现：Cross-Encoder vs Bi-Encoder

> 难度：中级
> 分类：RAG

## 简短回答

Bi-Encoder 将查询和文档独立编码为向量，速度快但精度有限，用于第一阶段的大规模检索（Retrieval）。Cross-Encoder 将查询和文档拼接后联合编码，精度高但速度慢，用于第二阶段的精排（Reranking）。生产 RAG 系统的标准架构是"Bi-Encoder 检索 Top-100 → Cross-Encoder 重排 Top-5"，这种两阶段流水线在效率和精度间取得最佳平衡。

## 详细解析

### 为什么需要重排序？

第一阶段检索（Bi-Encoder）从百万级文档中快速找出 Top-K 候选，但存在精度损失：

- Bi-Encoder 必须将文档的所有可能含义压缩进单一向量，信息不可避免地丢失
- Bi-Encoder 编码文档时没有查询的上下文信息（因为查询还未到达）
- 结果排序可能不够精确

重排序用更精确的模型对候选结果重新打分，确保送入 LLM 的上下文是最相关的。

### Bi-Encoder vs Cross-Encoder

```
Bi-Encoder（独立编码）:
  Query  → [Encoder] → q_vector ─┐
                                  ├── cosine_similarity → score
  Doc    → [Encoder] → d_vector ─┘

  文档向量可以预计算并缓存

Cross-Encoder（联合编码）:
  [Query + Doc] → [Transformer] → score (0~1)

  每对 query-doc 都需要完整的推理
```

| 维度 | Bi-Encoder | Cross-Encoder |
|------|-----------|---------------|
| **编码方式** | Query 和 Doc 独立编码 | Query 和 Doc 拼接后联合编码 |
| **精度** | 中等 | 高（查询感知的文档表示） |
| **速度** | 极快（文档向量预计算） | 极慢（每对需完整推理） |
| **可扩展性** | 10 万文档只需编码 10 万次 | 10 万文档需编码 10 万对！ |
| **用途** | 第一阶段：召回候选 | 第二阶段：精确重排 |
| **预计算** | 可以 | 不可以 |

### 可扩展性的本质差异

假设数据库有 100,000 个文档：
- **Bi-Encoder**：编码 100,000 个文档（一次性，可离线）+ 编码 1 个查询 = 100,001 次
- **Cross-Encoder**：编码 100,000 对 (query, doc) = 100,000 次（每次查询都要）

如果用小型 Cross-Encoder（如 BERT）在 V100 GPU 上，100,000 个文档的单次查询需要等待超过 50 小时！

这就是为什么 Cross-Encoder 只能用于重排少量候选（通常 20-100 个）。

### 两阶段检索流水线（标准架构）

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

# 第 1 阶段：Bi-Encoder 召回 Top-100
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
query_embedding = bi_encoder.encode(query)
candidates = vector_db.search(query_embedding, top_k=100)

# 第 2 阶段：Cross-Encoder 重排 → Top-5
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(query, doc.content) for doc in candidates]
scores = cross_encoder.predict(pairs)

# 按重排分数排序，取 Top-5
reranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)[:5]

# 将 Top-5 送入 LLM 生成回答
context = "\n\n".join([doc.content for doc, score in reranked])
```

### 重排序的实际效果

重排序阶段对高质量 RAG 是**不可妥协的**——LLM 的输出质量直接取决于上下文的质量。没有重排序的 RAG 系统通常会将不够相关的文档送入 LLM，导致回答偏离主题或包含无关信息。

### 常用重排序方案

| 方案 | 类型 | 特点 |
|------|------|------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 开源 | 轻量级，速度快 |
| Cohere Rerank API | 商业 API | 高质量，开箱即用 |
| Jina AI Rerank | 商业 API | 性价比高 |
| `bge-reranker-v2-m3` | 开源 | 多语言支持 |
| ColBERT | 开源 | 晚期交互模型，精度接近 Cross-Encoder，速度更快 |

#### 选择重排序模型

推荐使用 MTEB Leaderboard 的 Reranking 排行榜选择模型。Average 列是综合质量的良好代理指标。

### 框架集成

```python
# LangChain 集成
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_retriever  # 第一阶段检索器
)

# Cohere Rerank API
import cohere
co = cohere.Client("API_KEY")
reranked = co.rerank(
    model="rerank-english-v3.0",
    query=query,
    documents=[doc.content for doc in candidates],
    top_n=5
)
```

### 进阶：ColBERT（晚期交互模型）

ColBERT 是 Bi-Encoder 和 Cross-Encoder 的折中方案：

- 像 Bi-Encoder 一样独立编码 Query 和 Doc
- 但保留 token 级别的向量（不压缩为单一向量）
- 检索时做 token 级别的细粒度交互

```
Bi-Encoder:    doc → [single vector]     → dot product
Cross-Encoder: [query + doc] → [score]   → full attention
ColBERT:       doc → [token vectors]     → late interaction (MaxSim)
```

ColBERT 的精度接近 Cross-Encoder，速度接近 Bi-Encoder，是一个有前景的方向。

## 常见误区 / 面试追问

1. **误区："Bi-Encoder 精度低是因为模型小"** — 精度差异是架构本质决定的。Bi-Encoder 编码文档时没有查询上下文，无论模型多大，信息压缩的损失都存在。

2. **误区："直接用 Cross-Encoder 做全量检索更好"** — 计算上不可行。10 万文档 × 单次查询 = 10 万次推理，延迟不可接受。

3. **追问："Top-K 的 K 设多少？"** — 第一阶段：检索 50-100 个候选（K 太小会漏掉相关文档，太大会拖慢重排）。第二阶段：重排后取 3-10 个送入 LLM（取决于上下文窗口大小）。

4. **追问："Cross-Encoder 对长文档有什么问题？"** — 很多模型截断输入到 token 窗口长度（如 512 tokens），可能切掉文档最相关的部分。解决方案：先分块再重排，或使用支持长上下文的重排模型。

## 参考资料

- [Rerankers and Two-Stage Retrieval (Pinecone)](https://www.pinecone.io/learn/series/rag/rerankers/)
- [Bi-Encoder vs Cross-Encoder in IR and RAG (VeloDB)](https://www.velodb.io/glossary/bi-encoder-vs-cross-encoder)
- [Cross Encoder Reranker (LangChain)](https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/)
- [Semantic Reranking (Elastic)](https://www.elastic.co/docs/solutions/search/ranking/semantic-reranking)
- [Beyond Simple Embeddings: Bi-Encoders and Cross-Encoders (WaterCrawl)](https://watercrawl.dev/blog/Beyond-Simple-Embeddings)
