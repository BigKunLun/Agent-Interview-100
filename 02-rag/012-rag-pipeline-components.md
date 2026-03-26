# RAG Pipeline 的核心组件：Indexing、Retrieval、Generation

> 难度：基础
> 分类：RAG

## 简短回答

生产级 RAG 系统由三条流水线组成：**Indexing Pipeline**（离线——数据清洗、分块、Embedding、存储到向量数据库）、**Retrieval Pipeline**（在线——查询理解、向量检索、后检索优化如重排序和压缩）、**Generation Pipeline**（在线——上下文组装、Prompt 构建、LLM 生成、输出验证）。每条流水线都有独立的优化空间，理解它们的交互方式是构建高质量 RAG 的关键。

## 详细解析

### 整体架构

```
离线阶段                          在线阶段
┌──────────────────┐   ┌───────────────────────────────────────┐
│  Indexing        │   │  Retrieval          Generation        │
│  Pipeline        │   │  Pipeline           Pipeline          │
│                  │   │                                       │
│ 数据源 → 清洗    │   │ 用户查询 → 查询理解  → 检索 + 重排序  │
│   → 分块         │   │                        ↓              │
│   → Embedding    │   │               上下文组装 + Prompt 构建 │
│   → 向量数据库   │   │                        ↓              │
│                  │   │                  LLM 生成 → 输出验证   │
└──────────────────┘   └───────────────────────────────────────┘
```

### 1. Indexing Pipeline（索引流水线）

索引流水线是离线阶段，负责将原始数据转化为可检索的向量索引。

#### 数据摄取与清洗

```python
# 原始数据通常来自多种格式
sources = [
    PDFLoader("report.pdf"),
    WebLoader("https://docs.example.com"),
    DatabaseLoader("postgresql://..."),
    MarkdownLoader("docs/*.md"),
]

# 清洗：去除 HTML 标签、修正编码、标准化格式
for doc in documents:
    doc.content = remove_html_tags(doc.content)
    doc.content = normalize_whitespace(doc.content)
    doc.metadata = extract_metadata(doc)  # 保留元数据（来源、日期、作者）
```

#### 分块（Chunking）

将长文档切分为语义完整的小块，常见策略：
- **固定大小分块**：按 token 数切分，简单但可能破坏语义
- **递归分块**：按段落→句子→词的层次分割，平衡实用性
- **语义分块**：基于 Embedding 相似度在语义断点处分割

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # 每块约 512 tokens
    chunk_overlap=50,     # 相邻块重叠 50 tokens，保留上下文连续性
    separators=["\n\n", "\n", "。", " ", ""]
)
chunks = splitter.split_documents(documents)
```

#### Embedding 与存储

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectors = embeddings.embed_documents([chunk.content for chunk in chunks])

# 存入向量数据库（附带元数据，支持后续过滤）
vectorstore.upsert(
    ids=[chunk.id for chunk in chunks],
    embeddings=vectors,
    documents=[chunk.content for chunk in chunks],
    metadatas=[chunk.metadata for chunk in chunks]  # 来源、日期、类别等
)
```

### 2. Retrieval Pipeline（检索流水线）

检索流水线在每次用户查询时实时运行，负责找到最相关的文档块。

#### 预检索优化（Pre-retrieval）

提升检索质量的关键在于优化查询本身：

```python
# 查询改写：让 LLM 重新表述用户问题，提升匹配度
def rewrite_query(original_query: str) -> str:
    return llm.generate(
        f"将以下问题改写为更适合向量检索的形式，"
        f"保持核心语义：\n{original_query}"
    )

# 查询分解：将复杂问题拆分为多个子问题
def decompose_query(query: str) -> list[str]:
    return llm.generate(
        f"将以下复杂问题分解为 2-3 个独立的子问题：\n{query}"
    )

# HyDE：生成假设性回答，用回答而非问题去检索
def hypothetical_document(query: str) -> str:
    return llm.generate(f"为以下问题生成一个假设性回答：\n{query}")
```

#### 向量检索

```python
# 基本语义检索
results = vectorstore.similarity_search(
    query_embedding,
    top_k=20,                              # 先检索较多候选
    filter={"category": "technical_docs"}  # 元数据过滤
)
```

#### 后检索优化（Post-retrieval）

```python
# 重排序：用 Cross-Encoder 对候选结果精排
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rank(query, [r.content for r in results])
top_results = reranked[:5]  # 取 Top 5

# 上下文压缩：去除检索块中不相关的部分
compressed = llm.generate(
    f"从以下文档中提取与问题 '{query}' 直接相关的信息：\n{chunk}"
)
```

### 3. Generation Pipeline（生成流水线）

将检索结果与用户查询组合，送入 LLM 生成最终回答。

#### 上下文组装与 Prompt 构建

```python
def build_rag_prompt(query: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    return f"""基于以下参考文档回答用户问题。
如果文档中没有足够信息，请明确说明。
请在回答中引用相关来源。

参考文档：
{context_text}

用户问题：{query}

回答："""
```

#### 上下文窗口管理

当检索到的文档超出 LLM 的上下文窗口时，需要智能截断：
- 按重排序得分排列，优先保留高分文档
- 对低优先级文档进行摘要压缩
- 确保关键信息出现在上下文的开头和结尾（避免 "Lost in the Middle" 问题）

### 三条流水线的优化关系

| 阶段 | 优化方向 | 关键指标 |
|------|---------|---------|
| Indexing | 分块策略、Embedding 质量、元数据丰富度 | 索引覆盖率 |
| Retrieval | 查询改写、混合检索、重排序 | Recall@k, Precision@k |
| Generation | Prompt 工程、上下文窗口管理、输出验证 | 答案正确率、幻觉率 |

### RAG 的三个范式

1. **Naive RAG**：简单的索引→检索→生成链，最小可用
2. **Advanced RAG**：加入预检索和后检索优化，显著提升质量
3. **Modular RAG**：每个组件独立可替换、可组合，最大灵活性

## 常见误区 / 面试追问

1. **误区："RAG 的重点是 Generation"** — 实际上 Retrieval 的质量才是 RAG 效果的决定性因素。检索到的文档不相关，再强的 LLM 也救不回来。优化顺序应该是：Retrieval > Indexing > Generation。

2. **误区："原型能用 = 生产能用"** — 原型和生产系统的差异在于评估和监控能力。生产 RAG 需要 (1) 检索质量监控；(2) 生成质量评估；(3) 成本和延迟追踪。

3. **追问："如何评估 RAG Pipeline 的各个环节？"** — Indexing：覆盖率测试（是否所有关键信息都被索引）；Retrieval：Recall@k、MRR、NDCG；Generation：Faithfulness（忠实度）、Relevance（相关性）、Answer Correctness。

4. **追问："向量检索一定比关键词检索好吗？"** — 不一定。精确术语匹配（如产品型号、法律条款编号）时，BM25 等关键词检索可能更好。最佳实践是混合检索（Hybrid Search）。

## 参考资料

- [RAG 101: Demystifying Retrieval-Augmented Generation Pipelines (NVIDIA)](https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/)
- [Introduction to LLM RAG (Weaviate)](https://weaviate.io/blog/introduction-to-rag)
- [Retrieval-Augmented Generation: A Practical Guide (Comet)](https://www.comet.com/site/blog/retrieval-augmented-generation/)
- [RAG Pipelines Explained (Orq.ai)](https://orq.ai/blog/rag-pipelines)
- [RAG Pipelines in Production (Machine Learning Mastery)](https://machinelearningmastery.com/understanding-rag-part-x-rag-pipelines-in-production/)
