# 文档分块（Chunking）策略有哪些？各有什么优缺点？

> 难度：基础
> 分类：RAG

## 简短回答

主要有三种分块策略：**固定大小分块**（按字符/token 数切分，简单高效但破坏语义）、**递归分块**（按段落→句子→词层次切分，是最推荐的通用默认方案，Recall 85-90%）、**语义分块**（基于 Embedding 相似度在语义断点处切分，Recall 91-92% 但需要额外 Embedding 开销）。核心权衡是：**上下文保留 vs 检索精度**——块越大保留越多上下文但稀释相关性，块越小匹配越精确但丢失上下文。

## 详细解析

### 为什么分块很重要？

分块不是简单的预处理步骤，它是一个影响整个 RAG 系统性能的**设计决策**。分块策略决定了：
- Embedding 的语义质量（块是否表达完整的概念）
- 检索的精度和召回率（能否找到正确的信息）
- 生成的上下文充分性（LLM 是否有足够信息回答问题）
- Token 使用效率和成本

### 1. 固定大小分块（Fixed-Size Chunking）

**原理：** 按预定的字符数或 token 数均匀切分文本，通常带有重叠（overlap）。

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=500,      # 每块 500 字符
    chunk_overlap=50,    # 相邻块重叠 50 字符
    separator=""         # 按字符切
)
chunks = splitter.split_text(document)
```

| 优点 | 缺点 |
|------|------|
| 实现最简单，无需 NLP 库 | 完全忽略语义结构 |
| 计算成本最低 | 可能在句子甚至单词中间切断 |
| 块大小均匀，便于索引优化 | 可能混合不相关的主题 |
| 适合快速原型 | 即使有 overlap 也不能保证语义完整 |

**最佳场景：** 同质化数据集（新闻文章、博客等格式统一的内容）、快速原型验证。

### 2. 递归分块（Recursive Chunking）

**原理：** 按层次化分隔符递归切分——先尝试按段落（`\n\n`），如果段落仍然太大就按句子（`\n`），再不行按空格，最后按字符。确保尽可能保持段落和句子的完整性。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
    # 优先级：段落 > 换行 > 句号 > 逗号 > 空格 > 字符
)
chunks = splitter.split_text(document)
```

| 优点 | 缺点 |
|------|------|
| 实用的默认方案 | 仍可能在某些复杂结构处切断 |
| 显著减少句子被截断的概率 | 不处理表格、列表等特殊元素 |
| Chroma 测试：400-512 tokens 时 Recall 85-90% | 不同文档格式需要不同的分隔符集 |
| 计算成本仍然很低 | 不考虑语义相似度 |

**最佳场景：** 通用默认方案。结构化文本（技术文档、报告）、代码文件（配合语言特定分隔符）。

**推荐参数：** 400-512 tokens，10-20% overlap。

### 3. 语义分块（Semantic Chunking）

**原理：** 先对每个句子生成 Embedding，计算相邻句子的余弦相似度，在相似度急剧下降的地方切分——即在"主题转换点"分块。

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95  # 相似度低于 95 分位数时切分
)
chunks = splitter.split_text(document)
```

| 优点 | 缺点 |
|------|------|
| 最高的检索召回率（91-92%） | 需要对每个句子生成 Embedding |
| 保持主题连续性和语义完整性 | 计算成本高（API 调用或本地推理） |
| 块大小自适应内容 | 处理速度比固定/递归慢 |
| 适合叙述性文档 | 块大小不均匀，可能影响索引优化 |

**最佳场景：** 准确率优先的场景、叙述性/研究性文档、预算允许额外计算开销时。

### 其他分块策略

#### 4. 文档结构分块（Structure-Aware）

利用文档自身的结构（标题、章节、HTML 标签）作为分块边界：

```python
# Markdown 文档按标题分块
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"), ("##", "h2"), ("###", "h3")
    ]
)
```

#### 5. 小块检索 + 大块上下文（Parent-Child / Small-to-Big）

用小块做精确检索，返回时提供包含小块的大块作为上下文：

```python
# 检索时用小块（200 tokens）匹配精确
# 返回时给 LLM 提供大块（2000 tokens）保留上下文
small_chunks = split(doc, chunk_size=200)
large_chunks = split(doc, chunk_size=2000)
# 建立 small → large 的映射关系
```

### 核心权衡

```
小块（100-200 tokens）              大块（1000-2000 tokens）
├── 检索精度高                      ├── 上下文信息丰富
├── Embedding 语义集中              ├── 保持段落完整性
├── 但可能丢失上下文                ├── 但稀释 Embedding 相关性
└── 需要更多块 → 更多存储           └── 可能包含无关信息
```

**经验法则：** 400-512 tokens 是大多数场景的最佳起点。

### 关键问题："是否需要分块？"

一个常被忽略的问题：当文档本身就很小、聚焦、且直接匹配用户问题时，分块反而可能损害检索精度。评估是否需要分块，而不是默认总要分块。

## 常见误区 / 面试追问

1. **误区："语义分块一定最好"** — Chroma 的测试数据显示，递归分块（85-90% Recall）和语义分块（91-92% Recall）仅差 2-3%，但语义分块的计算成本显著更高。大多数场景递归分块就够了。

2. **误区："块越小越好"** — 太小的块会丢失上下文，导致 LLM 无法理解信息的含义。需要在精度和上下文之间找平衡。

3. **追问："overlap 有什么用？设多少合适？"** — Overlap 确保分块边界处的信息不会丢失。通常设置为 chunk_size 的 10-20%（如 512 tokens 的块用 50-100 tokens overlap）。

4. **追问："如何选择最佳的 chunk_size？"** — 没有通用最优值。应该在你的数据集上实验——用检索评估指标（Recall@k、MRR）比较不同 chunk_size 的效果。通常从 512 tokens 开始调。

## 参考资料

- [Best Chunking Strategies for RAG in 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Chunking Strategies for RAG (Weaviate)](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Breaking Up Is Hard to Do: Chunking in RAG Applications (Stack Overflow)](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)
- [Chunking Strategies for RAG: Best Practices (Unstructured)](https://unstructured.io/blog/chunking-for-rag-best-practices)
- [The Ultimate Guide to Chunking Strategies (Databricks)](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089)
