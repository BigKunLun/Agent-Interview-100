# 什么是 RAG？它解决了 LLM 的哪些核心局限？

> 难度：基础
> 分类：RAG

## 简短回答

RAG（Retrieval-Augmented Generation，检索增强生成）是一种在 LLM 生成回答之前，先从外部知识库中检索相关文档并注入到 Prompt 中的技术。它主要解决 LLM 的三大核心局限：知识截止（训练数据过时）、幻觉（编造看似合理的错误信息）、以及缺乏领域专有知识。

## 详细解析

### LLM 的三大核心局限

#### 1. 知识截止（Knowledge Cutoff）

LLM 的知识在训练完成后就被"冻结"了。当用户询问训练数据截止日期之后的信息时，模型要么承认不知道，要么自信地给出错误答案。LLM 的训练数据往往严重过时，而且当知识出现空白时，它们会进行外推，自信地说出听起来合理但实际错误的陈述。

#### 2. 幻觉（Hallucination）

由于依赖固定参数，LLM 在面对超出训练范围的任务时，经常产生与任务无关的输出或事实不一致的回答。这种现象被称为幻觉（hallucination）或虚构（confabulation），严重损害了 LLM 的可靠性和可信度。

#### 3. 缺乏私有/领域知识

LLM 无法访问企业内部文档、私有数据库或最新的领域知识。即使是最强大的通用模型，也无法回答关于公司内部流程、客户数据或专有技术的问题。

### RAG 的工作原理

```
用户提问
    ↓
[1. 检索] 将问题转化为向量，从知识库中检索相关文档
    ↓
[2. 增强] 将检索到的文档与原始问题组合成增强的 Prompt
    ↓
[3. 生成] LLM 基于增强的 Prompt 生成有据可查的回答
```

核心流程：
1. **索引（Indexing）**：离线阶段——将文档分块、生成 Embedding、存入向量数据库
2. **检索（Retrieval）**：在线阶段——将用户查询转为向量，从向量库中找到最相关的文档块
3. **生成（Generation）**：将检索到的文档块与原始问题拼接为 Prompt，送入 LLM 生成回答

### RAG 如何解决三大局限

| 局限 | RAG 的解决方式 |
|------|--------------|
| 知识截止 | 外部知识库可以随时更新，无需重新训练模型 |
| 幻觉 | 提供"事实锚点"，让 LLM 基于检索到的真实文档生成回答 |
| 缺乏领域知识 | 接入企业私有数据、行业文档、实时数据源 |

额外优势：
- **成本效益**：无需对 LLM 进行昂贵的微调或重新训练
- **来源可溯**：可以在回答中附带引用来源，用户可以验证
- **权限控制**：可以根据用户权限控制可检索的文档范围

### RAG 的局限性

RAG 并非万能。它自身也存在问题：

1. **不能完全消除幻觉**："RAG 不是直接的解决方案，因为 LLM 仍然可能围绕源材料进行幻觉。" LLM 可能从检索到的文档中断章取义，得出错误结论。

2. **检索质量瓶颈**：
   - 低精确率（Precision）：检索到的文档块与问题不匹配
   - 低召回率（Recall）：未能检索到所有相关文档块
   - 过时信息：知识库本身可能包含过时数据

3. **依赖知识库质量**：知识库中的偏见或错误会直接传导到 LLM 的回答中

### RAG 的三个演进阶段

| 阶段 | 特点 | 局限 |
|------|------|------|
| **Naive RAG** | 简单的"索引-检索-生成"链 | 检索精度低、上下文不足 |
| **Advanced RAG** | 加入预检索优化（查询改写）和后检索优化（重排序、压缩） | 仍是单次检索 |
| **Modular RAG** | 每个组件独立可替换、可组合 | 系统复杂度高 |

## 代码示例

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. 索引：分块 + Embedding + 存储
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 2. 检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. 生成：构建 RAG Chain
prompt = ChatPromptTemplate.from_template(
    "基于以下上下文回答问题。如果上下文中没有答案，请说'我不确定'。\n\n"
    "上下文：{context}\n\n问题：{question}"
)
llm = ChatOpenAI(model="gpt-4o")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

answer = rag_chain.invoke("公司的退款政策是什么？")
```

## 常见误区 / 面试追问

1. **误区："RAG 完全解决了幻觉问题"** — RAG 降低了幻觉概率，但 LLM 仍可能围绕检索内容进行幻觉，或忽略检索结果而使用自身知识。需要配合 Guardrails 和输出验证。

2. **误区："RAG 可以替代微调（Fine-tuning）"** — RAG 和微调解决不同问题。RAG 解决知识问题（"知道什么"），微调解决能力问题（"怎么做"）。如果需要改变模型的行为风格或推理模式，应该用微调。

3. **追问："RAG vs 长上下文窗口——如果模型能处理 100 万 token，还需要 RAG 吗？"** — 需要。(1) 长上下文的"大海捞针"问题——中间的信息容易被忽略；(2) 成本——100 万 token 的推理费用远高于 RAG 检索；(3) 延迟——长上下文增加推理时间。

4. **追问："RAG 的检索精度和 LLM 生成质量哪个更重要？"** — 检索精度。如果检索到的文档不相关，再强的 LLM 也无法生成正确答案。"Garbage in, garbage out" 在 RAG 中尤为适用。

## 参考资料

- [Retrieval-Augmented Generation (RAG) (Pinecone)](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [RAG for LLMs (Prompt Engineering Guide)](https://www.promptingguide.ai/research/rag)
- [Retrieval-Augmented Generation for Large Language Models: A Survey (arXiv:2312.10997)](https://arxiv.org/abs/2312.10997)
- [Retrieval Augmented Generation: Keeping LLMs Relevant and Current (Stack Overflow)](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)
- [RAG Hallucination: What Is It and How to Avoid It (K2View)](https://www.k2view.com/blog/rag-hallucination/)
