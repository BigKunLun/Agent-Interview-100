# 向量记忆 vs 结构化记忆：何时使用哪种？

> 难度：中级
> 分类：Memory & State

## 简短回答

向量记忆（Embedding + 向量数据库）擅长**语义相似度检索**——基于"意思相近"找到相关记忆，适合非结构化文本和模糊查询。结构化记忆（知识图谱 / 关系数据库）擅长**精确事实查询和关系推理**——支持多跳推理、时间推理和实体关系追踪。研究表明两者有互补优势：向量记忆对简单查询高效，结构化记忆对需要上下文和关系理解的复杂推理更优。生产系统推荐**混合架构**——用向量存储对话历史和非结构化内容，用知识图谱存储事实、实体和关系。Mem0 的混合方案实现了 26% 准确率提升和 90%+ token 成本降低。

## 详细解析

### 向量记忆（Vector / Embedding-based）

```python
# 向量记忆的工作原理
class VectorMemory:
    def store(self, text: str):
        embedding = embed_model.encode(text)  # 文本 → 向量
        self.index.add(embedding, metadata={"text": text})

    def retrieve(self, query: str, top_k=5):
        query_vec = embed_model.encode(query)
        # 基于余弦相似度找最近邻
        results = self.index.search(query_vec, top_k)
        return results

# 示例
memory.store("用户是 Python 开发者，偏好 FastAPI")
memory.store("上次讨论了数据库优化，使用了 PostgreSQL")
results = memory.retrieve("推荐一个 Web 框架")
# → 返回语义相似的记忆条目
```

**优势：**
- 语义理解：基于含义而非关键词匹配
- 灵活性：不需要预定义 Schema
- 对数级延迟：HNSW 索引在百万级数据上仍保持毫秒级检索

**劣势：**
- 基于相似度而非真正理解——语义相近但含义不同的内容可能被错误检索
- 不支持关系推理（"A 的上级是 B，B 的上级是谁？"）
- 不支持时间推理（"上周和这周的偏好有什么变化？"）

### 结构化记忆（Knowledge Graph / Relational）

```python
# 知识图谱记忆
class GraphMemory:
    def store(self, subject, predicate, object, valid_from=None, valid_to=None):
        # 存储三元组 + 时间有效性
        self.graph.add_edge(
            subject, object,
            relation=predicate,
            valid_from=valid_from or datetime.now(),
            valid_to=valid_to  # None = 当前有效
        )

    def query(self, query: str):
        # 支持图遍历和多跳查询
        return self.graph.traverse(query)

# 示例
graph.store("张三", "works_at", "公司A", valid_from="2024-01")
graph.store("张三", "works_at", "公司B", valid_from="2025-06")
# 可以查询："张三现在在哪家公司？" → 公司B
# 可以查询："张三之前在哪？" → 公司A（已过期但保留历史）
```

**优势：**
- 精确的实体和关系追踪
- 支持多跳推理（A→B→C 的关系链）
- 时间推理：追踪事实的变化历史
- 不会随数据增长产生检索噪声

**劣势：**
- 需要预定义 Schema 或本体（Ontology），限制适应性
- 构建和维护成本高
- 不擅长模糊的语义匹配

### 何时选择哪种？

| 需求场景 | 推荐方案 | 原因 |
|---------|---------|------|
| 语义搜索非结构化文本 | 向量 | 模糊匹配是强项 |
| 简单事实召回 | 向量 | 够用且简单 |
| RAG / 知识检索 | 向量 | 标准方案 |
| 实体关系追踪 | 知识图谱 | 关系推理是强项 |
| 时间推理 | 知识图谱 | 支持时间有效性 |
| 多跳推理 | 知识图谱 | 图遍历天然支持 |
| 用户画像 + 偏好 | 混合 | 结构化属性 + 语义历史 |
| 生产级个性化 | 混合 | 两者互补 |

### 混合架构（推荐）

```python
class HybridMemory:
    def __init__(self):
        self.vector_store = ChromaDB()       # 语义检索
        self.knowledge_graph = Neo4j()       # 关系推理
        self.profile_store = PostgreSQL()    # 结构化属性

    async def store(self, interaction):
        # 1. 对话历史 → 向量存储
        self.vector_store.add(interaction["text"])

        # 2. 提取实体和关系 → 知识图谱
        entities = extract_entities(interaction["text"])
        for entity in entities:
            self.knowledge_graph.upsert(entity)

        # 3. 用户属性 → 结构化存储
        profile_updates = extract_profile(interaction["text"])
        self.profile_store.update(interaction["user_id"], profile_updates)

    async def retrieve(self, query, user_id):
        # 双路检索 + 合并
        # 路径 1：语义检索
        semantic_results = self.vector_store.search(query, top_k=5)

        # 路径 2：图谱查询
        entities = extract_entities(query)
        graph_results = self.knowledge_graph.query(entities)

        # 路径 3：用户属性
        profile = self.profile_store.get(user_id)

        return merge(semantic_results, graph_results, profile)
```

Mem0 的混合方案正是这个思路：用向量存储做语义记忆，用图数据库做关系追踪，统一的 API 对上层透明。

### 性能对比数据

```
Mem0 混合方案 vs 纯全上下文方案：
- 准确率：+26%
- P95 延迟：-91%
- Token 成本：-90%+
```

## 常见误区 / 面试追问

1. **误区："向量数据库能解决所有记忆需求"** — 向量检索基于相似度，不是真正的"理解"。它无法回答"A 和 B 是什么关系？"或"这个信息是什么时候变的？"等需要结构化推理的问题。

2. **误区："知识图谱太复杂，不值得用"** — 对于简单应用确实如此。但当 Agent 需要追踪实体间关系、处理矛盾信息、或做时间推理时，知识图谱的价值会迅速超过构建成本。Graphiti 等框架已大幅降低了使用门槛。

3. **追问："向量检索返回了语义相似但实际无关的内容怎么办？"** — 两层解决：(1) 加 Reranker 对检索结果做精排；(2) 结合结构化元数据过滤（如按用户 ID、时间范围、类别筛选后再做语义检索）。

4. **追问："混合架构的一致性如何保证？"** — 同一条信息在向量存储和图谱中需要同步更新。实践中用事务性写入或最终一致性。删除操作尤其要注意——向量存储和图谱中都要清理。

## 参考资料

- [Comparing Memory Systems for LLM Agents: Vector, Graph, and Event Logs (MarkTechPost)](https://www.marktechpost.com/2025/11/10/comparing-memory-systems-for-llm-agents-vector-graph-and-event-logs/)
- [Mem0 Research: 26% Accuracy Boost for LLMs](https://mem0.ai/research)
- [How AI Agents Remember Things: Vector Stores in LLM Memory (freeCodeCamp)](https://www.freecodecamp.org/news/how-ai-agents-remember-things-vector-stores-in-llm-memory/)
- [A-MEM: Agentic Memory for LLM Agents (arXiv)](https://arxiv.org/pdf/2502.12110)
- [Beyond Short-term Memory: 3 Types of Long-term Memory (ML Mastery)](https://machinelearningmastery.com/beyond-short-term-memory-the-3-types-of-long-term-memory-ai-agents-need/)
