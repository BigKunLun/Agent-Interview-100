# 设计 Agent 的知识图谱记忆系统

> 难度：高级
> 分类：Memory & State

## 简短回答

知识图谱（Knowledge Graph, KG）作为 Agent 记忆系统，将信息表示为实体（节点）和关系（边）的网络，支持精确的事实存储、多跳推理和时间推理。与向量记忆的"语义相似度匹配"不同，KG 记忆能回答"A 的上级是谁？"、"这个信息什么时候变的？"等结构化问题。代表性框架 **Graphiti**（Zep 出品）实现了**双时间线模型**——同时追踪事件发生时间和数据入库时间，支持历史查询；**混合检索**——结合语义 Embedding、关键词 BM25 和图遍历，P95 延迟仅 300ms 且检索时无需 LLM 调用。

## 详细解析

### 为什么用知识图谱做记忆？

```
向量记忆的局限：
  "张三在公司A工作" → embedding → 存入向量数据库
  "张三跳槽到公司B" → embedding → 存入向量数据库
  查询"张三在哪工作？" → 两条记忆都被检索出来，LLM 无法判断哪个是最新的

知识图谱的优势：
  (张三) --[works_at, valid: 2024-01 ~ 2025-05]--> (公司A)
  (张三) --[works_at, valid: 2025-06 ~ now]-------> (公司B)
  查询"张三现在在哪？" → 图遍历直接找到当前有效的边 → 公司B
```

### 知识图谱记忆架构

```
用户交互 → LLM 提取实体和关系 → 知识图谱
                                    │
                              ┌─────┼─────┐
                              │     │     │
                           节点   边    时间线
                          (实体) (关系) (有效期)
                              │     │     │
                              └─────┼─────┘
                                    │
                          查询时：图遍历 + 语义搜索 + BM25
                                    │
                              检索结果 → 注入 LLM 上下文
```

### Graphiti 框架详解

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# 初始化 Graphiti（连接 Neo4j）
graphiti = Graphiti(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# 1. 摄入信息（自动提取实体和关系）
await graphiti.add_episode(
    name="用户对话",
    episode_body="张三说他刚从公司A跳槽到公司B，担任技术总监",
    source=EpisodeType.message,
    reference_time=datetime.now()
)
# Graphiti 自动：
# - 提取实体：张三、公司A、公司B
# - 提取关系：works_at、role_is
# - 设置旧关系(张三→公司A)的 valid_to = now
# - 创建新关系(张三→公司B)的 valid_from = now

# 2. 查询（混合检索，无需 LLM 调用）
results = await graphiti.search(
    query="张三现在在哪家公司？",
    num_results=5
)
# 返回当前有效的事实：张三 → works_at → 公司B（技术总监）
```

### 双时间线模型（Bi-temporal）

```python
# 时间线 T：事件实际发生的时间
# 时间线 T'：数据被系统记录的时间

class BiTemporalFact:
    subject: str        # 主体
    predicate: str      # 关系
    object: str         # 客体
    event_time: datetime      # T: 事件发生时间
    ingestion_time: datetime  # T': 系统记录时间
    valid_from: datetime      # 生效时间
    valid_to: datetime | None # 失效时间（None=当前有效）

# 示例
fact1 = BiTemporalFact(
    subject="张三", predicate="works_at", object="公司A",
    event_time="2024-01-15",   # 实际入职时间
    ingestion_time="2024-03-01",  # 我们获知这个信息的时间
    valid_from="2024-01-15", valid_to="2025-05-31"
)

# 支持的查询：
# "张三 2024 年 6 月在哪工作？" → 时间旅行查询
# "我们什么时候得知张三跳槽的？" → 数据溯源
```

### 混合检索（Hybrid Retrieval）

Graphiti 的检索不依赖 LLM 调用，保证低延迟：

```python
class HybridRetriever:
    """三路混合检索"""

    async def search(self, query: str, top_k: int = 10):
        # 路径 1：语义向量搜索
        semantic_results = await self.vector_index.search(
            embed(query), top_k=top_k
        )

        # 路径 2：关键词 BM25 搜索
        keyword_results = await self.bm25_index.search(
            query, top_k=top_k
        )

        # 路径 3：图遍历（实体 → 相关节点）
        entities = extract_entities(query)
        graph_results = []
        for entity in entities:
            neighbors = await self.graph.get_neighbors(
                entity, max_hops=2
            )
            graph_results.extend(neighbors)

        # 融合排序（RRF - Reciprocal Rank Fusion）
        return self.reciprocal_rank_fusion(
            semantic_results, keyword_results, graph_results
        )
```

P95 延迟仅 300ms——因为检索阶段完全不调用 LLM。

### 多 Agent 共享图谱

```python
# 知识图谱作为多 Agent 的共享记忆
class SharedKGMemory:
    """多 Agent 共享的知识图谱"""

    def __init__(self, graph_db):
        self.graph = graph_db

    async def agent_update(self, agent_id, fact):
        """任何 Agent 的更新对所有 Agent 可见"""
        fact["updated_by"] = agent_id
        fact["updated_at"] = datetime.now()

        # 检查冲突
        existing = await self.graph.find_conflicting(fact)
        if existing:
            # 标记旧事实失效
            await self.graph.invalidate(existing, reason=f"被 {agent_id} 更新")

        await self.graph.add(fact)
        # 所有 Agent 下次查询时自动看到最新事实
```

### 设计挑战与解决方案

| 挑战 | 解决方案 |
|------|---------|
| 实体消歧（同名不同人） | 上下文感知的实体链接 + 唯一 ID |
| Schema 进化 | Graphiti 支持 prescribed + learned ontology |
| 图谱膨胀 | 时间失效 + 定期清理低价值节点 |
| 提取质量 | LLM 提取 + 人工审核关键事实 |
| 查询延迟 | 混合索引（向量 + BM25 + 图遍历） |

### A-MEM：Zettelkasten 方法的记忆系统

```python
# A-MEM 将 Zettelkasten 卡片笔记法应用于 Agent 记忆
class AMemNote:
    """结构化记忆笔记"""
    content: str           # 原子化的知识点
    context: str           # 上下文描述
    keywords: list[str]    # 关键词标签
    links: list[str]       # 与其他笔记的关联
    created_at: datetime
    access_count: int

# 记忆笔记之间形成互联的知识网络
# 类似人脑的联想记忆——从一个记忆可以"联想"到相关记忆
```

## 常见误区 / 面试追问

1. **误区："知识图谱就是把所有信息都存成三元组"** — 不是所有信息都适合图谱。对话历史、非结构化文档适合向量存储。图谱应该只存储实体、关系和关键事实。混合架构是最佳实践。

2. **误区："构建知识图谱必须预定义完整 Schema"** — Graphiti 等现代框架支持 learned ontology——从数据中自动学习 Schema，同时支持预定义的 prescribed ontology 约束关键关系。

3. **追问："图谱记忆的实体提取准确率不够怎么办？"** — 三层保障：(1) 用专门的 NER 模型做初步提取；(2) LLM 做上下文理解和关系推理；(3) 对关键事实做人工审核或交叉验证。

4. **追问："图谱和 RAG 的关系是什么？"** — GraphRAG 是两者的结合——用知识图谱增强 RAG 检索。传统 RAG 只做文档级语义检索，GraphRAG 可以在检索后利用图结构做关系推理和多跳问答，提供更准确和完整的答案。

## 参考资料

- [Graphiti: Build Real-Time Knowledge Graphs for AI Agents (GitHub)](https://github.com/getzep/graphiti)
- [Graphiti: Knowledge Graph Memory for an Agentic World (Neo4j Blog)](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [Graph-based Agent Memory: Taxonomy, Techniques, and Applications (arXiv)](https://arxiv.org/html/2602.05665)
- [A-MEM: Agentic Memory for LLM Agents (arXiv)](https://arxiv.org/abs/2502.12110)
- [Zep: Temporal Knowledge Graph Architecture for Agent Memory](https://blog.getzep.com/content/files/2025/01/ZEP__USING_KNOWLEDGE_GRAPHS_TO_POWER_LLM_AGENT_MEMORY_2025011700.pdf)
