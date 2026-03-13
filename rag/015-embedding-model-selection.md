# Embedding 模型选择与微调策略

> 难度：中级
> 分类：RAG

## 简短回答

Embedding 模型的选择直接决定 RAG 检索质量。2025-2026 年的格局：**Voyage AI** 在 MTEB Benchmark 领先且性价比最高；**OpenAI text-embedding-3-large** 是最均衡的生产默认选择；**Cohere embed-v4** 在多语言和多模态场景最强。开源方面 **BGE-M3** 和 **Qwen3 Embedding** 表现出色。微调可带来 10-30% 的领域特定提升，但需要注意重新索引的成本。

## 详细解析

### Embedding 模型的作用

Embedding 模型将文本转化为高维向量，使得语义相似的文本在向量空间中距离更近。在 RAG 中，它用于两个环节：
1. **索引时**：将文档块转为向量存入向量数据库
2. **查询时**：将用户问题转为向量，与文档向量做相似度匹配

**关键约束：** 索引和查询必须使用**同一个** Embedding 模型，否则向量空间不一致，检索完全失效。

### 主流商业模型对比

| 模型 | 维度 | MTEB 排名 | 价格($/M tokens) | 特色 |
|------|------|-----------|-------------------|------|
| **Voyage AI voyage-3-large** | 1024 | 领先 | $0.06 | 性价比最高，多领域领先 |
| **OpenAI text-embedding-3-large** | 3072(可缩) | 前列 | $0.13 | 最均衡，支持维度缩减 |
| **Cohere embed-v4** | 1536 | 前列 | ~$0.10 | 多语言100+，多模态 |
| **OpenAI text-embedding-3-small** | 1536 | 中上 | $0.02 | 最便宜，适合成本敏感场景 |

#### Voyage AI — 性价比之王

Voyage AI 的 voyage-3-large 在 MTEB Benchmark 上超越 OpenAI text-embedding-3-large 9.74%，超越 Cohere embed-v3 20.71%。1024 维 Embedding 比 OpenAI 的 3072 维节省 3 倍存储空间，价格仅为 OpenAI 的一半。

特别值得注意：Voyage 提供领域特化模型：
- `voyage-law-2`：法律领域（案例检索提升 41%）
- `voyage-code-2`：代码搜索
- `voyage-finance-2`：金融领域

#### OpenAI text-embedding-3-large — 最均衡的默认选择

Agentset 的评测显示它在更多的直接对比中胜出，是最稳定的生产选择。独特优势：支持通过 `dimensions` 参数缩减维度（3072→1536→256），在精度和存储间灵活权衡。

```python
from openai import OpenAI
client = OpenAI()

# 全维度：最高精度
full = client.embeddings.create(
    model="text-embedding-3-large",
    input="查询文本"
)  # 3072 维

# 缩减维度：节省存储，轻微精度损失
compact = client.embeddings.create(
    model="text-embedding-3-large",
    input="查询文本",
    dimensions=1024  # 从 3072 缩减到 1024
)
```

#### Cohere embed-v4 — 多语言多模态之王

MTEB 最高分（65.2），支持 100+ 语言的跨语言检索，能在同一语义空间中嵌入文本和图像。支持 128K token 的超长输入。

### 主流开源模型

| 模型 | 维度 | 许可证 | 特色 |
|------|------|--------|------|
| **BAAI/bge-m3** | 1024 | MIT | 多语言，自托管首选 |
| **Qwen3 Embedding (4B/8B)** | 可变 | Apache 2.0 | 最新 SOTA，32K-128K 上下文 |
| **Jina Embeddings v3** | 1024 | Apache 2.0 | 多语言，任务特定适配 |
| **all-MiniLM-L6-v2** | 384 | Apache 2.0 | 极轻量，MVP 首选 |

**选择开源模型的理由：**
- 数据隐私：不需要将文档发送到第三方 API
- 成本控制：大量数据时自托管更经济
- 离线部署：无网络环境
- 完全可控：可以微调和定制

### 微调策略

领域特定微调可带来 **10-30%** 的检索质量提升。

#### 何时需要微调？

```
通用 Embedding 在你的数据上效果如何？
├── 满意 → 不需要微调
└── 不满意 → 分析原因
    ├── 数据质量问题 → 先优化数据，不是模型
    ├── 领域术语不理解 → 微调
    └── 检索逻辑需要定制 → 微调
```

#### 微调方法

```python
# 1. 准备训练数据：(query, positive_doc, negative_doc) 三元组
training_data = [
    {
        "query": "RLHF 的奖励模型如何训练？",
        "positive": "奖励模型通过人类偏好标注数据训练...",
        "negative": "强化学习是一种机器学习方法..."
    },
    # ... 至少 1000 个样本
]

# 2. 使用对比学习进行微调
from sentence_transformers import SentenceTransformer, losses

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
train_loss = losses.TripletLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3
)
```

#### 微调的关键注意事项

1. **必须重新索引**：微调改变了 Embedding 空间，所有已索引的向量都需要重新生成。这是不可忽略的成本。
2. **数据量要求**：至少 1000 个标注样本，理想情况 5000-10000 个
3. **评估闭环**：用 Recall@k、MRR 等指标对比微调前后的效果
4. **基础模型选择**：选与你的领域最接近的基础模型微调，效果更好

### 选型决策框架

```python
def choose_embedding_model(requirements: dict) -> str:
    if requirements["stage"] == "mvp":
        return "all-MiniLM-L6-v2"        # 免费、快、够用

    if requirements["multilingual"]:
        return "cohere-embed-v4"          # 100+ 语言

    if requirements["privacy_critical"]:
        return "bge-m3 (self-hosted)"     # MIT 许可，完全可控

    if requirements["budget"] == "tight":
        return "voyage-3-large"           # 最高性价比

    if requirements["domain_specific"]:
        if requirements["domain"] in ["legal", "code", "finance"]:
            return f"voyage-{domain}-2"   # 领域特化模型
        else:
            return "fine-tune bge-m3"     # 自定义微调

    return "openai-text-embedding-3-large"  # 最均衡的默认选择
```

### 关键原则

> "在你的数据上做 Benchmark——通用分数不一定适用于你的领域。"

> "价格更高的模型不一定准确率更高。最佳模型是在准确率和成本之间取得最好平衡的模型。"

## 常见误区 / 面试追问

1. **误区："选 MTEB 排名第一的就对了"** — MTEB 是通用基准，你的领域数据分布可能完全不同。必须在自己的数据集上评估。

2. **误区："维度越高越好"** — 3072 维比 1024 维精度更高，但存储 3 倍、检索更慢。很多场景下 1024 维甚至 768 维就足够了。OpenAI 的维度缩减功能正是为此设计。

3. **追问："如果换了 Embedding 模型怎么办？"** — 必须重新索引所有文档。索引和查询必须使用完全相同的模型。这是换模型的最大成本。

4. **追问："Embedding 和 LLM 需要来自同一家吗？"** — 不需要。Embedding 模型和 LLM 是独立组件。可以用 Voyage 的 Embedding + Anthropic 的 Claude 生成，完全没问题。

## 参考资料

- [Best Embedding Models 2025: MTEB Scores & Leaderboard (Ailog)](https://app.ailog.fr/en/blog/guides/choosing-embedding-models)
- [Top Embedding Models 2026: Complete Guide (ArtSmart)](https://artsmart.ai/blog/top-embedding-models-in-2025/)
- [Embedding Models Comparison 2026: OpenAI vs Cohere vs Voyage vs BGE (Reintech)](https://reintech.io/blog/embedding-models-comparison-2026-openai-cohere-voyage-bge)
- [9 Best Embedding Models for RAG (ZenML)](https://www.zenml.io/blog/best-embedding-models-for-rag)
- [Embedding Model Leaderboard (Agentset)](https://agentset.ai/embeddings)
