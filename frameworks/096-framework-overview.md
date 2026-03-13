# 主流 Agent 框架概览：LangChain、LlamaIndex、Haystack

> 难度：基础
> 分类：Frameworks

## 简短回答

2025 年 AI Agent 框架呈现三足鼎立格局：**LangChain** 是最全能的"瑞士军刀"——生态最大、功能最全、社区最活跃（GitHub 100k+ Stars），适合快速原型和复杂 Agent 编排，但抽象层多、学习曲线陡；**LlamaIndex** 是"数据专家"——专注于数据连接和 RAG，提供 160+ 数据源连接器，在检索场景下性能最优（检索速度最快），适合知识密集型应用；**Haystack** 是"生产派"——由 deepset 开发，Pipeline 架构清晰、模块化程度高，企业级生产部署最成熟（99.9% 可用性），适合需要稳定运行的生产环境。选择建议：快速原型+复杂 Agent → LangChain；数据密集型 RAG → LlamaIndex；生产稳定性优先 → Haystack。实际项目中三者并非互斥——LlamaIndex 可作为 LangChain 的检索后端，Haystack 的 Pipeline 可集成 LangChain 组件。2025 年新兴框架如 CrewAI（多 Agent 编排）、Semantic Kernel（微软企业级）、Agno（极速轻量）也在快速崛起。

## 详细解析

### 三大框架对比

```
┌──────────────┬─────────────┬──────────────┬──────────────┐
│ 维度         │ LangChain   │ LlamaIndex   │ Haystack     │
├──────────────┼─────────────┼──────────────┼──────────────┤
│ 定位         │ 通用 LLM 编排│ 数据+RAG 专家│ 生产级 NLP   │
│ GitHub Stars │ 100k+       │ 40k+         │ 18k+         │  ← 截至 2025 年
│ 语言         │ Python + TS │ Python + TS  │ Python       │
│ 核心抽象     │ Chain/Agent │ Index/Query  │ Pipeline/    │
│              │ /Tool       │ Engine       │ Component    │
├──────────────┼─────────────┼──────────────┼──────────────┤
│ Agent 能力   │ ★★★★★      │ ★★★☆☆       │ ★★★★☆       │
│ RAG 能力     │ ★★★★☆      │ ★★★★★       │ ★★★★☆       │
│ 生产就绪     │ ★★★☆☆      │ ★★★☆☆       │ ★★★★★       │
│ 学习曲线     │ 陡峭         │ 中等         │ 平缓         │
│ 灵活性       │ 极高         │ 高           │ 中高         │
├──────────────┼─────────────┼──────────────┼──────────────┤
│ 数据连接器   │ 多（社区驱动）│ 160+ 官方    │ 少（精选）   │
│ 模型支持     │ 最广泛       │ 广泛         │ 广泛         │
│ 监控/追踪    │ LangSmith   │ 内置追踪     │ Pipeline 日志│
│ 部署方式     │ LangServe   │ 自行部署     │ Hayhooks API │
└──────────────┴─────────────┴──────────────┴──────────────┘
```

### LangChain 核心架构

> **注意**：以下使用的 `create_tool_calling_agent` + `AgentExecutor` 是旧版 API。LangChain 官方现在更推荐使用 LangGraph 的 `create_react_agent` 来构建 Agent。

```python
# LangChain：链式编排 + Agent + 工具生态（旧版 API 示例）
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# 定义工具
@tool
def search_database(query: str) -> str:
    """搜索产品数据库"""
    return f"找到关于 {query} 的 3 条结果"

# 创建 Agent
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, [search_database], prompt)
executor = AgentExecutor(agent=agent, tools=[search_database])

# LangChain 优势：
# 1. 生态最大——几乎所有 LLM、向量数据库、工具都有集成
# 2. LangGraph 扩展——支持复杂的有状态 Agent 工作流
# 3. LangSmith——完整的追踪、评估、监控平台
# 4. 社区活跃——问题和示例最多

# LangChain 劣势：
# 1. 抽象层过多，调试困难（"框架深度"问题）
# 2. API 变更频繁，升级成本高
# 3. 过度封装导致性能损耗
# 4. 简单任务也需要理解大量概念
```

### LlamaIndex 核心架构

```python
# LlamaIndex：数据连接 + 索引 + 查询引擎
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# 加载数据并创建索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(similarity_top_k=5)

# 将查询引擎包装为工具
tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="knowledge_base",
    description="搜索内部知识库",
)

# 创建 Agent
agent = ReActAgent.from_tools([tool], verbose=True)
response = agent.chat("最新的产品定价是多少？")

# LlamaIndex 优势：
# 1. 160+ 数据连接器（PDF、数据库、API、Notion 等）
# 2. 多种索引类型（向量、关键词、知识图谱、树状）
# 3. 检索性能最优——专为数据查询场景优化
# 4. 与 LangChain 互操作——可作为其检索后端

# LlamaIndex 劣势：
# 1. Agent 能力相对弱（依赖外部框架）
# 2. 复杂编排能力不如 LangChain
# 3. 社区规模较小
```

### Haystack 核心架构

```python
# Haystack：Pipeline 架构——组件化 + 生产优先
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

# 定义 Pipeline（DAG 结构）
pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
pipeline.add_component("prompt", PromptBuilder(template=template))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-4o"))

# 连接组件
pipeline.connect("retriever", "prompt")
pipeline.connect("prompt", "llm")

# 运行
result = pipeline.run({"retriever": {"query": "什么是 RAG？"}})

# Haystack 优势：
# 1. Pipeline 架构清晰——DAG 图式编排，易于理解和调试
# 2. 类型安全——组件间的输入输出有严格类型检查
# 3. 生产成熟——deepset Cloud 提供企业级托管
# 4. 序列化——Pipeline 可导出为 YAML，支持版本控制

# Haystack 劣势：
# 1. 生态较小，第三方集成少
# 2. 仅支持 Python
# 3. Agent 能力发展较晚
```

### 选择决策树

```
需要构建什么类型的应用？
│
├── 复杂 Agent（多步推理、工具编排）
│   └── → LangChain + LangGraph
│
├── 知识密集型（RAG、文档问答、数据检索）
│   └── → LlamaIndex（可搭配 LangChain）
│
├── 生产部署优先（企业级、高可用）
│   └── → Haystack
│
├── 多 Agent 协作
│   └── → CrewAI 或 AutoGen
│
├── 微软生态
│   └── → Semantic Kernel
│
└── 极致性能 + 轻量
    └── → Agno（前 Phidata，号称最快）
```

## 常见误区 / 面试追问

1. **误区："LangChain 是唯一选择"** — LangChain 生态最大但不一定最适合。对于 RAG 专项任务，LlamaIndex 更专业；对于生产稳定性，Haystack 更成熟。框架选择应该基于具体需求，而非社区热度。

2. **误区："框架越全能越好"** — 全能框架意味着更多抽象层和更高复杂度。如果只需要一个 RAG Pipeline，使用 LangChain 全家桶就是杀鸡用牛刀。选择最小够用的框架可以降低维护成本。

3. **追问："这些框架可以混用吗？"** — 可以，且推荐。典型组合：LlamaIndex 处理数据索引和检索 → LangChain/LangGraph 编排 Agent 逻辑 → Haystack Pipeline 处理特定流程。框架间通常通过工具接口互操作。

4. **追问："2025 年有什么新兴框架值得关注？"** — (1) **CrewAI**：多 Agent 角色扮演，简单直观；(2) **Agno**：号称最快的 Agent 框架，模型无关；(3) **Semantic Kernel**：微软出品，企业级 .NET/Python 支持；(4) **Google ADK**：Google Agent Development Kit，集成 Gemini 生态；(5) **OpenAI Agents SDK**：轻量级，原生 OpenAI 集成。

## 参考资料

- [LangChain vs Haystack vs LlamaIndex: RAG Showdown 2025 (Medium)](https://mayur-ds.medium.com/langchain-vs-haystack-vs-llamaindex-rag-showdown-2025-28c222d34b0a)
- [We Tested 14 AI Agent Frameworks. Here's How to Choose (Softcery)](https://softcery.com/lab/top-14-ai-agent-frameworks-of-2025-a-founders-guide-to-building-smarter-systems)
- [Best 10 AI Agent Frameworks for 2025 (Deepchecks)](https://deepchecks.com/best-ai-agent-frameworks/)
- [AI Agent Frameworks Compared: LangChain, CrewAI, and More (Arsum)](https://arsum.com/blog/posts/ai-agent-frameworks/)
- [Compare the Top 7 RAG Frameworks in 2025 (Pathway)](https://pathway.com/rag-frameworks)
