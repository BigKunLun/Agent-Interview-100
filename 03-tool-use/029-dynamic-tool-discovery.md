# 如何实现动态工具发现和注册？

> 难度：高级
> 分类：Tool Use

## 简短回答

动态工具发现是指 Agent 在运行时（而非编译时）发现和使用新工具的能力。这解决了静态工具加载的两大问题：**上下文窗口浪费**（50+ 工具定义可消耗 55k+ tokens）和**选择准确率下降**（工具越多，模型选错的概率越高）。主流实现方式包括：**MCP 动态发现**（通过 `tools/list` 端点和 `list_changed` 通知实时更新工具列表）、**语义检索**（用 Embedding + FAISS 按用户意图检索最相关工具）、**上下文感知过滤**（根据认证状态、权限、会话阶段动态暴露不同工具集）、**Instruction-Tool Retrieval (ITR)**（每步只检索最小必要工具子集，减少 95% 上下文开销）。

## 详细解析

### 为什么需要动态工具发现？

静态工具加载的问题随规模暴露：

```
静态方式（所有工具塞进 System Prompt）：
┌────────────────────────────────────┐
│ System Prompt                      │
│ 工具 1 定义 (500 tokens)           │
│ 工具 2 定义 (500 tokens)           │
│ ...                                │
│ 工具 50 定义 (500 tokens)          │
│ ──────────────────                 │
│ 总计: 25,000+ tokens 仅用于工具定义 │
└────────────────────────────────────┘

动态方式（按需加载相关工具）：
┌────────────────────────────────────┐
│ System Prompt                      │
│ 工具 A 定义 (500 tokens) ← 相关    │
│ 工具 B 定义 (500 tokens) ← 相关    │
│ 工具 C 定义 (500 tokens) ← 可能相关│
│ ──────────────────                 │
│ 总计: 1,500 tokens                 │
└────────────────────────────────────┘
```

Anthropic 内部测试显示 58 个工具可消耗约 55k tokens。工具数量增加不仅浪费成本，还会降低工具选择准确率。

### 方式 1：MCP 动态发现

MCP 协议原生支持工具的动态发现和热更新：

```python
# MCP Server 端：工具列表变化时通知客户端
class DynamicMCPServer:
    def __init__(self):
        self.tools = {}

    async def register_tool(self, tool):
        """运行时注册新工具"""
        self.tools[tool.name] = tool
        # 通知所有连接的 Client 工具列表已变更
        await self.notify("notifications/tools/list_changed")

    async def unregister_tool(self, tool_name):
        """运行时移除工具"""
        del self.tools[tool_name]
        await self.notify("notifications/tools/list_changed")

    async def handle_tools_list(self):
        """响应 tools/list 请求"""
        return [tool.to_schema() for tool in self.tools.values()]

# MCP Client 端：监听变更并更新可用工具
class MCPClient:
    async def on_notification(self, method, params):
        if method == "notifications/tools/list_changed":
            # 重新获取工具列表
            updated_tools = await self.call("tools/list")
            self.agent.update_tools(updated_tools)
```

**典型场景：** 用户登录后，Server 根据权限暴露不同工具；Session 从"浏览"阶段进入"购买"阶段时，新增支付相关工具。

### 方式 2：语义检索（Tool RAG）

当工具数量超过 50 个时，用 Embedding 检索最相关的工具子集：

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class ToolRegistry:
    def __init__(self, tools: list[dict]):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.tools = tools

        # 将工具描述编码为向量
        descriptions = [t["description"] for t in tools]
        self.embeddings = self.model.encode(descriptions)

        # 构建 FAISS 索引
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # 内积相似度
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def discover(self, query: str, top_k: int = 5) -> list[dict]:
        """根据用户意图检索最相关的工具"""
        query_vec = self.model.encode([query])
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return [self.tools[i] for i in indices[0] if scores[0][list(indices[0]).index(i)] > 0.3]

# 使用：只将相关工具传给 LLM
registry = ToolRegistry(all_200_tools)
relevant = registry.discover("查询用户订单状态", top_k=5)
response = llm.generate(query=user_input, tools=relevant)
```

### 方式 3：上下文感知过滤

根据运行时上下文动态调整可用工具：

```python
class ContextAwareToolFilter:
    def __init__(self, all_tools):
        self.all_tools = all_tools

    def get_available_tools(self, context: dict) -> list:
        available = []
        for tool in self.all_tools:
            # 权限检查
            if not self.user_has_permission(context["user"], tool):
                continue
            # 认证状态检查
            if tool.requires_auth and not context.get("authenticated"):
                continue
            # 会话阶段检查
            if tool.stage and tool.stage != context.get("session_stage"):
                continue
            # Feature Flag 检查
            if tool.feature_flag and not is_enabled(tool.feature_flag):
                continue
            available.append(tool)
        return available
```

### 方式 4：Instruction-Tool Retrieval (ITR)

最新研究方法——每一步只检索最小必要的系统指令片段和工具子集：

```python
class InstructionToolRetriever:
    """每步检索最小必要的指令和工具"""

    def retrieve_for_step(self, agent_state, step_context):
        # 基于当前步骤的上下文，检索相关指令片段
        relevant_instructions = self.retrieve_instructions(step_context)
        # 检索相关工具（而非加载全部）
        relevant_tools = self.retrieve_tools(step_context)

        return {
            "system_prompt": relevant_instructions,  # 精简的指令
            "tools": relevant_tools,                 # 最小工具集
        }
        # 效果：每步上下文 tokens 减少 95%
        #       工具选择准确率提升 32%
        #       端到端成本降低 70%
```

### 方式 5：集中式工具注册中心

生产环境的规模化方案：

```python
class CentralToolRegistry:
    """集中式工具注册中心"""

    def __init__(self):
        self.registry = {}       # 工具元数据
        self.health_checks = {}  # 健康状态
        self.search_index = None # 语义搜索索引

    def register(self, server_id: str, tools: list[dict]):
        """MCP Server 注册其工具"""
        for tool in tools:
            self.registry[tool["name"]] = {
                "server": server_id,
                "schema": tool,
                "registered_at": datetime.now(),
                "status": "active"
            }
        self._rebuild_index()  # 重建搜索索引

    def deregister(self, server_id: str):
        """Server 下线时移除工具"""
        self.registry = {
            k: v for k, v in self.registry.items()
            if v["server"] != server_id
        }
        self._rebuild_index()

    def search(self, query: str, top_k: int = 5) -> list:
        """语义搜索发现工具"""
        return self.search_index.query(query, top_k)
```

### 各方式对比

| 方式 | 工具规模 | 延迟开销 | 灵活性 | 实现复杂度 |
|------|---------|---------|--------|-----------|
| MCP 动态发现 | 中等(10-50) | 低 | 高 | 中 |
| 语义检索 | 大(50-1000+) | 中(检索耗时) | 高 | 中 |
| 上下文过滤 | 中等(10-50) | 极低 | 中 | 低 |
| ITR | 大(50+) | 中 | 高 | 高 |
| 集中注册中心 | 大(100+) | 中 | 极高 | 高 |

## 常见误区 / 面试追问

1. **误区："把所有工具都放进 System Prompt 最简单可靠"** — 工具超过 20 个后，选择准确率明显下降，上下文窗口被大量占用。动态发现是规模化的必要手段。

2. **误区："动态发现只是性能优化"** — 它也是安全机制。通过动态过滤，可以根据用户权限、认证状态实时调整可用工具，实现最小权限原则。

3. **追问："语义检索选错了工具怎么办？"** — 两层保护：(1) 检索时取 top-k（如 5-8 个）而非 top-1，给 LLM 更多选项；(2) LLM 仍然做最终选择——检索只是缩小范围，不是替代 LLM 的判断。

4. **追问："动态工具发现和 Agent 编排有什么关系？"** — 动态发现影响 Agent 的能力边界——Agent 能做什么取决于它能发现哪些工具。在多 Agent 系统中，不同 Agent 可能连接不同的 MCP Server，拥有不同的工具发现范围。

## 参考资料

- [Dynamic Tool Discovery in MCP (Speakeasy)](https://www.speakeasy.com/mcp/tool-design/dynamic-tool-discovery)
- [7 Benefits of a Centralized MCP Tool Registry (Nordic APIs)](https://nordicapis.com/7-benefits-of-a-centralized-mcp-tool-registry/)
- [Dynamic System Instructions and Tool Exposure for Efficient Agentic LLMs (arXiv)](https://arxiv.org/abs/2602.17046)
- [MCP Gateway Registry: Dynamic Tool Discovery (GitHub)](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/dynamic-tool-discovery.md)
- [How Dynamic Tool Discovery with MCP Is Rewriting the Rules of Autonomy (Medium)](https://medium.com/ai-simplified-in-plain-english/how-dynamic-tool-discovery-with-mcp-is-rewriting-the-rules-of-autonomy-5cce7475d6e2)
