# 比较主流多 Agent 框架：CrewAI、AutoGen、LangGraph

> 难度：中级
> 分类：Multi-Agent

## 简短回答

三大多 Agent 框架各有侧重：**CrewAI** 以角色为核心，用 Crew（团队）+ Flow（流程）两层架构实现快速搭建，适合业务流程自动化和快速原型；**LangGraph** 以图为核心，用节点-边-状态机实现精确的流程控制，可追踪可调试，适合复杂生产系统；**AutoGen**（Microsoft Research）以对话为核心，Agent 通过自然语言消息协作，最灵活但最难调试，适合群体决策和辩论场景。选择原则：需要快速上手选 CrewAI，需要精确控制选 LangGraph，需要灵活对话选 AutoGen。

## 详细解析

### 设计哲学对比

```
CrewAI:  角色驱动 —— "谁做什么"
         Crew（团队）= Agent + Task + Process
         Flow（流程）= 确定性任务编排

LangGraph: 图驱动 —— "怎么流转"
           Graph = Node + Edge + State
           支持条件分支、循环、并行

AutoGen:  对话驱动 —— "怎么讨论"
          GroupChat = Agent + Message + Orchestration
          Agent 通过对话自然协作
```

### 架构详解

#### CrewAI

```python
from crewai import Agent, Task, Crew, Process

# 定义角色
researcher = Agent(
    role="高级研究分析师",
    goal="发现关于 {topic} 的最新趋势",
    backstory="你是一位经验丰富的研究者...",
    tools=[search_tool],
)

writer = Agent(
    role="技术写作专家",
    goal="将研究成果写成引人入胜的文章",
    backstory="你擅长将复杂技术概念简化...",
)

# 定义任务
research_task = Task(
    description="研究 {topic} 的最新进展",
    agent=researcher,
    expected_output="详细的研究报告"
)

write_task = Task(
    description="基于研究报告撰写博客文章",
    agent=writer,
    expected_output="1500 字的技术博客"
)

# 组装团队
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential  # 或 Process.hierarchical
)

result = crew.kickoff(inputs={"topic": "AI Agent"})
```

#### LangGraph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

class State(TypedDict):
    messages: list
    research_data: str
    draft: str

# 定义节点（每个节点可以是 Agent）
async def research_node(state: State) -> dict:
    data = await researcher.invoke(state["messages"])
    return {"research_data": data}

async def write_node(state: State) -> dict:
    draft = await writer.invoke(state["research_data"])
    return {"draft": draft}

def should_revise(state: State) -> str:
    if quality_check(state["draft"]):
        return "end"
    return "revise"  # 条件路由

# 构建图
graph = StateGraph(State)
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_node("revise", revise_node)

graph.add_edge(START, "research")
graph.add_edge("research", "write")
graph.add_conditional_edges("write", should_revise, {
    "end": END,
    "revise": "revise"
})
graph.add_edge("revise", "write")  # 循环

app = graph.compile()
```

#### AutoGen

> **注意**：以下代码示例对应 AutoGen 0.2.x 版本 API。0.4+（AG2）已重构 API，import 路径变为 `from autogen_agentchat.agents import AssistantAgent` 等。

```python
from autogen import ConversableAgent

# 定义对话 Agent
researcher = ConversableAgent(
    name="Researcher",
    system_message="你是研究分析师...",
    llm_config={"model": "gpt-4"},
)

writer = ConversableAgent(
    name="Writer",
    system_message="你是技术写作专家...",
    llm_config={"model": "gpt-4"},
)

critic = ConversableAgent(
    name="Critic",
    system_message="你是内容审核专家...",
    llm_config={"model": "gpt-4"},
)

# 群聊模式
from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(
    agents=[researcher, writer, critic],
    messages=[],
    max_round=10
)
manager = GroupChatManager(groupchat=groupchat)
researcher.initiate_chat(manager, message="研究 AI Agent 的最新趋势")
```

### 核心维度对比

| 维度 | CrewAI | LangGraph | AutoGen |
|------|--------|-----------|---------|
| **核心抽象** | 角色/团队 | 图/状态机 | 对话/消息 |
| **学习曲线** | 低 | 高 | 中 |
| **流程控制** | 顺序/层级 | 任意图（条件、循环、并行） | 对话驱动 |
| **状态管理** | 内置简单状态 | 精细状态 + Reducer | 对话历史 |
| **调试性** | 中 | 高（图可视化） | 低（非确定性对话） |
| **生产就绪度** | 中 | 高（Klarna、Replit 使用） | 中（v0.4 仍在稳定） |
| **工具生态** | 中 | 300+ 集成 + LangSmith | Azure AI 集成 |
| **Human-in-Loop** | 支持 | 原生支持 | 原生支持 |
| **配置方式** | YAML 驱动 | 代码驱动 | 代码/Studio GUI |

### 各框架最佳适用场景

```python
framework_guide = {
    "CrewAI": [
        "业务流程自动化（角色清晰的团队协作）",
        "快速原型和 MVP",
        "非技术团队参与的项目（YAML 配置）",
    ],
    "LangGraph": [
        "复杂生产系统（需要精确流程控制）",
        "需要条件分支和循环的工作流",
        "对可观测性要求高的企业应用",
    ],
    "AutoGen": [
        "群体决策和多 Agent 辩论",
        "研究探索和创意生成",
        "Microsoft/Azure 生态的项目",
    ],
}
```

### 新兴竞争者

除了三大框架，值得关注的还有：
- **Google ADK**：SequentialAgent、ParallelAgent、LoopAgent 内置编排
- **OpenAI Agents SDK**：轻量级 Handoff 机制，代码优先编排
- **AWS Agent Squad**：Agent-as-Tools 架构，Lead Agent 协调团队

## 常见误区 / 面试追问

1. **误区："选最流行的框架就对了"** — 框架选择应基于具体需求：需要可控性选 LangGraph，需要快速迭代选 CrewAI，需要灵活对话选 AutoGen。没有万能框架。

2. **误区："框架 = 生产就绪"** — 框架提供基础抽象，但生产环境还需要自行解决可观测性、错误处理、安全性等横切关注点。LangGraph 在这方面最成熟（配合 LangSmith），其他框架可能需要更多自定义。

3. **追问："能否混合使用多个框架？"** — 可以。例如用 LangGraph 做整体编排，单个节点内部用 CrewAI 的 Crew 完成子任务。但需要注意状态同步和调试复杂度的增加。

4. **追问："自研 vs 使用框架，如何取舍？"** — 如果需求与框架的抽象契合，使用框架；如果框架的限制导致大量 hack，考虑自研。OpenAI 的建议是：先用代码编排（最可控），只在需要灵活性时才引入 LLM 编排。

## 参考资料

- [CrewAI vs LangGraph vs AutoGen (DataCamp)](https://www.datacamp.com/tutorial/crewai-vs-langgraph-vs-autogen)
- [AI Agent Framework Comparison (Latenode)](https://latenode.com/blog/platform-comparisons-alternatives/automation-platform-comparisons/langgraph-vs-autogen-vs-crewai-complete-ai-agent-framework-comparison-architecture-analysis-2025)
- [Top AI Agent Frameworks (Codecademy)](https://www.codecademy.com/article/top-ai-agent-frameworks-in-2025)
- [Detailed Comparison of Top 6 AI Agent Frameworks (Turing)](https://www.turing.com/resources/ai-agent-frameworks)
- [Best AI Agent Frameworks 2025 (Maxim AI)](https://www.getmaxim.ai/articles/top-5-ai-agent-frameworks-in-2025-a-practical-guide-for-ai-builders/)
