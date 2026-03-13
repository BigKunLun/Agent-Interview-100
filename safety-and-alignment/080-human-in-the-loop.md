# Human-in-the-Loop：何时以及如何引入人工审核？

> 难度：中级
> 分类：Safety & Alignment

## 简短回答

Human-in-the-Loop (HITL) 是在 AI Agent 工作流中**战略性嵌入人类判断**的设计模式——不是所有决策都该自动化，也不是所有操作都需要人工审核，关键是找到**自动化效率和人类监督安全之间的最佳平衡点**。核心设计模式包括：(1) **审批/检查点模式**——在关键操作前暂停等待人类确认（如删除数据、发送邮件、支付）；(2) **置信度路由**——Agent 对自身置信度评分，低于阈值自动升级给人类；(3) **Human-as-a-Tool**——Agent 将"人类"视为可调用的工具，遇到不确定时主动提问；(4) **角色审批**——只有特定角色（审核员、管理员）才能批准敏感操作；(5) **异步升级**——非阻塞性地将决策路由到 Slack/邮件等异步审核渠道。何时使用的决策原则：**不可逆操作、合规监管要求、高风险领域（医疗/金融/法律）、需要同理心的场景**必须加人工介入。主流框架 LangGraph（`interrupt()`）、HumanLayer、Amazon Bedrock Agents 都提供了开箱即用的 HITL 支持。

## 详细解析

### 何时需要 Human-in-the-Loop

```
必须加 HITL 的场景：
│
├── 不可逆/破坏性操作
│   ├── 删除数据、文件、资源
│   ├── 发送外部通信（邮件、消息、通知）
│   ├── 金融交易（支付、转账）
│   └── 修改权限或配置
│
├── 合规监管要求
│   ├── 合同审核（法律）
│   ├── 医疗建议（医疗）
│   ├── 投资建议（金融）
│   └── 涉及隐私数据的操作
│
├── 高风险/高影响
│   ├── 影响客户的决策
│   ├── 影响业务指标的变更
│   └── 安全关键操作
│
└── 主观判断
    ├── 需要同理心的客服场景
    ├── 创意内容的最终审定
    └── 复杂的优先级权衡

可以完全自动化的场景：
├── 信息查询和检索
├── 数据格式转换
├── 日志分析和报告生成
├── 内部代码的 lint/格式化
└── 低风险的重复性操作
```

### 五种 HITL 设计模式

```python
# 模式 1：审批/检查点模式（最常用）
class ApprovalCheckpoint:
    """在关键操作前暂停等待人类确认"""

    async def execute_with_approval(self, agent, task):
        plan = await agent.plan(task)

        for step in plan.steps:
            if step.requires_approval:
                # 暂停执行，展示计划给人类
                approval = await self.request_approval(
                    action=step.action,
                    params=step.params,
                    context=step.reasoning,
                )
                if not approval.approved:
                    return self.handle_rejection(approval.feedback)

            await agent.execute_step(step)


# 模式 2：置信度路由
class ConfidenceBasedRouting:
    """Agent 置信度低时自动升级给人类"""

    def __init__(self, confidence_threshold=0.7):
        self.threshold = confidence_threshold

    async def route(self, agent_result):
        if agent_result.confidence >= self.threshold:
            return {"action": "auto_execute", "result": agent_result}
        elif agent_result.confidence >= 0.4:
            return {"action": "human_review", "result": agent_result}
        else:
            return {"action": "human_takeover", "result": agent_result}
        # 高置信度 → 自动执行
        # 中等置信度 → 人工审核 Agent 的建议
        # 低置信度 → 人工完全接管


# 模式 3：Human-as-a-Tool
class HumanTool:
    """Agent 将人类视为可调用的工具"""

    name = "ask_human"
    description = "当你不确定如何处理时，向人类提问"

    async def execute(self, question: str) -> str:
        """Agent 主动向人类提问"""
        response = await self.send_to_human(
            question=question,
            channel="slack",  # 或邮件、UI 弹窗
            timeout=300,      # 等待 5 分钟
        )
        return response.answer


# 模式 4：角色审批
class RoleBasedApproval:
    """根据操作类型要求不同角色审批"""

    approval_matrix = {
        "delete_data":     {"role": "admin",    "required": True},
        "send_email":      {"role": "reviewer", "required": True},
        "read_database":   {"role": "any",      "required": False},
        "modify_config":   {"role": "admin",    "required": True},
        "generate_report": {"role": "any",      "required": False},
    }

    async def check_approval(self, action, user):
        rule = self.approval_matrix.get(action)
        if not rule or not rule["required"]:
            return True
        return user.role == rule["role"] or user.role == "admin"


# 模式 5：异步升级
class AsyncEscalation:
    """非阻塞的异步人工审核"""

    async def escalate(self, issue, priority="normal"):
        if priority == "critical":
            # 关键问题：同步等待
            return await self.sync_review(issue)
        else:
            # 一般问题：异步通知，Agent 继续其他工作
            ticket = await self.create_review_ticket(issue)
            await self.notify_slack(issue, ticket)
            return {"status": "escalated", "ticket": ticket.id}
```

### LangGraph HITL 实现

```python
from langgraph.graph import StateGraph
from langgraph.types import interrupt

def agent_node(state):
    """Agent 决策节点"""
    result = llm.invoke(state["messages"])
    return {"messages": [result], "pending_action": result.tool_calls}

def human_review_node(state):
    """人工审核节点——使用 interrupt() 暂停"""
    action = state["pending_action"]

    # 暂停图执行，等待人类输入
    human_decision = interrupt({
        "action": action,
        "question": f"Agent 想要执行: {action}，是否批准？",
    })

    if human_decision["approved"]:
        return {"approved": True}
    else:
        return {"approved": False, "feedback": human_decision["reason"]}

def should_review(state):
    """路由：是否需要人工审核"""
    action = state.get("pending_action")
    if action and action["tool"] in HIGH_RISK_TOOLS:
        return "human_review"
    return "execute"

# 构建图
graph = StateGraph()
graph.add_node("agent", agent_node)
graph.add_node("human_review", human_review_node)
graph.add_node("execute", execute_node)
graph.add_conditional_edges("agent", should_review)
graph.add_edge("human_review", "execute")
```

### 决策框架

```
如何决定是否需要 HITL？

                    质量重要 vs 速度重要？
                    /                    \
              质量优先                 速度优先
              /                          \
    操作可逆？                    低风险操作？
    /        \                    /        \
  不可逆    可逆               是          否
   ↓         ↓                 ↓           ↓
 必须HITL  可选HITL       全自动化     加HITL

推荐的渐进策略：
Phase 1: 所有操作都需人工确认（最安全）
Phase 2: 低风险操作自动化，高风险保留人工
Phase 3: 基于置信度动态路由
Phase 4: 只在异常/边界情况升级人工

关键指标：
├── 人工介入率（目标：逐步降低到 10-20%）
├── 人工审核响应时间（影响用户体验）
├── 人工否决率（反映 Agent 质量）
└── 自动化后的质量对比（确保质量不下降）
```

### HITL 工具生态

```
┌──────────────────┬──────────────────────────────────┐
│ 工具             │ HITL 能力                        │
├──────────────────┼──────────────────────────────────┤
│ LangGraph        │ interrupt() 暂停图执行           │
│                  │ 完全控制路由和恢复               │
├──────────────────┼──────────────────────────────────┤
│ HumanLayer       │ 框架无关的 HITL API/SDK          │
│                  │ Slack/Email/UI 多渠道审批        │
├──────────────────┼──────────────────────────────────┤
│ Amazon Bedrock   │ 内置 user confirmation 功能      │
│ Agents           │ 可配置哪些工具需要确认           │
├──────────────────┼──────────────────────────────────┤
│ CrewAI           │ human_input=True 参数            │
│                  │ Agent 级别的人工输入开关          │
├──────────────────┼──────────────────────────────────┤
│ Permit.io        │ 基于 RBAC/ABAC 的细粒度审批     │
│                  │ 与 Agent 框架集成的权限管理      │
└──────────────────┴──────────────────────────────────┘
```

## 常见误区 / 面试追问

1. **误区："HITL 只是临时方案，最终目标是完全自动化"** — HITL 不是权宜之计，而是长期的架构模式。即使 AI 能力不断提升，高风险决策、合规审核和同理心判断仍然需要人类。随着 Agent 能力提升，人类的角色从"逐一审核"转向"战略监督和异常处理"。

2. **误区："加了 HITL 就安全了"** — HITL 本身也有失败模式：审核疲劳（人类不仔细看就点批准）、响应延迟（人类不在线导致 Agent 阻塞）、能力不匹配（审核者不理解技术细节）。需要配合良好的 UI 设计、轮值机制和明确的审核标准。

3. **追问："如何降低 HITL 对用户体验的影响？"** — (1) 异步审核——非阻塞操作用异步通知；(2) 批量审核——收集多个待审项一次性处理；(3) 预审批——用户提前授权某类操作；(4) 置信度路由——只有低置信度的才需人工，大部分自动通过。

4. **追问："如何量化 HITL 的投入产出？"** — 跟踪三个指标：(1) 人工介入率及趋势（应该随 Agent 优化而下降）；(2) 人工否决后的实际影响（避免了多少潜在问题）；(3) HITL 引入的延迟对业务指标的影响。如果否决率 < 2% 且历史无重大事故，可以考虑降低该类操作的审核要求。

## 参考资料

- [Human-in-the-Loop for AI Agents: Best Practices, Frameworks, Use Cases (Permit.io)](https://www.permit.io/blog/human-in-the-loop-for-ai-agents-best-practices-frameworks-use-cases-and-demo)
- [Agents with Human in the Loop: Everything You Need to Know (CAMEL AI)](https://dev.to/camelai/agents-with-human-in-the-loop-everything-you-need-to-know-3fo5)
- [Human-in-the-Loop AI in 2025: Proven Design Patterns (Ideafloats)](https://blog.ideafloats.com/human-in-the-loop-ai-in-2025/)
- [Humans and Agents in Software Engineering Loops (Martin Fowler)](https://martinfowler.com/articles/exploring-gen-ai/humans-and-agents.html)
- [Implement Human-in-the-Loop Confirmation with Amazon Bedrock Agents (AWS)](https://aws.amazon.com/blogs/machine-learning/implement-human-in-the-loop-confirmation-with-amazon-bedrock-agents/)
