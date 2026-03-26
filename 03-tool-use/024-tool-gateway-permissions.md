# 如何设计 Tool Gateway 和工具权限管理？

> 难度：中级
> 分类：Tool Use

## 简短回答

Tool Gateway 是 Agent 与工具之间的安全中间层——Agent 不直接调用工具，所有请求都经过 Gateway 做身份验证、权限检查、速率限制和审计日志。核心设计原则是**最小权限（Least Privilege）**和**将 Agent 视为不可信请求者**。权限管理采用 Policy-as-Code（如 OPA/Cedar）将授权逻辑外部化，配合 RBAC/ABAC 做细粒度控制。

## 详细解析

### 为什么需要 Tool Gateway？

没有 Gateway 的架构中，Agent 直接持有 API 密钥并调用外部服务。风险包括：
- Agent 被 Prompt Injection 操纵，调用不该调用的工具
- Agent 无意中执行破坏性操作（删除数据、发送未经审核的消息）
- 无法追踪和审计 Agent 的行为
- 凭证暴露在 Agent 的上下文中

### Gateway 架构

```
┌──────────┐     ┌────────────────────┐     ┌──────────────┐
│  Agent   │────→│    Tool Gateway    │────→│  实际工具/API │
│ (LLM)   │     │                    │     │              │
│          │     │ 1. 身份验证        │     │  - 天气 API  │
│ 不持有   │     │ 2. 权限检查(OPA)   │     │  - 数据库    │
│ 任何凭证 │     │ 3. 输入验证        │     │  - 邮件服务  │
│          │     │ 4. 速率限制        │     │  - 文件系统  │
│          │←────│ 5. 审计日志        │←────│              │
└──────────┘     │ 6. 结果过滤        │     └──────────────┘
                 └────────────────────┘
```

关键原则：**Agent 永远不直接与基础设施 API 通信。** Gateway 拦截每个请求，做验证、授权、执行。Agent 不持有任何凭证。

```python
class ToolGateway:
    def __init__(self, policy_engine, rate_limiter, audit_logger):
        self.policy = policy_engine      # OPA / Cedar
        self.limiter = rate_limiter
        self.logger = audit_logger
        self.tools = {}                  # 注册的工具

    async def execute(self, agent_id: str, tool_name: str, params: dict) -> dict:
        # 1. 身份验证
        agent = await self.authenticate(agent_id)

        # 2. 权限检查
        decision = self.policy.evaluate({
            "agent": agent,
            "tool": tool_name,
            "params": params,
            "context": {"time": now(), "user": agent.delegated_user}
        })
        if not decision.allowed:
            self.logger.log_denied(agent_id, tool_name, decision.reason)
            return {"error": f"权限不足: {decision.reason}"}

        # 3. 输入验证
        validated = self.validate_input(tool_name, params)

        # 4. 速率限制
        if not self.limiter.allow(agent_id, tool_name):
            return {"error": "请求过于频繁，请稍后重试"}

        # 5. 执行工具
        result = await self.tools[tool_name].execute(validated)

        # 6. 审计日志
        self.logger.log_execution(agent_id, tool_name, params, result)

        # 7. 结果过滤（去除敏感信息）
        return self.filter_sensitive(result, agent.permission_level)
```

### 权限模型

#### RBAC（基于角色的访问控制）

```python
# 角色定义
roles = {
    "reader_agent": {
        "allowed_tools": ["search_docs", "get_weather", "calculate"],
        "denied_tools": ["send_email", "delete_record", "execute_sql"],
    },
    "support_agent": {
        "allowed_tools": ["search_docs", "get_order", "create_ticket"],
        "denied_tools": ["delete_record", "modify_pricing"],
    },
    "admin_agent": {
        "allowed_tools": ["*"],  # 全部权限
        "requires_approval": ["delete_*", "modify_*"],  # 高危操作需审批
    },
}
```

#### ABAC（基于属性的访问控制）

更细粒度——基于请求的上下文属性做决策：

```python
# OPA Policy (Rego 语言)
# policy.rego
"""
package tool_access

default allow = false

# 工作时间内允许发送邮件
allow {
    input.tool == "send_email"
    input.context.hour >= 9
    input.context.hour <= 18
    input.agent.role == "support_agent"
}

# 只允许查询自己负责的客户数据
allow {
    input.tool == "get_customer_data"
    input.params.customer_id in input.agent.assigned_customers
}

# 高危操作需要人工审批
allow {
    input.tool == "delete_record"
    input.context.human_approved == true
}
"""
```

### 委托授权（Delegated Authorization）

Agent 的权限不应超过委托它的用户的权限：

```python
class DelegatedAuth:
    def check(self, agent: Agent, tool: str, params: dict) -> bool:
        # Agent 的权限 = min(Agent 角色权限, 委托用户权限)
        agent_allowed = self.check_agent_role(agent.role, tool)
        user_allowed = self.check_user_permission(agent.delegated_user, tool)

        # 两者都允许才放行
        return agent_allowed and user_allowed
```

关键原则：人类用户常常被过度授权，Agent 的权限应该独立审查，不能简单继承用户的全部权限。

### Human-in-the-Loop 审批

```python
class ApprovalGateway:
    HIGH_RISK_TOOLS = ["send_email", "delete_record", "transfer_funds"]

    async def execute_with_approval(self, agent_id, tool, params):
        if tool in self.HIGH_RISK_TOOLS:
            # 暂停执行，等待人工审批
            approval_request = await self.request_human_approval(
                agent_id=agent_id,
                tool=tool,
                params=params,
                timeout=300  # 5 分钟超时
            )
            if not approval_request.approved:
                return {"error": "操作被人工拒绝", "reason": approval_request.reason}

        return await self.gateway.execute(agent_id, tool, params)
```

### 多 Agent 场景的权限挑战

多 Agent 工作流中，权限管理更复杂：
- Agent A 可能把任务委托给 Agent B
- Agent B 的权限不应超过 Agent A
- 需要追踪完整的委托链

```python
# 委托链追踪
class DelegationChain:
    def validate_delegation(self, from_agent, to_agent, tool):
        # to_agent 的权限不能超过 from_agent
        if not self.is_subset(to_agent.permissions, from_agent.permissions):
            raise PermissionError("被委托 Agent 权限不能超过委托者")
```

### 运行时护栏

除了权限控制，还需要内容级别的护栏：

```
输入护栏：扫描用户输入是否有恶意意图、越狱尝试、PII
    ↓
Agent 处理
    ↓
输出护栏：验证 Agent 输出是否有幻觉、毒性内容、敏感数据
```

### 治理流水线模式

将策略更新像软件发布一样管理：
1. 安全/法律团队编写策略 → 翻译为机器可读代码
2. 对历史日志做影响分析（模拟部署）
3. 确认无误后自动部署到全部 Agent

## 常见误区 / 面试追问

1. **误区："在 Agent 代码里做权限检查就够了"** — Agent 代码可能被 Prompt Injection 绕过。权限检查必须在 Agent 外部（Gateway 层）执行，作为独立的安全边界。

2. **误区："给 Agent 和用户一样的权限"** — 用户常常被过度授权。Agent 应该只获得完成当前任务所需的最小权限集。

3. **追问："Policy-as-Code 有什么好处？"** — (1) 授权逻辑与应用代码分离，安全团队可以独立管理；(2) 策略可以版本控制和审计；(3) 策略变更不需要重新部署应用代码。

4. **追问："AWS Bedrock 的 Policy 和自建 Gateway 有什么区别？"** — 云服务商的方案（如 Bedrock AgentCore Policy）在 Agent 代码外部执行安全控制，开箱即用但绑定生态。自建 Gateway 更灵活但需要维护，适合多云或自托管场景。

## 参考资料

- [Building a Least-Privilege AI Agent Gateway (InfoQ)](https://www.infoq.com/articles/building-ai-agent-gateway-mcp/)
- [AI Agent Access Control: How to Handle Permissions (Noma Security)](https://noma.security/resources/access-control-for-ai-agents/)
- [Best Practices of Authorizing AI Agents (Oso)](https://www.osohq.com/learn/best-practices-of-authorizing-ai-agents)
- [Access Control and Permission Management for AI Agents (Cerbos)](https://www.cerbos.dev/blog/permission-management-for-ai-agents)
- [Agent Governance Patterns: Policy-as-Code for Live Systems (a21.ai)](https://a21.ai/agent-governance-patterns-policy-as-code-for-live-systems/)
