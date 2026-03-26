# Agent 系统的灾难恢复与高可用设计

> 难度：高级
> 分类：Production & Deployment

## 简短回答

Agent 系统的高可用（HA）和灾难恢复（DR）比传统服务更关键——因为 AI Agent 正从后台走向前台，直接面对客户。如果 Agent 背后的 LLM 服务或数据库宕机，用户立即感知。核心设计原则：(1) **多提供商冗余**——不依赖单一 LLM 提供商，主备切换延迟 < 1 秒；(2) **状态持久化与检查点**——每步保存 Agent 执行状态，支持从任意步骤恢复，跨越小时甚至天级别的长任务；(3) **多区域部署**——Active-Passive 架构，故障区域的流量自动切换到备用区域；(4) **优雅降级**——LLM 不可用时降级到规则引擎/缓存/人工接管，而非完全不可用；(5) **熔断器模式**——检测到异常时自动停止 Agent 执行，防止级联故障。Microsoft Foundry Agent Service 推荐 Active-Passive 配合 Gateway Routing 模式，备用区域保持温备。LangGraph 提供了内置的检查点和恢复机制。实战教训：Wells Fargo 2.45 亿次 AI 交互无人工介入证明了正确架构的价值，MD Anderson 在 IBM Watson 上 6200 万美元的失败则警示了架构决策的重要性。

## 详细解析

### 高可用架构设计

```
Agent 系统高可用架构：

┌─────────────────────────────────────────────┐
│              全球负载均衡 (DNS/CDN)           │
├───────────────────┬─────────────────────────┤
│   Region A (主)   │   Region B (备)         │
│                   │                         │
│ ┌───────────────┐ │ ┌───────────────┐      │
│ │ API Gateway   │ │ │ API Gateway   │ (温备)│
│ ├───────────────┤ │ ├───────────────┤      │
│ │ Agent Workers │ │ │ Agent Workers │ (缩容)│
│ ├───────────────┤ │ ├───────────────┤      │
│ │ Model Gateway │ │ │ Model Gateway │      │
│ │ (多提供商)    │ │ │ (多提供商)    │      │
│ ├───────────────┤ │ ├───────────────┤      │
│ │ Redis (主)    │←→│ Redis (副本)  │ 同步  │
│ │ PostgreSQL(主)│←→│ PostgreSQL(副)│ 复制  │
│ └───────────────┘ │ └───────────────┘      │
└───────────────────┴─────────────────────────┘

故障切换流程：
1. 健康检查检测到 Region A 不可用
2. DNS/负载均衡将流量切换到 Region B
3. Region B 的 Worker 自动扩容
4. 从 Redis/PG 副本中恢复会话状态
5. 用户感知的中断时间 < 30 秒
```

### 多提供商故障转移

```python
class LLMFailoverManager:
    """LLM 提供商级别的故障转移"""

    def __init__(self):
        self.providers = [
            {"name": "openai",    "priority": 1, "healthy": True},
            {"name": "anthropic", "priority": 2, "healthy": True},
            {"name": "google",    "priority": 3, "healthy": True},
        ]
        self.circuit_breakers = {}

    async def call_with_failover(self, request):
        """带故障转移的 LLM 调用"""
        for provider in sorted(self.providers, key=lambda p: p["priority"]):
            if not provider["healthy"]:
                continue

            breaker = self.circuit_breakers[provider["name"]]
            if breaker.is_open:
                continue  # 熔断器打开，跳过

            try:
                response = await self.call_provider(provider["name"], request)
                breaker.record_success()
                return response

            except RateLimitError:
                breaker.record_failure()
                continue  # 限流，尝试下一个

            except ServiceUnavailableError:
                breaker.record_failure()
                provider["healthy"] = False
                self.schedule_health_check(provider, delay=60)
                continue

        # 所有提供商都不可用 → 降级
        return await self.degrade(request)


class CircuitBreaker:
    """熔断器——防止故障级联"""

    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.threshold = failure_threshold
        self.timeout = recovery_timeout
        self.state = "CLOSED"  # CLOSED → OPEN → HALF_OPEN

    @property
    def is_open(self):
        if self.state == "OPEN":
            if time.time() - self.opened_at > self.timeout:
                self.state = "HALF_OPEN"
                return False
            return True
        return False

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
            self.opened_at = time.time()

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
```

### 状态持久化与检查点

```python
class AgentCheckpointing:
    """Agent 执行的检查点与恢复"""

    async def execute_with_checkpoints(self, task_id, task):
        """带检查点的 Agent 执行"""

        # 检查是否有未完成的执行（恢复场景）
        checkpoint = await self.load_checkpoint(task_id)
        if checkpoint:
            # 从上次中断的步骤恢复
            step_index = checkpoint["completed_steps"]
            plan = checkpoint["plan"]
            results = checkpoint["results"]
        else:
            # 全新执行
            plan = await self.agent.plan(task)
            step_index = 0
            results = []

        # 逐步执行，每步保存检查点
        for i in range(step_index, len(plan.steps)):
            step = plan.steps[i]

            result = await self.agent.execute_step(step)
            results.append(result)

            # 保存检查点
            await self.save_checkpoint(task_id, {
                "plan": plan,
                "completed_steps": i + 1,
                "results": results,
                "timestamp": datetime.utcnow(),
            })

        return self.agent.synthesize(results)

    # 检查点使能的能力：
    # 1. 服务重启后从断点恢复（不丢失已完成的步骤）
    # 2. 跨越小时/天的长任务
    # 3. 人工审核暂停后恢复
    # 4. Worker 迁移（从一个实例转到另一个）
```

### 优雅降级策略

```python
class GracefulDegradation:
    """Agent 系统的优雅降级"""

    degradation_levels = {
        "Level 0: 完全正常": {
            "条件": "所有服务正常",
            "行为": "完整 Agent 功能",
        },
        "Level 1: LLM 部分降级": {
            "条件": "主 LLM 提供商限流或慢",
            "行为": [
                "切换到备用提供商",
                "降级到更小的模型",
                "增加缓存使用",
            ],
        },
        "Level 2: LLM 完全不可用": {
            "条件": "所有 LLM 提供商不可用",
            "行为": [
                "使用缓存回答高频问题",
                "切换到规则引擎处理简单请求",
                "复杂请求排队等待恢复",
                "显示'AI 助手暂时不可用，已转接人工'",
            ],
        },
        "Level 3: 数据库不可用": {
            "条件": "状态存储不可用",
            "行为": [
                "只处理无状态的单轮请求",
                "禁用多步 Agent 和工具调用",
                "内存中临时存储（不持久）",
            ],
        },
        "Level 4: 完全降级": {
            "条件": "核心服务全部不可用",
            "行为": [
                "静态降级页面",
                "联系方式和人工支持入口",
                "自动通知运维团队",
            ],
        },
    }
```

### RTO/RPO 设计

```
Agent 系统的 RTO/RPO 目标：

┌──────────────────┬──────────┬──────────┬──────────────┐
│ 组件             │ RTO      │ RPO      │ 策略         │
│                  │(恢复时间)│(数据丢失)│              │
├──────────────────┼──────────┼──────────┼──────────────┤
│ LLM API 访问     │ < 1s     │ 0        │ 多提供商自动 │
│                  │          │          │ 故障转移     │
├──────────────────┼──────────┼──────────┼──────────────┤
│ Agent 服务       │ < 30s    │ 0        │ 多实例 + 健康│
│                  │          │          │ 检查 + 自愈  │
├──────────────────┼──────────┼──────────┼──────────────┤
│ 会话状态(Redis)  │ < 5s     │ < 1s     │ Redis Cluster│
│                  │          │          │ + 异步复制   │
├──────────────────┼──────────┼──────────┼──────────────┤
│ 持久存储(PG)     │ < 5min   │ 0        │ 同步复制 +   │
│                  │          │          │ 自动故障转移 │
├──────────────────┼──────────┼──────────┼──────────────┤
│ 向量数据库(RAG)  │ < 10min  │ 可重建   │ 定期快照 +   │
│                  │          │          │ 从源数据重建 │
└──────────────────┴──────────┴──────────┴──────────────┘
```

## 常见误区 / 面试追问

1. **误区："LLM API 是云服务，不需要考虑可用性"** — LLM API 提供商也会宕机或限流。2024-2025 年 OpenAI 和 Anthropic 都有过多次服务中断。必须有多提供商故障转移策略，不能把鸡蛋放在一个篮子里。

2. **误区："Agent 没有状态，不需要灾难恢复"** — 多步 Agent 有大量状态：执行计划、已完成步骤的结果、对话历史、工具调用结果。丢失这些状态意味着任务从头开始，用户体验极差。检查点机制是必需的。

3. **追问："如何测试灾难恢复方案？"** — (1) **Chaos Engineering**——随机中断 LLM API 连接，验证故障转移；(2) **故障注入**——模拟数据库宕机、网络分区；(3) **DR 演练**——每季度进行一次完整的区域切换演练；(4) **恢复时间测量**——记录实际 RTO 并与目标对比。

4. **追问："优雅降级和完全不可用之间如何选择？"** — 原则：部分功能可用 > 完全不可用。用户宁可看到"AI 助手暂时简化模式运行"也不愿看到"服务不可用"。降级策略应该预先设计和测试，而不是在事故中临时决定。

## 参考资料

- [Foundry Agent Service Platform Outage Recovery (Microsoft)](https://learn.microsoft.com/en-us/azure/foundry/how-to/agent-service-platform-disaster-recovery)
- [Agentic AI in Production: 10 Patterns That Ship in 2025 (Medium)](https://medium.com/@ThinkingLoop/d3-1-agentic-ai-in-production-10-patterns-that-ship-in-2025-d9c367827e58)
- [5 Most Popular Agentic AI Design Patterns in 2025 (Azilen)](https://www.azilen.com/blog/agentic-ai-design-patterns/)
- [20 Agentic AI Workflow Patterns That Actually Work in 2025 (Skywork AI)](https://skywork.ai/blog/agentic-ai-examples-workflow-patterns-2025/)
- [How AI is Transforming IT Disaster Recovery (Cutover)](https://www.cutover.com/blog/how-ai-transforming-it-disaster-recovery)
