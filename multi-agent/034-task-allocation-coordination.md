# 如何设计 Agent 间的任务分配与协调？

> 难度：中级
> 分类：Multi-Agent

## 简短回答

多 Agent 任务分配与协调的核心挑战是：如何将复杂任务拆分为子任务、分配给合适的 Agent、并确保它们协同工作产出正确结果。主要策略包括：**集中式规划-分散执行**（中央规划器分解任务，Agent 独立执行）、**LLM 作为协调器**（利用 LLM 推理能力做动态任务分配）、**角色定义 + 能力匹配**（根据 Agent 能力描述自动路由）、**动态分配**（基于运行时状态自适应调整）。关键设计原则：明确定义每个 Agent 的角色和能力边界、选择匹配任务结构的通信模式、以及实现健壮的错误处理和降级机制。

## 详细解析

### 任务分配的基本流程

```
复杂任务
    ↓
┌──────────────┐
│  任务分解     │ → 将大任务拆成子任务
│ (Decompose)  │
└──────┬───────┘
       ↓
┌──────────────┐
│  任务分配     │ → 将子任务匹配给合适的 Agent
│ (Allocate)   │
└──────┬───────┘
       ↓
┌──────────────┐
│  协调执行     │ → 管理依赖、同步、冲突
│ (Coordinate) │
└──────┬───────┘
       ↓
┌──────────────┐
│  结果聚合     │ → 合并各 Agent 的输出
│ (Aggregate)  │
└──────────────┘
```

### 策略 1：集中式规划，分散执行

```python
class CentralPlanner:
    """中央规划器负责全局任务分解和分配"""

    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents

    async def plan_and_execute(self, task: str):
        # Step 1: LLM 分解任务为子任务
        subtasks = await self.decompose(task)

        # Step 2: 根据 Agent 能力分配子任务
        assignments = self.assign(subtasks)

        # Step 3: 按依赖关系编排执行
        results = {}
        for batch in self.topological_sort(assignments):
            # 同一批次的任务无依赖，可并行
            batch_results = await asyncio.gather(
                *[self.agents[a.agent_id].execute(a.subtask, results)
                  for a in batch]
            )
            results.update(zip([a.subtask.id for a in batch], batch_results))

        return results

    def assign(self, subtasks):
        """基于 Agent 能力描述的自动匹配"""
        assignments = []
        for subtask in subtasks:
            best_agent = max(
                self.agents.values(),
                key=lambda a: self.capability_match(a, subtask)
            )
            assignments.append(Assignment(subtask, best_agent.id))
        return assignments
```

### 策略 2：LLM 作为协调器

利用 LLM 的推理能力做动态决策：

```python
class LLMCoordinator:
    async def coordinate(self, task, agents):
        # 给 LLM 提供所有 Agent 的能力描述
        agent_descriptions = "\n".join([
            f"- {a.name}: {a.description}, 擅长: {a.capabilities}"
            for a in agents
        ])

        plan = await self.llm.generate(f"""
        任务: {task}

        可用 Agent:
        {agent_descriptions}

        请分解任务并分配给合适的 Agent。
        输出格式:
        1. [Agent名] → 子任务描述 (依赖: 无/步骤N)
        2. ...
        """)

        return self.parse_and_execute(plan)
```

**Planner vs Orchestrator：** 研究发现 Planner 方法（先生成完整计划再执行）在处理并发操作时优于 Orchestrator 方法（逐步决策），因为 Planner 能更好地识别可并行的子任务。

### 策略 3：角色定义与能力匹配

```python
# CAMEL 框架的角色扮演方法
class RoleBasedAllocation:
    def __init__(self):
        self.agents = {
            "researcher": Agent(
                role="研究分析师",
                capabilities=["web_search", "paper_analysis", "data_collection"],
                constraints=["只能读取数据，不能修改"],
            ),
            "developer": Agent(
                role="软件工程师",
                capabilities=["code_generation", "debugging", "testing"],
                constraints=["只能修改 src/ 目录下的文件"],
            ),
            "reviewer": Agent(
                role="质量审核员",
                capabilities=["code_review", "fact_checking"],
                constraints=["不能直接修改代码，只能提出建议"],
            ),
        }

    def match(self, subtask: str) -> str:
        """基于语义匹配找到最合适的 Agent"""
        subtask_embedding = embed(subtask)
        scores = {
            name: cosine_similarity(subtask_embedding, embed(a.capabilities))
            for name, a in self.agents.items()
        }
        return max(scores, key=scores.get)
```

### 策略 4：动态分配（DRAMA 方法）

静态分配的局限：Agent 能力固定、任务分配策略不适应环境变化。

```python
class DynamicAllocator:
    """基于亲和度的事件驱动动态分配"""

    def __init__(self, agents):
        self.agents = agents
        self.affinity_scores = {}  # (agent, task_type) → 历史成功率

    async def allocate(self, subtask):
        # 考虑多个因素动态选择
        candidates = []
        for agent in self.agents:
            score = self.compute_score(agent, subtask)
            candidates.append((agent, score))

        # 选择得分最高的 Agent
        best = max(candidates, key=lambda x: x[1])
        return best[0]

    def compute_score(self, agent, subtask):
        return (
            0.4 * self.capability_match(agent, subtask) +   # 能力匹配
            0.3 * self.affinity_scores.get((agent.id, subtask.type), 0.5) +  # 历史表现
            0.2 * (1 - agent.current_load / agent.max_load) +  # 当前负载
            0.1 * self.recency_bonus(agent)  # 最近是否空闲
        )

    def update_affinity(self, agent_id, task_type, success: bool):
        """根据执行结果更新亲和度"""
        key = (agent_id, task_type)
        old = self.affinity_scores.get(key, 0.5)
        self.affinity_scores[key] = old * 0.8 + (1.0 if success else 0.0) * 0.2
```

### 协调的关键挑战

#### 1. 依赖管理

```python
# 用 DAG（有向无环图）表示任务依赖
task_graph = {
    "search":    {"depends_on": []},
    "analyze":   {"depends_on": ["search"]},
    "visualize": {"depends_on": ["search"]},      # 和 analyze 并行
    "report":    {"depends_on": ["analyze", "visualize"]},  # 等两者完成
}

# 拓扑排序确定执行顺序
# Batch 1: search（无依赖）
# Batch 2: analyze, visualize（并行）
# Batch 3: report（等待前两个）
```

#### 2. Token 冗余控制

研究显示多 Agent 框架中 token 重复率达 53-86%。解决方案：

```python
# 传递摘要而非完整输出
def compress_for_handoff(agent_output: str, max_tokens: int = 500):
    if count_tokens(agent_output) > max_tokens:
        return summarize(agent_output, max_tokens=max_tokens)
    return agent_output
```

#### 3. 失败处理

```python
class ResilientCoordinator:
    async def execute_with_fallback(self, subtask, primary_agent, backup_agent):
        try:
            return await asyncio.wait_for(
                primary_agent.execute(subtask),
                timeout=30
            )
        except (TimeoutError, AgentError):
            # 降级到备选 Agent
            return await backup_agent.execute(subtask)
```

### 设计原则总结

| 原则 | 说明 |
|------|------|
| 明确角色边界 | 每个 Agent 有清晰的能力和约束定义 |
| 从小开始 | 先 3-5 个 Agent，验证后再扩展 |
| 匹配通信模式 | 简单链用 Pipeline，复杂协调用 Hierarchical |
| 健壮的错误处理 | 断路器、重试、降级回退 |
| 可观测性 | 追踪每个 Agent 的输入/输出/耗时 |

## 常见误区 / 面试追问

1. **误区："Agent 越多越好"** — 每增加一个 Agent 都增加通信开销和协调复杂度。最佳实践是从一个 Agent 开始，当遇到明确瓶颈时才拆分。研究建议初始阶段限制在 3-5 个 Agent。

2. **误区："用 LLM 做所有协调决策"** — LLM 协调灵活但慢且贵。对于确定性的任务路由，用代码规则更可靠。混合方式最佳：简单路由用代码，复杂判断用 LLM。

3. **追问："如何处理 Agent 间的目标冲突？"** — 明确优先级规则（如安全 > 效率 > 成本）；引入仲裁 Agent 或投票机制；将冲突升级给人类决策者。

4. **追问："分布式 Agent 系统如何保证一致性？"** — 对于非关键系统，接受最终一致性即可。对于关键操作（如金融交易），使用事务性协调：先锁定资源 → 执行操作 → 确认或回滚。

## 参考资料

- [Coordination Mechanisms in Multi-Agent Systems (APXML)](https://apxml.com/courses/agentic-llm-memory-architectures/chapter-5-multi-agent-systems/coordination-mechanisms-mas)
- [Multi-Agent Coordination: Fix With 10 Strategies (Galileo)](https://galileo.ai/blog/multi-agent-coordination-strategies)
- [DRAMA: Dynamic and Robust Allocation-based Multi-Agent System (arXiv)](https://arxiv.org/html/2508.04332v1)
- [Self-Resource Allocation in Multi-Agent LLM Systems (arXiv)](https://arxiv.org/html/2504.02051v1)
- [Multi-Agent Collaboration Mechanisms: A Survey (arXiv)](https://arxiv.org/html/2501.06322v1)
