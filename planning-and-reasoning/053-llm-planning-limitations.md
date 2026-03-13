# LLM 作为规划器的局限性与缓解方案

> 难度：中级
> 分类：Planning & Reasoning

## 简短回答

LLM 作为规划器面临根本性挑战：Kambhampati (ASU, 2024) 的核心论点是 **"LLM 不能真正规划，但可以在规划中发挥建设性作用"**。主要局限包括：(1) **约束满足能力弱**——LLM 难以同时满足多个相互制约的条件；(2) **长程规划退化**——随着步骤增多，累积错误导致计划质量急剧下降；(3) **环境模型缺失**——LLM 没有真正的世界模型，无法准确预测动作的后果；(4) **幻觉导致无效计划**——生成看似合理但实际不可执行的步骤。缓解方案的核心思路是 **LLM-Modulo 框架**：LLM 生成候选计划，外部验证器检查正确性，不合格则反馈修正。在 Blocksworld 基准上，纯 LLM 规划仅 30-40% 可行率，而 LLM-Modulo 框架可将可行率提升到 80%+。

## 详细解析

### LLM 规划的核心局限

```python
planning_limitations = {
    "约束满足": {
        "问题": "LLM 难以同时满足多个约束条件",
        "示例": "安排5人会议，每人有不同的空闲时间段 → LLM 经常忽略部分约束",
        "原因": "自回归生成逐 token 产生，无法全局优化",
    },
    "长程规划退化": {
        "问题": "步骤越多，计划质量越差",
        "示例": "10 步计划的可行率远低于 3 步计划",
        "原因": "每步的小错误会累积放大",
    },
    "世界模型缺失": {
        "问题": "LLM 无法准确模拟动作的后果",
        "示例": "移动积木 A 到 B 上 → LLM 可能忘记 A 原来下面的积木会暴露出来",
        "原因": "LLM 是语言模型，不是物理/逻辑模拟器",
    },
    "幻觉与虚构": {
        "问题": "生成看似合理但不可执行的步骤",
        "示例": "计划中引用不存在的 API、使用不可用的工具",
        "原因": "LLM 优化的是语言流畅度，不是计划可行性",
    },
    "回溯能力缺失": {
        "问题": "自回归模型无法回头修改已生成的步骤",
        "示例": "发现第 3 步需要第 1 步的不同输出时，无法修改第 1 步",
        "原因": "单向生成架构的根本限制",
    },
}
```

### 实证数据：LLM 规划的真实表现

```
Blocksworld 规划任务（2024 基准测试）：
┌──────────────────────┬────────────┬─────────────┐
│ 方法                 │ 可行率     │ 最优率       │
├──────────────────────┼────────────┼─────────────┤
│ GPT-4 直接规划       │ ~35%       │ ~15%        │
│ GPT-4 + CoT          │ ~42%       │ ~20%        │
│ GPT-4 + Self-Verify  │ ~55%       │ ~30%        │
│ LLM-Modulo (外部验证)│ ~82%       │ ~65%        │
│ 传统规划器 (PDDL)    │ 100%       │ 100%        │
└──────────────────────┴────────────┴─────────────┘

结论：LLM 单独做规划远不如传统规划器可靠
     但 LLM + 外部验证可以显著提升
```

### 缓解方案 1：LLM-Modulo 框架

```python
class LLMModuloPlanner:
    """LLM 生成 + 外部验证器检查"""

    async def plan(self, task, max_attempts=5):
        for attempt in range(max_attempts):
            # 1. LLM 生成候选计划
            candidate = await self.llm.generate_plan(
                task=task,
                feedback=self.previous_feedback if attempt > 0 else None
            )

            # 2. 外部验证器检查
            validation = await self.verify(candidate)

            if validation.is_valid:
                return candidate
            else:
                # 3. 将验证反馈回传给 LLM
                self.previous_feedback = validation.errors
                # "步骤 3 违反了约束 X：..."

        return None  # 所有尝试都失败

    async def verify(self, plan):
        """多层验证"""
        errors = []

        # 语法验证：步骤格式是否正确
        errors += self.syntax_check(plan)

        # 约束验证：是否满足所有约束
        errors += self.constraint_check(plan)

        # 可执行性验证：每个动作是否可执行
        errors += await self.executability_check(plan)

        # 模拟验证：在模拟环境中执行
        errors += await self.simulation_check(plan)

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

### 缓解方案 2：混合规划架构

```python
class HybridPlanner:
    """LLM 处理灵活部分 + 传统算法处理确定性部分"""

    async def plan(self, task):
        # 1. LLM 负责高层目标分解（擅长理解自然语言需求）
        high_level_goals = await self.llm.decompose(task)

        # 2. 传统规划器负责具体步骤排列（擅长约束满足）
        for goal in high_level_goals:
            if goal.is_structured:
                # 调度、排列组合等 → 用确定性算法
                goal.plan = self.classical_planner.solve(goal)
            else:
                # 创意、判断等 → 用 LLM
                goal.plan = await self.llm.plan_steps(goal)

        return self.merge_plans(high_level_goals)
```

### 缓解方案 3：Plan-Verify-Correct 循环

```python
class PlanVerifyCorrect:
    """生成 → 验证 → 纠正 的迭代循环"""

    async def solve(self, task):
        plan = await self.generate_initial_plan(task)

        for iteration in range(self.max_iterations):
            # 验证
            issues = await self.llm.verify_plan(
                plan=plan,
                prompt="""
                审查以下计划，检查：
                1. 是否有遗漏的步骤？
                2. 步骤顺序是否正确？
                3. 是否有不可执行的步骤？
                4. 是否满足所有约束？
                列出发现的问题。
                """
            )

            if not issues:
                break  # 验证通过

            # 纠正
            plan = await self.llm.correct_plan(
                plan=plan, issues=issues,
                prompt="根据以下问题修正计划：..."
            )

        return plan
```

### 缓解方案 4：分层规划降低复杂度

```
完整任务（LLM 难以一次性规划）
│
├── 抽象层（LLM 擅长）：高层目标和策略
│   "先收集数据，再分析，最后生成报告"
│
├── 中间层（LLM + 约束）：具体子任务
│   "搜索竞品A → 提取定价 → 搜索竞品B → 提取定价"
│
└── 执行层（工具/代码）：原子操作
    "调用 search_api('竞品A 定价')"

每一层只处理 3-5 个步骤 → 在 LLM 的能力范围内
```

### 关键研究发现

```python
key_findings = {
    "Kambhampati 2024": (
        "LLMs cannot plan but can help planning. "
        "LLM-Modulo 框架让 LLM 作为候选生成器，"
        "外部验证器保证正确性"
    ),
    "Valmeekam et al. 2023": (
        "在 Blocksworld 上，GPT-4 直接生成的计划 "
        "仅有约 35% 可行，远低于人们的预期"
    ),
    "DeepPlanning 2026": (
        "隐含约束比显式约束更难被 LLM 检测到。"
        "环境中未明确说明的限制是 LLM 规划的最大盲区"
    ),
    "PlanGenLLMs Survey": (
        "LLM 在需要精确状态追踪的规划任务上表现最差，"
        "但在需要常识推理的规划任务上有独特优势"
    ),
}
```

## 常见误区 / 面试追问

1. **误区："用更大的模型就能解决规划问题"** — 规划能力的局限是自回归架构的根本问题，不是模型大小的问题。更大的模型可以改善但无法根本解决约束满足和状态追踪的弱点。正确做法是用外部工具补偿 LLM 的结构性弱点。

2. **误区："LLM 不能规划 = LLM 在规划中没用"** — 恰恰相反。LLM 在规划中的角色是：理解自然语言需求、生成候选方案、提供常识知识。它不擅长的是精确约束满足和状态追踪——这些交给传统算法或验证器。

3. **追问："如何判断一个规划任务是否适合 LLM？"** — 两个维度：(1) 约束的数量和复杂度——少约束适合 LLM，多约束需要验证器；(2) 步骤数——3-5 步 LLM 可以直接处理，10+ 步需要分层规划或外部辅助。

4. **追问："LLM-Modulo 框架的验证器从哪来？"** — 取决于领域：代码规划用单元测试和类型检查；数学规划用符号计算验证；现实世界规划用模拟器；通用场景用另一个 LLM 做交叉验证（效果有限但成本低）。

## 参考资料

- [LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks (arXiv, Kambhampati)](https://arxiv.org/html/2402.01817v2)
- [PlanGenLLMs: A Survey of LLM Planning Capabilities (ACL 2025)](https://aclanthology.org/2025.acl-long.958.pdf)
- [DeepPlanning: Benchmark Exposing Limits of LLM Planning](https://co-r-e.com/method/deepplanning-benchmark-llm-20260128)
- [Planning for Agents (LangChain Blog)](https://blog.langchain.com/planning-for-agents/)
- [LLM Planner Agent: Adaptive Modular Planning (Emergent Mind)](https://www.emergentmind.com/topics/llm-planner-agent)
