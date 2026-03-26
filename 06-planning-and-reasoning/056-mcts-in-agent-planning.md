# Monte Carlo Tree Search 在 Agent 规划中的应用

> 难度：高级
> 分类：Planning & Reasoning

## 简短回答

Monte Carlo Tree Search (MCTS) 是一种结合树搜索与随机模拟的决策算法，通过四步循环——**选择（Selection）、扩展（Expansion）、模拟（Simulation）、反向传播（Backpropagation）**——在庞大的决策空间中找到近似最优解。在 Agent 规划中，MCTS 将 LLM 的推理能力与系统化搜索结合，克服了 LLM 单次生成无法回溯的根本局限。代表性框架 **LATS（Language Agent Tree Search, ICML 2024）** 将 MCTS 应用于 LLM Agent，用 LLM 同时作为推理引擎、行动策略和价值函数，在 HumanEval 编程任务上达到 94.4%（GPT-4），超越了所有已知的 prompting 方法。MCTS 的核心优势是**探索-利用平衡**——既深入挖掘有前景的方案，又不忽略潜在的好路径。

## 详细解析

### MCTS 的四步循环

```
        根节点（初始状态）
           │
    ┌──────┼──────┐
    │      │      │
   A(3/5) B(1/4) C(2/3)    ← 选择：UCB1 选 C（胜率高+访问少）
                  │
              ┌───┼───┐
              │       │
            C1(new) C2(new)  ← 扩展：生成新子节点
              │
          [模拟到终局]        ← 模拟：LLM 评估结果
              │
          反向传播 ↑↑↑        ← 更新路径上所有节点的统计

四步循环：
1. Selection（选择）：从根节点用 UCB1 策略向下选择
2. Expansion（扩展）：在叶节点生成新的子节点
3. Simulation（模拟）：评估新节点的价值
4. Backpropagation（反向传播）：更新路径上所有节点的统计值
```

### MCTS 在 LLM Agent 中的实现

```python
import math

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state          # 当前推理状态
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def ucb1(self, exploration_weight=1.414):
        """UCB1：平衡探索与利用"""
        if self.visits == 0:
            return float('inf')  # 未访问的节点优先
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

class LLM_MCTS:
    """将 MCTS 与 LLM 结合的规划器"""

    def __init__(self, llm, num_iterations=50):
        self.llm = llm
        self.num_iterations = num_iterations

    async def search(self, problem):
        root = MCTSNode(state=problem)

        for _ in range(self.num_iterations):
            # 1. 选择：沿 UCB1 最高的路径向下
            node = self.select(root)

            # 2. 扩展：用 LLM 生成可能的下一步
            child = await self.expand(node)

            # 3. 模拟：用 LLM 评估这条路径的价值
            value = await self.simulate(child)

            # 4. 反向传播：更新路径上的统计值
            self.backpropagate(child, value)

        # 返回访问次数最多的子节点（最稳健的选择）
        return max(root.children, key=lambda c: c.visits)

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda c: c.ucb1())
        return node

    async def expand(self, node):
        """用 LLM 生成多个候选动作"""
        actions = await self.llm.invoke(f"""
        当前状态：{node.state}
        请生成 3 个可能的下一步行动。
        """)
        for action in actions:
            child = MCTSNode(
                state=self.apply_action(node.state, action),
                parent=node
            )
            node.children.append(child)
        return node.children[0]  # 返回第一个新节点用于模拟

    async def simulate(self, node):
        """用 LLM 评估当前路径的价值（0-1）"""
        evaluation = await self.llm.invoke(f"""
        评估以下推理路径达到最终目标的可能性（0-1分）：
        路径：{self.get_path(node)}
        目标：{self.goal}
        """)
        return float(evaluation)

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent
```

### LATS 框架：MCTS + LLM Agent

```python
class LATS:
    """Language Agent Tree Search (ICML 2024)"""

    def __init__(self, llm, environment, n_samples=5, depth=7):
        self.llm = llm
        self.env = environment
        self.n_samples = n_samples
        self.depth = depth

    async def solve(self, task):
        root = LATSNode(observation=task)

        for iteration in range(self.n_samples):
            node = self.select(root)

            # LATS 的关键创新：LLM 同时充当三个角色
            # 角色 1：Action Generator（生成候选动作）
            actions = await self.generate_actions(node)

            # 角色 2：Environment Simulator（模拟执行结果）
            for action in actions:
                observation = await self.env.step(action)
                child = LATSNode(observation=observation, action=action)
                node.add_child(child)

            # 角色 3：Value Function（评估状态价值）
            for child in node.children:
                value = await self.evaluate_state(child)
                child.value = value

            # 角色 4（额外）：Self-Reflection（从失败中学习）
            if self.is_terminal_failure(node):
                reflection = await self.reflect_on_failure(node)
                # 反思被存储，影响后续搜索
                self.reflections.append(reflection)

            self.backpropagate(node)

        return self.get_best_trajectory(root)
```

### LATS 的性能对比

```
HumanEval 编程任务（GPT-4）：
┌──────────────────────┬──────────┐
│ 方法                 │ Pass@1   │
├──────────────────────┼──────────┤
│ 直接生成             │ 82.0%    │
│ CoT                  │ 83.5%    │
│ Reflexion            │ 91.0%    │
│ ToT (DFS)            │ 89.0%    │
│ LATS (MCTS)          │ 94.4%    │
└──────────────────────┴──────────┘

WebShop（网页导航任务）：
  ReAct: 40%  →  LATS: 75% (提升 87.5%)
```

### MCTS vs 其他搜索策略

```python
comparison = {
    "Greedy (CoT)": {
        "搜索方式": "单路径，无回溯",
        "优势": "最快、最便宜",
        "劣势": "容易卡在局部最优",
        "LLM 调用": "1 次",
    },
    "BFS (ToT)": {
        "搜索方式": "逐层扩展所有候选",
        "优势": "保证找到最浅的解",
        "劣势": "内存消耗大，不适合深搜索",
        "LLM 调用": "O(b^d)",
    },
    "DFS (ToT)": {
        "搜索方式": "深入探索一条路，失败回溯",
        "优势": "内存高效",
        "劣势": "可能陷入无解的深分支",
        "LLM 调用": "O(b*d)",
    },
    "MCTS (LATS)": {
        "搜索方式": "UCB1 引导的自适应搜索",
        "优势": "探索-利用平衡，渐进最优",
        "劣势": "需要大量迭代，成本最高",
        "LLM 调用": "O(N * b)，N=迭代次数",
    },
}
```

### 实际应用中的成本-效果权衡

```python
# MCTS 的成本很高——每次迭代都需要多次 LLM 调用
# 实际使用时需要策略性地控制搜索空间

cost_optimization = {
    "减少迭代次数": "从 50 降到 10-20，牺牲少量质量换取大幅降低成本",
    "缩小分支因子": "每步生成 2-3 个候选而非 5-10 个",
    "用小模型做模拟": "扩展用大模型，模拟评估用小模型",
    "早期终止": "找到足够好的方案就停止，不追求最优",
    "缓存重复状态": "相同状态不重复评估",
}
```

## 常见误区 / 面试追问

1. **误区："MCTS 只适合棋类游戏"** — MCTS 是通用的搜索框架。LATS 证明它在编程、网页导航、推理等 Agent 任务上都有效。关键是需要定义好"状态"、"动作"和"价值函数"——LLM 可以同时充当这三者。

2. **误区："MCTS 保证找到最优解"** — MCTS 是近似算法，理论上无限迭代才收敛到最优。实际中迭代次数有限，只能找到"足够好"的方案。但相比贪心搜索，MCTS 通过探索-利用平衡显著降低了陷入局部最优的风险。

3. **追问："LATS 和 Reflexion 有什么区别？"** — Reflexion 是从失败中学习并重试（线性），LATS 是在搜索树中探索多条路径并利用反思信息引导搜索（树形）。LATS 包含了 Reflexion 的自我反思能力，但搜索策略更系统化。

4. **追问："MCTS 在生产 Agent 系统中实用吗？"** — 对于高价值、允许高延迟的任务（如代码生成、复杂分析）非常实用。对于实时对话等低延迟场景，MCTS 的成本太高。生产中常用简化版——减少迭代次数、缩小搜索空间。

## 参考资料

- [Language Agent Tree Search (LATS) - ICML 2024 (arXiv)](https://arxiv.org/abs/2310.04406)
- [LATS Official Implementation (GitHub)](https://github.com/lapisrocks/LanguageAgentTreeSearch)
- [ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/76ec4dc30e9faaf0e4b6093eaa377218-Paper-Conference.pdf)
- [DSG-MCTS: Dynamic Strategy-Guided MCTS for LLM Reasoning (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.532.pdf)
- [ThoughtSculpt: Reasoning with MCTS (Prompt Engineering Guide)](https://www.promptingguide.ai/research/thoughtsculpt)
