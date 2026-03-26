# Tree-of-Thought (ToT) vs Chain-of-Thought：何时使用哪种？

> 难度：中级
> 分类：Planning & Reasoning

## 简短回答

Chain-of-Thought (CoT) 是**线性推理**——模型沿单一路径逐步思考，适合有明确解题步骤的任务。Tree-of-Thought (ToT) 是**探索式推理**——模型在每一步生成多个候选思路，用搜索算法（BFS/DFS）探索不同分支，并通过评估函数剪枝和回溯，适合需要全局搜索和创造性思考的任务。Yao et al. (Princeton, NeurIPS 2023) 的论文表明 ToT 在 Game of 24 上将成功率从 CoT 的 4% 提升到 74%。**选择原则**：任务有唯一解题路径 → CoT；任务需要探索、回溯或创造性方案 → ToT；需要兼顾成本和准确率 → CoT + Self-Consistency。ToT 的代价是 LLM 调用次数显著增加（通常 10-100 倍），因此只在高价值复杂任务上使用。

## 详细解析

### CoT 与 ToT 的核心区别

```
Chain-of-Thought (线性)：
  思路 A → 步骤 1 → 步骤 2 → 步骤 3 → 答案
  （一条路走到底，不回头）

Tree-of-Thought (树形)：
  问题
  ├── 思路 A → 评估: 0.8 → 继续探索
  │   ├── A1 → 评估: 0.9 → ★ 最优
  │   └── A2 → 评估: 0.3 → 剪枝 ✂
  ├── 思路 B → 评估: 0.5 → 继续探索
  │   └── B1 → 评估: 0.2 → 剪枝 ✂
  └── 思路 C → 评估: 0.1 → 剪枝 ✂
```

### ToT 的工作原理

```python
class TreeOfThoughts:
    """ToT 的核心实现：生成、评估、搜索"""

    async def solve(self, problem, max_depth=3, breadth=3):
        # 初始化根节点
        root = ThoughtNode(state=problem, depth=0)

        if self.search_strategy == "BFS":
            return await self.bfs(root, max_depth, breadth)
        else:
            return await self.dfs(root, max_depth, breadth)

    async def bfs(self, root, max_depth, breadth):
        """广度优先搜索：每层保留最优的 k 个节点"""
        current_level = [root]

        for depth in range(max_depth):
            candidates = []
            for node in current_level:
                # 1. 生成：每个节点生成多个候选思路
                thoughts = await self.generate_thoughts(node, n=breadth)
                # 2. 评估：对每个思路打分
                for thought in thoughts:
                    score = await self.evaluate(thought)
                    thought.score = score
                    candidates.append(thought)

            # 3. 选择：保留得分最高的 k 个
            current_level = sorted(candidates, key=lambda x: x.score, reverse=True)[:breadth]

        return current_level[0]  # 返回最优方案

    async def evaluate(self, thought):
        """用 LLM 评估当前思路的可行性"""
        response = await self.llm.invoke(f"""
        评估以下问题解决思路的可行性（1-10分）：
        问题：{thought.root_problem}
        当前思路：{thought.reasoning_path}

        评分标准：
        - 逻辑是否正确？
        - 是否有可能达到最终答案？
        - 是否存在明显矛盾？
        """)
        return float(response)
```

### 两种搜索策略对比

```python
# BFS（广度优先）：适合解空间较浅但较宽的问题
# - Game of 24：每步可选的运算组合多
# - 创意写作：需要比较多种风格方向

# DFS（深度优先）：适合解空间较深的问题
# - 数独求解：需要深入推导
# - 代码调试：需要沿一条思路深入追踪
# - 支持回溯：发现死路可以退回上一步

bfs_config = {"breadth": 5, "depth": 2}   # 宽搜索，浅深度
dfs_config = {"breadth": 2, "depth": 5}   # 窄搜索，深探索
```

### 关键性能对比

```
任务：Game of 24（用四个数字通过加减乘除得到 24）
┌────────────────────┬───────────┬──────────┐
│ 方法               │ 成功率    │ LLM 调用  │
├────────────────────┼───────────┼──────────┤
│ Standard (IO)      │ 7.3%      │ 1        │
│ CoT                │ 4.0%      │ 1        │  ← CoT 的线性推理在需要回溯搜索的问题上反而成为限制，因为它强制模型沿单一路径推理而无法探索多个分支
│ CoT + SC (k=100)   │ 9.0%      │ 100      │
│ ToT (BFS, b=5)     │ 74.0%     │ ~O(b^d)  │
└────────────────────┴───────────┴──────────┘

任务：创意写作（Coherent Passage）
┌────────────────────┬───────────┐
│ 方法               │ 一致性分   │
├────────────────────┼───────────┤
│ Standard (IO)      │ 6.19      │
│ CoT                │ 6.93      │
│ ToT                │ 7.56      │
└────────────────────┴───────────┘
```

### 何时选择哪种？

```python
decision_guide = {
    "使用 CoT": [
        "数学计算：有明确的解题步骤（算术、代数）",
        "逻辑推理：前提 → 结论的线性推导",
        "信息提取：从文本中逐步提取关键信息",
        "成本敏感：每次只需 1 次 LLM 调用",
        "实时应用：需要低延迟响应",
    ],
    "使用 ToT": [
        "组合优化：如 Game of 24、数独",
        "创意任务：需要探索多种方案的写作、设计",
        "规划问题：需要比较不同路径的决策",
        "约束满足：多个约束需要同时满足",
        "准确率优先：愿意用更多计算换取更好结果",
    ],
    "使用 CoT + Self-Consistency": [
        "需要比 CoT 更好的准确率",
        "但 ToT 的成本太高",
        "问题有明确的最终答案（可以投票）",
    ],
}
```

### 实际应用中的 ToT 简化版

```python
# 生产环境中的 ToT 通常不需要完整实现
# 用 Prompt 模拟即可

tot_prompt = """
问题：{problem}

请用以下方式推理：
1. 生成 3 种不同的解题思路
2. 对每种思路评估可行性（1-10分）
3. 选择最佳思路，展开详细推理
4. 如果遇到矛盾，回到步骤1尝试其他方向

思路 1：
"""

# 这种 "Prompt-based ToT" 比完整 ToT 便宜很多
# 虽然效果不如算法级 ToT，但对大多数场景够用
```

### 方法谱系总结

```
简单 ←─────────────────────────────────→ 复杂
成本低                                    成本高

IO → Zero-shot CoT → Few-shot CoT → Self-Consistency → ToT → GoT
 │         │               │              │              │     │
 │    "逐步思考"      提供示例       多次采样+投票    树搜索  图搜索
 │                                                     │
 │                                              包含评估+回溯
 1次调用    1次             1次           k次        O(b^d)次
```

## 常见误区 / 面试追问

1. **误区："ToT 总是比 CoT 好"** — 在简单任务上 ToT 不仅成本高，甚至可能因为过度思考而降低准确率。CoT 在 GSM8K 等标准数学推理上已经足够好。ToT 的优势主要体现在需要全局搜索和回溯的问题上。

2. **误区："ToT 就是多次调用 CoT"** — ToT 的关键不是"多次"，而是"结构化搜索"——包括生成候选、评估打分、剪枝和回溯。Self-Consistency 也是多次调用但没有搜索结构。

3. **追问："ToT 和 Reasoning Model (o1/o3) 的关系是什么？"** — Reasoning Model 将类似 ToT 的搜索和回溯内化到模型内部——模型自主探索多条推理链并选择最优。ToT 是外部搜索（在 API 层面实现），Reasoning Model 是内部搜索（在模型训练层面实现）。

4. **追问："Graph-of-Thought 比 ToT 好在哪里？"** — GoT 允许非线性推理——不同推理分支可以合并、交叉引用。比如"思路 A 的结论可以帮助思路 B"。但实现复杂度更高，实际应用较少。

## 参考资料

- [Tree of Thoughts: Deliberate Problem Solving with LLMs (arXiv, Yao et al.)](https://arxiv.org/pdf/2305.10601)
- [What is Tree Of Thoughts Prompting? (IBM)](https://www.ibm.com/think/topics/tree-of-thoughts)
- [Tree of Thoughts (ToT) - Prompt Engineering Guide](https://www.promptingguide.ai/techniques/tot)
- [Something-of-Thought in LLM Prompting: An Overview (Towards Data Science)](https://towardsdatascience.com/something-of-thought-in-llm-prompting-an-overview-of-structured-llm-reasoning-70302752b390/)
- [Demystifying Chains, Trees, and Graphs of Thoughts (arXiv)](https://arxiv.org/html/2401.14295v3)
