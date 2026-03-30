# 推理策略详解：Chain-of-Thought 与 Tree-of-Thought

> 难度：中级
> 分类：Planning & Reasoning

## 简短回答

Chain-of-Thought (CoT) 和 Tree-of-Thought (ToT) 是两种主流的 LLM 推理策略。**CoT 是线性推理**——通过引导模型"一步步思考"而非直接给出答案，显著提升复杂推理任务的准确率，核心类比人类解题时列出中间步骤。CoT 有两种形式：**Few-shot CoT**（提供带推理步骤的示例）和 **Zero-shot CoT**（在 prompt 末尾加 "Let's think step by step"）。**ToT 是探索式推理**——在每一步生成多个候选思路，用搜索算法（BFS/DFS）探索不同分支，通过评估函数剪枝和回溯。Wei et al. (Google, NeurIPS 2022) 证明 CoT 在算术、常识和符号推理上带来显著提升；Yao et al. (Princeton, NeurIPS 2023) 表明 ToT 在 Game of 24 上将成功率从 CoT 的 4% 提升到 74%。**选择原则**：任务有明确解题路径 → CoT；任务需要探索、回溯或创造性思考 → ToT；兼顾成本与准确率 → CoT + Self-Consistency。

## 详细解析

### 一、Chain-of-Thought (CoT)

#### CoT 的工作原理

```
标准 Prompting（直接回答）：
Q: 小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
A: 6

CoT Prompting（逐步推理）：
Q: 小明有 5 个苹果，给了小红 2 个，又买了 3 个，现在有几个？
A: 让我一步步分析：
   1. 小明初始有 5 个苹果
   2. 给了小红 2 个：5 - 2 = 3
   3. 又买了 3 个：3 + 3 = 6
   所以小明现在有 6 个苹果。
```

看似结果一样，但在更复杂的问题上，有中间步骤的推理会大幅减少错误。

#### Few-shot CoT

在 prompt 中提供带推理步骤的示例：

```python
few_shot_cot_prompt = """
问题：一个商店有 15 箱苹果，每箱 20 个。卖掉了 120 个，还剩多少？
推理：
1. 总共有 15 × 20 = 300 个苹果
2. 卖掉了 120 个
3. 剩余 300 - 120 = 180 个
答案：180 个

问题：一辆车以 60km/h 的速度行驶了 2.5 小时，然后以 80km/h 行驶了 1.5 小时。总距离是多少？
推理：
1. 第一段距离：60 × 2.5 = 150 km
2. 第二段距离：80 × 1.5 = 120 km
3. 总距离：150 + 120 = 270 km
答案：270 km

问题：{user_question}
推理：
"""
```

#### Zero-shot CoT

不需要示例，只需在 prompt 末尾加一句话：

```python
# Zero-shot CoT：最简单的形式
prompt = f"""
{user_question}

Let's think step by step.
"""

# 变体
prompts = [
    f"{question}\nLet's think step by step.",
    f"{question}\nLet's work this out in a step by step way to be sure we have the right answer.",
    f"{question}\n请一步步分析这个问题。",
]
```

Kojima et al. (2022) 发现这个简单的添加将 MultiArith 准确率从 17.7% 提升到 78.7%。

#### 为什么 CoT 有效？

```python
reasons_cot_works = {
    "问题分解": (
        "复杂问题被拆分为更小的子问题，"
        "每个子问题对 LLM 来说更容易处理"
    ),
    "更多推理计算": (
        "生成中间步骤 = 更多的 token = 更多的计算。"
        "模型获得了更多'思考时间'来处理信息"
    ),
    "减少跳跃式错误": (
        "直接给答案容易跳过关键逻辑步骤，"
        "CoT 强制模型不跳步"
    ),
    "自我纠正机会": (
        "中间步骤产生的错误可能在后续步骤中被发现和修正"
    ),
    "透明性与可调试性": (
        "推理过程可见 → 可以定位错误发生在哪一步"
    ),
}
```

#### CoT 在 Agent 系统中的应用

```python
# ReAct 模式就是 CoT 的 Agent 化应用
react_prompt = """
用户问题：{question}

请按以下格式推理和行动：

Thought: 我需要思考下一步做什么
Action: 使用工具 [工具名]
Action Input: 工具输入参数
Observation: 工具返回结果
... (可以重复多次)
Thought: 我现在有了足够的信息来回答
Final Answer: 最终答案
"""

# CoT 让 Agent 的决策过程可解释
# 每个 Thought 步骤都展示了 Agent 为什么选择这个工具
```

#### CoT 的变体与扩展

```
CoT (Chain-of-Thought)
 ├── Few-shot CoT：提供示例
 ├── Zero-shot CoT："Let's think step by step"
 ├── Self-Consistency：多次采样 + 多数投票
 │    (同一问题生成多条推理链，取最常见答案)
 ├── Tree-of-Thought (ToT)：探索多条推理路径
 │    (每步生成多个候选，评估后选择最优)
 ├── Graph-of-Thought (GoT)：非线性推理图
 └── Auto-CoT：自动生成推理示例
```

#### Self-Consistency：CoT 的增强版

```python
async def self_consistency(question, num_samples=5):
    """多次采样 + 多数投票"""
    answers = []
    for _ in range(num_samples):
        # 每次独立生成一条推理链（temperature > 0）
        response = await llm.invoke(
            f"{question}\nLet's think step by step.",
            temperature=0.7
        )
        answer = extract_final_answer(response)
        answers.append(answer)

    # 多数投票
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

#### CoT 的局限性

| 局限 | 说明 |
|------|------|
| 模型规模要求 | 传统上认为仅在 ~100B+ 参数模型上有效，但随着蒸馏技术的发展（如 DeepSeek-R1 蒸馏到 7B/14B），较小模型也能通过蒸馏获得显著的 CoT 能力；未经蒸馏的小模型仍可能产生错误推理链 |
| 成本增加 | 中间步骤消耗更多 output token |
| 推理链质量不保证 | 模型可能生成"看似合理实则错误"的推理步骤 |
| 不适合所有任务 | 简单任务加 CoT 反而增加不必要的复杂度 |
| 可被攻击 | 对抗样本可以诱导错误的推理链 |

### 二、Tree-of-Thought (ToT)

#### CoT 与 ToT 的核心区别

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

#### ToT 的工作原理

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

#### 两种搜索策略对比

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

#### 实际应用中的 ToT 简化版

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

### 三、对比分析与选择指南

#### 关键性能对比

```
任务：Game of 24（用四个数字通过加减乘除得到 24）
┌────────────────────┬───────────┬──────────┐
│ 方法               │ 成功率    │ LLM 调用  │
├────────────────────┼───────────┼──────────┤
│ Standard (IO)      │ 7.3%      │ 1        │
│ CoT                │ 4.0%      │ 1        │  ← CoT 的线性推理在需要回溯搜索的问题上反而成为限制
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

#### 何时选择哪种？

```python
decision_guide = {
    "使用 CoT": [
        "数学计算：有明确的解题步骤（算术、代数）",
        "逻辑推理：前提 → 结论的线性推导",
        "信息提取：从文本中逐步提取关键信息",
        "Agent 的工具选择决策（ReAct 的 Thought 步骤）",
        "代码调试（逐步分析错误原因）",
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
    "不需要 CoT/ToT": [
        "简单的事实查询",
        "翻译和改写",
        "情感分析等分类任务",
        "小模型（<10B 参数，除非经过蒸馏训练）",
    ],
}
```

#### 方法谱系总结

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

1. **误区："CoT 只是让模型输出更长"** — CoT 的核心不是长度，而是结构化的中间推理步骤。重要的是推理的质量而非数量。一条简洁但正确的推理链比冗长但偏题的推理更有效。

2. **误区："Zero-shot CoT 总是有效的"** — 只在足够大的模型上有效。小模型使用 CoT 反而会因为生成错误的推理链而降低准确率。另外，对于简单任务，CoT 增加成本但不提升质量。

3. **误区："ToT 总是比 CoT 好"** — 在简单任务上 ToT 不仅成本高，甚至可能因为过度思考而降低准确率。CoT 在 GSM8K 等标准数学推理上已经足够好。ToT 的优势主要体现在需要全局搜索和回溯的问题上。

4. **误区："ToT 就是多次调用 CoT"** — ToT 的关键不是"多次"，而是"结构化搜索"——包括生成候选、评估打分、剪枝和回溯。Self-Consistency 也是多次调用但没有搜索结构。

5. **追问："CoT 和 Reasoning Models（o1/o3/R1）是什么关系？"** — Reasoning Models 将 CoT 内化到了模型的推理过程中（internal chain-of-thought），不需要用户显式提示。模型自动生成"思考 token"，然后再输出答案。本质上是 CoT 的模型级实现。同样，Reasoning Model 也将类似 ToT 的搜索和回溯内化到模型内部——ToT 是外部搜索（在 API 层面实现），Reasoning Model 是内部搜索（在模型训练层面实现）。

6. **追问："Self-Consistency 比 CoT 好多少？"** — Self-Consistency 通过多次采样 + 投票显著提升准确率，特别是在数学推理上。但代价是成本增加 N 倍（N 次 LLM 调用）。适合准确率要求高且成本不敏感的场景。

7. **追问："Graph-of-Thought 比 ToT 好在哪里？"** — GoT 允许非线性推理——不同推理分支可以合并、交叉引用。比如"思路 A 的结论可以帮助思路 B"。但实现复杂度更高，实际应用较少。

## 参考资料

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arXiv, Wei et al.)](https://arxiv.org/abs/2201.11903)
- [Tree of Thoughts: Deliberate Problem Solving with LLMs (arXiv, Yao et al.)](https://arxiv.org/pdf/2305.10601)
- [What is Chain of Thought Prompting? (IBM)](https://www.ibm.com/think/topics/chain-of-thoughts)
- [What is Tree Of Thoughts Prompting? (IBM)](https://www.ibm.com/think/topics/tree-of-thoughts)
- [Chain-of-Thought Prompting (Prompt Engineering Guide)](https://www.promptingguide.ai/techniques/cot)
- [Tree of Thoughts (ToT) - Prompt Engineering Guide](https://www.promptingguide.ai/techniques/tot)
- [Chain-of-Thought Prompting: Step-by-Step Reasoning with LLMs (DataCamp)](https://www.datacamp.com/tutorial/chain-of-thought-prompting)
- [Chain-of-Thought Prompting Guide (PromptHub)](https://www.prompthub.us/blog/chain-of-thought-prompting-guide)
- [Something-of-Thought in LLM Prompting: An Overview (Towards Data Science)](https://towardsdatascience.com/something-of-thought-in-llm-prompting-an-overview-of-structured-llm-reasoning-70302752b390/)
- [Demystifying Chains, Trees, and Graphs of Thoughts (arXiv)](https://arxiv.org/html/2401.14295v3)
