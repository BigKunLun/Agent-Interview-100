# Chain-of-Thought (CoT) 推理是什么？为什么有效？

> 难度：基础
> 分类：Planning & Reasoning

## 简短回答

Chain-of-Thought（思维链）是一种 Prompt 技术，通过引导 LLM **"一步步思考"**而非直接给出答案，显著提升复杂推理任务的准确率。核心思想类比人类解题——拿到一道数学题不会直接写答案，而是列出中间步骤。CoT 有两种形式：**Few-shot CoT**（在 prompt 中提供带推理步骤的示例）和 **Zero-shot CoT**（简单地在 prompt 末尾加上"Let's think step by step"）。Wei et al. (Google, NeurIPS 2022) 的开创性论文证明 CoT 在算术、常识和符号推理任务上带来了显著提升。Zero-shot CoT 在 MultiArith 数学基准上将准确率从 17.7% 提升到 78.7%。但 CoT 有一个重要限制：只在大模型（~100B+ 参数）上有效，小模型反而会因为错误的推理链导致更差的结果。

## 详细解析

### CoT 的工作原理

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

### Few-shot CoT

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

### Zero-shot CoT

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

### 为什么 CoT 有效？

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

### CoT 在 Agent 系统中的应用

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

### CoT 的变体与扩展

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

### Self-Consistency：CoT 的增强版

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

### CoT 的局限性

| 局限 | 说明 |
|------|------|
| 模型规模要求 | 传统上认为仅在 ~100B+ 参数模型上有效，但随着蒸馏技术的发展（如 DeepSeek-R1 蒸馏到 7B/14B），较小模型也能通过蒸馏获得显著的 CoT 能力；未经蒸馏的小模型仍可能产生错误推理链 |
| 成本增加 | 中间步骤消耗更多 output token |
| 推理链质量不保证 | 模型可能生成"看似合理实则错误"的推理步骤 |
| 不适合所有任务 | 简单任务加 CoT 反而增加不必要的复杂度 |
| 可被攻击 | 对抗样本可以诱导错误的推理链 |

### 何时使用 CoT？

```python
use_cot_when = [
    "多步数学计算",
    "逻辑推理和演绎",
    "需要综合多个信息源的分析",
    "Agent 的工具选择决策（ReAct 的 Thought 步骤）",
    "代码调试（逐步分析错误原因）",
]

skip_cot_when = [
    "简单的事实查询",
    "翻译和改写",
    "情感分析等分类任务",
    "小模型（<10B 参数，除非经过蒸馏训练）",
]
```

## 常见误区 / 面试追问

1. **误区："CoT 只是让模型输出更长"** — CoT 的核心不是长度，而是结构化的中间推理步骤。重要的是推理的质量而非数量。一条简洁但正确的推理链比冗长但偏题的推理更有效。

2. **误区："Zero-shot CoT 总是有效的"** — 只在足够大的模型上有效。小模型使用 CoT 反而会因为生成错误的推理链而降低准确率。另外，对于简单任务，CoT 增加成本但不提升质量。

3. **追问："CoT 和 Reasoning Models（o1/o3/R1）是什么关系？"** — Reasoning Models 将 CoT 内化到了模型的推理过程中（internal chain-of-thought），不需要用户显式提示。模型自动生成"思考 token"，然后再输出答案。本质上是 CoT 的模型级实现。

4. **追问："Self-Consistency 比 CoT 好多少？"** — Self-Consistency 通过多次采样 + 投票显著提升准确率，特别是在数学推理上。但代价是成本增加 N 倍（N 次 LLM 调用）。适合准确率要求高且成本不敏感的场景。

## 参考资料

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (arXiv, Wei et al.)](https://arxiv.org/abs/2201.11903)
- [What is Chain of Thought Prompting? (IBM)](https://www.ibm.com/think/topics/chain-of-thoughts)
- [Chain-of-Thought Prompting (Prompt Engineering Guide)](https://www.promptingguide.ai/techniques/cot)
- [Chain-of-Thought Prompting: Step-by-Step Reasoning with LLMs (DataCamp)](https://www.datacamp.com/tutorial/chain-of-thought-prompting)
- [Chain-of-Thought Prompting Guide (PromptHub)](https://www.prompthub.us/blog/chain-of-thought-prompting-guide)
