# DSPy 等编程化 Prompt 优化工具的原理与使用

> 难度：中级
> 分类：Prompt Engineering

## 简短回答

DSPy（Declarative Self-improving Language Programs in Python）是 Stanford NLP 组开发的框架，核心理念是**"编程而非提示"**——用 Python 代码声明式地定义 LLM 管道，由框架自动优化 Prompt，取代手工 Prompt Engineering。传统方式中，开发者需要反复手动调整 Prompt 措辞来提升效果；DSPy 将这个过程自动化——开发者定义"做什么"（签名 + 指标），DSPy 的优化器自动发现"怎么做"（最优 Prompt + Few-shot 示例）。核心概念包括：**Signature**（定义输入输出的语义描述）、**Module**（可组合的 LLM 调用单元）、**Optimizer**（自动优化 Prompt 的算法，如 BootstrapFewShot、MIPROv2）。DSPy 让 Prompt 从"脆弱的字符串"变成"可编译的程序"，显著提升了 LLM 管道的可维护性和可移植性。

## 详细解析

### 传统 Prompt Engineering 的问题

```python
# 传统方式：手工编写和迭代 Prompt
prompt_v1 = "回答以下问题：{question}"           # 效果差
prompt_v2 = "你是专家。详细回答：{question}"      # 好一点
prompt_v3 = "你是资深专家。\n请分步骤回答：\n{question}\n先分析再总结"  # 更好
# ...手动迭代数十个版本

# 问题：
# 1. 脆弱：换个模型可能失效
# 2. 不可复现：Prompt 调优依赖个人经验
# 3. 难以组合：多步骤管道中每个 Prompt 互相影响
# 4. 无法系统优化：缺少自动化的优化反馈循环
```

### DSPy 的核心概念

```python
import dspy

# 1. Signature（签名）：声明输入输出的语义
# 最简形式："question -> answer"
# 等价于告诉 LLM "给定 question，生成 answer"

class QA(dspy.Signature):
    """回答关于 AI Agent 的技术问题"""
    question: str = dspy.InputField(desc="技术面试问题")
    answer: str = dspy.OutputField(desc="详细的技术回答，包含示例")

# 2. Module（模块）：LLM 调用的基本单元
class SimpleQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(QA)  # 自动加 CoT

    def forward(self, question):
        return self.generate(question=question)

# 3. 配置 LLM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 4. 使用
qa = SimpleQA()
result = qa(question="什么是 ReAct 模式？")
print(result.answer)
```

### DSPy 的优化器（核心创新）

```python
# 优化器自动为你的 Module 找到最佳 Prompt

# 准备训练数据（少量示例即可）
trainset = [
    dspy.Example(
        question="什么是 RAG？",
        answer="RAG 是检索增强生成..."
    ).with_inputs("question"),
    # ... 10-50 个示例
]

# 定义评估指标
def accuracy_metric(example, prediction, trace=None):
    """评估回答质量"""
    # 可以用 LLM 评分、关键词匹配等
    return dspy.evaluate.answer_exact_match(example, prediction)

# 选择优化器
optimizer = dspy.BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=4,  # 最多 4 个自动生成的示例
    max_labeled_demos=4,       # 最多 4 个标注示例
)

# 编译（自动优化）
optimized_qa = optimizer.compile(
    SimpleQA(),
    trainset=trainset
)

# optimized_qa 现在包含了自动优化后的 Prompt 和 Few-shot 示例
# 直接使用即可，无需手动调 Prompt
```

### 主要优化器对比

```python
optimizers = {
    "BootstrapFewShot": {
        "原理": "自动生成和筛选 Few-shot 示例",
        "过程": "用 LLM 生成候选示例 → 用指标筛选好的 → 加入 Prompt",
        "适用": "小数据集、快速优化",
        "成本": "低（少量 LLM 调用）",
    },
    "MIPROv2": {
        "原理": "贝叶斯优化搜索最佳 Prompt 指令 + 示例组合",
        "过程": "生成候选指令 → 搜索最优组合 → 迭代优化",
        "适用": "需要高质量优化的生产场景",
        "成本": "中等",
    },
    "BootstrapFinetune": {
        "原理": "用优化后的 trace 数据微调小模型",
        "过程": "先用大模型生成高质量 trace → 用 trace 微调小模型",
        "适用": "需要降低推理成本",
        "成本": "高（需要微调）",
    },
}
```

### 多步骤管道示例

```python
class RAGPipeline(dspy.Module):
    """DSPy 实现的 RAG 管道"""

    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought(
            "context, question -> answer"
        )

    def forward(self, question):
        # Step 1: 检索
        context = self.retrieve(question).passages

        # Step 2: 生成（自动带 CoT）
        answer = self.generate(
            context=context,
            question=question
        )
        return answer

# 优化整个管道——不只是单个 Prompt
# 优化器会同时优化检索和生成的配合
optimized_rag = optimizer.compile(
    RAGPipeline(),
    trainset=trainset
)
```

### DSPy vs 手工 Prompt Engineering

```
┌──────────────────┬─────────────────────┬──────────────────────┐
│ 维度             │ 手工 Prompt         │ DSPy                 │
├──────────────────┼─────────────────────┼──────────────────────┤
│ 开发方式         │ 反复修改字符串       │ 编写 Python 模块     │
│ 优化方式         │ 人工试错            │ 算法自动优化          │
│ 可移植性         │ 换模型需重写        │ 重新编译即可          │
│ 可组合性         │ 难以组合            │ 模块化组合            │
│ 可复现性         │ 依赖个人经验        │ 代码 + 数据可复现     │
│ 版本管理         │ 管理字符串版本      │ 管理代码版本          │
│ 学习曲线         │ 低                  │ 中等                 │
│ 适用场景         │ 简单任务、原型      │ 复杂管道、生产环境    │
└──────────────────┴─────────────────────┴──────────────────────┘
```

### 类似框架

```python
similar_frameworks = {
    "DSPy": {
        "机构": "Stanford NLP",
        "核心": "声明式签名 + 自动优化器",
        "语言": "Python",
    },
    "TextGrad": {
        "机构": "Stanford",
        "核心": "用文本反馈做梯度下降式优化",
        "特点": "将 Prompt 优化类比为神经网络训练",
    },
    "OPRO": {
        "机构": "Google DeepMind",
        "核心": "用 LLM 本身来优化 Prompt",
        "方法": "LLM 生成 Prompt 变体 → 评估 → 选择最优",
    },
    "PromptBreeder": {
        "机构": "Google DeepMind",
        "核心": "进化算法优化 Prompt",
        "方法": "Prompt 变异 + 选择 + 交叉",
    },
}
```

## 常见误区 / 面试追问

1. **误区："DSPy 完全不需要 Prompt Engineering 知识"** — DSPy 自动化了 Prompt 措辞的优化，但你仍需要设计好 Signature（输入输出语义）、选择合适的 Module（是否需要 CoT、是否需要检索），以及定义好评估指标。框架自动化的是"调词"，不是"设计"。

2. **误区："DSPy 的优化结果总是更好"** — 优化器需要足够好的评估指标和训练数据。如果指标定义不当或数据太少/有偏差，优化可能适得其反。始终要对比优化前后的效果。

3. **追问："DSPy 的优化后的 Prompt 可以导出吗？"** — 可以。用 `optimized_module.save(path)` 保存，用 `module.load(path)` 加载。也可以 inspect 看到优化后的实际 Prompt 文本。生产中可以将优化后的 Prompt 提取出来直接使用，不依赖 DSPy 运行时。

4. **追问："什么时候不该用 DSPy？"** — (1) 简单的一次性任务——手写 Prompt 更快；(2) 没有评估数据——优化器需要指标来判断好坏；(3) 任务频繁变化——每次变化都需要重新编译。DSPy 最适合需要长期维护和迭代优化的生产管道。

## 参考资料

- [DSPy Official Website](https://dspy.ai/)
- [DSPy: Programming—not prompting—language models (GitHub, Stanford NLP)](https://github.com/stanfordnlp/dspy)
- [Programming, Not Prompting: A Hands-on Guide to DSPy (Medium)](https://miptgirl.medium.com/programming-not-prompting-a-hands-on-guide-to-dspy-04ea2d966e6d)
- [Systematic LLM Prompt Engineering Using DSPy Optimization (Towards Data Science)](https://towardsdatascience.com/systematic-llm-prompt-engineering-using-dspy-optimization/)
- [DSPy Prompt Optimization (Weights & Biases)](https://docs.wandb.ai/weave/cookbooks/dspy_prompt_optimization)
