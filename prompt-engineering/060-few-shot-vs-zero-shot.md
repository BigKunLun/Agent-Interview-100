# Few-Shot vs Zero-Shot Prompting：如何选择？

> 难度：基础
> 分类：Prompt Engineering

## 简短回答

**Zero-shot Prompting** 不提供任何示例，直接描述任务让 LLM 执行——依赖模型从预训练中学到的通用能力。**Few-shot Prompting** 在 Prompt 中提供 2-5 个输入输出示例，让模型通过"模仿"理解任务格式和期望。选择原则：**任务简单且模型理解充分 → Zero-shot**（更省 token、更灵活）；**任务有特定格式/逻辑或模型表现不稳定 → Few-shot**（更准确、更一致）。研究表明 Few-shot 在分类和格式化任务上优势显著，但在复杂推理任务上，Few-shot 的优势被 CoT 等推理增强技术压缩。随着模型能力提升（GPT-4、Claude Opus 等），Zero-shot 的能力边界在不断扩大——很多过去需要 Few-shot 的任务现在 Zero-shot 就能做好。

## 详细解析

### 核心区别

```python
# Zero-shot：直接描述任务，无示例
zero_shot_prompt = """
将以下客户评价分类为"正面"、"负面"或"中性"。

评价：这个产品用了一周就坏了，太失望了。
分类：
"""
# 模型需要自己理解"分类"的含义和格式

# Few-shot：提供示例，模型模仿模式
few_shot_prompt = """
将客户评价分类为"正面"、"负面"或"中性"。

评价：非常好用，超出预期！
分类：正面

评价：还行吧，没什么特别的。
分类：中性

评价：快递太慢了，包装也破损了。
分类：负面

评价：这个产品用了一周就坏了，太失望了。
分类：
"""
# 模型通过示例理解输出格式和分类标准
```

### 何时选择 Zero-shot

```python
use_zero_shot_when = [
    "任务在模型预训练中常见（翻译、摘要、问答）",
    "使用足够强的模型（GPT-4、Claude Opus）",
    "任务描述本身足够清晰明确",
    "需要灵活处理多样化的输入",
    "Token 预算有限（节省 Prompt token）",
    "快速原型验证阶段",
]

# Zero-shot 的增强技巧
zero_shot_enhanced = {
    "角色设定": "你是一个资深的情感分析专家...",
    "详细指令": "分类为正面/负面/中性，基于以下标准：...",
    "格式约束": "只输出分类标签，不要解释",
    "CoT 增强": "Let's think step by step",
}
```

### 何时选择 Few-shot

```python
use_few_shot_when = [
    "任务有特定的输出格式（JSON、表格、特殊结构）",
    "分类标签有领域特定含义（不是通用的正面/负面）",
    "模型在 Zero-shot 下表现不稳定",
    "需要教模型一个它不熟悉的模式",
    "使用较小的模型（Few-shot 对小模型帮助更大）",
    "输出一致性非常重要（生产环境）",
]

# Few-shot 示例
custom_classification = """
将代码审查评论分类为以下类型：
- BUG: 发现了实际的代码缺陷
- STYLE: 代码风格建议
- PERF: 性能优化建议
- QUESTION: 需要澄清的问题
- NITPICK: 微小的改进建议

评论：这里的循环应该用 dict 而不是 list，查找复杂度从 O(n) 降到 O(1)
类型：PERF

评论：变量名 x 太模糊了，建议改为 user_count
类型：STYLE

评论：这个空指针检查缺失了，传入 None 时会崩溃
类型：BUG

评论：{new_comment}
类型：
"""
```

### Few-shot 的最佳实践

```python
few_shot_best_practices = {
    "示例数量": {
        "建议": "2-5 个示例最佳",
        "原因": "太少不够学习模式，太多浪费 token 且可能引入噪声",
        "例外": "复杂任务可以用 5-10 个示例",
    },
    "示例质量": {
        "多样性": "覆盖不同的输入类型和边界情况",
        "代表性": "包含典型案例，不只是简单案例",
        "平衡性": "每个分类标签的示例数量大致相等",
    },
    "示例顺序": {
        "建议": "将与待预测输入最相似的示例放在最后",
        "原因": "Recency Effect——模型对最近的示例记忆最深",
    },
    "格式一致": {
        "建议": "所有示例的格式必须完全一致",
        "原因": "模型会精确模仿示例的格式，包括标点和空格",
    },
}
```

### 动态 Few-shot：结合两者优势

```python
class DynamicFewShot:
    """根据输入动态选择最相关的示例"""

    def __init__(self, example_bank, embedding_model):
        self.examples = example_bank  # 预存的示例库
        self.embedder = embedding_model

    def build_prompt(self, user_input, k=3):
        # 1. 从示例库中检索最相关的 k 个示例
        input_embedding = self.embedder.encode(user_input)
        similar_examples = self.examples.search(
            input_embedding, top_k=k
        )

        # 2. 构建 Few-shot Prompt
        prompt = "根据以下示例完成任务：\n\n"
        for example in similar_examples:
            prompt += f"输入：{example.input}\n"
            prompt += f"输出：{example.output}\n\n"

        prompt += f"输入：{user_input}\n输出："
        return prompt

# 优势：
# - 示例与当前输入高度相关 → 准确率更高
# - 示例库可以持续扩充 → 覆盖更多场景
# - 不需要手动挑选示例 → 自动化
```

### 性能对比数据

```
情感分类（SST-2）：
┌──────────────────┬──────────┬──────────┐
│ 方法             │ GPT-3.5  │ GPT-4    │
├──────────────────┼──────────┼──────────┤
│ Zero-shot        │ 88%      │ 95%      │
│ Few-shot (3)     │ 93%      │ 96%      │
│ Few-shot (5)     │ 94%      │ 96%      │
└──────────────────┴──────────┴──────────┘

观察：
1. Few-shot 对弱模型帮助更大（+6% vs +1%）
2. 强模型 Zero-shot 已经很好
3. 3 个示例就已获得大部分提升

数学推理（GSM8K）：
┌──────────────────┬──────────┐
│ 方法             │ GPT-4    │
├──────────────────┼──────────┤
│ Zero-shot        │ ~80%     │
│ Few-shot (8)     │ ~82%     │
│ Zero-shot CoT    │ ~90%     │
│ Few-shot CoT     │ ~92%     │
└──────────────────┴──────────┘

观察：CoT 的提升 > Few-shot 的提升
对于推理任务，推理方式比示例数量更重要
```

### 决策流程图

```
任务到来
  │
  ├── 任务简单+模型强？ ─── 是 → Zero-shot
  │
  ├── 需要特定输出格式？ ── 是 → Few-shot（确保格式一致）
  │
  ├── 模型表现不稳定？ ─── 是 → Few-shot（用示例稳定行为）
  │
  ├── 需要推理？ ────────── 是 → Zero-shot CoT 或 Few-shot CoT
  │
  ├── Token 预算紧张？ ─── 是 → Zero-shot + 详细指令
  │
  └── 不确定？ ──────────── 先试 Zero-shot，不够再加示例
```

## 常见误区 / 面试追问

1. **误区："Few-shot 总是比 Zero-shot 好"** — 不一定。(1) 在推理任务上，CoT 的提升远大于示例的提升；(2) 不好的示例可能误导模型（示例质量比数量更重要）；(3) 强模型的 Zero-shot 已经接近人类水平。

2. **误区："示例越多越好"** — 超过 5-10 个示例后，边际收益递减且成本急增。更多示例 = 更多 input token = 更高成本 + 更少剩余空间给模型思考。

3. **追问："如何选择 Few-shot 的示例？"** — 三个原则：(1) 多样性——覆盖不同类型的输入；(2) 代表性——选典型案例而非极端案例；(3) 相关性——最好与待处理的输入相关。动态 Few-shot（检索最相关的示例）是生产中的最佳实践。

4. **追问："One-shot（只给一个示例）有用吗？"** — 有用，尤其在教模型特定输出格式时。一个精心选择的示例可能比三个普通示例更有效。但对于分类任务，一个示例可能引入偏差（模型倾向于输出示例中的标签）。

## 参考资料

- [Zero-Shot vs Few-Shot Prompting: A Guide with Examples (Vellum)](https://www.vellum.ai/blog/zero-shot-vs-few-shot-prompting-a-guide-with-examples)
- [Zero-Shot, One-Shot, and Few-Shot Prompting (Learn Prompting)](https://learnprompting.org/docs/basics/few_shot)
- [Zero-shot and few-shot learning (.NET - Microsoft Learn)](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/zero-shot-learning)
- [How to Choose Your GenAI Prompting Strategy (Matillion)](https://www.matillion.com/blog/gen-ai-prompt-strategy-zero-shot-few-shot-prompt)
- [Harness the Power of LLMs: Zero-shot and Few-shot Prompting (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2023/09/power-of-llms-zero-shot-and-few-shot-prompting/)
