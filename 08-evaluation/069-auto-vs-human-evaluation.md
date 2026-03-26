# LLM 评估的基本方法：自动评估 vs 人工评估

> 难度：基础
> 分类：Evaluation

## 简短回答

LLM 评估分为三大类：(1) **自动指标评估**——用算法计算的确定性指标（如 BLEU、ROUGE、精确匹配），速度快、成本低，但只能衡量表面特征（如词汇重叠），无法评判语义质量；(2) **人工评估**——由人类标注者评判输出质量（如流畅度、有用性、准确性），是"金标准"但成本高、速度慢、难以规模化；(3) **LLM-as-Judge**——用强 LLM 评估其他 LLM 的输出，是近年来的主流趋势，在成本和质量间取得平衡，与人工评估的一致性达 80%+。生产环境推荐**混合方案**：自动指标做初筛和持续监控，LLM-as-Judge 做质量评估，人工评估做最终决策和校准。Sebastian Raschka 将 LLM 评估总结为四种方法：多选基准、人类偏好、自动化 LLM 评估、和编程基准。

## 详细解析

### 评估方法全景

```
┌──────────────────────────────────────────────────────┐
│                LLM 评估方法全景                       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  自动指标        LLM-as-Judge       人工评估          │
│  ─────────      ──────────────     ─────────         │
│  BLEU/ROUGE     GPT-4 打分         专家评审          │
│  精确匹配       多维度评估          众包标注          │
│  F1 Score       Pairwise 对比      用户反馈          │
│  Perplexity     Rubric 评分        A/B 测试          │
│                                                      │
│  速度: 最快      速度: 中等         速度: 最慢        │
│  成本: 最低      成本: 中等         成本: 最高        │
│  质量: 有限      质量: 较好         质量: 最好        │
│  规模: 无限      规模: 大           规模: 小          │
└──────────────────────────────────────────────────────┘
```

### 自动指标评估

```python
# 常见自动评估指标

# 1. 精确匹配（Exact Match）
def exact_match(prediction, reference):
    return prediction.strip() == reference.strip()
# 适用：数学题答案、事实性问题、代码输出

# 2. BLEU（机器翻译质量）
from nltk.translate.bleu_score import sentence_bleu
score = sentence_bleu([reference.split()], prediction.split())
# 衡量 n-gram 重叠度，0-1 分

# 3. ROUGE（摘要质量）
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
scores = scorer.score(reference, prediction)
# ROUGE-1: 单词重叠, ROUGE-L: 最长公共子序列

# 4. F1 Score（信息提取）
def token_f1(prediction, reference):
    pred_tokens = set(prediction.split())
    ref_tokens = set(reference.split())
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1

# 自动指标的局限：
# "北京是中国的首都" vs "中国的首都是北京"
# → 语义完全相同，但 BLEU/ROUGE 可能不是满分
# → 无法评估回答的有用性、创造性、安全性
```

### LLM-as-Judge

```python
async def llm_as_judge(question, answer, reference=None):
    """用 LLM 评估回答质量"""

    # 方式 1：直接评分（Pointwise）
    pointwise_prompt = f"""
    请评估以下回答的质量（1-5分）：

    问题：{question}
    回答：{answer}
    {"参考答案：" + reference if reference else ""}

    评分维度：
    - 准确性 (1-5)：事实是否正确？
    - 完整性 (1-5)：是否全面回答了问题？
    - 有用性 (1-5)：对提问者是否有帮助？
    - 清晰度 (1-5)：表达是否清楚？

    请给出每个维度的分数和简要理由，最后给出总分。
    """

    # 方式 2：对比评分（Pairwise）
    pairwise_prompt = f"""
    问题：{question}

    回答 A：{answer_a}
    回答 B：{answer_b}

    哪个回答更好？请从准确性、完整性和清晰度三个维度比较。
    输出：A 更好 / B 更好 / 差不多
    """

    return await judge_llm.invoke(pointwise_prompt)

# LLM-as-Judge 的已知偏差：
biases = {
    "位置偏差": "倾向于给排在前面的回答更高分",
    "冗长偏差": "倾向于给更长的回答更高分",
    "自我偏好": "GPT-4 作为 Judge 倾向于给 GPT-4 的输出更高分",
    "格式偏差": "倾向于给格式更好看的回答更高分",
}

# 缓解偏差：
# - 交换 A/B 位置做两次评估取平均
# - 使用与被评估模型不同的 Judge 模型
# - 提供明确的评分 Rubric
```

### 人工评估

```python
human_evaluation_methods = {
    "专家评审": {
        "方法": "领域专家按预定标准打分",
        "优势": "质量最高，能评估专业领域的细微差别",
        "劣势": "成本高，速度慢，难规模化",
        "适用": "高风险场景（医疗、法律、金融）",
    },
    "众包标注": {
        "方法": "通过 Scale AI、Toloka 等平台招募标注者",
        "优势": "可规模化，成本相对可控",
        "劣势": "标注者质量参差不齐，需要质量控制",
        "适用": "大规模偏好数据收集",
    },
    "用户反馈": {
        "方法": "收集真实用户的点赞/点踩/投诉",
        "优势": "最真实的质量信号",
        "劣势": "反馈稀疏（大部分用户不反馈），有偏差",
        "适用": "生产环境的持续监控",
    },
}
```

### 混合评估框架（推荐）

```python
class HybridEvaluator:
    """混合评估：自动指标 + LLM Judge + 人工抽检"""

    async def evaluate(self, test_set):
        results = []

        for example in test_set:
            prediction = await self.model.invoke(example.input)
            scores = {}

            # Layer 1: 自动指标（全量，毫秒级）
            scores["exact_match"] = exact_match(prediction, example.reference)
            scores["f1"] = token_f1(prediction, example.reference)

            # Layer 2: LLM-as-Judge（全量或采样，秒级）
            scores["llm_judge"] = await llm_as_judge(
                example.input, prediction, example.reference
            )

            # Layer 3: 标记需要人工审核的案例
            if scores["llm_judge"]["total"] < 3 or scores["f1"] < 0.5:
                scores["needs_human_review"] = True

            results.append(scores)

        # Layer 3 继续：人工审核低分和边界案例
        flagged = [r for r in results if r.get("needs_human_review")]
        # 送人工审核队列...

        return results
```

### 评估指标选择指南

```
┌──────────────────┬─────────────────┬──────────────────┐
│ 任务类型         │ 推荐指标        │ 评估方式          │
├──────────────────┼─────────────────┼──────────────────┤
│ 事实性问答       │ 精确匹配/F1     │ 自动             │
│ 文本摘要         │ ROUGE + LLM     │ 自动 + LLM Judge │
│ 翻译             │ BLEU + 人工     │ 自动 + 人工       │
│ 创意写作         │ 人工 + LLM      │ LLM Judge + 人工  │
│ 对话质量         │ LLM 多维评分     │ LLM Judge        │
│ 代码生成         │ Pass@k          │ 自动（运行测试）  │
│ Agent 任务       │ 任务完成率       │ 自动 + 轨迹评估   │
│ 安全性           │ 拒绝率/攻击成功  │ 自动 + 红队测试   │
└──────────────────┴─────────────────┴──────────────────┘
```

## 常见误区 / 面试追问

1. **误区："BLEU/ROUGE 分数高就说明质量好"** — 这些指标只衡量表面词汇重叠，无法评估语义正确性、逻辑合理性和实用性。两个语义相同但措辞不同的回答可能得到很不同的 BLEU 分数。LLM 时代这些传统指标的参考价值有限。

2. **误区："LLM-as-Judge 完全可以替代人工"** — LLM Judge 有系统性偏差（冗长偏好、位置偏差、自我偏好），且在专业领域（医学、法律）的判断可能不可靠。生产中应该定期用人工评估校准 LLM Judge 的准确性。

3. **追问："如何提高 LLM-as-Judge 的可靠性？"** — (1) 提供详细的评分 Rubric（标准）而非让 Judge 自由打分；(2) 交换位置做两次评估取平均（消除位置偏差）；(3) 用多个 Judge 模型投票；(4) 定期用人工标注校准。

4. **追问："评估数据集从哪里来？"** — 三个来源：(1) 从生产日志中采样真实问题；(2) 人工构造边界案例和对抗样本；(3) 使用公开基准（MMLU、GSM8K 等）。最佳实践是三者结合——公开基准评估通用能力，私有数据集评估业务场景。

## 参考资料

- [Understanding the 4 Main Approaches to LLM Evaluation (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches)
- [LLM Evaluation Metrics: The Ultimate Guide (Confident AI)](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [LLM Evaluation: Benchmarks vs. Human Judgment (Medium)](https://medium.com/@lmpo/llm-evaluation-benchmarks-vs-human-judgment-f1cdd16098c0)
- [LLM Evaluation Metrics and Methods, Explained Simply (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-evaluation-metrics)
- [An Analysis of Automated, Human, and LLM-Based Approaches (arXiv)](https://arxiv.org/pdf/2406.03339)
