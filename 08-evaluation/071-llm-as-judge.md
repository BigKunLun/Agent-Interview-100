# LLM-as-Judge：使用 LLM 评估 LLM 输出

> 难度：中级
> 分类：Evaluation

## 简短回答

LLM-as-Judge 是用一个强大的 LLM（如 GPT-4）自动评估另一个 LLM 输出质量的技术，在成本和质量之间取得了最佳平衡。两种核心模式：**Pointwise 评分**（对单个输出按维度打分，如 1-5 分）和 **Pairwise 对比**（比较两个输出哪个更好）。**衡量 Judge 可靠性的正确指标是 Cohen's Kappa（κ）**——顶级 Judge（GPT-4o / Claude Opus）的 κ ≈ 0.78-0.84，逼近人类-人类一致性（κ ≈ 0.80）。⚠️ **常见误区**：很多文章引用"Judge 与人类一致率 80%+"看似很高，但 raw percent agreement 容易虚高（κ=0.62 也能 >80% 一致率），学术界（"Judging the Judges" arXiv:2406.12624）明确指出应优先报告 Kappa。已知系统性偏差：**位置偏差**（倾向于给排在前面的答案更高分）、**冗长偏差**（偏好更长的回答）、**自我偏好**（GPT-4 Judge 偏好 GPT-4 的输出）。缓解策略包括：交换位置多次评估、提供详细的评分 Rubric、使用多个 Judge 投票、定期用人工标注校准。2025 年的新趋势是 **Agent-as-Judge**——用 Agent 代替单纯的 LLM 做评估，Agent 可以执行代码验证、搜索事实等操作来辅助判断。

## 详细解析

### 两种核心评估模式

```python
# 模式 1：Pointwise 评分（直接打分）
async def pointwise_judge(question, answer, rubric=None):
    prompt = f"""
    请评估以下回答的质量。

    问题：{question}
    回答：{answer}

    评分标准（Rubric）：
    {rubric or '''
    5分：完全正确、完整、清晰
    4分：基本正确，有小瑕疵
    3分：部分正确，有明显遗漏
    2分：大部分不正确或不相关
    1分：完全错误或无关
    '''}

    请先分析回答的优缺点，然后给出分数。
    输出 JSON：{{"analysis": "...", "score": 1-5}}
    """
    return await judge_llm.invoke(prompt)

# 模式 2：Pairwise 对比（两两比较）
async def pairwise_judge(question, answer_a, answer_b):
    prompt = f"""
    比较以下两个回答，判断哪个更好。

    问题：{question}
    回答 A：{answer_a}
    回答 B：{answer_b}

    比较维度：准确性、完整性、清晰度、实用性。
    输出：A更好 / B更好 / 差不多，并说明理由。
    """
    return await judge_llm.invoke(prompt)
```

### LLM Judge 的已知偏差

```python
known_biases = {
    "位置偏差 (Position Bias)": {
        "现象": "Pairwise 中倾向于给排在前面的答案更高分",
        "缓解": "交换 A/B 位置做两次评估，取平均或一致性结果",
    },
    "冗长偏差 (Verbosity Bias)": {
        "现象": "倾向于给更长、更详细的回答更高分",
        "缓解": "在 Rubric 中明确'简洁也是优点'",
    },
    "自我偏好 (Self-Preference)": {
        "现象": "GPT-4 做 Judge 时偏好 GPT-4 生成的内容",
        "缓解": "用与被评估模型不同家族的模型做 Judge",
    },
    "格式偏差 (Format Bias)": {
        "现象": "偏好格式更好（如有列表、标题）的回答",
        "缓解": "在评估时统一格式，或在 Rubric 中降低格式权重",
    },
    "锚定效应 (Anchoring)": {
        "现象": "提供参考答案时 Judge 被参考答案锚定",
        "缓解": "不提供参考答案做 reference-free 评估",
    },
    "修辞偏差 (Rhetoric Bias)": {
        "现象": "内容不变、只改修辞表达即可系统性抬高 Judge 打分（可被对手主动利用）",
        "缓解": "对同一内容做多版修辞改写做稳健性测试；Rubric 锚定可核验的实质断言而非表达力",
        "出处": "How Can Rhetoric Reward-Hack AI Reviewers? arXiv:2608.08975（2026-08-14）",
    },
}
```

**关于修辞偏差的补充说明。** 这一条与前面几条的性质不同，值得单独展开。arXiv:2608.08975《How Can Rhetoric Reward-Hack AI Reviewers?》（2026-08-14 登上 HuggingFace Daily Papers，48 upvote）用 4200 篇改写稿（源自 120 篇 ICLR 2026 投稿）做实验，**该文报告**：在科学内容完全不变的前提下，纯修辞维度的改写可以系统性地操纵 AI 审稿的打分。

差别在攻击者的位置：

| | 位置 / 冗长 / 格式偏差 | 修辞偏差 |
|---|---|---|
| 层次 | **格式层**——排序、长度、排版这类表层特征 | **修辞层**——论证语气、确定性措辞、框架叙述 |
| 性质 | 被动干扰，方向大体固定，接近**随机噪声** | 主动攻击，攻击者可**定向优化**（reward hacking，即针对评分机制本身刷分而非真正提升质量） |
| 缓解 | 交换位置、统一格式等**机械手段**基本可消除 | 机械手段无效——改写本身合法合规，没有可对齐的「正确格式」 |

含义是：**只要打分会带来收益（论文录用、模型排行榜位次、RLAIF 奖励信号），Judge 面对的就不是随机噪声，而是有动机的对手。**据其实验，被评估方无需改动任何实质内容即可拿到更高分，这直接威胁到把 LLM Judge 用作奖励模型或准入门槛的场景。

需要说明的是，这是社区的单次实验结果，截至本文更新时尚未见独立复现，实际影响幅度和跨 Judge 模型的普适性仍待验证——引用时应作为「需要防范的风险面」而非已确立的定量结论。

### 构建可靠的 LLM Judge 系统

```python
class ReliableLLMJudge:
    """带偏差缓解的 LLM 评估系统"""

    async def evaluate(self, question, answer, rubric):
        scores = []

        # 策略 1：多次评估取平均（减少随机性）
        for _ in range(3):
            score = await self.pointwise_judge(question, answer, rubric)
            scores.append(score)
        avg_score = np.mean(scores)

        # 策略 2：多 Judge 投票（减少单模型偏差）
        judges = ["gpt-4o", "claude-sonnet-4-5", "gemini-pro"]
        multi_scores = []
        for judge in judges:
            s = await self.judge_with_model(judge, question, answer, rubric)
            multi_scores.append(s)
        consensus = np.median(multi_scores)

        return {
            "single_judge_avg": avg_score,
            "multi_judge_consensus": consensus,
            "agreement": self.compute_agreement(multi_scores),
        }

    async def pairwise_debiased(self, question, answer_a, answer_b):
        """消除位置偏差的 Pairwise 评估"""
        # 正序评估
        result_1 = await self.pairwise_judge(question, answer_a, answer_b)
        # 交换位置再评估
        result_2 = await self.pairwise_judge(question, answer_b, answer_a)

        if result_1 == "A" and result_2 == "B":
            return "A更好"  # 两次都选了同一个答案
        elif result_1 == "B" and result_2 == "A":
            return "B更好"
        else:
            return "不确定"  # 结果不一致，可能真的差不多
```

### 评分 Rubric 的设计

```python
# 好的 Rubric 是 LLM Judge 可靠性的关键
rubric_example = """
评估维度及标准：

1. 事实准确性 (0-3分)
   3: 所有事实完全正确
   2: 大部分正确，有 1-2 处小错误
   1: 有多处事实错误
   0: 大部分内容不准确

2. 完整性 (0-3分)
   3: 全面覆盖问题的所有方面
   2: 覆盖了主要方面，遗漏了次要点
   1: 只回答了部分问题
   0: 严重遗漏关键内容

3. 清晰度 (0-2分)
   2: 结构清晰，逻辑连贯
   1: 基本可读但组织可改进
   0: 混乱难懂

4. 实用性 (0-2分)
   2: 包含可操作的建议或代码
   1: 提供了方向但缺少细节
   0: 纯理论无实用价值

总分 = 准确性 + 完整性 + 清晰度 + 实用性（满分 10）
"""

# 具体的 Rubric 比模糊的"评估质量"可靠得多
```

### Agent-as-Judge（前沿趋势）

```python
class AgentJudge:
    """用 Agent 替代纯 LLM 做评估——可以调用工具验证"""

    async def evaluate(self, question, answer):
        # Agent 不仅用推理评判，还可以执行验证动作

        # 1. 事实核查：搜索验证关键声明
        claims = await self.extract_claims(answer)
        fact_checks = []
        for claim in claims:
            evidence = await self.web_search(claim)
            is_true = await self.verify(claim, evidence)
            fact_checks.append({"claim": claim, "verified": is_true})

        # 2. 代码验证：运行代码检查正确性
        code_blocks = self.extract_code(answer)
        for code in code_blocks:
            result = await self.execute_code(code)
            # 检查是否能运行、输出是否正确

        # 3. 综合评分
        return await self.synthesize_score(fact_checks, code_results)
```

## 常见误区 / 面试追问

1. **误区："LLM Judge 的评分就是客观事实"** — LLM Judge 的评分包含系统性偏差，不应被视为绝对真理。它是"有偏差的专家意见"，需要用人工标注定期校准。**衡量 Judge 可靠性必须用 Cohen's Kappa**（消除随机一致性），而非 raw percent agreement——后者非常容易虚高（κ=0.62 即可对应 >80% percent agreement，但实际可靠性远低于看起来）。基准参考：人类-人类 κ ≈ 0.80，顶级 LLM Judge κ ≈ 0.78-0.84（"Judging the Judges", arXiv:2406.12624）。

2. **误区："Pairwise 比 Pointwise 总是更好"** — Pairwise 在主观评估（风格、偏好）上更稳定，但对于有明确标准的评估（事实正确性），Pointwise + 详细 Rubric 更高效。且 ACL 2025 研究表明 Pairwise 实际上会放大偏差。

3. **追问："如何选择 Judge 模型？"** — 原则：(1) Judge 应比被评估模型更强；(2) Judge 应与被评估模型来自不同家族（减少自我偏好）；(3) 对于关键评估用多个 Judge 投票。常见选择：GPT-4o 评估 Claude 输出，反之亦然。

4. **追问："LLM Judge 的成本如何控制？"** — (1) 对大规模评估采样而非全量评估；(2) 先用便宜模型做初筛，只对边界案例用强 Judge；(3) 缓存相同输入的评估结果。通常 LLM Judge 成本是人工评估的 1/10。

5. **追问："缓解了位置偏差和冗长偏差，Judge 就可信了吗？"** — 不够。位置/冗长/格式属于**格式层**的被动干扰，交换位置、统一排版这类机械手段基本可以消除；但 arXiv:2608.08975（2026-08-14）指出还存在**修辞层**的主动攻击面——**该文报告**用 4200 篇改写稿（源自 120 篇 ICLR 2026 投稿）验证，在科学内容完全不变的前提下，纯修辞改写可以系统性操纵 AI 审稿打分。区别在于前者接近随机噪声、可用固定流程抵消，后者是有动机的对手针对评分机制做的定向优化（reward hacking），改写本身合法合规，没有可对齐的「正确格式」可言。**实践含义**：凡是打分会带来实际收益的场景（排行榜、准入门槛、RLAIF 奖励信号），都必须假设被评估方会针对 Judge 优化，评估设计要包含对抗性改写的稳健性测试，并保留人工抽检。**需注意**该结论来自社区单次实验、尚未见独立复现，应作为风险面对待而非定量事实。

## 参考资料

- [LLM-as-a-Judge Simply Explained (Confident AI)](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)
- [LLM-as-a-Judge: A Complete Guide (Evidently AI)](https://www.evidentlyai.com/llm-guide/llm-as-a-judge)
- [Comprehensive Guide to LLM-as-a-Judge Evaluation (Galileo AI)](https://galileo.ai/blog/llm-as-a-judge-guide-evaluation)
- [The Rise of Agent-as-a-Judge Evaluation for LLMs (arXiv)](https://arxiv.org/html/2508.02994v1)
- [LLM As a Judge: Tutorial and Best Practices (Patronus AI)](https://www.patronus.ai/llm-testing/llm-as-a-judge)
- [How Can Rhetoric Reward-Hack AI Reviewers? (arXiv:2608.08975，2026-08-14；社区单次实验，未见独立复现)](https://arxiv.org/abs/2608.08975)
