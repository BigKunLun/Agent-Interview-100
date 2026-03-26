# 内容过滤与毒性检测在 Agent 系统中的实现

> 难度：中级
> 分类：Safety & Alignment

## 简短回答

内容过滤（Content Filtering）和毒性检测（Toxicity Detection）是 Agent 系统防止生成或传播有害内容的关键安全机制。在 GenAI 时代，内容审核需要从"事后审查"转变为**实时拦截**——在 Agent 的输入和输出环节同时部署。实现方式分为三种：(1) **规则型过滤**——关键词黑名单、正则匹配（快速低成本，但易被变体绕过）；(2) **专用分类器**——训练专门的毒性分类模型（如 Perspective API、Llama Guard、Granite HAP），准确率可达 90%+ 且延迟低；(3) **LLM-as-Moderator**——用大模型理解上下文语义来判断（最智能但最慢最贵）。2025 年的最佳实践是**多层级联**：规则型快速过滤明显违规 → 分类器处理常见毒性模式 → LLM 处理需要上下文理解的边界情况。开源安全模型（Llama Guard 3、ShieldGemma、Granite Guardian）已达到商用水准。一个重要发现：在毒性分类任务上，传统轻量 NN 分类器（38M 参数）在召回率上显著优于 LLM in-context learning（0.96 vs 0.78），且计算成本低得多。

## 详细解析

### 内容过滤的工作位置

```
Agent 系统中的内容过滤点：

用户输入 → [输入过滤] → Agent 推理 → [中间过滤] → 工具调用
              │                           │
              ├── 毒性检测                 ├── 工具参数检查
              ├── 仇恨言论过滤             └── 生成内容检查
              ├── 成人内容检测
              └── PII/敏感信息

工具返回 → Agent 总结 → [输出过滤] → 返回用户
                           │
                           ├── 毒性检测
                           ├── 偏见检测
                           ├── PII 脱敏
                           └── 合规检查

关键：GenAI 的内容审核需要实时且上下文感知，
      传统的静态规则无法应对 LLM 生成内容的多样性
```

### 三种实现方式

```python
# 方式 1：规则型过滤（速度快，第一道防线）
class RuleBasedFilter:
    """基于规则的内容过滤——毫秒级延迟"""

    def __init__(self):
        self.keyword_blacklist = self.load_blacklist()  # 关键词黑名单
        self.regex_patterns = [
            r'(?i)(kill|murder|attack)\s+(people|person)',  # 暴力威胁
            r'(?i)how\s+to\s+(hack|exploit|attack)',        # 攻击教程
        ]

    def filter(self, text):
        # 关键词匹配
        for keyword in self.keyword_blacklist:
            if keyword.lower() in text.lower():
                return {"blocked": True, "reason": f"keyword: {keyword}"}

        # 正则匹配
        for pattern in self.regex_patterns:
            if re.search(pattern, text):
                return {"blocked": True, "reason": "pattern match"}

        return {"blocked": False}

    # 局限：容易被变体绕过（"k1ll"、Unicode 替换、隐喻表达）


# 方式 2：专用分类器（准确率高，推荐主力方案）
class ClassifierFilter:
    """基于专用模型的毒性检测"""

    def __init__(self):
        # 开源安全模型选择
        self.models = {
            "llama_guard": {
                "模型": "Llama Guard 3 (8B)",
                "来源": "Meta",
                "能力": "多类别安全分类（暴力、色情、仇恨、自伤等）",
                "特色": "可自定义安全策略",
            },
            "shield_gemma": {
                "模型": "ShieldGemma",
                "来源": "Google",
                "能力": "输入输出双向安全过滤",
                "特色": "与 Gemma 生态集成",
            },
            "granite_guardian": {
                "模型": "Granite Guardian (3B)",
                "来源": "IBM",
                "能力": "安全分类 + RAG 幻觉检测",
                "特色": "轻量级，适合边缘部署",
            },
            "granite_hap": {
                "模型": "Granite HAP (38M)",
                "来源": "IBM",
                "能力": "Hate/Abuse/Profanity 检测",
                "特色": "极轻量，实时检测",
            },
        }

    async def classify(self, text):
        """使用 Llama Guard 分类"""
        result = await self.llama_guard.invoke(
            f"[INST] Task: Check if the content is safe.\n"
            f"Content: {text} [/INST]"
        )
        return {
            "safe": result.label == "safe",
            "categories": result.violated_categories,
            "confidence": result.confidence,
        }


# 方式 3：LLM-as-Moderator（最智能，处理边界情况）
class LLMModerator:
    """用 LLM 理解上下文语义来审核内容"""

    async def moderate(self, text, context=None):
        prompt = f"""
        你是内容安全审核员。请评估以下内容是否安全。

        上下文：{context or '无'}
        待审内容：{text}

        评估维度：
        1. 暴力/威胁 (0-1)
        2. 仇恨/歧视 (0-1)
        3. 色情/成人 (0-1)
        4. 自伤/自杀 (0-1)
        5. 违法/犯罪 (0-1)

        注意上下文：医疗讨论中的"注射"不是暴力，
        安全教育中的"攻击方式"是合理讨论。

        输出 JSON：{{"safe": bool, "scores": {{...}}, "reasoning": "..."}}
        """
        return await self.judge.invoke(prompt)
```

### 多层级联架构

```python
class CascadingContentFilter:
    """多层级联过滤架构——平衡速度、准确性和成本"""

    async def filter(self, text, context=None):
        # Layer 1: 规则过滤（< 1ms）
        rule_result = self.rule_filter.filter(text)
        if rule_result["blocked"]:
            return {"blocked": True, "layer": "rule", **rule_result}

        # Layer 2: 轻量分类器（~10ms）
        classifier_result = await self.hap_classifier.classify(text)
        if not classifier_result["safe"] and classifier_result["confidence"] > 0.9:
            return {"blocked": True, "layer": "classifier", **classifier_result}

        # Layer 3: 安全大模型（~100ms，仅处理不确定情况）
        if not classifier_result["safe"] and classifier_result["confidence"] < 0.9:
            llm_result = await self.llama_guard.classify(text)
            if not llm_result["safe"]:
                return {"blocked": True, "layer": "llm_guard", **llm_result}

        # Layer 4: 上下文感知审核（~500ms，仅高风险场景）
        if context and context.risk_level == "high":
            moderator_result = await self.llm_moderator.moderate(text, context)
            if not moderator_result["safe"]:
                return {"blocked": True, "layer": "moderator", **moderator_result}

        return {"blocked": False}

    # 统计：~95% 在 Layer 1-2 决定（< 10ms）
    #       ~4% 在 Layer 3 决定（~100ms）
    #       ~1% 需要 Layer 4（~500ms）
```

### 评估数据集与基准

```python
safety_datasets = {
    "ToxiGen": {
        "规模": "274,000 条",
        "内容": "13 个少数群体的毒性/良性声明",
        "用途": "评估毒性检测模型的偏见覆盖",
    },
    "RealToxicityPrompts": {
        "规模": "100,000 条",
        "内容": "可能触发 LLM 生成有毒内容的 Prompt",
        "用途": "压力测试 LLM 的毒性倾向",
    },
    "Toxic-Chat": {
        "规模": "10,000+ 条",
        "内容": "真实对话中的毒性样本",
        "用途": "训练和评估对话场景的毒性检测",
    },
    "Jigsaw Toxic Comments": {
        "规模": "150,000+ 条",
        "内容": "多标签毒性分类（6 类）",
        "用途": "训练多类别毒性分类器",
    },
}
```

## 常见误区 / 面试追问

1. **误区："用最大的 LLM 做内容审核最好"** — 研究表明轻量 NN 分类器（38M 参数）在毒性分类的召回率上显著优于 8B LLM 的 in-context learning（0.96 vs 0.78）。对于明确定义的分类任务，专用小模型比通用大模型更快、更准、更便宜。LLM 的优势在于需要上下文理解的边界情况。

2. **误区："内容过滤只需要检查用户输入"** — Agent 系统需要在多个点过滤：用户输入、LLM 生成的中间输出、工具调用参数、最终输出。间接 Prompt Injection 可以通过检索到的外部内容注入有害内容，必须在输出端也做过滤。

3. **追问："如何处理多语言毒性检测？"** — (1) 使用多语言安全模型（Llama Guard 支持多语言）；(2) 对于不支持的语言，先翻译再检测；(3) 注意文化差异——同一表达在不同文化中毒性不同。这是一个活跃的研究领域。

4. **追问："如何衡量过滤器的效果？"** — 两个核心指标：**召回率**（真正有害的内容被拦截的比例——越高越安全）和**精确率**（被拦截的内容中真正有害的比例——越高误伤越少）。生产系统通常优先保证高召回率（宁可误拦不能漏放），同时通过多层级联降低误报率。

## 参考资料

- [AI Guardrails: Content Moderation and Safety with Open Language Models (Haystack)](https://haystack.deepset.ai/cookbook/safety_moderation_open_lms)
- [Content Moderation for GenAI: A New Layer of Defense (Lakera)](https://www.lakera.ai/blog/content-moderation)
- [Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation (arXiv)](https://arxiv.org/html/2508.05775v1)
- [Top 10 Open Datasets for LLM Safety, Toxicity & Bias Evaluation (Promptfoo)](https://www.promptfoo.dev/blog/top-llm-safety-bias-benchmarks/)
- [Innovative Guardrails for Generative AI: Designing an Intelligent Filter (MDPI)](https://www.mdpi.com/2076-3417/15/13/7298)
