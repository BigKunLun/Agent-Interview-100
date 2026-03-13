# 跨模型 Prompt 迁移：如何编写模型无关的 Prompt？

> 难度：高级
> 分类：Prompt Engineering

## 简短回答

跨模型 Prompt 迁移（Cross-Model Prompt Portability）是指让同一个 Prompt 在不同 LLM（GPT-4、Claude、Gemini、Llama 等）上都能有效工作的能力。现实中 Prompt 高度模型特异——为 GPT-4 精心调优的 Prompt 迁移到 Claude 可能效果骤降。核心挑战包括：(1) **Tokenization 差异**——不同模型的分词方式不同；(2) **指令偏好差异**——有的模型偏好详细指令，有的偏好简洁指令；(3) **格式偏好差异**——有的偏好 JSON，有的偏好 XML；(4) **能力边界差异**——推理能力、上下文长度、多语言能力不同。解决方案包括：编写**模型无关的核心 Prompt** + **模型特定的适配层**，或使用 **PromptBridge** 等自动化迁移工具。Stanford 的 PromptBridge 研究在多 Agent 场景下将迁移后的准确率损失控制在 5% 以内。

## 详细解析

### 为什么 Prompt 不可移植？

```python
# 同一个 Prompt，不同模型的表现差异
portability_challenges = {
    "格式偏好": {
        "GPT-4": "偏好 JSON 格式的工具参数",
        "Claude": "偏好 XML 标签分隔结构",
        "Gemini": "对 Markdown 结构响应最好",
        "Llama": "对简洁直接的指令响应最好",
    },
    "指令理解": {
        "问题": "同一句指令，不同模型的理解方式不同",
        "示例": "'简洁回答' → GPT-4 输出 2-3 句话，"
                "Claude 输出 1 段话，Llama 可能输出 1 个词",
    },
    "CoT 行为": {
        "问题": "有的模型自带 CoT，有的需要显式提示",
        "示例": "o1/o3 自动思考，GPT-4 需要 'Let's think step by step'",
    },
    "安全边界": {
        "问题": "不同模型的安全过滤阈值不同",
        "示例": "Claude 拒绝的请求 GPT-4 可能回答（反之亦然）",
    },
}
```

### 模型无关的 Prompt 设计原则

```python
model_agnostic_principles = {
    "使用清晰的自然语言": {
        "原则": "避免依赖特定模型的'魔法短语'",
        "❌": "Let's think step by step（只对特定模型有效）",
        "✓": "请分步骤分析这个问题，每步列出推理过程",
    },
    "显式化所有期望": {
        "原则": "不依赖模型的默认行为",
        "❌": "回答这个问题（依赖模型自己决定格式和长度）",
        "✓": "用 2-3 句话回答这个问题，使用中文，不加标题",
    },
    "结构化分隔": {
        "原则": "用通用分隔符而非模型特定标签",
        "✓ 通用": "--- 分隔线、Markdown 标题、编号列表",
        "⚠ 特定": "<thinking> 标签（Claude 特有）",
    },
    "避免极端依赖": {
        "原则": "不依赖最大上下文长度或特定能力",
        "✓": "Prompt 长度控制在 2000 token 以内（所有模型都支持）",
    },
}
```

### 适配层架构

```python
class PromptAdapter:
    """核心 Prompt + 模型特定适配层"""

    def __init__(self):
        # 核心 Prompt（模型无关）
        self.core_prompts = {
            "customer_service": """
            你是客服助手。
            职责：回答产品问题、处理退款、查询订单。
            规则：
            1. 使用友好专业的语言
            2. 不确定时建议联系人工
            3. 退款超过 500 元转人工

            用户消息：{user_input}
            """,
        }

        # 模型特定适配
        self.model_adapters = {
            "gpt-4": {
                "wrapper": "以 JSON 格式输出你的回复：\n"
                           '{{"response": "...", "action": "..."}}\n',
                "system_role": "system",
                "temperature": 0.7,
            },
            "claude": {
                "wrapper": "请在 <response> 标签中输出回复，"
                           "在 <action> 标签中输出操作。\n",
                "system_role": "system",  # Claude 也支持 system
                "temperature": 0.7,
            },
            "llama": {
                "wrapper": "回复格式：\n回复：...\n操作：...\n",
                "system_role": "system",
                "temperature": 0.6,  # 开源模型可能需要更低温度
            },
        }

    def get_prompt(self, prompt_id, model, **kwargs):
        core = self.core_prompts[prompt_id]
        adapter = self.model_adapters.get(model, self.model_adapters["gpt-4"])

        # 组合核心 Prompt 和适配层
        full_prompt = core.format(**kwargs) + "\n" + adapter["wrapper"]
        return full_prompt, adapter
```

### PromptBridge：自动化迁移

```python
class PromptBridge:
    """自动将 Prompt 从一个模型迁移到另一个模型"""

    async def migrate(self, prompt, source_model, target_model, eval_set):
        # Step 1: 分析源 Prompt 的意图和结构
        analysis = await self.analyze_prompt(prompt, source_model)

        # Step 2: 生成目标模型的候选适配
        candidates = await self.generate_adaptations(
            prompt=prompt,
            analysis=analysis,
            target_model=target_model,
            k=5
        )

        # Step 3: 在验证集上评估
        best_candidate = None
        best_score = 0
        for candidate in candidates:
            score = await self.evaluate(candidate, target_model, eval_set)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    async def analyze_prompt(self, prompt, model):
        """分析 Prompt 的核心意图（模型无关的表达）"""
        return await self.analyzer_llm.invoke(f"""
        分析以下针对 {model} 的 Prompt：
        {prompt}

        提取：
        1. 核心任务意图
        2. 输出格式要求
        3. 约束和规则
        4. 模型特定的技巧（如果有）
        5. 可以通用化的部分
        """)
```

### 迁移清单

```python
migration_checklist = {
    "格式迁移": [
        "JSON ↔ XML ↔ Markdown 格式转换",
        "特定标签替换（<thinking> → 通用的'推理过程'）",
        "函数调用格式适配",
    ],
    "指令迁移": [
        "调整详细程度（有的模型需要更详细/简洁的指令）",
        "添加/移除 CoT 提示",
        "调整 Few-shot 示例的数量和格式",
    ],
    "参数迁移": [
        "Temperature 调整",
        "Max tokens 适配",
        "Stop sequences 更新",
    ],
    "验证": [
        "在标准测试集上对比新旧模型的表现",
        "检查边界情况（极长输入、特殊字符、多语言）",
        "安全测试（Prompt Injection 防御是否有效）",
    ],
}
```

### 多模型 Agent 系统

```python
class MultiModelAgent:
    """支持多模型的 Agent 系统"""

    def __init__(self):
        self.models = {
            "planning": "claude-opus-4-6",      # 规划用强模型
            "execution": "gpt-4o-mini",          # 执行用快模型
            "evaluation": "claude-sonnet-4-6",   # 评估用中等模型
        }
        self.adapter = PromptAdapter()

    async def run(self, task):
        # 每个阶段使用不同模型，Prompt 自动适配
        plan_prompt = self.adapter.get_prompt(
            "planning", model=self.models["planning"], task=task
        )
        plan = await self.call(self.models["planning"], plan_prompt)

        exec_prompt = self.adapter.get_prompt(
            "execution", model=self.models["execution"], plan=plan
        )
        result = await self.call(self.models["execution"], exec_prompt)

        return result
```

## 常见误区 / 面试追问

1. **误区："好的 Prompt 在所有模型上都好用"** — 研究表明，为一个模型优化的 Prompt 迁移到另一个模型时，性能平均下降 10-30%。尤其是利用了模型特定行为的 Prompt（如 Claude 的 XML 标签、GPT 的 JSON mode），迁移后几乎必然失效。

2. **误区："只需要换 API endpoint 就能切换模型"** — API 格式只是表层差异。深层差异包括：模型对指令的理解方式、默认行为、安全边界、以及推理风格。真正的模型切换需要 Prompt 适配 + 验证测试。

3. **追问："为什么企业需要多模型支持？"** — 三个原因：(1) 避免供应商锁定——单一供应商的 API 中断会导致业务停滞；(2) 成本优化——不同任务用不同价位的模型；(3) 性能优化——某些任务在特定模型上效果更好。

4. **追问："如何在不手动迁移的情况下支持新模型？"** — 核心 Prompt + 适配层架构。核心 Prompt 用模型无关的自然语言编写，适配层处理格式和参数差异。新模型只需添加新的适配层。PromptBridge 等工具可以自动化这个过程。

## 参考资料

- [PromptBridge: Cross-Model Prompt Transfer for LLMs (arXiv)](https://arxiv.org/abs/2512.01420)
- [Model-Agnostic Prompts: Port Without Rewrites (Medium)](https://medium.com/@connect.hashblock/model-agnostic-prompts-port-without-rewrites-fb1144267bb6)
- [Model Agnostic Prompts: Future-Proof AI Applications (PromptLayer)](https://blog.promptlayer.com/model-agnostic/)
- [Cross-Model Prompting: Adapting Techniques for Different AI Systems (Qolaba)](https://blog.qolaba.ai/prompt-engineering-by-qolaba/cross-model-prompting-adapting-techniques-for-different-ai-systems/)
- [Key Considerations in Cross-Model Migration (DZone)](https://dzone.com/articles/key-considerations-in-cross-model-migration)
