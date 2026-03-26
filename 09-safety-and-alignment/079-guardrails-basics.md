# 什么是 Guardrails？如何为 Agent 设置安全护栏？

> 难度：基础
> 分类：Safety & Alignment

## 简短回答

Guardrails（安全护栏）是放置在 LLM/Agent 输入和输出之间的**规则和过滤器**，用于确保 AI 系统的安全性、合规性和可靠性。它们就像 Agent 和外部世界之间的"安检门"——在请求到达 LLM 之前检查输入是否安全（输入护栏），在 LLM 生成回答后检查输出是否合规（输出护栏）。护栏可以防御六大类威胁：**Prompt Injection**（恶意注入）、**数据泄露**（PII 暴露）、**Jailbreak**（绕过安全限制）、**偏见和毒性**（有害内容）、**幻觉**（虚假信息）、**隐私违规**。实现方式分为两种：**规则型**（正则、关键词匹配——快速低成本）和**模型型**（用 LLM/分类器语义理解——更智能但更慢更贵）。主流工具：**Guardrails AI**（开源验证框架）、**NVIDIA NeMo Guardrails**（状态机+可编程 rails）、**LangChain Middleware**（中间件模式）。最佳实践是组合使用规则型和模型型护栏，形成多层防御。

## 详细解析

### 护栏的工作位置

```
用户请求 → [输入护栏] → Prompt 构造 → [LLM] → [输出护栏] → 返回用户
                │                                    │
                ├── PII 检测与脱敏                    ├── 敏感信息过滤
                ├── Prompt Injection 检测             ├── 幻觉检测
                ├── 话题合规检查                      ├── 毒性检测
                └── 输入长度/格式验证                 └── 格式/结构验证

对于 Agent，还有：
LLM 输出 → [工具调用护栏] → 工具执行
                │
                ├── 工具权限检查
                ├── 参数合法性验证
                └── 危险操作审批
```

### 两种实现方式

```python
# 方式 1：规则型护栏（Rule-based）
class RuleBasedGuardrail:
    """基于规则的护栏——快速、可预测、低成本"""

    def __init__(self):
        self.pii_patterns = {
            "email": r'\b[\w.-]+@[\w.-]+\.\w+\b',
            "phone": r'\b\d{3}[-.]?\d{4}[-.]?\d{4}\b',
            "id_card": r'\b\d{17}[\dXx]\b',
        }
        self.blocked_keywords = ["密码是", "信用卡号", "DROP TABLE"]
        self.topic_whitelist = ["技术咨询", "产品使用", "账户管理"]

    def check_input(self, text):
        """输入检查"""
        # PII 检测
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                return {"blocked": True, "reason": f"包含 {pii_type}"}

        # 关键词过滤
        for keyword in self.blocked_keywords:
            if keyword in text:
                return {"blocked": True, "reason": f"包含敏感关键词"}

        return {"blocked": False}

    def check_output(self, text):
        """输出检查"""
        # 检查输出中是否泄露 PII
        for pii_type, pattern in self.pii_patterns.items():
            text = re.sub(pattern, f"[{pii_type}_REDACTED]", text)
        return text


# 方式 2：模型型护栏（Model-based）
class ModelBasedGuardrail:
    """基于模型的护栏——语义理解更强，但更慢更贵"""

    async def check_input(self, text):
        """用 LLM 判断输入是否安全"""
        result = await self.classifier.classify(
            text=text,
            categories=[
                "prompt_injection",
                "jailbreak_attempt",
                "off_topic",
                "toxic_content",
            ],
        )
        return {
            "blocked": any(r.score > 0.8 for r in result),
            "details": result,
        }

    async def check_output(self, question, answer):
        """用 LLM 检查输出质量"""
        prompt = f"""
        检查以下回答是否存在问题：
        问题：{question}
        回答：{answer}

        检查项：
        1. 是否包含虚假信息（幻觉）
        2. 是否包含有害或偏见内容
        3. 是否泄露了不应公开的信息
        4. 是否偏离了问题主题

        输出 JSON：{{"safe": true/false, "issues": [...]}}
        """
        return await self.judge.invoke(prompt)
```

### 主流工具对比

```python
# 1. Guardrails AI（开源验证框架）
guardrails_ai = {
    "核心能力": "Input/Output Guards，检测和量化特定风险",
    "特色": [
        "Guardrails Hub：预构建的验证器市场",
        "Guardrails Index：首个护栏性能基准（24 个护栏 × 6 类风险）",
        "支持自定义验证器",
    ],
    "使用示例": """
    from guardrails import Guard
    from guardrails.hub import ToxicLanguage, DetectPII

    guard = Guard().use_many(
        ToxicLanguage(threshold=0.8, on_fail="exception"),
        DetectPII(pii_entities=["EMAIL", "PHONE"], on_fail="fix"),
    )

    result = guard(
        llm_api=openai.chat.completions.create,
        prompt="用户问题...",
    )
    """,
}

# 2. NVIDIA NeMo Guardrails（状态机方法）
nemo_guardrails = {
    "核心能力": "基于 Colang 语言的可编程 rails",
    "特色": [
        "状态机驱动：定义对话流的合法路径",
        "多种 rail 类型：input/output/dialog/topical",
        "GPU 加速：低延迟性能",
        "框架集成：LangChain、LlamaIndex 等",
    ],
    "Colang 示例": """
    define user ask about competitors
        "你们和竞品 X 比怎么样？"
        "竞品 Y 好不好？"

    define bot refuse competitor comparison
        "我专注于帮助您了解我们的产品。"

    define flow
        user ask about competitors
        bot refuse competitor comparison
    """,
}

# 3. LangChain Middleware（中间件方法）
langchain_middleware = {
    "核心能力": "在 Agent 执行链路的关键点插入护栏",
    "拦截点": [
        "Agent 启动前（输入验证）",
        "模型调用前后（Prompt/输出检查）",
        "工具调用前后（权限和参数检查）",
        "Agent 完成后（最终输出验证）",
    ],
    "内置护栏": ["PII 检测", "Human-in-the-Loop", "输出格式验证"],
}
```

### Agent 专用护栏

```python
class AgentGuardrails:
    """Agent 特有的护栏——超越简单的输入输出过滤"""

    def tool_call_guard(self, tool_name, params, context):
        """工具调用护栏"""
        # 1. 权限检查：Agent 是否有权调用此工具
        if tool_name not in context.allowed_tools:
            return {"blocked": True, "reason": "未授权的工具"}

        # 2. 参数验证：参数是否合法且安全
        if tool_name == "database_query":
            if "DROP" in params["query"].upper():
                return {"blocked": True, "reason": "危险 SQL 操作"}

        # 3. 频率限制：防止循环调用或资源耗尽
        if self.call_count[tool_name] > self.rate_limits[tool_name]:
            return {"blocked": True, "reason": "调用频率超限"}

        # 4. 高风险操作需人工确认
        if tool_name in self.high_risk_tools:
            return {"needs_approval": True, "action": tool_name, "params": params}

        return {"blocked": False}

    def trajectory_guard(self, steps):
        """轨迹护栏——检测 Agent 的行为模式"""
        # 检测循环：Agent 是否在重复相同操作
        if self.detect_loop(steps):
            return {"blocked": True, "reason": "检测到循环行为"}

        # 检测偏离：Agent 是否偏离原始目标
        if self.detect_drift(steps):
            return {"blocked": True, "reason": "行为偏离原始目标"}

        # 成本检查：总成本是否超出预算
        total_cost = sum(s.cost for s in steps)
        if total_cost > self.budget_limit:
            return {"blocked": True, "reason": f"成本超限 ${total_cost}"}

        return {"blocked": False}
```

### 护栏设计最佳实践

```
┌─────────────────────┬──────────────────────────────────┐
│ 原则                │ 说明                             │
├─────────────────────┼──────────────────────────────────┤
│ 分层防御            │ 规则型 + 模型型组合，不依赖单层 │
│ 先规则后模型        │ 规则型快速拦截明显问题，         │
│                     │ 模型型处理需要语义理解的场景     │
│ 失败安全            │ 护栏出错时默认拒绝而非放行       │
│ 可观测              │ 记录所有拦截事件，用于分析和优化 │
│ 定期 Red Team       │ 定期测试护栏是否能被绕过         │
│ 持续更新            │ 根据新攻击手法更新规则和模型     │
│ 延迟权衡            │ 护栏增加的延迟要在可接受范围内   │
└─────────────────────┴──────────────────────────────────┘
```

## 常见误区 / 面试追问

1. **误区："护栏会让 Agent 变得很慢"** — 规则型护栏（正则、关键词）延迟通常 < 5ms，几乎无感。模型型护栏延迟较高（100-500ms），但可以异步执行（先返回结果，后台评估并在发现问题时追回）。关键是选择合适的护栏组合。

2. **误区："护栏只需要加在输入端"** — 输入护栏防不住间接注入（恶意指令隐藏在检索内容中）和模型自身的幻觉。必须同时有输出护栏。对于 Agent，还需要工具调用护栏和轨迹护栏。

3. **追问："如何评估护栏的有效性？"** — (1) 用 Guardrails AI Index 基准测试性能；(2) 计算两个关键指标：拦截率（真正拦截了多少攻击）和误报率（错误拦截了多少正常请求）；(3) 定期进行 Red Team 测试验证护栏的鲁棒性。

4. **追问："规则型和模型型如何搭配？"** — 推荐的组合模式：规则型做"第一道门"（快速拦截明确的违规：PII、SQL 注入、关键词黑名单），模型型做"第二道门"（处理需要语义理解的问题：Prompt Injection 变体、隐含毒性、话题偏离）。两层串联，规则型过滤掉 80% 的问题，模型型处理剩余 20% 的边界情况。

## 参考资料

- [LLM Guardrails: Best Practices for Deploying LLM Apps Securely (Datadog)](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [Mastering LLM Guardrails: Complete 2025 Guide (Orq.ai)](https://orq.ai/blog/llm-guardrails)
- [LLM Guardrails for Data Leakage, Prompt Injection, and More (Confident AI)](https://www.confident-ai.com/blog/llm-guardrails-the-ultimate-guide-to-safeguard-llm-systems)
- [Guardrails AI — Open Source Framework](https://guardrailsai.com/)
- [NVIDIA NeMo Guardrails](https://developer.nvidia.com/nemo-guardrails)
