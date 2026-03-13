# 如何检测和缓解 Agent 的幻觉（Hallucination）？

> 难度：中级
> 分类：Safety & Alignment

## 简短回答

幻觉（Hallucination）是 LLM 生成看似合理但事实错误的内容的现象，在 Agent 系统中尤其危险——因为 Agent 会基于幻觉内容做出**实际行动**（调用工具、修改数据、返回用户）。行业报告显示幻觉相关事故每年造成超过 2.5 亿美元损失。检测方法分三类：(1) **不确定性估计**——通过 logprob 分析、多次采样一致性检查识别低置信度输出；(2) **知识验证**——将 LLM 输出与外部知识库/搜索结果交叉验证（RAG 可减少 35-60% 幻觉）；(3) **自我一致性检查**——让 LLM 对同一问题生成多个回答，不一致的部分可能是幻觉（FINCH-ZK 方法）。缓解策略：**RAG 接地**（用检索事实约束生成）、**Multi-Agent 验证**（Guardian Agent 架构可将幻觉率降至 1% 以下）、**结构化输出约束**（限制自由生成的空间）、**人工闭环**（关键场景由人类确认事实）。Agent 特有的幻觉问题：即使工具返回了正确数据，Agent 在解读和总结时仍可能引入幻觉。

## 详细解析

### 幻觉的类型

```
LLM 幻觉分类：

├── 内在幻觉（Intrinsic Hallucination）
│   ├── 与输入矛盾：生成的内容与提供的上下文冲突
│   └── 示例：文档说"收入增长10%"，Agent 总结为"收入下降"
│
├── 外在幻觉（Extrinsic Hallucination）
│   ├── 凭空捏造：生成无法从输入或已知知识验证的内容
│   └── 示例：虚构不存在的论文、编造 API 参数
│
└── Agent 特有的幻觉
    ├── 工具输出误解：工具返回 A，Agent 总结成 B
    ├── 跨步骤信息丢失：多步推理中遗忘或歪曲之前的结果
    └── 虚构工具能力：Agent 声称工具可以做某事但实际不能
```

### 检测技术

```python
class HallucinationDetector:
    """幻觉检测系统"""

    # 方法 1：Log Probability 分析
    def logprob_detection(self, response, logprobs):
        """低 logprob 的 token 序列可能是幻觉"""
        suspicious_spans = []
        for i, (token, logprob) in enumerate(zip(response.tokens, logprobs)):
            if logprob < -5.0:  # 极低概率的 token
                suspicious_spans.append({
                    "position": i,
                    "token": token,
                    "logprob": logprob,
                    "confidence": "low",
                })
        return suspicious_spans

    # 方法 2：多次采样一致性检查
    async def consistency_check(self, question, n_samples=5):
        """多次生成，检查一致性"""
        responses = []
        for _ in range(n_samples):
            resp = await self.llm.invoke(question, temperature=0.7)
            responses.append(resp)

        # 提取关键声明
        claims_per_response = [self.extract_claims(r) for r in responses]

        # 检查声明一致性
        consistent_claims = []
        inconsistent_claims = []
        for claim in claims_per_response[0]:
            support_count = sum(
                1 for claims in claims_per_response[1:]
                if self.is_supported(claim, claims)
            )
            if support_count >= (n_samples - 1) * 0.6:
                consistent_claims.append(claim)
            else:
                inconsistent_claims.append(claim)

        return {
            "consistent": consistent_claims,
            "potentially_hallucinated": inconsistent_claims,
            "consistency_rate": len(consistent_claims) / max(len(claims_per_response[0]), 1),
        }

    # 方法 3：FINCH-ZK（零知识幻觉检测）
    async def finch_zk_detection(self, response):
        """分段检测 + 交叉一致性评估"""
        # 1. 将回答分成语义段落
        segments = self.segment_response(response)

        # 2. 对每个段落独立验证
        segment_scores = []
        for segment in segments:
            # 不依赖外部知识，只看内部一致性
            score = await self.cross_consistency_eval(segment, response)
            segment_scores.append({"text": segment, "score": score})

        # 3. 加权评分
        hallucination_blocks = [
            s for s in segment_scores if s["score"] < 0.5
        ]
        return {
            "overall_score": np.mean([s["score"] for s in segment_scores]),
            "hallucinated_blocks": hallucination_blocks,
        }
```

### 缓解策略

```python
class HallucinationMitigation:
    """幻觉缓解策略"""

    # 策略 1：RAG 接地（最有效的通用方案）
    async def rag_grounding(self, question):
        """用检索到的事实约束生成"""
        # 检索相关文档
        docs = await self.retriever.search(question, top_k=5)

        # 带接地约束的 Prompt
        prompt = f"""
        基于以下参考文档回答问题。
        规则：
        - 只使用参考文档中的信息
        - 如果文档中没有相关信息，明确说"根据现有资料无法回答"
        - 引用具体的文档段落作为依据

        参考文档：{docs}
        问题：{question}
        """
        return await self.llm.invoke(prompt)

    # 策略 2：Guardian Agent（多 Agent 验证）
    async def guardian_agent_verify(self, question, answer):
        """用独立的 Guardian Agent 验证回答"""
        # 1. 提取声明
        claims = await self.extract_claims(answer)

        # 2. 对每个声明独立验证
        verified_claims = []
        for claim in claims:
            # 搜索验证
            evidence = await self.web_search(claim)
            verification = await self.verify_claim(claim, evidence)

            verified_claims.append({
                "claim": claim,
                "verified": verification.is_true,
                "evidence": verification.evidence,
                "confidence": verification.confidence,
            })

        # 3. 如果有未验证的声明，修正回答
        unverified = [c for c in verified_claims if not c["verified"]]
        if unverified:
            corrected = await self.correct_answer(answer, unverified)
            return corrected
        return answer

    # 策略 3：结构化输出约束
    async def structured_constraint(self, question):
        """用结构化输出减少自由生成空间"""
        from pydantic import BaseModel

        class FactualAnswer(BaseModel):
            answer: str
            sources: list[str]     # 必须列出来源
            confidence: float      # 自评置信度
            caveats: list[str]     # 注意事项/不确定之处

        # 强制结构化输出减少随意编造
        return await self.llm.invoke(
            question,
            response_format=FactualAnswer,
        )

    # 策略 4：Agent 工具输出验证
    async def verify_tool_interpretation(self, tool_output, agent_summary):
        """验证 Agent 对工具输出的解读是否准确"""
        prompt = f"""
        工具返回了以下原始数据：
        {tool_output}

        Agent 将其总结为：
        {agent_summary}

        请检查：总结是否准确反映了原始数据？
        是否有遗漏、歪曲或添加了原始数据中没有的信息？
        输出 JSON：{{"accurate": true/false, "issues": [...]}}
        """
        return await self.verifier.invoke(prompt)
```

### 幻觉缓解效果对比

```
┌─────────────────────────┬────────────┬───────────┬──────────┐
│ 缓解策略                │ 幻觉减少率 │ 延迟影响  │ 成本影响 │
├─────────────────────────┼────────────┼───────────┼──────────┤
│ RAG 接地                │ 35-60%     │ +200ms    │ 低       │
│ Guardian Agent 验证     │ 80-99%     │ +2-5s     │ 高       │
│ 多次采样一致性          │ 40-60%     │ ×N 倍    │ ×N 倍   │
│ 结构化输出约束          │ 20-30%     │ 无        │ 无       │
│ CoT 推理               │ 15-30%     │ +50%      │ 中       │
│ RAG + NeMo Guardrails   │ 90-97%     │ +300ms    │ 中       │
│ 温度设为 0             │ 10-20%     │ 无        │ 无       │
└─────────────────────────┴────────────┴───────────┴──────────┘
```

## 常见误区 / 面试追问

1. **误区："用 RAG 就能消除幻觉"** — RAG 显著减少幻觉但不能消除。LLM 仍可能忽略检索到的文档而"自由发挥"，或错误解读文档内容。RAG 需要配合输出验证和 Faithfulness 评估（如 Ragas 的 Faithfulness 指标）。

2. **误区："Agent 调了工具就不会幻觉"** — Agent 在解读工具输出时仍然会引入幻觉。例如：数据库返回"Q3 收入 1200 万"，Agent 可能总结成"Q3 收入增长 20%"（增长率是 Agent 自己编的）。需要对工具输出的解读进行专门验证。

3. **追问："如何在成本和幻觉检测率之间取平衡？"** — 分级策略：(1) 所有输出用结构化约束 + 低温度（免费）；(2) 重要场景用 RAG 接地（低成本）；(3) 关键决策用 Guardian Agent 验证（高成本但最可靠）。按场景风险级别分配检测预算。

4. **追问："幻觉能被完全消除吗？"** — 从当前技术看，不能。幻觉是 LLM 生成机制的固有特性（基于概率采样而非事实检索）。但可以通过多层缓解将关键场景的幻觉率降到极低水平（< 1%）。长期方向：Guardian Agent 架构 + 神经符号混合方法。

## 参考资料

- [Mitigating Hallucination in LLMs: An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems (arXiv)](https://arxiv.org/html/2510.24476v1)
- [LLM Hallucination Detection and Mitigation: Best Techniques (Deepchecks)](https://deepchecks.com/llm-hallucination-detection-and-mitigation-best-techniques/)
- [Reducing Hallucinations with Custom Intervention Using Amazon Bedrock Agents (AWS)](https://aws.amazon.com/blogs/machine-learning/reducing-hallucinations-in-large-language-models-with-custom-intervention-using-amazon-bedrock-agents/)
- [Zero-Knowledge LLM Hallucination Detection and Mitigation (EMNLP 2025)](https://aclanthology.org/2025.emnlp-industry.139.pdf)
- [From Illusion to Insight: A Taxonomic Survey of Hallucination Mitigation Techniques (MDPI)](https://www.mdpi.com/2673-2688/6/10/260)
