# LLMOps 是什么？与传统 MLOps 有何区别？

> 难度：基础
> 分类：Production & Deployment

## 简短回答

LLMOps（Large Language Model Operations）是专门为 LLM 驱动的应用设计的运维实践体系，是 MLOps 在大语言模型时代的演进。与传统 MLOps 的核心区别：(1) **开发方式不同**——MLOps 以训练模型为核心（数据收集→特征工程→训练→部署），LLMOps 以使用模型为核心（Prompt 工程→RAG→评估→部署），大多数团队不需要训练模型；(2) **成本结构不同**——MLOps 的主要成本在训练阶段，LLMOps 的主要成本在**运行时推理**（按 token 计费），需要持续的成本优化；(3) **迭代速度不同**——MLOps 的迭代周期是周/月级（重新训练），LLMOps 的迭代是小时/天级（修改 Prompt 即生效）；(4) **核心产物不同**——MLOps 管理的是模型文件和特征管道，LLMOps 管理的是 **Prompt、向量索引、工具配置、护栏规则**等新型产物；(5) **评估方式不同**——MLOps 用精确指标（AUC、F1），LLMOps 需要 LLM-as-Judge 等语义评估。LLMOps 不替代 MLOps，而是在其基础上扩展，企业通常需要两者协同工作。

## 详细解析

### 核心差异对比

```
┌──────────────────┬──────────────────┬──────────────────┐
│ 维度             │ MLOps            │ LLMOps           │
├──────────────────┼──────────────────┼──────────────────┤
│ 核心模型         │ 自训练模型       │ 预训练 LLM（API）│
│ 数据类型         │ 结构化数据       │ 非结构化文本     │
│ 主要产物         │ 模型文件、特征   │ Prompt、向量索引 │
│ 迭代周期         │ 周/月级          │ 小时/天级        │
│ 主要成本         │ 训练（GPU 时间） │ 推理（API 调用） │
│ 评估方式         │ AUC/F1/MSE       │ LLM Judge/人工   │
│ 版本管理         │ 模型+数据版本    │ Prompt+配置版本  │
│ 可观测性         │ 模型漂移         │ Prompt 漂移+Trace│
│ 部署模式         │ 模型服务器       │ API Gateway      │
│ 安全关注         │ 数据隐私         │ Prompt Injection │
└──────────────────┴──────────────────┴──────────────────┘
```

### LLMOps 的核心组件

```python
llmops_components = {
    "Prompt 管理": {
        "内容": "Prompt 版本控制、A/B 测试、模板管理",
        "工具": "LangSmith, Humanloop, PromptLayer",
        "类比": "MLOps 中的特征工程",
    },
    "模型网关": {
        "内容": "API 路由、负载均衡、故障转移、成本控制",
        "工具": "LiteLLM, Portkey, Kong AI Gateway",
        "类比": "MLOps 中的模型服务器",
    },
    "RAG 基础设施": {
        "内容": "向量数据库、文档处理、索引更新",
        "工具": "Pinecone, Weaviate, Chroma",
        "类比": "MLOps 中的特征存储",
    },
    "评估管道": {
        "内容": "自动化评估、回归测试、LLM Judge",
        "工具": "DeepEval, Ragas, Braintrust",
        "类比": "MLOps 中的模型验证",
    },
    "可观测性": {
        "内容": "Trace/Span 追踪、成本监控、质量监控",
        "工具": "Langfuse, LangSmith, Arize Phoenix",
        "类比": "MLOps 中的模型监控",
    },
    "安全护栏": {
        "内容": "输入/输出过滤、PII 检测、内容安全",
        "工具": "Guardrails AI, NeMo Guardrails",
        "类比": "MLOps 中的数据验证（但范围更广）",
    },
}
```

### LLMOps 工作流

```python
class LLMOpsWorkflow:
    """LLMOps 的典型工作流"""

    def development_cycle(self):
        """开发迭代循环"""
        return [
            "1. Prompt 设计与迭代",
            "   - 编写/修改 System Prompt",
            "   - 在 Playground 中测试",
            "   - 版本化保存",

            "2. RAG 配置（如需要）",
            "   - 文档处理和分块",
            "   - 向量索引构建",
            "   - 检索策略调优",

            "3. 评估",
            "   - 在 Golden Dataset 上运行评估",
            "   - LLM Judge 自动评分",
            "   - 对比基线版本",

            "4. 部署",
            "   - Prompt 和配置推送到生产",
            "   - 灰度发布（5% → 25% → 100%）",
            "   - 实时监控",

            "5. 监控与优化",
            "   - 追踪质量指标和成本",
            "   - 收集用户反馈",
            "   - 识别优化机会",
        ]

    def key_metrics(self):
        """LLMOps 核心监控指标"""
        return {
            "质量": ["回答准确率", "用户满意度", "幻觉率"],
            "性能": ["延迟 P50/P95", "TTFT", "吞吐量"],
            "成本": ["每请求成本", "每用户日均成本", "Token 使用量"],
            "安全": ["护栏触发率", "注入检测率", "PII 泄露率"],
        }
```

### 企业协同：MLOps + LLMOps

```
实际企业 AI 系统中 MLOps 和 LLMOps 的协同：

保险行业示例：
├── MLOps 管理：
│   ├── 定价模型（结构化数据 → 风险评分）
│   ├── 欺诈检测模型（交易数据 → 欺诈概率）
│   └── 客户分群模型（行为数据 → 用户画像）
│
└── LLMOps 管理：
    ├── 智能客服（用户问题 → 自然语言回答）
    ├── 保单解释助手（保单文档 → RAG 回答）
    └── 理赔报告生成（结构化数据 → 文本报告）

两者共享：CI/CD 管道、监控基础设施、数据治理框架
```

## 常见误区 / 面试追问

1. **误区："LLMOps 就是 MLOps 加个 Prompt 管理"** — LLMOps 引入了全新的挑战维度：非确定性输出评估、Prompt Injection 安全、运行时成本控制、向量数据库管理等。这些不是简单地在 MLOps 上"加功能"，而是需要不同的思维方式和工具链。

2. **误区："用了 LLM API 就不需要 MLOps 了"** — 大多数企业的 AI 系统同时包含传统 ML 模型和 LLM 应用。推荐系统、风控模型仍然需要 MLOps。LLMOps 和 MLOps 是互补关系，不是替代关系。

3. **追问："LLMOps 的最大挑战是什么？"** — 评估。传统 ML 有明确的量化指标（准确率、F1），但 LLM 输出的质量是主观且多维度的。如何自动化、可靠地评估 LLM 输出质量是 LLMOps 的核心难题，也是 LLM-as-Judge 等技术兴起的原因。

4. **追问："小团队如何起步 LLMOps？"** — 最小可行方案：(1) Prompt 用 Git 版本管理；(2) 用 Langfuse（免费开源）记录所有请求和成本；(3) 维护 50 条 Golden Dataset 做回归测试；(4) 用 LiteLLM 统一多模型 API。这四步一周内可以搭好。

## 参考资料

- [MLOps vs LLMOps: What's the Difference? (ZenML)](https://www.zenml.io/blog/mlops-vs-llmops)
- [What is LLMOps Compared to MLOps (Pluralsight)](https://www.pluralsight.com/resources/blog/ai-and-data/what-is-llmops)
- [From MLOps to LLMOps: The Evolution of Automation (CircleCI)](https://circleci.com/blog/from-mlops-to-llmops/)
- [LLMOps vs MLOps: Key Differences and Evolution (Ideas2IT)](https://www.ideas2it.com/blogs/llmops-vs-mlops-key-differences-and-evolution)
- [What is LLMOps? Key Components & Differences to MLOps (lakeFS)](https://lakefs.io/blog/llmops/)
