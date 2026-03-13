# 评估工具对比：Ragas、LangSmith、Braintrust

> 难度：中级
> 分类：Evaluation

## 简短回答

LLM/Agent 评估工具分为两大类：**评估框架**（定义指标和运行评估，如 Ragas、DeepEval）和**评估平台**（提供完整的评估+监控+协作能力，如 LangSmith、Braintrust、Langfuse）。**Ragas** 专注于 RAG 评估，提供 Faithfulness、Answer Relevancy、Context Precision 等 RAG 专用指标；**LangSmith** 是 LangChain 生态的全栈平台，强在 Trace + 评估 + Playground 一体化；**Braintrust** 专注于评估和 Prompt 迭代，强在 A/B 测试和在线评估；**Langfuse** 是最流行的开源替代方案，支持 OTel 和多种框架。选择原则：RAG 专项评估 → Ragas；LangChain 生态 → LangSmith；框架无关 + 开源 → Langfuse；专业评估工作流 → Braintrust。

## 详细解析

### 工具全景分类

```
LLM 评估生态：
│
├── 评估框架（定义指标、运行评估）
│   ├── Ragas：RAG 专用评估
│   ├── DeepEval：通用 LLM 评估（开源）
│   ├── Promptfoo：Prompt 对比测试
│   └── Giskard：安全和偏差测试
│
├── 评估平台（评估 + 监控 + 协作）
│   ├── LangSmith：LangChain 全栈平台
│   ├── Braintrust：专业评估和迭代
│   ├── Langfuse：开源可观测性平台
│   ├── Arize Phoenix：ML/LLM 观测平台
│   └── Maxim AI：Agent 评估专注
│
└── 通用观测（可扩展到 LLM）
    ├── Datadog LLM Observability
    ├── New Relic AI Monitoring
    └── Grafana + OpenTelemetry
```

### Ragas：RAG 评估专家

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Ragas 的 RAG 专用指标
ragas_metrics = {
    "Faithfulness": {
        "含义": "回答是否基于检索到的上下文（非幻觉）",
        "计算": "提取回答中的声明 → 检查每个声明是否被上下文支持",
        "范围": "0-1，越高越好",
    },
    "Answer Relevancy": {
        "含义": "回答与问题的相关程度",
        "计算": "从回答反向生成问题 → 比较生成问题与原问题的相似度",
        "范围": "0-1",
    },
    "Context Precision": {
        "含义": "检索到的上下文中有多少是相关的",
        "计算": "在检索结果中，相关段落的排名越高分越高",
        "范围": "0-1",
    },
    "Context Recall": {
        "含义": "相关信息是否都被检索到了",
        "计算": "参考答案中的信息是否都能在检索上下文中找到",
        "范围": "0-1",
    },
}

# 使用示例
result = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=ChatOpenAI(model="gpt-4o"),
)
print(result)  # {'faithfulness': 0.85, 'answer_relevancy': 0.92, ...}
```

### LangSmith：LangChain 全栈平台

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# LangSmith 的核心能力
langsmith_features = {
    "Tracing": "自动追踪 LangChain/LangGraph 的每步执行",
    "Datasets": "创建和管理评估数据集",
    "Evaluators": "内置和自定义评估器",
    "Playground": "在线调试和测试 Prompt",
    "Monitoring": "生产环境的实时监控",
    "Annotation": "人工标注和反馈收集",
}

# 运行评估
def correctness_evaluator(run, example):
    """自定义评估器"""
    prediction = run.outputs["output"]
    reference = example.outputs["answer"]
    # 用 LLM 评估正确性
    score = llm_judge(prediction, reference)
    return {"score": score, "key": "correctness"}

results = evaluate(
    target=my_agent,
    data="my-eval-dataset",
    evaluators=[correctness_evaluator],
)
```

### Braintrust：专业评估和迭代

```python
import braintrust

# Braintrust 的核心能力
braintrust_features = {
    "Eval Framework": "声明式评估框架，简洁易用",
    "Online Evals": "生产流量的实时评估",
    "Prompt Playground": "Prompt 在线编辑和对比",
    "A/B Testing": "内置的 Prompt A/B 测试",
    "Logging": "自动记录所有 LLM 调用",
}

# 运行评估
@braintrust.traced
def my_task(input):
    return agent.invoke(input)

experiment = braintrust.Eval(
    "my-agent-eval",
    data=lambda: [
        {"input": "什么是 RAG？", "expected": "检索增强生成..."},
        # ...
    ],
    task=my_task,
    scores=[
        braintrust.Score.factuality,  # 内置评分器
        braintrust.Score.relevance,
    ],
)
```

### Langfuse：开源可观测性

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Langfuse 的核心能力
langfuse_features = {
    "开源": "完全开源，可自部署",
    "OTel 兼容": "支持 OpenTelemetry 标准",
    "框架无关": "支持 LangChain、LlamaIndex、OpenAI SDK 等",
    "Tracing": "完整的 Trace/Span 追踪",
    "Evaluations": "在线和离线评估",
    "Cost Tracking": "自动追踪 LLM 成本",
    "Prompt Management": "Prompt 版本管理",
}

# 使用示例
trace = langfuse.trace(name="agent-task")
span = trace.span(name="llm-call", input=prompt)
# ... LLM 调用 ...
span.end(output=response, metadata={"model": "gpt-4o"})

# 评估
trace.score(name="quality", value=0.85, comment="回答准确完整")
```

### 工具对比总结

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│ 维度         │ Ragas    │ LangSmith│ Braintrust│ Langfuse │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 类型         │ 评估框架 │ 全栈平台 │ 评估平台 │ 观测平台 │
│ 开源         │ ✓        │ ✗ 商业   │ ✗ 商业   │ ✓        │
│ RAG 评估     │ ★★★★★  │ ★★★     │ ★★★     │ ★★      │
│ Agent 评估   │ ★★      │ ★★★★   │ ★★★★   │ ★★★    │
│ Tracing      │ ✗        │ ★★★★★ │ ★★★     │ ★★★★  │
│ A/B 测试     │ ✗        │ ★★★     │ ★★★★★ │ ★★      │
│ Prompt 管理  │ ✗        │ ★★★★   │ ★★★★   │ ★★★    │
│ OTel 支持    │ ✗        │ 部分     │ 部分     │ ✓        │
│ 框架依赖     │ 无       │ LangChain│ 无       │ 无       │
│ 自部署       │ N/A      │ ✗        │ ✗        │ ✓        │
│ 定价         │ 免费     │ $$$      │ $$       │ 免费/$$  │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

### 选择决策树

```
你的场景是什么？
│
├── 主要做 RAG → Ragas（指标最全）+ Langfuse（追踪）
│
├── 使用 LangChain/LangGraph → LangSmith（最佳集成）
│
├── 需要开源 + 自部署 → Langfuse
│
├── 需要专业的 Prompt A/B 测试 → Braintrust
│
├── 需要企业级 + 已有 Datadog → Datadog LLM Observability
│
└── 预算有限 + 快速开始 → Langfuse（开源）+ DeepEval（评估）
```

## 常见误区 / 面试追问

1. **误区："选一个工具就够了"** — 评估框架（Ragas）和评估平台（LangSmith/Langfuse）是互补的。Ragas 定义 RAG 评估的指标，LangSmith/Langfuse 提供运行评估和追踪的基础设施。生产系统通常需要两者结合。

2. **误区："商业平台一定比开源好"** — Langfuse 作为开源方案在很多场景下已足够好，且可以自部署保证数据安全。商业平台的优势在于企业级支持、更丰富的 UI 和更好的协作功能。

3. **追问："这些工具如何处理数据隐私？"** — 评估工具会记录 LLM 的输入输出，可能包含敏感数据。解决方案：(1) 自部署 Langfuse 保证数据不出组织；(2) 配置 PII 脱敏规则；(3) 只在开发环境记录完整数据，生产环境只记录指标。

4. **追问："如何从零开始搭建评估体系？"** — 三步走：(1) 先用 Langfuse/LangSmith 加 Tracing，了解 Agent 的行为；(2) 构建 Golden Dataset 并用 DeepEval/Ragas 定义指标；(3) 集成到 CI/CD 实现自动化回归测试。

## 参考资料

- [Best LLM Evaluation Tools: 7 Platforms for 2026 (Rhesis AI)](https://rhesis.ai/post/best-llm-evaluation-testing-tools)
- [Comparing LLM Evaluation Platforms: Top Frameworks (Arize AI)](https://arize.com/llm-evaluation-platforms-top-frameworks/)
- [LangWatch vs LangSmith vs Braintrust vs Langfuse (LangWatch)](https://langwatch.ai/blog/langwatch-vs-langsmith-vs-braintrust-vs-langfuse-choosing-the-best-llm-evaluation-monitoring-tool-in-2025)
- [The 5 Best RAG Evaluation Tools (Braintrust)](https://www.braintrust.dev/articles/best-rag-evaluation-tools)
- [Top LLM Observability Platforms 2025 (Agenta)](https://agenta.ai/blog/top-llm-observability-platforms)
