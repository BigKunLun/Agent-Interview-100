<p align="center">
  <h1 align="center">🤖 Agent Interview 100</h1>
  <p align="center"><strong>AI Agent 面试知识库 — 100 篇精选面试题，助你拿下 Agent 方向 Offer</strong></p>
  <p align="center">
    <img src="https://img.shields.io/badge/题目数量-108-blue?style=flat-square" />
    <img src="https://img.shields.io/badge/主题模块-11-green?style=flat-square" />
    <img src="https://img.shields.io/badge/难度覆盖-基础 · 中级 · 高级-orange?style=flat-square" />
    <img src="https://img.shields.io/badge/语言-中文-red?style=flat-square" />
    <img src="https://img.shields.io/badge/许可证-MIT-purple?style=flat-square" />
  </p>
</p>

---

## ✨ 项目亮点

- 📚 **108 篇精选面试题** — 覆盖 AI Agent 开发的方方面面，从基础概念到生产实践，包含场景故障题、系统设计题、代码 Review 题、量化计算题、面试追问链等多种题型
- 🏗️ **11 大主题模块** — 系统化组织知识体系，循序渐进
- 🎯 **三级难度分布** — 基础 (30%) · 中级 (50%) · 高级 (20%)，适配不同阶段候选人
- 📝 **统一四段式结构** — 每篇文章均包含：简短回答 → 详细解析 → 常见误区/面试追问 → 参考资料
- 💻 **丰富的代码示例** — Python 实现、伪代码、架构图，理论与实践结合
- 🔬 **前沿研究引用** — 引用最新论文与行业实践，紧跟技术前沿（含 Proactive Agent、RAG 评估指标等 2025 最新内容）

---

## 📖 目录结构

### 一、🏛️ Agent 架构 (`01-agent-architecture/`) — 12 题

Agent 系统的核心设计理念，从基本概念到生产级架构设计。

| # | 文章 |
|---|------|
| 001 | [什么是 LLM Agent？与传统 LLM 应用有何区别？](01-agent-architecture/001-what-is-llm-agent.md) |
| 002 | [解释 Agent 的核心组件：感知、推理、行动、记忆](01-agent-architecture/002-agent-core-components.md) |
| 003 | [ReAct 模式是什么？它如何结合推理与行动？](01-agent-architecture/003-react-pattern.md) |
| 004 | [比较不同的 Agent 架构模式：ReAct、Plan-and-Execute、LATS、Proactive](01-agent-architecture/004-agent-architecture-patterns.md) |
| 005 | [如何设计一个分层 Agent 架构（Orchestrator / Worker 模式）？](01-agent-architecture/005-layered-agent-architecture.md) |
| 006 | [Agent Loop 的设计：如何处理循环终止、最大步数限制？](01-agent-architecture/006-agent-loop-design.md) |
| 007 | [Workflow vs Agent：什么时候用确定性工作流，什么时候用自主 Agent？](01-agent-architecture/007-workflow-vs-agent.md) |
| 008 | [如何设计 Agent 的错误恢复与重试机制？](01-agent-architecture/008-error-recovery-retry.md) |
| 009 | [如何实现 Agent 的自我反思（Self-Reflection）和自我纠正？](01-agent-architecture/009-self-reflection-correction.md) |
| 010 | [设计一个生产级 Agent 系统：从单轮对话到多步任务执行](01-agent-architecture/010-production-agent-system-design.md) |
| 106 | [🆕 系统设计题：设计一个日均 10 万对话的智能客服 Agent 系统](01-agent-architecture/106-agent-system-design-customer-service.md) |
| 108 | [🆕 面试追问链：从「什么是 Agent」到系统设计的 10 层递进追问](01-agent-architecture/108-interview-deep-dive-chain.md) |

### 二、🔍 RAG (`02-rag/`) — 10 题

检索增强生成的全链路知识，从文档处理到评估优化（含 Context Relevancy、MRR、MAP 等评估指标）。

| # | 文章 |
|---|------|
| 011 | [什么是 RAG？它解决了 LLM 的哪些核心局限？](02-rag/011-what-is-rag.md) |
| 012 | [RAG Pipeline 的核心组件：Indexing、Retrieval、Generation](02-rag/012-rag-pipeline-components.md) |
| 013 | [文档分块（Chunking）策略有哪些？各有什么优缺点？](02-rag/013-chunking-strategies.md) |
| 014 | [向量数据库选型：Pinecone vs Weaviate vs Chroma vs Milvus](02-rag/014-vector-database-comparison.md) |
| 015 | [Embedding 模型选择与微调策略](02-rag/015-embedding-model-selection.md) |
| 016 | [混合检索：如何结合语义检索和关键词检索？](02-rag/016-hybrid-retrieval.md) |
| 017 | [Re-ranking 的原理与实现：Cross-Encoder vs Bi-Encoder](02-rag/017-reranking-strategies.md) |
| 018 | [什么是 Agentic RAG？它与传统 RAG 有何不同？](02-rag/018-agentic-rag.md) |
| 019 | [高级 RAG 变体：Corrective RAG、Self-RAG、Adaptive RAG](02-rag/019-advanced-rag-variants.md) |
| 020 | [如何评估 RAG 系统的性能？关键指标与评估框架](02-rag/020-rag-evaluation-metrics.md) |

### 三、🔧 工具使用 (`03-tool-use/`) — 10 题

LLM 与外部工具的交互，从 Function Calling 到安全管控。

| # | 文章 |
|---|------|
| 021 | [什么是 Function Calling？它是如何工作的？](03-tool-use/021-function-calling-basics.md) |
| 022 | [如何为 LLM 定义和描述工具（Tool Schema）？](03-tool-use/022-tool-schema-design.md) |
| 023 | [Tool Use 的常见模式：API 调用、数据库查询、代码执行](03-tool-use/023-common-tool-patterns.md) |
| 024 | [如何设计 Tool Gateway 和工具权限管理？](03-tool-use/024-tool-gateway-permissions.md) |
| 025 | [工具选择策略：LLM 如何决定使用哪个工具？](03-tool-use/025-tool-selection-strategy.md) |
| 026 | [如何处理工具调用失败和超时？](03-tool-use/026-tool-failure-handling.md) |
| 027 | [MCP（Model Context Protocol）是什么？它如何标准化工具集成？](03-tool-use/027-model-context-protocol.md) |
| 028 | [并行工具调用 vs 顺序工具调用的设计考量](03-tool-use/028-parallel-vs-sequential-tools.md) |
| 029 | [如何实现动态工具发现和注册？](03-tool-use/029-dynamic-tool-discovery.md) |
| 030 | [工具使用的安全性：防止注入攻击和越权操作](03-tool-use/030-tool-use-security.md) |

### 四、🤝 多 Agent (`04-multi-agent/`) — 10 题

多 Agent 协作系统的设计、编排、调试与场景故障处理。

| # | 文章 |
|---|------|
| 031 | [什么是多 Agent 系统？与单 Agent 相比有何优势？](04-multi-agent/031-what-is-multi-agent.md) |
| 032 | [多 Agent 通信模式：消息传递、共享状态、黑板模式](04-multi-agent/032-communication-patterns.md) |
| 033 | [Agent 编排模式：Hub-Spoke、Pipeline、Hierarchical](04-multi-agent/033-orchestration-patterns.md) |
| 034 | [如何设计 Agent 间的任务分配与协调？](04-multi-agent/034-task-allocation-coordination.md) |
| 035 | [多 Agent 系统中的冲突解决机制](04-multi-agent/035-conflict-resolution.md) |
| 036 | [比较主流多 Agent 框架：CrewAI、AutoGen、LangGraph](04-multi-agent/036-multi-agent-frameworks.md) |
| 037 | [如何实现 Agent 间的 Handoff（任务交接）？](04-multi-agent/037-agent-handoff.md) |
| 038 | [多 Agent 系统中的涌现行为与可控性](04-multi-agent/038-emergent-behavior.md) |
| 039 | [如何调试和监控多 Agent 系统？](04-multi-agent/039-debugging-monitoring-multi-agent.md) |
| 101 | [🆕 A2A（Agent-to-Agent）协议是什么？它与 MCP 有何区别？](04-multi-agent/101-a2a-protocol.md) |

### 五、🧠 记忆与状态 (`05-memory-and-state/`) — 9 题

Agent 的记忆管理，从上下文窗口到知识图谱。

| # | 文章 |
|---|------|
| 040 | [Agent 记忆的类型：短期记忆、长期记忆、工作记忆](05-memory-and-state/040-memory-types.md) |
| 041 | [如何管理对话上下文窗口？](05-memory-and-state/041-context-window-management.md) |
| 042 | [对话历史的摘要与压缩策略](05-memory-and-state/042-conversation-summarization.md) |
| 043 | [如何实现 Agent 的持久化记忆（Persistent Memory）？](05-memory-and-state/043-persistent-memory.md) |
| 044 | [状态管理在 Agent 系统中的设计模式](05-memory-and-state/044-state-management-patterns.md) |
| 045 | [如何实现跨会话的用户偏好学习？](05-memory-and-state/045-cross-session-preferences.md) |
| 046 | [向量记忆 vs 结构化记忆：何时使用哪种？](05-memory-and-state/046-vector-vs-structured-memory.md) |
| 047 | [设计 Agent 的知识图谱记忆系统](05-memory-and-state/047-knowledge-graph-memory.md) |
| 048 | [记忆的遗忘与更新机制：如何处理过时信息？](05-memory-and-state/048-memory-forgetting-updating.md) |

### 六、🧩 规划与推理 (`06-planning-and-reasoning/`) — 11 题

LLM 的推理能力增强与任务规划策略。

| # | 文章 |
|---|------|
| 049 | [Chain-of-Thought (CoT) 推理是什么？为什么有效？](06-planning-and-reasoning/049-chain-of-thought.md) |
| 050 | [任务分解（Task Decomposition）的基本方法](06-planning-and-reasoning/050-task-decomposition.md) |
| 051 | [Tree-of-Thought (ToT) vs Chain-of-Thought：何时使用哪种？](06-planning-and-reasoning/051-tot-vs-cot.md) |
| 052 | [Plan-and-Solve 策略：先规划再执行](06-planning-and-reasoning/052-plan-and-solve.md) |
| 053 | [LLM 作为规划器的局限性与缓解方案](06-planning-and-reasoning/053-llm-planning-limitations.md) |
| 054 | [如何实现动态重规划（Replanning）？](06-planning-and-reasoning/054-dynamic-replanning.md) |
| 055 | [Reasoning 模型（o1/o3/DeepSeek-R1）vs 标准模型：架构差异与适用场景](06-planning-and-reasoning/055-reasoning-models.md) |
| 056 | [Monte Carlo Tree Search 在 Agent 规划中的应用](06-planning-and-reasoning/056-mcts-in-agent-planning.md) |
| 057 | [如何评估 Agent 的推理质量？](06-planning-and-reasoning/057-reasoning-quality-evaluation.md) |
| 058 | [因果推理在 Agent 决策中的作用](06-planning-and-reasoning/058-causal-reasoning.md) |
| 103 | [🆕 Agentic-RL 是什么？如何用 GRPO 训练 Agent 的决策能力？](06-planning-and-reasoning/103-agentic-rl-grpo.md) |

### 七、✍️ 提示工程 (`07-prompt-engineering/`) — 11 题

Prompt 设计、优化与管理的最佳实践。

| # | 文章 |
|---|------|
| 059 | [System Prompt 设计的核心原则](07-prompt-engineering/059-system-prompt-principles.md) |
| 060 | [Few-Shot vs Zero-Shot Prompting：如何选择？](07-prompt-engineering/060-few-shot-vs-zero-shot.md) |
| 061 | [结构化输出（Structured Output）：如何让 LLM 返回 JSON/XML？](07-prompt-engineering/061-structured-output.md) |
| 062 | [Agentic Prompting：如何编写让 LLM 自主执行任务的 Prompt？](07-prompt-engineering/062-agentic-prompting.md) |
| 063 | [Prompt Chaining：多步骤 Prompt 的设计与编排](07-prompt-engineering/063-prompt-chaining.md) |
| 064 | [如何防止 Prompt Injection 攻击？](07-prompt-engineering/064-prompt-injection-defense.md) |
| 065 | [DSPy 等编程化 Prompt 优化工具的原理与使用](07-prompt-engineering/065-dspy-programmatic-prompting.md) |
| 066 | [Prompt 版本管理与 A/B 测试](07-prompt-engineering/066-prompt-versioning-ab-testing.md) |
| 067 | [Meta-Prompting：让 LLM 自动生成和优化 Prompt](07-prompt-engineering/067-meta-prompting.md) |
| 068 | [跨模型 Prompt 迁移：如何编写模型无关的 Prompt？](07-prompt-engineering/068-cross-model-prompt-portability.md) |
| 102 | [🆕 什么是 Context Engineering？它与 Prompt Engineering 有何本质区别？](07-prompt-engineering/102-context-engineering.md) |

### 八、📊 评估 (`08-evaluation/`) — 10 题

Agent 系统的评估方法、基准测试、可观测性与 RAG 评估指标。

| # | 文章 |
|---|------|
| 069 | [LLM 评估的基本方法：自动评估 vs 人工评估](08-evaluation/069-auto-vs-human-evaluation.md) |
| 070 | [Agent 评估与 LLM 评估有何不同？](08-evaluation/070-agent-vs-llm-evaluation.md) |
| 071 | [LLM-as-Judge：使用 LLM 评估 LLM 输出](08-evaluation/071-llm-as-judge.md) |
| 072 | [Agent Benchmark：如何设计端到端的 Agent 测试？](08-evaluation/072-agent-benchmarks.md) |
| 073 | [回归测试：如何检测 Agent 性能退化？](08-evaluation/073-regression-testing.md) |
| 074 | [Trace 和 Span：Agent 执行的可观测性](08-evaluation/074-traces-and-spans.md) |
| 075 | [评估工具对比：Ragas、LangSmith、Braintrust](08-evaluation/075-evaluation-tools-comparison.md) |
| 076 | [静态 Benchmark 的陷阱：为什么 95% 准确率在生产中会失效？](08-evaluation/076-static-benchmark-trap.md) |
| 077 | [如何构建持续评估（Continuous Evaluation）流水线？](08-evaluation/077-continuous-evaluation-pipeline.md) |
| 105 | [🆕 量化计算题：手算 RAG 评估指标（Precision@K、MRR、NDCG、Context Recall）](08-evaluation/105-rag-metrics-calculation.md) |

### 九、🛡️ 安全与对齐 (`09-safety-and-alignment/`) — 8 题

Agent 系统的安全风险防控与对齐策略。

| # | 文章 |
|---|------|
| 078 | [LLM Agent 的主要安全风险有哪些？](09-safety-and-alignment/078-agent-safety-risks.md) |
| 079 | [什么是 Guardrails？如何为 Agent 设置安全护栏？](09-safety-and-alignment/079-guardrails-basics.md) |
| 080 | [Human-in-the-Loop：何时以及如何引入人工审核？](09-safety-and-alignment/080-human-in-the-loop.md) |
| 081 | [Agent 的权限最小化原则与沙箱执行](09-safety-and-alignment/081-least-privilege-sandboxing.md) |
| 082 | [如何检测和缓解 Agent 的幻觉（Hallucination）？](09-safety-and-alignment/082-hallucination-detection.md) |
| 083 | [内容过滤与毒性检测在 Agent 系统中的实现](09-safety-and-alignment/083-content-filtering-toxicity.md) |
| 084 | [Agent 对齐问题：如何确保 Agent 行为符合人类意图？](09-safety-and-alignment/084-agent-alignment.md) |
| 085 | [Red Teaming：如何对 Agent 系统进行对抗测试？](09-safety-and-alignment/085-red-teaming-agents.md) |

### 十、🚀 生产部署 (`10-production-and-deployment/`) — 12 题

从开发到生产的全链路工程实践。

| # | 文章 |
|---|------|
| 086 | [LLMOps 是什么？与传统 MLOps 有何区别？](10-production-and-deployment/086-llmops-basics.md) |
| 087 | [Agent 系统的基本部署架构](10-production-and-deployment/087-deployment-architecture.md) |
| 088 | [LLM API 的成本优化策略](10-production-and-deployment/088-cost-optimization.md) |
| 089 | [模型路由（Model Routing）：如何根据任务复杂度选择模型？](10-production-and-deployment/089-model-routing.md) |
| 090 | [Agent 系统的延迟优化：Streaming、缓存、批处理](10-production-and-deployment/090-latency-optimization.md) |
| 091 | [Prompt Drift 管理：如何避免 Prompt 退化？](10-production-and-deployment/091-prompt-drift-management.md) |
| 092 | [Agent 系统的日志、监控与告警设计](10-production-and-deployment/092-logging-monitoring-alerting.md) |
| 093 | [如何实现 Agent 的灰度发布和 A/B 测试？](10-production-and-deployment/093-canary-ab-testing.md) |
| 094 | [高并发场景下的 Agent 系统扩展策略](10-production-and-deployment/094-scaling-strategies.md) |
| 095 | [Agent 系统的灾难恢复与高可用设计](10-production-and-deployment/095-disaster-recovery-ha.md) |
| 104 | [🆕 场景题：你的 Agent 在生产环境出了故障，如何系统性排查和修复？](10-production-and-deployment/104-agent-production-troubleshooting.md) |
| 107 | [🆕 代码 Review 题：找出这段 Agent 代码中的设计问题并修复](10-production-and-deployment/107-agent-code-review.md) |

### 十一、🧰 框架选型 (`11-frameworks/`) — 5 题

主流 Agent 框架的对比分析与自研决策。

| # | 文章 |
|---|------|
| 096 | [主流 Agent 框架概览：LangChain、LlamaIndex、Haystack](11-frameworks/096-framework-overview.md) |
| 097 | [LangGraph 的核心概念：节点、边、状态](11-frameworks/097-langgraph-concepts.md) |
| 098 | [框架 vs 自研：什么时候应该自己构建 Agent 框架？](11-frameworks/098-framework-vs-custom.md) |
| 099 | [OpenAI Assistants API vs Anthropic Claude Agent SDK 对比](11-frameworks/099-assistants-api-vs-claude-sdk.md) |
| 100 | [如何设计可测试、可扩展的 Agent 框架抽象层？](11-frameworks/100-testable-extensible-framework.md) |

---

## 📋 使用说明

### 💡 推荐学习方式

- **先答后看**：先尝试自己回答问题，再对照文章查漏补缺
- **关注追问**：每篇文章末尾的「面试追问」是高频考点，务必准备
- **动手实践**：文章中的代码示例建议实际运行，配合上方「实践项目推荐」动手练习
- **横向关联**：注意不同模块间的知识联系（如 RAG + 评估、安全 + 工具使用）
- **新题型优先**：场景题、系统设计题、代码 Review 题更接近真实面试，建议重点准备

---

## 🏷️ 模块概览

| 模块 | 题数 | 简介 |
|------|------|------|
| 🏛️ Agent 架构 | 12 | Agent 定义、架构模式、循环设计、生产级系统设计、**系统设计题**、**面试追问链** |
| 🔍 RAG | 10 | 检索增强生成全流程：文档分块、向量数据库、混合检索、Re-ranking、Agentic RAG、评估指标 |
| 🔧 工具使用 | 10 | Function Calling、Tool Schema、MCP 协议、工具安全与动态发现 |
| 🤝 多 Agent | 10 | 多 Agent 通信与编排模式、任务协调、Handoff、涌现行为、**A2A 协议** |
| 🧠 记忆与状态 | 9 | 短期/长期/工作记忆、上下文窗口管理、持久化记忆、知识图谱记忆 |
| 🧩 规划与推理 | 11 | CoT/ToT 推理、任务分解、动态重规划、MCTS、因果推理、**Agentic-RL/GRPO** |
| ✍️ 提示工程 | 11 | System Prompt 设计、结构化输出、Prompt Injection 防御、DSPy、**Context Engineering** |
| 📊 评估 | 10 | 自动/人工评估、LLM-as-Judge、Benchmark 设计、可观测性、**RAG 指标量化计算** |
| 🛡️ 安全与对齐 | 8 | 安全风险、Guardrails、权限最小化、幻觉检测、Red Teaming |
| 🚀 生产部署 | 12 | LLMOps、成本优化、模型路由、延迟优化、**场景故障排查**、**代码 Review** |
| 🧰 框架选型 | 5 | LangChain/LlamaIndex/LangGraph 对比、框架 vs 自研决策、可扩展架构设计 |

---

## 🆕 新题型说明

本项目在传统问答题的基础上，引入了 **5 种创新题型**，模拟真实面试的多种考察方式：

| 题型 | 代表题目 | 考察重点 |
|------|---------|---------|
| 🔥 **场景故障排查** | [#104](10-production-and-deployment/104-agent-production-troubleshooting.md) | 给出生产故障场景，要求排查根因并修复。考察工程经验和系统性思维 |
| 📐 **量化计算** | [#105](08-evaluation/105-rag-metrics-calculation.md) | 手算 Precision@K、MRR、NDCG 等指标。考察对评估体系的深度理解 |
| 🏗️ **系统设计** | [#106](01-agent-architecture/106-agent-system-design-customer-service.md) | 从零设计完整的 Agent 系统。考察架构能力和工程全局观 |
| 🔍 **代码 Review** | [#107](10-production-and-deployment/107-agent-code-review.md) | 审查有问题的 Agent 代码并修复。考察代码质量意识和最佳实践 |
| 🔗 **面试追问链** | [#108](01-agent-architecture/108-interview-deep-dive-chain.md) | 10 层递进追问，从基础到系统设计。模拟真实面试的深度挖掘过程 |

---

## 🗺️ 推荐学习路径

### 按岗位推荐

#### 🛠️ Agent 应用工程师（偏工程实现）

> 核心能力：能独立开发和部署 Agent 应用

```
Week 1: 基础概念
001 → 002 → 003 → 021 → 022 → 040 → 041

Week 2: 核心技能
006 → 027 → 028 → 062 → 063 → 044 → 059

Week 3: 生产实践
086 → 087 → 090 → 092 → 026 → 104（场景题）→ 107（代码 Review）

Week 4: 进阶提升
008 → 009 → 037 → 074 → 094 → 095 → 108（追问链）
```

#### 🔍 RAG 工程师（偏检索增强）

> 核心能力：设计和优化 RAG 系统

```
Week 1: RAG 基础
011 → 012 → 013 → 014 → 015

Week 2: RAG 进阶
016 → 017 → 018 → 019 → 061

Week 3: RAG 评估与优化
020 → 069 → 071 → 075 → 105（量化计算题）

Week 4: 生产化
076 → 077 → 090 → 088 → 102（Context Engineering）
```

#### 🏗️ AI 架构师（偏系统设计）

> 核心能力：设计大规模 Agent 系统架构

```
Week 1: 架构基础
001 → 004 → 005 → 007 → 010

Week 2: 多 Agent 与编排
031 → 033 → 034 → 036 → 037 → 101（A2A 协议）

Week 3: 生产架构
087 → 089 → 093 → 094 → 095 → 106（系统设计题）

Week 4: 前沿技术
055 → 103（Agentic-RL）→ 098 → 100 → 108（追问链）
```

#### 🛡️ AI 安全工程师（偏安全对齐）

> 核心能力：保障 Agent 系统的安全性

```
Week 1-2: 安全基础
078 → 079 → 080 → 081 → 082 → 083 → 084 → 085

Week 3: 安全实践
030 → 064 → 024 → 104（场景题 - 含危险操作场景）

Week 4: 评估与监控
070 → 073 → 074 → 076 → 092 → 107（代码 Review - 含安全审查）
```

### 按难度推荐

- **入门阶段**（1-2 周）：先通读所有「基础」难度题目（30 篇），建立知识全景
- **进阶阶段**（2-3 周）：深入「中级」题目（54 篇，含新增），掌握核心技术细节
- **冲刺阶段**（1-2 周）：攻克「高级」题目（24 篇，含新增），构建系统设计能力

---

## 🔨 实践项目推荐

配合面试题学习，建议动手实践以下 mini-project：

| 项目 | 描述 | 涉及面试题 |
|------|------|-----------|
| **tiny-agent-loop** | 从零实现一个最小 Agent 循环（50 行 Python），包含 Tool Calling、循环终止、错误处理 | #001 #006 #008 #021 |
| **tiny-rag** | 搭建简单 RAG 系统，支持文档分块、向量检索、Re-ranking | #011 #012 #013 #016 #017 |
| **tiny-multi-agent** | 实现两个 Agent 的协作对话（Router + Worker 模式），体验 Handoff | #031 #033 #037 #101 |
| **tiny-eval** | 用 Ragas 框架评估你的 RAG 系统，手算并验证评估指标 | #020 #075 #105 |
| **tiny-mcp-server** | 实现一个 MCP Server，让 Agent 调用自定义工具 | #027 #029 #024 |

> 💡 参考项目：[wdndev/tiny-rag](https://github.com/wdndev/llm_interview_note)、[datawhalechina/hello-agents](https://github.com/datawhalechina/hello-agents)

---

## 🌟 内容特色

每篇文章均采用 **统一的四段式结构**，确保学习效率最大化：

```
📌 简短回答        →  30 秒快速回答面试官，展现核心理解
📖 详细解析        →  深入原理、架构图、代码示例，构建完整认知
⚠️ 常见误区/面试追问 →  避坑指南 + 高频追问，模拟真实面试
📚 参考资料        →  论文、博客、官方文档，便于深入研究
```

**其他特色：**
- 🔗 文章间交叉引用，形成知识网络
- 📊 难度标注（基础/中级/高级），便于制定学习计划
- 🆕 引用最新研究成果与行业实践（2024-2025）

---

## 👥 适用人群

- 🎯 **求职者** — 准备 AI Agent / LLM 工程师面试的候选人
- 🏗️ **架构师** — 设计和构建 Agent 系统的技术负责人
- 🔬 **研究者** — 关注 Agent 技术前沿的研究人员
- 📖 **学习者** — 希望系统了解 AI Agent 技术栈的开发者
- 👨‍💼 **面试官** — 需要出题和评估候选人的技术面试官

---

## 📄 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

你可以自由地使用、修改和分发本项目的内容，但请保留原始版权声明。

---

<p align="center">
  <sub>⭐ 如果这个项目对你有帮助，欢迎 Star 支持！</sub>
</p>
