# 解释 Agent 的核心组件：感知、推理、行动、记忆

> 难度：基础
> 分类：Agent 架构

## 简短回答

一个完整的 LLM Agent 由四大核心组件构成：**感知模块**（接收和解析输入）、**推理模块**（LLM 作为"大脑"进行思考和规划）、**行动模块**（调用工具执行操作）、**记忆模块**（存储和检索上下文信息）。这四个模块形成一个持续运转的认知闭环：感知→推理→行动→记忆→感知...

## 详细解析

### 1. 感知模块（Perception）

感知模块是 Agent 的"感官系统"，负责接收和预处理来自环境的各种信号。

**职责：**
- 接收用户自然语言输入
- 解析多模态数据（文本、图像、音频、视频）
- 处理工具返回的结果（Observation）
- 解析外部事件和通知（如 Webhook、消息队列）

**关键设计要点：**
- 感知不是被动接收，而是主动过滤和转换。原始数据需要被结构化为 Agent 可理解的格式
- 感知质量直接影响下游所有决策。如果 Agent 误解了用户意图或错误解析了工具输出，后续的推理和行动都会出错
- 现代 Agent 通常支持多模态感知：文本（NLP）、视觉（Computer Vision）、结构化数据（API Response Parsing）

```python
# 感知模块示例：解析用户输入并提取意图
class PerceptionModule:
    def process_input(self, raw_input: dict) -> dict:
        """将原始输入转化为结构化的感知数据"""
        return {
            "user_query": raw_input.get("text", ""),
            "images": raw_input.get("images", []),
            "tool_results": raw_input.get("observations", []),
            "context": self._extract_context(raw_input)
        }
```

### 2. 推理模块（Reasoning / Cognitive Core）

推理模块是 Agent 的"大脑"，通常由 LLM 担任，负责理解任务、制定计划、做出决策。

**核心能力：**
- **任务理解**：解析用户目标，理解隐含需求
- **任务分解**：将复杂任务拆解为可执行的子任务
- **规划**：制定行动序列，确定工具使用顺序
- **决策**：在多个可选方案中选择最优路径
- **反思**：评估中间结果，判断是否需要调整策略

**主流推理范式：**

| 范式 | 描述 | 适用场景 |
|------|------|---------|
| ReAct | 交替进行思考和行动 | 需要动态探索的任务 |
| Plan-and-Execute | 先完整规划，再逐步执行 | 结构化的多步任务 |
| Chain-of-Thought | 逐步推理，但不执行行动 | 纯推理问题 |
| Tree-of-Thought | 探索多条推理路径 | 需要回溯的复杂问题 |
| ReWOO | 一次性生成完整计划，减少 LLM 调用 | 成本敏感场景 |

```python
# 推理模块：LLM 作为决策核心
class ReasoningModule:
    def think(self, perception: dict, memory: dict) -> dict:
        """基于感知和记忆进行推理"""
        prompt = self._build_prompt(perception, memory)
        response = self.llm.generate(prompt)
        return {
            "thought": response.reasoning,      # 思考过程
            "plan": response.plan,              # 行动计划
            "next_action": response.action,     # 下一步行动
            "should_stop": response.is_final    # 是否完成
        }
```

### 3. 行动模块（Action / Execution）

行动模块是 Agent 与外部世界交互的"手脚"，负责执行推理模块做出的决策。

**行动类型：**
- **工具调用**：执行 Function Calling（搜索、计算、API 请求）
- **代码执行**：在沙箱中运行代码
- **数据操作**：读写数据库、文件系统
- **通信**：发送消息、邮件、触发通知
- **最终输出**：向用户返回最终答案

**关键设计考量：**
- 行动需要有明确的成功/失败反馈，作为下一轮推理的 Observation
- 工具调用应有超时、重试和降级机制
- 危险操作（删除数据、发送邮件）需要 Human-in-the-Loop 确认

```python
# 行动模块：执行工具调用
class ActionModule:
    def execute(self, action: dict) -> dict:
        """执行推理模块决定的行动"""
        tool_name = action["tool"]
        tool_input = action["input"]

        try:
            result = self.tools[tool_name].run(tool_input)
            return {"status": "success", "observation": result}
        except ToolError as e:
            return {"status": "error", "observation": str(e)}
```

### 4. 记忆模块（Memory）

记忆模块为 Agent 提供跨步骤和跨会话的上下文保持能力。

**记忆类型：**

| 类型 | 类比 | 实现 | 生命周期 |
|------|------|------|---------|
| 工作记忆 | 当前任务的"便签纸" | LLM Context Window | 单次任务 |
| 短期记忆 | 对话上下文 | 对话历史 + 摘要 | 单次会话 |
| 长期记忆 | 经验知识库 | 向量数据库 / 知识图谱 | 持久化 |

**关键设计要点：**
- 上下文窗口有限，需要摘要和压缩策略
- 长期记忆的检索质量（Recall + Precision）直接影响 Agent 表现
- 需要遗忘机制来处理过时信息

```python
# 记忆模块：管理不同层次的记忆
class MemoryModule:
    def __init__(self):
        self.working_memory = []      # 当前任务步骤
        self.conversation_history = [] # 对话历史
        self.vector_store = VectorDB() # 长期记忆

    def store(self, entry: dict):
        """存储新记忆"""
        self.working_memory.append(entry)
        if entry.get("persist"):
            self.vector_store.upsert(entry)

    def recall(self, query: str, k: int = 5) -> list:
        """检索相关记忆"""
        return self.vector_store.search(query, top_k=k)
```

### 四大模块的协作闭环

```
用户输入 → [感知] → 结构化信息
                        ↓
              [记忆] ← [推理] → 决策
                        ↓
              [行动] → 执行结果（Observation）
                        ↓
              反馈回 [感知]，开始下一轮循环
```

这个闭环持续运转，直到推理模块判断任务已完成（或达到最大步数限制）。四个模块的集成质量决定了 Agent 的整体能力——推理依赖记忆提供上下文，记忆依赖感知获取新信息，行动依赖推理做决策，感知处理行动的结果。

## 常见误区 / 面试追问

1. **误区："LLM 就是 Agent 的全部"** — LLM 只是推理模块的核心，一个完整的 Agent 还需要感知、行动、记忆模块的协同工作。

2. **误区："记忆就是对话历史"** — 对话历史只是短期记忆的一部分。完整的记忆系统还包括工作记忆（当前任务状态）和长期记忆（持久化知识库）。

3. **追问："如果推理模块（LLM）出错了怎么办？"** — 需要 Guardrails（安全护栏）+ 自我反思机制 + 人工审核介入。好的 Agent 设计应该假设 LLM 会犯错，并在架构层面做好容错。

4. **追问："四个模块的耦合度应该怎么设计？"** — 模块间应该通过标准化接口通信（如统一的消息格式），实现松耦合。这样可以独立替换某个模块（比如换一个 LLM、换一个向量数据库），不影响其他部分。

## 参考资料

- [AI Agent Core Components (IBM)](https://www.ibm.com/think/topics/components-of-ai-agents)
- [Traditional Agent Architecture: Perceive, Reason, Act (AWS)](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-foundations/traditional-agents.html)
- [The Architecture of Autonomous AI Agents (Deepak Gupta)](https://guptadeepak.com/the-rise-of-autonomous-ai-agents-a-comprehensive-guide-to-their-architecture-applications-and-impact/)
- [Agentic AI Architecture: Types, Components, Best Practices (Exabeam)](https://www.exabeam.com/explainers/agentic-ai/agentic-ai-architecture-types-components-best-practices/)
- [Agentic AI: Architectures, Taxonomies, and Evaluation (arXiv 2601.12560)](https://arxiv.org/html/2601.12560v1)
