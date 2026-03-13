# Agent Benchmark：如何设计端到端的 Agent 测试？

> 难度：中级
> 分类：Evaluation

## 简短回答

Agent Benchmark 是用于端到端评估 AI Agent 在真实或模拟环境中完成任务能力的标准化测试。与传统 LLM 基准（如 MMLU 测知识）不同，Agent Benchmark 评估的是**完整的任务执行过程**——包括规划、工具使用、多步推理和错误恢复。代表性基准包括：**SWE-bench**（修复真实 GitHub Issue，代码 Agent 标杆）、**WebArena**（在真实网站完成复杂操作）、**GAIA**（通用助手的多步推理和工具组合任务）、**OSWorld**（操作系统级别的计算机使用任务）。设计 Agent Benchmark 的关键原则：(1) 使用真实任务而非人造题目；(2) 评估过程而非仅评估结果；(3) 包含多种难度层次；(4) 防止数据泄露和过拟合。当前面临的挑战：基准饱和（模型快速刷榜）、可游戏性（针对基准优化而非真实能力提升）。

## 详细解析

### 主要 Agent 基准全景

```
Agent 基准分类：
│
├── 代码 Agent
│   ├── SWE-bench：修复真实 GitHub Issue（最权威）
│   ├── SWE-bench Verified：人工验证的高质量子集
│   ├── Multi-SWE-bench：多语言扩展
│   └── HumanEval / MBPP：函数级代码生成
│
├── Web Agent
│   ├── WebArena：真实网站交互任务
│   ├── VisualWebArena：需要视觉理解的网页任务
│   └── Mind2Web：跨网站通用操作
│
├── 通用 Agent
│   ├── GAIA：多步推理 + 多工具组合
│   ├── ALFWorld：文本家庭环境任务
│   └── WebShop：模拟电商购物
│
├── 计算机使用
│   ├── OSWorld：操作系统级任务
│   └── Computer Use benchmarks
│
└── 工具使用
    ├── ToolBench：API 工具选择和使用
    ├── API-Bank：API 调用正确性
    └── TaskBench：多工具组合
```

### SWE-bench 详解

```python
# SWE-bench：代码 Agent 的标杆基准
swe_bench = {
    "任务": "给定一个真实的 GitHub Issue，修改代码库使相关测试通过",
    "来源": "12 个流行 Python 开源项目的真实 Issue",
    "规模": "2294 个 Issue（Verified 子集 500 个）",
    "评估": "自动化——运行项目测试套件",
    "难度": "非常高——需要理解大型代码库、定位 bug、编写修复",

    "评估指标": {
        "Resolved Rate": "成功修复的 Issue 比例",
        "当前 SOTA": "约 72%（SWE-bench Verified）",
    },

    "为什么重要": [
        "使用真实世界的软件工程任务",
        "需要理解数千行代码的上下文",
        "修复必须通过真实的测试套件",
        "无法靠记忆训练数据作弊",
    ],
}
```

### GAIA 基准详解

```python
gaia_benchmark = {
    "任务": "回答需要多步推理和工具组合的复杂问题",
    "特点": "答案是确定性的（可以精确匹配）",
    "三个难度等级": {
        "Level 1": "需要 1-2 步推理，1 个工具",
        "Level 2": "需要 3-5 步推理，多个工具组合",
        "Level 3": "需要 5+ 步推理，复杂工具链和长上下文",
    },
    "示例问题": (
        "'找到 2024 年诺贝尔物理学奖获得者的本科毕业院校，"
        "这所院校的现任校长是谁？'"
        "→ 需要：搜索→提取→再搜索→提取"
    ),
}
```

### 设计 Agent Benchmark 的原则

```python
benchmark_design_principles = {
    "真实性": {
        "原则": "使用真实任务而非人造题目",
        "方法": "从生产日志、GitHub Issue、真实网站中采集",
        "反例": "人工构造的'玩具问题'不能反映真实复杂度",
    },
    "可验证性": {
        "原则": "评估结果必须可自动化验证",
        "方法": "定义明确的成功标准（测试通过、精确匹配等）",
        "挑战": "开放式任务的评估需要 LLM Judge",
    },
    "防泄露": {
        "原则": "防止基准数据出现在训练集中",
        "方法": [
            "动态基准（定期更新题目）",
            "使用私有测试集",
            "基于时间的切分（只用模型训练后的数据）",
        ],
    },
    "多维度": {
        "原则": "评估多种能力，不只是最终结果",
        "维度": ["推理质量", "工具使用", "效率", "安全性"],
    },
    "抗游戏性": {
        "原则": "防止针对基准优化而非真实能力提升",
        "方法": "大规模多样化的测试集 + 动态更新",
    },
}
```

### 自定义 Agent 评估套件

```python
class CustomAgentBenchmark:
    """为自己的 Agent 设计评估套件"""

    def __init__(self):
        self.test_cases = []

    def add_test(self, task, expected_result, difficulty, category,
                 required_tools=None, max_steps=None):
        self.test_cases.append({
            "task": task,
            "expected": expected_result,
            "difficulty": difficulty,      # easy/medium/hard
            "category": category,          # coding/search/analysis
            "required_tools": required_tools,
            "max_steps": max_steps,
        })

    async def run(self, agent):
        results = []
        for case in self.test_cases:
            trajectory = await agent.execute_with_trace(case["task"])

            result = {
                "task_success": self.check_result(
                    trajectory.final_output, case["expected"]
                ),
                "steps_used": len(trajectory.steps),
                "tools_used": [s.tool for s in trajectory.steps if s.tool],
                "cost": trajectory.total_cost,
                "latency": trajectory.total_time,
                "correct_tools": self.check_tools(
                    trajectory, case["required_tools"]
                ),
            }
            results.append(result)

        return self.aggregate_results(results)

    def generate_report(self, results):
        """按维度和难度分组的评估报告"""
        return {
            "overall_success_rate": np.mean([r["task_success"] for r in results]),
            "by_difficulty": self.group_by("difficulty", results),
            "by_category": self.group_by("category", results),
            "avg_cost": np.mean([r["cost"] for r in results]),
            "avg_steps": np.mean([r["steps_used"] for r in results]),
        }
```

### 基准测试的挑战

```python
current_challenges = {
    "数据泄露": "基准题目可能出现在 LLM 的训练数据中",
    "基准饱和": "模型快速刷满分，基准失去区分能力",
    "过拟合基准": "针对基准优化 ≠ 真实能力提升",
    "评估成本": "端到端 Agent 测试需要真实环境，成本高",
    "非确定性": "Agent 每次运行路径不同，评估结果有方差",
}

# SWE-MERA 的解决方案：动态基准
swe_mera = {
    "创新": "持续从最新 GitHub Issue 中自动收集测试用例",
    "优势": "永远不会被训练数据污染",
    "挑战": "质量控制——自动收集的题目可能质量不一",
}
```

## 常见误区 / 面试追问

1. **误区："在基准上得分高就说明 Agent 好用"** — 基准测试是受控环境，生产场景更复杂（网络问题、意外输入、安全攻击等）。SWE-bench 上 70% 的模型在实际开发中可能远达不到这个表现。基准是必要但不充分的。

2. **误区："一个基准就够了"** — 不同基准测试不同能力。代码能力强（SWE-bench 高分）不代表网页操作好（WebArena）。需要根据 Agent 的实际使用场景选择或组合多个基准。

3. **追问："如何防止 Agent 针对基准过拟合？"** — (1) 使用动态更新的基准（如 SWE-MERA）；(2) 保留私有测试集不公开；(3) 评估时加入从未见过的新类型任务；(4) 关注轨迹质量而非仅结果。

4. **追问："小团队如何设计自己的 Agent 评估？"** — 从生产日志中采样 50-100 个典型任务，定义明确的成功标准（可自动验证的优先），标注难度和类别。每次 Agent 更新后运行这个测试套件作为回归测试。不需要从头建造大规模基准。

## 参考资料

- [Agent Evaluation: Metrics, Benchmarks and Safety Standards](https://mbrenndoerfer.com/writing/agent-evaluation-metrics-benchmarks-safety)
- [AI Agent Benchmark Compendium (Phil Schmid, GitHub)](https://github.com/philschmid/ai-agent-benchmark-compendium)
- [SWE-bench Leaderboards](https://www.swebench.com/)
- [AI Agent Benchmarks are Broken (Daniel Kang)](https://medium.com/@danieldkang/ai-agent-benchmarks-are-broken-c1fedc9ea071)
- [SWE-MERA: A Dynamic Benchmark for Evaluating LLMs (arXiv)](https://arxiv.org/html/2507.11059v1)
