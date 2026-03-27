# 代码 Review 题：找出这段 Agent 代码中的设计问题并修复

> 难度：中级
> 分类：Production & Deployment

## 简短回答

Agent 代码的 Review 与普通业务代码有本质区别，因为 Agent 运行在一个**自主循环（Autonomous Loop）**中，每一轮迭代都涉及 **LLM 调用**和**外部工具执行**，错误代价远高于普通函数调用。一次合格的 Agent Code Review 至少需要检查六个维度：**循环保护（Loop Guard）**防止无限消耗 Token；**工具调用超时（Timeout）**避免单次调用阻塞整个 Agent；**结构化错误处理（Structured Error Handling）**而非简单吞掉异常；**Prompt 版本管理**确保可复现与可回滚；**可观测性（Observability）**覆盖 logging、tracing、metrics 三大支柱；以及**敏感操作确认（Human-in-the-Loop）**防止 Agent 自主执行破坏性操作。掌握这六个维度，就能系统性地发现初级 Agent 代码中的典型问题。

## 详细解析

### 问题代码：一个 "看起来能跑" 的 SimpleAgent

以下代码能跑通 demo，但隐藏了六个生产级设计问题。请先通读，尝试自己找出问题。

```python
import json
import openai

class SimpleAgent:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = openai.OpenAI()
        self.messages = [
            {"role": "system", "content": "你是一个全能助手，可以调用工具完成任务。"
             "当用户让你做某件事时，直接调用对应工具执行。"
             "如果不确定，就多试几次。"}
        ]
        self.tools = {
            "search_web": self.search_web,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "delete_file": self.delete_file,
            "execute_sql": self.execute_sql,
        }

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self._get_tool_schemas(),
            )
            msg = response.choices[0].message
            self.messages.append(msg)
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    try:
                        result = self.tools[name](**args)
                    except Exception as e:
                        result = f"Error: {e}"
                        print(f"工具调用失败: {e}")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })
            else:
                return msg.content

    def search_web(self, query: str) -> str:
        import requests
        resp = requests.get("https://api.search.com/v1/search", params={"q": query})
        return resp.json()["results"]

    def read_file(self, path: str) -> str:
        with open(path) as f:
            return f.read()

    def write_file(self, path: str, content: str) -> str:
        with open(path, "w") as f:
            f.write(content)
        return f"已写入 {path}"

    def delete_file(self, path: str) -> str:
        import os
        os.remove(path)
        return f"已删除 {path}"

    def execute_sql(self, sql: str) -> str:
        import sqlite3
        conn = sqlite3.connect("app.db")
        cursor = conn.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return str(results)

    def _get_tool_schemas(self) -> list:
        return [...]  # 省略 schema 定义
```

### 问题逐项解析

#### 问题 1：无循环保护（No Loop Guard）

`while True` 没有退出条件。LLM 若反复生成 tool_calls 而不返回最终回答（模型幻觉、prompt 缺陷），Agent 将无限循环消耗 Token，一次失控可能烧掉数百美元。

```python
# ✅ 修复：max_steps 保护
def run(self, user_input: str, max_steps: int = 25) -> str:
    for step in range(max_steps):
        response = self.client.chat.completions.create(...)
        # ...正常逻辑...
    raise AgentLoopLimitError(f"Agent 在 {max_steps} 步后未完成")
```

#### 问题 2：工具调用无超时（No Timeout）

`requests.get` 默认无超时，外部 API 无响应时阻塞整个进程，连带影响其他用户请求。

```python
# ✅ 修复：httpx + 显式超时
import httpx
resp = httpx.get("https://api.search.com/v1/search",
                 params={"q": query}, timeout=10.0)
```

#### 问题 3：错误处理太粗暴（Catch-All Swallowing）

`except Exception` + `print` 存在三个缺陷：吞掉所有异常（含 `MemoryError`）；`print` 无法被监控系统捕获；LLM 收到模糊的 Error 字符串，无法判断重试还是放弃。

```python
# ✅ 修复：分层捕获 + 结构化返回
try:
    result = self.tools[name](**args)
except KeyError:
    result = json.dumps({"error": "tool_not_found", "name": name})
    logger.error("未知工具: %s", name)
except TimeoutError as e:
    result = json.dumps({"error": "timeout", "tool": name})
    logger.warning("工具超时: %s", name, exc_info=True)
except Exception as e:
    result = json.dumps({"error": "internal", "tool": name, "detail": str(e)})
    logger.exception("工具异常: %s", name)
```

#### 问题 4：Prompt 硬编码（Hardcoded Prompt）

System prompt 直接写在 `__init__`，无法 A/B 测试、无法版本管理、无法热更新。且 "如果不确定，就多试几次" 直接鼓励盲目重试，加剧循环失控。

```python
# ✅ 修复：Prompt 外部化 + 版本管理
class PromptRegistry:
    def __init__(self, prompt_dir: str = "prompts/"):
        self.prompt_dir = Path(prompt_dir)

    def load(self, name: str, version: str = "latest") -> str:
        path = self.prompt_dir / f"{name}.{version}.yaml"
        with open(path) as f:
            return yaml.safe_load(f)["content"]
```

#### 问题 5：无可观测性（No Observability）

零 logging、零 tracing、零 metrics。生产出问题时无法回溯 Agent 走了几步、调用了哪些工具、每步耗时和 Token 消耗。LLM 的不确定性使问题难以复现，可观测性是唯一的事后诊断手段。

```python
# ✅ 修复：trace_id + 结构化日志
trace_id = str(uuid.uuid4())[:8]
logger.info("[%s] Step %d | latency=%.2fs | tokens=%d+%d",
            trace_id, step, llm_latency,
            usage.prompt_tokens, usage.completion_tokens)
```

#### 问题 6：敏感操作无确认（No Human-in-the-Loop）

`delete_file`、`execute_sql` 直接执行，LLM 产生幻觉或遭遇 prompt injection 时，破坏性操作不可逆。

```python
# ✅ 修复：敏感工具标记 + 确认拦截
SENSITIVE_TOOLS = {"delete_file", "execute_sql", "write_file"}

def _execute_tool(self, name: str, args: dict) -> str:
    if name in SENSITIVE_TOOLS:
        if not self._request_human_approval(name, args):
            return json.dumps({"error": "rejected", "detail": "用户拒绝了此操作"})
    return self.tools[name](**args)
```

### 修复后的完整代码

```python
import json, time, uuid, yaml, httpx, logging
from pathlib import Path
import openai

logger = logging.getLogger(__name__)

class AgentLoopLimitError(Exception):
    pass

class PromptRegistry:
    def __init__(self, prompt_dir="prompts/"):
        self.prompt_dir = Path(prompt_dir)
    def load(self, name: str, version: str = "latest") -> str:
        with open(self.prompt_dir / f"{name}.{version}.yaml") as f:
            return yaml.safe_load(f)["content"]

SENSITIVE_TOOLS = {"delete_file", "execute_sql", "write_file"}

class ProductionAgent:
    """生产级 Agent，覆盖六大设计维度"""
    def __init__(self, model="gpt-4o", prompt_version="latest"):
        self.client = openai.OpenAI()
        self.model = model
        # ✅ Prompt 外部化
        prompt = PromptRegistry().load("agent_system", prompt_version)
        self.messages = [{"role": "system", "content": prompt}]
        self.tools = {
            "search_web": self.search_web,  "read_file": self.read_file,
            "write_file": self.write_file,  "delete_file": self.delete_file,
            "execute_sql": self.execute_sql,
        }

    def run(self, user_input: str, max_steps: int = 25) -> str:
        trace_id = str(uuid.uuid4())[:8]
        self.messages.append({"role": "user", "content": user_input})
        logger.info("[%s] Agent 启动 | input=%s", trace_id, user_input[:100])
        # ✅ 循环保护
        for step in range(max_steps):
            t0 = time.monotonic()
            response = self.client.chat.completions.create(
                model=self.model, messages=self.messages,
                tools=self._get_tool_schemas(),
            )
            msg = response.choices[0].message
            self.messages.append(msg)
            # ✅ 可观测性
            usage = response.usage
            logger.info("[%s] Step %d/%d | %.2fs | tokens=%d+%d",
                        trace_id, step+1, max_steps, time.monotonic()-t0,
                        usage.prompt_tokens, usage.completion_tokens)
            if not msg.tool_calls:
                logger.info("[%s] 完成 | steps=%d", trace_id, step+1)
                return msg.content
            for tc in msg.tool_calls:
                result = self._safe_call(tc.function.name,
                    json.loads(tc.function.arguments), trace_id)
                self.messages.append({"role": "tool",
                    "tool_call_id": tc.id, "content": result})
        raise AgentLoopLimitError(f"[{trace_id}] 超过 {max_steps} 步")

    def _safe_call(self, name: str, args: dict, trace_id: str) -> str:
        # ✅ 敏感操作拦截
        if name in SENSITIVE_TOOLS:
            if not self._human_approve(name, args):
                return json.dumps({"error": "rejected"})
        # ✅ 分层错误处理
        try:
            t0 = time.monotonic()
            result = self.tools[name](**args)
            logger.info("[%s] tool=%s | %.2fs", trace_id, name, time.monotonic()-t0)
            return str(result)
        except KeyError:
            logger.error("[%s] 未知工具: %s", trace_id, name)
            return json.dumps({"error": "tool_not_found", "name": name})
        except (httpx.TimeoutException, TimeoutError) as e:
            logger.warning("[%s] 超时: %s", trace_id, name)
            return json.dumps({"error": "timeout", "tool": name})
        except Exception as e:
            logger.exception("[%s] 异常: %s", trace_id, name)
            return json.dumps({"error": "internal", "detail": str(e)})

    def _human_approve(self, tool: str, args: dict) -> bool:
        print(f"\n⚠️  敏感操作: {tool}\n    参数: {json.dumps(args, ensure_ascii=False)}")
        return input("允许? [y/N]: ").strip().lower() == "y"

    # ✅ 工具实现（带超时）
    def search_web(self, query: str) -> str:
        return httpx.get("https://api.search.com/v1/search",
                         params={"q": query}, timeout=10.0).json()["results"]
    def read_file(self, path: str) -> str:
        with open(path) as f: return f.read()
    def write_file(self, path: str, content: str) -> str:
        with open(path, "w") as f: f.write(content)
        return f"已写入 {path}"
    def delete_file(self, path: str) -> str:
        import os; os.remove(path); return f"已删除 {path}"
    def execute_sql(self, sql: str) -> str:
        import sqlite3
        conn = sqlite3.connect("app.db")
        r = conn.execute(sql).fetchall(); conn.close(); return str(r)
    def _get_tool_schemas(self) -> list:
        return [...]
```

### 六大问题对比总结

| # | 问题 | 原始代码 | 修复方案 | 风险等级 |
|---|------|----------|----------|----------|
| 1 | 无循环保护 | `while True` 无退出条件 | `for step in range(max_steps)` + 异常 | 严重 |
| 2 | 工具无超时 | `requests.get()` 无 timeout | `httpx.get(timeout=10.0)` | 严重 |
| 3 | 错误处理粗暴 | `except Exception` + `print` | 分层捕获 + `logging` + JSON 返回 | 中等 |
| 4 | Prompt 硬编码 | 字符串写死在 `__init__` | `PromptRegistry` 外部加载 + 版本管理 | 中等 |
| 5 | 无可观测性 | 零日志、零追踪 | `trace_id` + `logging` + Token 埋点 | 中等 |
| 6 | 敏感操作无确认 | `delete_file` 直接执行 | `SENSITIVE_TOOLS` + Human-in-the-Loop | 严重 |

### Agent Review 与普通 Review 的核心差异

```
┌─────────────────────────────────────────────────────┐
│               普通代码 Review                        │
│  输入 ──→ [ 函数逻辑 ] ──→ 输出                      │
│            确定性、单次执行                            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│               Agent 代码 Review                      │
│  输入 ──→ ┌────────────────────────────┐            │
│           │ LLM 推理 → Tool 调用 → 结果 │ ←─ 循环   │
│           │    ↑                    │   │            │
│           │    └────────────────────┘   │            │
│           └────────────────────────────┘            │
│            不确定性、多步循环、外部副作用               │
│                                                     │
│  额外 Review 维度:                                   │
│  ┌────────┐ ┌────────┐ ┌──────────┐                │
│  │循环边界 │ │成本上限 │ │权限 & 安全│                │
│  └────────┘ └────────┘ └──────────┘                │
│  ┌────────┐ ┌────────┐ ┌──────────┐                │
│  │可观测性 │ │Prompt  │ │超时 & 重试│                │
│  │        │ │版本管理 │ │          │                │
│  └────────┘ └────────┘ └──────────┘                │
└─────────────────────────────────────────────────────┘
```

## 常见误区 / 面试追问

1. **误区："Agent 代码和普通代码的 Review 标准一样"** — Agent 代码的核心特征是**非确定性循环 + 外部副作用**。普通代码 Review 关注逻辑正确性和代码风格，但 Agent 代码必须额外审查循环边界、成本控制、权限隔离、可观测性等维度。忽略这些维度的 Review 等于没有 Review，建议维护专门的 Agent Code Review Checklist。

2. **误区："加了 try-catch 就算处理了错误"** — `except Exception` 配 `print` 是最危险的"伪处理"。生产级错误处理需要：**分层捕获**（区分可重试与致命错误）、**结构化返回**（让 LLM 根据错误类型决策）、**持久化记录**（logging 而非 print）。Agent 的错误信息会反馈给 LLM，格式化 JSON 远比 "Error: something went wrong" 有用。

3. **追问："如何设计 Agent 代码审查 Checklist？"** — 可按 SECURE 助记法组织：**S**teps Limit（最大步数）、**E**rror Handling（分层处理 + 结构化返回）、**C**ost Guard（Token / 费用上限）、**U**ser Confirmation（敏感操作确认）、**R**eplay & Trace（trace_id 可回溯）、**E**xternal Timeout（外部调用超时）。每次 PR Review 逐条过一遍即可系统性覆盖。

4. **追问："如何为 Agent 编写单元测试？"** — 分三层：第一层**工具函数单测**，Mock 外部依赖验证正常 / 超时 / 异常行为；第二层 **Agent 循环单测**，Mock LLM 返回固定 tool_calls 序列，验证 max_steps 触发和敏感工具拦截；第三层**集成 Eval**，真实 LLM + 沙箱环境验证端到端完成率。关键原则：不断言 LLM 的具体输出，而是断言 Agent 的**行为属性**——是否限步完成、是否拦截危险操作、是否生成完整 trace。

## 参考资料

- [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)
- [LLM Powered Autonomous Agents (Lilian Weng)](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [OpenAI Function Calling Guide (OpenAI)](https://platform.openai.com/docs/guides/function-calling)
- [Observability for LLM Applications (Langfuse Docs)](https://langfuse.com/docs)
- [Human-in-the-Loop Patterns for AI Agents (LangChain Blog)](https://blog.langchain.dev/human-in-the-loop/)
