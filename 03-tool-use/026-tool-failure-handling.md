# 如何处理工具调用失败和超时？

> 难度：中级
> 分类：Tool Use

## 简短回答

工具调用失败在生产环境中不可避免——网络超时、API 限流、参数错误、服务宕机都会发生。核心处理策略是**分层防御**：首先对每个工具调用设置超时，然后用**指数退避重试**处理瞬时错误，用**断路器模式**防止持续请求失败的服务，用**降级回退**在主工具不可用时切换备选方案，最后通过**错误信息回传 LLM** 让 Agent 自主调整策略。关键原则：永远不要让 Agent 无限等待或无限重试。

## 详细解析

### 工具调用的常见失败模式

```
失败来源分类：
├── 网络层：超时、DNS 解析失败、连接拒绝
├── 服务层：HTTP 4xx（参数错误/未授权）、HTTP 5xx（服务端错误）、429（限流）
├── 输入层：LLM 生成的参数不符合 Schema、类型错误、值越界
├── 执行层：工具内部逻辑 bug、未处理的边界条件
└── Agent 层：无限循环调用、重复调用同一个失败工具
```

### 策略 1：超时控制

每个外部调用都必须设置超时。如果服务无响应，工具不应无限挂起。

```python
import httpx
import asyncio

async def call_external_api(url: str, params: dict, timeout: float = 10.0):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
    except httpx.TimeoutException:
        return {"status": "error", "error": "请求超时，服务可能暂时不可用"}
    except httpx.HTTPStatusError as e:
        return {"status": "error", "error": f"HTTP {e.response.status_code}"}
```

Agent 级别的超时同样重要：

```python
# Agent Loop 级别的防护
agent_config = {
    "max_iterations": 10,        # 最大步数，防止无限循环
    "max_execution_time": 120,   # 整体超时（秒）
    "single_tool_timeout": 30,   # 单次工具调用超时
}
```

### 策略 2：指数退避重试（Exponential Backoff）

瞬时错误（网络抖动、临时过载）适合重试，但不能立即重试——这会加剧服务压力。

```python
import asyncio
import random

async def retry_with_backoff(func, max_retries=3, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return await func()
        except TransientError as e:
            if attempt == max_retries - 1:
                raise  # 最后一次仍失败，抛出异常
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # 加 jitter
            await asyncio.sleep(delay)
            # 等待: ~1s, ~3s, ~7s
```

**重试决策矩阵：**

| 错误类型 | 是否重试 | 原因 |
|---------|---------|------|
| 网络超时 | 是 | 瞬时问题 |
| HTTP 429 (限流) | 是（按 Retry-After） | 等待后可恢复 |
| HTTP 500 | 是（有限次） | 服务可能临时异常 |
| HTTP 400 (参数错误) | 否 | 重试不会改变结果 |
| HTTP 401/403 | 否 | 权限问题，重试无意义 |
| Schema 验证失败 | 否（但可让 LLM 重新生成参数） | 需要修正输入 |

### 策略 3：断路器模式（Circuit Breaker）

对持续失败的服务，不断重试会浪费资源和 LLM 上下文。断路器提供更结构化的保护：

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "closed"       # closed → open → half_open
        self.last_failure_time = None

    async def call(self, func):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"  # 尝试恢复
            else:
                return {"error": "服务暂时不可用（断路器打开）"}

        try:
            result = await func()
            if self.state == "half_open":
                self.state = "closed"  # 恢复成功
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.threshold:
                self.state = "open"    # 触发熔断
            raise
```

三种状态的含义：
- **Closed（正常）**：请求正常通过，记录失败次数
- **Open（熔断）**：所有请求立即拒绝，进入冷却期
- **Half-Open（试探）**：允许少量请求通过，测试服务是否恢复

### 策略 4：降级回退（Fallback）

当主工具不可用时，切换到备选方案：

```python
class ToolWithFallback:
    def __init__(self, primary_tool, fallback_tool):
        self.primary = primary_tool
        self.fallback = fallback_tool

    async def execute(self, params):
        try:
            return await self.primary.execute(params)
        except ToolError:
            # 主工具失败，尝试降级
            return await self.fallback.execute(params)

# 示例：主用 Google Search API，降级用 Bing Search API
search_tool = ToolWithFallback(
    primary_tool=GoogleSearchTool(),
    fallback_tool=BingSearchTool()
)
```

### 策略 5：错误信息回传 LLM

最重要的策略——将有意义的错误信息返回给 Agent，让它自主调整：

```python
# 差的做法：吞掉错误
except Exception:
    return None  # Agent 不知道发生了什么

# 好的做法：返回可操作的错误信息
except Exception as e:
    return {
        "status": "error",
        "error_type": "timeout",
        "message": "天气 API 超时，可能是服务暂时不可用",
        "suggestion": "可以尝试使用 search_web 工具搜索天气信息作为替代"
    }
```

Agent 收到错误后可以：生成修正后的参数重试、选择替代工具、直接告知用户工具不可用。

### 防止 Agent 无限循环

```python
class AgentExecutor:
    def __init__(self, max_iterations=10, max_tool_retries=3):
        self.max_iterations = max_iterations
        self.tool_call_counts = {}  # 每个工具的调用计数

    def should_continue(self, tool_name):
        count = self.tool_call_counts.get(tool_name, 0)
        if count >= self.max_tool_retries:
            return False, f"工具 {tool_name} 已连续失败 {count} 次，停止重试"
        self.tool_call_counts[tool_name] = count + 1
        return True, None
```

### 生产级分层防御架构

```
请求 → [断路器] → [速率限制] → [重试 + 退避] → 工具执行
                                                      │
                                               成功 ←──┤──→ 失败
                                                       │
                                              [降级回退] → [错误回传 LLM]
```

## 常见误区 / 面试追问

1. **误区："所有错误都应该重试"** — 只有瞬时错误（超时、429、5xx）才值得重试。参数错误（400）、权限错误（401/403）重试不会改变结果，反而浪费资源和上下文窗口。

2. **误区："重试应该立即执行"** — 立即重试会加剧服务压力。使用指数退避 + 随机 jitter 分散请求。对 429 错误，应遵循 `Retry-After` 头。

3. **追问："如何防止 Agent 在工具失败时陷入无限循环？"** — 设置 `max_iterations`（限制总步数）和 `max_tool_retries`（限制单工具重试次数）。同时追踪工具调用历史，检测重复失败模式。

4. **追问："断路器和重试有什么区别？"** — 重试是在单次失败后立即的短期策略；断路器是在多次失败后的长期保护。重试处理偶发故障，断路器处理持续故障。两者应组合使用：断路器内部包含重试逻辑。

5. **场景追问："你的 Agent 反复调用 search_web 工具但每次结果都不满足需求，Token 消耗不断增加。如何修复？"** — 这是"搜索无果死循环"问题。修复路径：(1) 限制 search_web 调用次数，超过后强制切换策略；(2) 优化工具返回格式 → 明确告知 Agent"已搜索 X 次，无相关结果，建议更换查询策略"；(3) 加入查询反思 → 让 Agent 分析为什么搜索失败，是查询太宽泛还是太狭窄；(4) 设计降级策略 → 搜索失败后转而使用知识库检索或直接询问用户更多细节；(5) 加入人工介入点 → 多次失败后主动询问用户是否需要调整查询方向。

6. **场景追问："你的工具调用成功但返回数据格式与 LLM 期望不符，导致解析错误并重试。如何解决？"** — 这是"成功但失败"的问题。解决路径：(1) 工具 Schema 必须明确定义输出格式；（2）工具内部加入输出验证，确保返回数据符合 Schema；（3）在工具返回时附带示例格式，帮助 LLM 理解；（4）LLM 端加入容错解析，处理边界情况；（5）对不稳定的外部 API 加入 Wrapper 层，标准化输出格式。

7. **场景追问："你的数据库查询工具因参数注入攻击而被封禁，Agent 无法再访问数据。如何防范和恢复？"** — 这是安全故障场景。防范路径：(1) 实施严格的参数验证和转义；（2）限制工具权限，遵循最小权限原则；（3）加入查询模板机制，禁止动态构造完整 SQL；（4）实施速率限制，防止单一 Agent 过度请求；（5）监控异常查询模式，提前识别攻击行为。恢复路径：(1) 紧急启用备用数据库连接；（2) 暂时切换到只读模式；（3）与数据库厂商沟通解封；（4）事后分析攻击来源，加强防护。

## 参考资料

- [Retries, Fallbacks, and Circuit Breakers in LLM Apps (Portkey)](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/)
- [Error Handling for LLM Agent Tools (APXML)](https://apxml.com/courses/building-advanced-llm-agent-tools/chapter-1-llm-agent-tooling-foundations/tool-error-handling)
- [Error Recovery and Fallback Strategies in AI Agent Development (GoCodeo)](https://www.gocodeo.com/post/error-recovery-and-fallback-strategies-in-ai-agent-development)
- [LLM Tool-Calling in Production: Rate Limits, Retries, and the "Infinite Loop" Failure Mode (Medium)](https://medium.com/@komalbaparmar007/llm-tool-calling-in-production-rate-limits-retries-and-the-infinite-loop-failure-mode-you-must-2a1e2a1e84c8)
- [Handling Tool Errors and Agent Recovery (APXML)](https://apxml.com/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools/agent-error-handling)
