# Tool Use 的常见模式：API 调用、数据库查询、代码执行

> 难度：基础
> 分类：Tool Use

## 简短回答

LLM 工具使用有三大类模式：**数据访问**（API 调用获取外部数据、SQL 查询数据库、向量检索知识库）、**计算与代码执行**（在沙箱中运行代码进行数据处理、计算或可视化）、**写操作与动作执行**（发邮件、创建工单、修改数据库）。每类模式的安全级别和设计要点不同——读操作相对安全，写操作需要严格的权限控制和确认机制。

## 详细解析

### 模式 1：API 调用（外部数据获取）

最常见的工具类型——让 Agent 调用外部 REST API 获取实时数据。

```python
# 工具定义
{
    "name": "get_stock_price",
    "description": "获取股票的实时价格。输入股票代码（如 AAPL, GOOGL）。",
    "input_schema": {
        "properties": {
            "symbol": {"type": "string", "description": "股票代码"},
            "market": {"type": "string", "enum": ["US", "HK", "CN"]}
        },
        "required": ["symbol"]
    }
}

# 工具实现
import httpx

async def get_stock_price(symbol: str, market: str = "US") -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.stockdata.com/v1/quote",
            params={"symbol": symbol, "market": market},
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10.0  # 超时设置
        )
        response.raise_for_status()
        data = response.json()
        return {
            "symbol": symbol,
            "price": data["price"],
            "change": data["change_percent"],
            "timestamp": data["timestamp"]
        }
```

**设计要点：**
- 设置合理的超时（避免 Agent 卡住）
- 返回值精简（不要把整个 API 响应都扔给 LLM，消耗 token）
- 错误信息有意义（让 LLM 能根据错误决定下一步）

### 模式 2：数据库查询（结构化数据）

让 Agent 用自然语言查询数据库，分为两种实现方式：

#### 方式 A：Text-to-SQL

```python
# 给 Agent 提供数据库 Schema，让它生成 SQL
tools = [
    {
        "name": "list_tables",
        "description": "列出数据库中所有表名",
    },
    {
        "name": "get_table_schema",
        "description": "获取指定表的列定义",
        "input_schema": {"properties": {"table_name": {"type": "string"}}}
    },
    {
        "name": "execute_sql",
        "description": "执行只读 SQL 查询并返回结果。仅支持 SELECT 语句。",
        "input_schema": {"properties": {"query": {"type": "string"}}}
    }
]

# Agent 的工作流：
# 1. list_tables() → 了解有哪些表
# 2. get_table_schema("orders") → 了解表结构
# 3. execute_sql("SELECT ... FROM orders WHERE ...") → 查询数据
```

#### 方式 B：预定义查询工具

```python
# 不让 LLM 写 SQL，而是提供预定义的查询工具
{
    "name": "get_order_status",
    "description": "根据订单号查询订单状态",
    "input_schema": {
        "properties": {
            "order_id": {"type": "string"},
        }
    }
}

def get_order_status(order_id: str) -> dict:
    # 内部执行参数化 SQL，防止注入
    result = db.execute(
        "SELECT status, updated_at FROM orders WHERE id = %s",
        [order_id]
    )
    return {"order_id": order_id, "status": result.status}
```

**Text-to-SQL vs 预定义查询：**

| 维度 | Text-to-SQL | 预定义查询 |
|------|-----------|----------|
| 灵活性 | 高（任意查询） | 低（固定查询） |
| 安全性 | 低（SQL 注入风险） | 高（参数化查询） |
| 准确性 | 中（LLM 可能写错 SQL） | 高（人工验证过） |
| 适用场景 | 数据探索、分析 | 生产环境、敏感数据 |

### 模式 3：代码执行（计算与数据处理）

让 Agent 在沙箱中运行代码，用于计算、数据处理或可视化。

```python
{
    "name": "run_python",
    "description": "在安全沙箱中执行 Python 代码。可用库：pandas, numpy, matplotlib。用于数据分析、计算和图表生成。",
    "input_schema": {
        "properties": {
            "code": {"type": "string", "description": "Python 代码"},
        },
        "required": ["code"]
    }
}

import subprocess
import tempfile

def run_python(code: str) -> dict:
    """在隔离环境中执行代码"""
    # 安全检查：禁止危险操作
    forbidden = ["os.system", "subprocess", "eval(", "exec(", "__import__"]
    if any(f in code for f in forbidden):
        return {"error": "代码包含禁止的操作"}

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as f:
        f.write(code)
        f.flush()
        result = subprocess.run(
            ["python", f.name],
            capture_output=True, text=True,
            timeout=30,  # 30 秒超时
            # 生产环境应使用 Docker 容器隔离
        )
    return {
        "stdout": result.stdout[:2000],  # 截断防止 token 爆炸
        "stderr": result.stderr[:500],
        "returncode": result.returncode
    }
```

**安全要点：**
- 使用 Docker 或 gVisor 等容器隔离
- 设置 CPU/内存/时间限制
- 禁止网络访问（除非明确需要）
- 禁止文件系统写入（或限制路径）

### 模式 4：写操作（动作执行）

最需要谨慎的工具类型——它们会改变外部系统的状态。

```python
# 低风险写操作：创建草稿
{
    "name": "create_draft_email",
    "description": "创建邮件草稿（不会发送）。创建后用户需确认才会发送。",
}

# 高风险写操作：直接发送
{
    "name": "send_email",
    "description": "发送邮件。这是不可逆操作，务必在调用前确认收件人和内容正确。",
}

# Read-Before-Write 模式
async def update_record(record_id: str, updates: dict):
    # 1. 先读取当前状态
    current = await db.get(record_id)
    # 2. 验证更新是否合理
    if not validate_update(current, updates):
        return {"error": "更新不合理，请检查"}
    # 3. 执行更新
    await db.update(record_id, updates)
    return {"status": "updated", "previous": current, "current": updates}
```

### 各类工具的风险等级

| 模式 | 风险 | 需要确认？ | 示例 |
|------|------|-----------|------|
| API 读取 | 低 | 否 | 天气、股价查询 |
| 数据库读取 | 低-中 | 视数据敏感度 | 查询订单状态 |
| 代码执行 | 中 | 视场景 | 数据分析、计算 |
| API 写入 | 高 | 是 | 发邮件、创建工单 |
| 数据库写入 | 高 | 是 | 修改/删除记录 |

### 工具返回值设计

```python
# 差的返回：原始 API 响应（太长，浪费 token）
return api_response.json()  # 可能有 KB 级数据

# 好的返回：精简、结构化、有上下文
return {
    "status": "success",
    "data": {
        "order_id": "12345",
        "status": "shipped",
        "tracking_url": "https://..."
    },
    "hint": "如果用户需要更多详情，可以使用 get_order_details 工具"
}
```

## 常见误区 / 面试追问

1. **误区："让 LLM 直接写 SQL 很方便"** — 在生产环境中极其危险。LLM 可能生成恶意 SQL 或写错 SQL。推荐预定义查询工具 + 参数化查询，除非是面向数据分析师的内部工具。

2. **误区："代码执行不需要沙箱"** — 即使 Agent 没有代码解释器，LLM 生成的任何字符串用于系统调用前都应视为不可信输入。必须做输入验证、容器隔离和运行时监控。

3. **追问："如何处理工具链？（一个工具的结果是另一个的输入）"** — 用 Agent Loop：LLM 调用工具 A → 观察结果 → 决定是否需要调用工具 B。不要在工具之间硬编码依赖。

4. **追问："工具的返回值应该多详细？"** — 够用就好。返回太多数据浪费 token 且可能干扰 LLM 判断。返回太少则 LLM 无法做出正确决策。关键信息 + 下一步提示 是最佳实践。

## 参考资料

- [Mastering LLM Tool Calling (Machine Learning Mastery)](https://machinelearningmastery.com/mastering-llm-tool-calling-the-complete-framework-for-connecting-models-to-the-real-world/)
- [LLM Agents (Prompt Engineering Guide)](https://www.promptingguide.ai/research/llm-agents)
- [Function Calling with LLMs (Prompt Engineering Guide)](https://www.promptingguide.ai/applications/function_calling)
- [Preventing Unexpected Code Execution in AI Agents (Will Velida)](https://www.willvelida.com/posts/preventing-unexpected-code-execution-in-agents)
- [LLM Agents (Google ADK)](https://google.github.io/adk-docs/agents/llm-agents/)
