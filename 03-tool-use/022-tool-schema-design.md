# 如何为 LLM 定义和描述工具（Tool Schema）？

> 难度：基础
> 分类：Tool Use

## 简短回答

工具 Schema 使用 JSON Schema 格式定义三要素：**名称**（唯一标识符）、**描述**（告诉 LLM 何时以及如何使用此工具）、**参数**（输入的类型、约束、是否必填）。描述是最关键的字段——它直接影响 LLM 的工具选择准确率。最佳实践包括：使用清晰的命名空间、利用 Pydantic 自动生成 Schema、处理多工具调用、以及对 LLM 输出做参数验证。

## 详细解析

### 工具 Schema 的三要素

```json
{
  "name": "search_knowledge_base",
  "description": "在内部知识库中搜索技术文档。当用户询问产品功能、API 文档或技术规范时使用此工具。不要用于搜索客户订单或财务数据。",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "搜索关键词或自然语言问题"
      },
      "category": {
        "type": "string",
        "enum": ["api_docs", "user_guide", "release_notes"],
        "description": "文档类别，缩小搜索范围"
      },
      "max_results": {
        "type": "integer",
        "description": "返回的最大结果数量",
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

### 编写高质量描述的原则

描述（description）是 LLM 选择工具的核心依据。差的描述 → 错误的工具选择 → 错误的结果。

```python
# 差的描述
{"name": "search", "description": "搜索东西"}
# 问题：搜什么？什么时候用？什么时候不用？

# 好的描述
{
    "name": "search_product_catalog",
    "description": (
        "在产品目录中搜索产品信息，包括名称、价格、库存和规格。"
        "当用户询问产品详情、价格比较或库存查询时使用此工具。"
        "不适用于搜索订单状态或用户账户信息——请使用 query_orders 或 get_user_profile。"
    )
}
```

**高质量描述的检查清单：**
1. 说明工具做什么（功能）
2. 说明什么时候应该使用（触发条件）
3. 说明什么时候不该使用（排除条件）
4. 如果有相似工具，说明区别

### 利用 JSON Schema 的丰富特性

```json
{
  "type": "object",
  "properties": {
    "date_range": {
      "type": "object",
      "properties": {
        "start": {"type": "string", "format": "date", "description": "起始日期 YYYY-MM-DD"},
        "end": {"type": "string", "format": "date", "description": "结束日期 YYYY-MM-DD"}
      },
      "required": ["start", "end"]
    },
    "status": {
      "type": "string",
      "enum": ["pending", "completed", "cancelled"],
      "description": "按订单状态过滤"
    },
    "amount": {
      "type": "number",
      "minimum": 0,
      "description": "最低金额过滤（美元）"
    },
    "tags": {
      "type": "array",
      "items": {"type": "string"},
      "description": "标签列表，取交集过滤"
    }
  }
}
```

可用特性：`type`、`enum`、`description`、`default`、`minimum/maximum`、`format`、嵌套对象、数组、`required`。

### 使用 Pydantic 自动生成 Schema

手动编写 JSON Schema 在工具多了以后容易出错。用 Pydantic 自动生成更可靠：

```python
from pydantic import BaseModel, Field
from typing import Optional
import json

class SearchParams(BaseModel):
    """在知识库中搜索文档"""
    query: str = Field(description="搜索关键词或自然语言问题")
    category: Optional[str] = Field(
        default=None,
        description="文档类别",
        json_schema_extra={"enum": ["api_docs", "user_guide", "release_notes"]}
    )
    max_results: int = Field(default=5, description="最大返回数量", ge=1, le=50)

# 自动生成 JSON Schema
schema = SearchParams.model_json_schema()
print(json.dumps(schema, indent=2))

# 自动验证 LLM 的输出
def execute_tool(llm_output: dict):
    params = SearchParams(**llm_output)  # 自动类型验证
    return knowledge_base.search(**params.model_dump())
```

**Pydantic 的优势：**
- 自动类型验证和默认值处理
- description 从 Field 自动提取
- 数据解析和序列化一体
- Schema 更新时自动同步

### 工具命名空间

当工具数量增多时，用命名空间组织：

```python
tools = [
    # CRM 域
    {"name": "crm.search_contacts", "description": "搜索 CRM 中的联系人"},
    {"name": "crm.create_lead", "description": "创建新的销售线索"},

    # 计费域
    {"name": "billing.get_invoice", "description": "获取发票详情"},
    {"name": "billing.process_refund", "description": "处理退款"},

    # 物流域
    {"name": "shipping.track_package", "description": "追踪包裹状态"},
]
```

命名空间帮助 LLM 理解工具的归属领域，减少跨域误选。

### 处理 LLM 输出验证

LLM 生成的参数可能不完全符合 Schema。始终在执行前验证：

```python
import jsonschema

def validate_and_execute(tool_name: str, llm_args: dict):
    schema = tool_schemas[tool_name]["input_schema"]

    # 1. Schema 验证
    try:
        jsonschema.validate(llm_args, schema)
    except jsonschema.ValidationError as e:
        return {"error": f"参数验证失败: {e.message}"}

    # 2. 安全检查（防注入）
    if contains_injection(llm_args):
        return {"error": "检测到潜在注入攻击"}

    # 3. 执行
    return tools[tool_name](**llm_args)
```

### 常见的 Schema 设计模式

```python
# 模式 1：确认型工具（危险操作前确认）
{
    "name": "delete_record",
    "description": "删除数据库记录。这是不可逆操作，调用前请确认用户意图。",
    "input_schema": {
        "properties": {
            "record_id": {"type": "string"},
            "confirmation": {
                "type": "string",
                "enum": ["confirmed"],
                "description": "必须传入 'confirmed' 才会执行删除"
            }
        },
        "required": ["record_id", "confirmation"]
    }
}

# 模式 2：分页型工具
{
    "name": "list_items",
    "input_schema": {
        "properties": {
            "page": {"type": "integer", "default": 1, "minimum": 1},
            "page_size": {"type": "integer", "default": 20, "maximum": 100}
        }
    }
}
```

## 常见误区 / 面试追问

1. **误区："名称比描述重要"** — 恰恰相反。描述是 LLM 决定是否使用工具的核心依据。名称只是标识符。好的描述应明确说明用途、触发条件和排除条件。

2. **误区："Schema 越详细越好"** — 过度复杂的 Schema 会消耗上下文窗口且增加 LLM 出错概率。保持简洁，只定义必要的参数。

3. **追问："工具太多（50+）怎么办？"** — (1) 用命名空间分组；(2) 动态加载——根据用户意图只加载相关工具；(3) 用 Embedding 做工具检索（RAG for tools）；(4) 分层——先让 LLM 选类别，再加载该类别的工具。

4. **追问："如何测试工具 Schema 的质量？"** — 构建测试用例集：一组用户查询 + 期望被选中的工具。运行后检查 LLM 的工具选择准确率。低于 90% 就需要优化描述。

## 参考资料

- [How JSON Schema Works for LLM Tools & Structured Outputs (PromptLayer)](https://blog.promptlayer.com/how-json-schema-works-for-structured-outputs-and-tool-integration/)
- [Function Calling (OpenAI API Docs)](https://platform.openai.com/docs/guides/function-calling)
- [Schema Generation for LLM Function Calling (Medium)](https://medium.com/@wangxj03/schema-generation-for-llm-function-calling-5ab29cecbd49)
- [awesome-llm-json (GitHub)](https://github.com/imaurer/awesome-llm-json)
- [Configure Tool and Function Calling for LLMs (Anyscale)](https://docs.anyscale.com/llm/serving/tool-function-calling)
