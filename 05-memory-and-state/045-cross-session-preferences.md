# 如何实现跨会话的用户偏好学习？

> 难度：中级
> 分类：Memory & State

## 简短回答

跨会话用户偏好学习使 Agent 能像人类助手一样"了解用户"——记住用户的习惯、风格偏好和工作方式，并在后续交互中自动适配。核心技术架构是**持久化记忆 + 偏好提取 + 动态适配**：从对话中自动提取用户偏好（显式和隐式），存入持久化存储，在新会话开始时检索并注入上下文。前沿研究包括 **PAMU（偏好感知记忆更新）**——用滑动窗口 + 指数移动平均融合短期波动和长期趋势，以及 **Memory-R1**——用强化学习训练 Agent 学习最优的记忆操作策略。

## 详细解析

### 偏好学习的闭环

```
会话 1                    会话 2                    会话 N
用户交互 → 偏好提取       检索偏好 → 个性化响应      偏好持续进化
    ↓                        ↑                        ↑
[持久化存储] ←──── 更新 ────┘──── 更新 ────────────────┘
```

### 偏好的类型

```python
preference_types = {
    # 显式偏好：用户直接说出的
    "explicit": [
        "请用中文回复",
        "我喜欢简洁的回答",
        "代码示例用 Python",
    ],

    # 隐式偏好：从行为推断的
    "implicit": [
        "用户总是追问技术细节 → 偏好深度分析",
        "用户经常要求修改语气 → 偏好正式/非正式风格",
        "用户多次选择方案 A 而非 B → 偏好某种技术栈",
    ],

    # 进化偏好：随时间变化的
    "evolving": [
        "三个月前用 React → 最近开始学 Vue",
        "从初学者成长为中级开发者",
    ],
}
```

### 实现步骤

#### Step 1：偏好提取

```python
class PreferenceExtractor:
    """从对话中提取用户偏好"""

    async def extract(self, conversation: list) -> list:
        prompt = f"""
        分析以下对话，提取用户的偏好和特征。
        只提取明确表达或可以从行为合理推断的偏好。

        对话：
        {self.format(conversation)}

        输出格式（JSON）：
        [
            {{"category": "技术栈", "preference": "偏好 Python", "confidence": 0.9, "source": "explicit"}},
            {{"category": "回复风格", "preference": "喜欢简洁回答", "confidence": 0.7, "source": "implicit"}}
        ]
        """
        return await self.llm.invoke(prompt)
```

#### Step 2：持久化存储

```python
class UserPreferenceStore:
    """用户偏好的持久化存储"""

    def __init__(self, db):
        self.db = db

    async def save_preference(self, user_id: str, pref: dict):
        existing = await self.get_similar(user_id, pref)

        if existing:
            # 偏好已存在 → 更新置信度和时间戳
            await self.update(existing["id"], {
                "confidence": max(existing["confidence"], pref["confidence"]),
                "last_confirmed": datetime.now(),
                "confirmation_count": existing["confirmation_count"] + 1
            })
        else:
            # 新偏好 → 直接存储
            await self.db.insert({
                "user_id": user_id,
                "category": pref["category"],
                "preference": pref["preference"],
                "confidence": pref["confidence"],
                "created_at": datetime.now(),
                "last_confirmed": datetime.now(),
                "confirmation_count": 1
            })

    async def get_preferences(self, user_id: str) -> list:
        """获取用户的活跃偏好"""
        prefs = await self.db.find({
            "user_id": user_id,
            "confidence": {"$gte": 0.5}  # 只返回高置信度偏好
        })
        return sorted(prefs, key=lambda p: p["confidence"], reverse=True)
```

#### Step 3：动态适配

```python
class PersonalizedAgent:
    """基于偏好的个性化 Agent"""

    async def respond(self, user_id: str, message: str):
        # 1. 加载用户偏好
        preferences = await self.pref_store.get_preferences(user_id)

        # 2. 构建个性化 System Prompt
        pref_context = self.format_preferences(preferences)
        system_prompt = f"""
        你是一个个性化助手。以下是该用户的已知偏好：
        {pref_context}

        请根据这些偏好调整你的回复风格和内容。
        如果偏好与当前请求无关，忽略即可。
        """

        # 3. 生成个性化回复
        response = await self.llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ])

        # 4. 从本次交互中提取新偏好
        new_prefs = await self.extractor.extract([message, response])
        for pref in new_prefs:
            await self.pref_store.save_preference(user_id, pref)

        return response
```

### 前沿研究：PAMU（偏好感知记忆更新）

PAMU 解决了偏好变化的追踪问题——用户偏好不是静态的：

```python
class PreferenceAwareMemory:
    """PAMU: 融合短期波动和长期趋势"""

    def __init__(self, alpha=0.3):
        self.alpha = alpha  # EMA 平滑因子

    def update_preference(self, user_id, new_signal):
        # 滑动窗口平均（捕捉短期变化）
        sw_avg = self.sliding_window_average(
            self.recent_signals[user_id], window=5
        )

        # 指数移动平均（捕捉长期趋势）
        ema = self.ema_values.get(user_id, new_signal)
        ema = self.alpha * new_signal + (1 - self.alpha) * ema
        self.ema_values[user_id] = ema

        # 融合两种信号
        fused = 0.6 * ema + 0.4 * sw_avg

        # 检测偏好变化
        if self.detect_shift(user_id, fused):
            self.trigger_adaptation(user_id, fused)

    def detect_shift(self, user_id, current):
        """检测渐变和突变"""
        history = self.preference_history[user_id]
        # 突变检测：当前值与历史均值偏差大
        if abs(current - np.mean(history)) > 2 * np.std(history):
            return True
        # 渐变检测：最近值（含当前）呈连续上升趋势
        recent = list(history[-4:]) + [current]
        if len(recent) >= 5 and all(
            recent[i] < recent[i+1] for i in range(len(recent) - 1)
        ):
            return True
        return False
```

### User Profile 动态构建

```python
class DynamicUserProfile:
    """动态演进的用户画像"""

    def __init__(self, user_id):
        self.profile = {
            "user_id": user_id,
            "technical_level": "unknown",     # 初始未知
            "preferred_language": "unknown",
            "communication_style": "unknown",
            "domain_expertise": [],
            "interaction_history_summary": "",
            "last_updated": None
        }

    async def evolve(self, new_interaction):
        """每次交互后更新画像"""
        prompt = f"""
        当前用户画像：{json.dumps(self.profile)}

        最新交互：{new_interaction}

        请更新用户画像。规则：
        1. 只更新有明确证据支持的字段
        2. 将 "unknown" 更新为具体值
        3. 如果新信息与旧画像矛盾，以新信息为准
        4. 不要凭猜测填充字段
        """
        self.profile = await llm.invoke(prompt)
```

### MultiSessionCollab 研究结果

研究表明：
- 记忆使 Agent 能持续学习用户交互偏好，提升协作质量
- 结果：更高的任务成功率、更高效的交互、更少的用户纠正
- 基于 RL 的框架训练 Agent 生成更全面的反思和更有效的记忆更新

此外，**[Memory-R1](https://arxiv.org/abs/2508.19828)** 进一步验证了 RL 在记忆管理中的有效性——它通过强化学习训练 Memory Manager（学习 ADD/UPDATE/DELETE 操作）和 Answer Agent（学习检索和推理），仅用 152 条训练样本即在多个长期记忆基准上超越 Mem0 等强基线。

## 常见误区 / 面试追问

1. **误区："直接把所有对话历史存起来就是偏好学习"** — 对话历史 ≠ 偏好。需要主动提取和结构化。存储原始对话占用大量空间且检索效率低。偏好应该是精炼后的结构化信息。

2. **误区："偏好一旦提取就不变了"** — 用户偏好会进化。三个月前用 React 的用户可能已经转向 Vue。需要偏好衰减机制和变化检测（如 PAMU）。

3. **追问："如何处理多用户共享 Agent 的场景？"** — Collaborative Memory 框架提出了基于二部图的动态访问控制——不同用户的记忆隔离，但在授权范围内可共享。例如团队共享的项目偏好 vs 个人偏好需要区分。

4. **追问："偏好学习的评估指标是什么？"** — (1) 适配准确率：Agent 的回复是否符合用户偏好；(2) 用户纠正频率：用户需要纠正 Agent 的次数是否随会话数减少；(3) 任务完成效率：个性化后任务是否完成更快。

## 参考资料

- [Preference-Aware Memory Update for Long-Term LLM Agents (arXiv)](https://arxiv.org/html/2510.09720)
- [Learning User Preferences Through Interaction for Long-Term Collaboration (arXiv)](https://arxiv.org/html/2601.02702v1)
- [Toward Personalized LLM-Powered Agents (arXiv)](https://arxiv.org/html/2602.22680)
- [Enabling Personalized Long-term Interactions Through Persistent Memory (arXiv)](https://arxiv.org/abs/2510.07925)
- [Collaborative Memory: Multi-User Memory Sharing with Dynamic Access Control (arXiv)](https://arxiv.org/abs/2505.18279)
