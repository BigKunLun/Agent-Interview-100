# 主流 Coding Agent Harness 横评：Claude Code / Cursor / Aider / Cline / Codex CLI 在 Context / Tool / Permission / Sandbox 四维对比

> **难度**: 高级
> 🆕 2026 新增（Harness 主题）

## 简短回答

2026 年 Coding Agent Harness 已从"单一 CLI 工具"演化为多层架构产业。Claude Code（Anthropic）、Cursor（Cursor Inc.）、Aider（Paul Gauthier）、Cline（Cline.bot）、Codex CLI（OpenAI）这五大 harness 在四个维度上 **故意选择了完全不同的策略**：Context（Claude Code 自动 compact + Memory tool vs Aider tree-sitter + PageRank vs Cursor 云端 Composer vs Cline IDE context vs Codex AGENTS.md）、Tool（MCP / 自家 SDK / Function calling / Apply_patch）、Permission（Claude Code allow-list + Hook 硬拦 vs Cline 显式 Plan/Act vs Codex 三档 approval mode）、Sandbox（无沙箱 vs OS 原生 Seatbelt/bubblewrap vs 云端 VM）。任何"哪个最好"的回答都必须先问"在哪个维度"。Claude Code 的扩展生态（Skill / MCP / Slash / Hook / Subagent 五机制协同）是当前最完整、最值得深入研究的范本。

2026 年中值得关注的两个新动向：一是**工具融合趋势**——The New Stack 报道 Claude Code / Cursor / Codex 三大工具正通过 MCP、Skill 等共享开放标准"merge into one stack"，能力趋同而非分化；二是**自主性光谱的另一极**——Devin 2.0（2026.2）走"多模型全自主异步工程师"路线，自主任务完成率约 40%，与五大本地 harness 的"人机协作"路线形成对照。

**Cheat sheet**：
- **四维对比口诀**：Context（怎么记）/ Tool（怎么用）/ Permission（怎么审）/ Sandbox（怎么隔）
- **Context 三流派**：全塞 + compact（Claude Code）/ 检索式（Aider repo map）/ 云端（Cursor / Devin）
- **Sandbox 两路线**：OS 原生（Codex Seatbelt/bubblewrap）vs 云端 VM（Devin / Replit / Cursor 3）vs 无沙箱靠人审批（Cline / Claude Code）
- **Claude Code 五机制口诀**：MCP=plumbing / Skill=how-to / Slash=button / Hook=guardrail / Subagent=isolation
- **Hook vs Prompt 本质差**：Hook 确定性（regardless of model） / Prompt 概率性（model may skip）
- **关键数据**：Cline CLI + Claude Opus 4.7 在 Terminal Bench 2.0 跑出 74.2%，超过 Claude Code 同模型 69.4%
- **性价比维度（2026-09）**：GPT-5.6 家族进 Kiro 后 Terminal-Bench 2.1 成功任务成本降约 82%（OpenAI + AWS 联合优化）——harness 与模型协同调优的性价比实证
- **模型-工具面耦合（2026-09）**：Claude Agent SDK（TS）v0.3.268 起 TaskCreate/TodoWrite 类工具仅在旧模型（Claude 3.x 与 4.0–4.7 系）默认提供，新模型须显式列入 tools/allowedTools——升级模型即静默丢任务工具，模型与 harness 耦合的又一面
- **2026 三强新进展**：Claude Code（Ultrathink 深思考模式 + 语音输入，2026.3）/ Cursor 3（重建界面 + 并行 Agent 编排）/ Codex CLI（每周多 alpha 迭代）
- **2026 工具融合趋势**：三大 harness 通过 MCP、Skill 等共享标准走向 "merge into one stack"，能力趋同
- **自主性另一极**：Devin 2.0（2026.2，多模型，全自主异步工程师）自主完成率约 40%，$500/mo；社区共识是「组合两个工具」（如 Aider/Cline + Devin）而非单押

## 详细解析

### 一、五大 Coding Agent Harness 形态总览

| Harness | 形态 | 模型策略 | 标志特征 | SWE-bench Verified 2026-Q1 |
|---------|------|---------|---------|---------------------------|
| **Claude Code** | 全屏 CLI + IDE 集成 | 锁定 Claude（Opus 4.7 / Sonnet 4.6 / Haiku 4.5） | Hook + Skill + MCP + Subagent + Plugin 五件套；75% auto-compact；2026.3 新增 Ultrathink 深思考模式 + 语音输入 | Opus 4.7 87.6% / Sonnet 4.5 80.9% |
| **Cursor** | VS Code Fork + Composer | 自家 Composer 模型 + 多模型可选 | Cursor 3（2026-04）重建界面，引入 Agents Window（云端 VM）+ 并行 Agent 编排 | Composer 78.4%（Cursor 3 发布数据） |
| **Aider** | Git-first 终端 | BYOM（OpenAI / Claude / 本地） | tree-sitter + PageRank repo map | 70%+（Sonnet 4.6 配置） |
| **Cline** | VS Code 扩展 + 2026-05 独立 SDK | BYOM（任意） | Plan/Act 显式分离；checkpoint 每步可回滚 | Cline CLI + Opus 4.7 = Terminal Bench 2.0 74.2% |
| **Codex CLI** | Rust 终端 App | OpenAI（gpt-5 / o5 / codex models） | OS 原生 Seatbelt/bubblewrap 沙箱 + AGENTS.md 标准 | gpt-5-codex ~85% |

> Devin / Replit Agent 因为是"云端容器 VM + 浏览器入口"而非"本地 harness"，本题不作主比较对象，仅作为云端 sandbox 路线的参照。
>
> **Devin 2.0（2026.2）补充**：Cognition 把 Devin 定位为"全自主异步工程师"——2026.2 起支持多模型（不再锁单家），自主任务完成率约 40%，入门价 $500/mo。它代表与五大本地 harness 相反的一极：**用「自治长度 + 异步交付」换「人工介入」**。社区共识是顶级开发者倾向于**组合使用两个工具**（如 Aider + Devin、Cline + Devin）而非押注单一 Agent——本地协作做精细改动，云端 Devin 跑长周期大任务。Codex CLI 同期保持每周多个 alpha 版本的高频迭代。

### 二、四维对比详表

#### 维度 1：Context 管理

| Harness | 策略 | 关键机制 | Token 经济 |
|---------|------|---------|-----------|
| **Claude Code** | 全塞 + 自动 compact | 75% 利用率时 compact；Memory tool 持久化；Skill progressive disclosure | 200K 窗口，compact 后基本不丢；CLAUDE.md 自动 re-inject |
| **Cursor** | 云端 Composer 上下文池 | @-mention 文件、Composer 自动 retrieve；Cursor 3 多窗口并发 | 自家模型适配，不公开具体策略 |
| **Aider** | 外部检索 + 显式控制 | tree-sitter AST 解析全 repo → PageRank 排序 → token 预算内塞入 | 默认 `--map-tokens=1024`，无 chat 文件时膨胀 8x |
| **Cline** | IDE 上下文 + 文件批准 | 显式 add file / read file；每步全 transcript 可见 | 用户完全可控，配 BYOM 模型选 |
| **Codex CLI** | AGENTS.md + workdir 文件读取 | 自动读 monorepo 嵌套 AGENTS.md；32 KiB 硬上限静默截断 | AGENTS.md 是核心 context 入口 |

**关键对比 — Claude Code 的 "全塞 + compact" vs Aider 的 "检索式 repo map"**：

| | Claude Code 流派（全塞 + 智能压缩） | Aider 流派（外部检索） |
|---|---|---|
| 优势 | context 不丢，cross-file 推理强；模型直接看到原文 | 小窗口模型也能在大 repo 工作；显式控制，可解释 |
| 劣势 | 200K 窗口 + compact 算法依赖；长 trajectory 可能丢早期细节 | tree-sitter 不支持的语言失效；仅给 outline 不给原文 |

#### 维度 2：Tool Registry & Discovery

| Harness | 工具协议 | Discovery | 扩展机制 |
|---------|---------|-----------|---------|
| **Claude Code** | 内置 + MCP（开放） + Skill（按需加载） | MCP server 自动发现；Skill metadata 启动时常驻 | MCP / Skill / Subagent / Hook / Plugin 五件套 |
| **Cursor** | 内置 + MCP（2025-10 起支持） | 编辑器 UI 配 MCP | Cursor Rules（`.cursor/rules/*.mdc`） |
| **Aider** | git + 终端 + LLM-driven file edit | 无插件机制，靠 prompt 触发 | 几乎不可扩展（哲学：保持简单） |
| **Cline** | 内置 read/write/exec + MCP | MCP marketplace；2026-05 SDK 后开放 plugin | `@cline/sdk` plugin 注册 tool / 监听事件 |
| **Codex CLI** | `shell_command` + `apply_patch`（深度耦合模型权重） | AGENTS.md / 自家 cookbook | Custom tools via `[tools.<name>]` 配置 |

**关键观察 — `apply_patch` 的 post-training coupling**：

Codex 模型在 post-training 阶段被 *绑定* 到 `apply_patch` 工具的语法（OpenCode 团队 + HumanLayer 双重证实）。要让 Codex 在自家 harness 里发挥水平，必须模仿 Codex 的 `apply_patch` 签名。**Claude Code 模型类似——Claude Opus 4.x / Sonnet 4.x 对 `str_replace_based_edit_tool` 与 `bash` 工具的偏好已写进权重**。这就是 SDK / harness / 模型三者深度耦合的本质。

#### 维度 3：Permission & Approval

| Harness | 默认审批策略 | 升级机制 | "硬拦"能力 |
|---------|------------|---------|-----------|
| **Claude Code** | Permission allowlist（`settings.json`） | `--dangerously-skip-permissions` 跳过 | **PreToolUse Hook `deny` 即使 bypass 模式也强拦** |
| **Cursor** | 编辑器内"Apply / Reject" diff | Composer 自动模式可批量批准 | 无强制硬拦 |
| **Aider** | 自动 commit（git audit trail） | `--yes` 跳过批准 | 无（靠 git revert） |
| **Cline** | **每动作弹批准框（Plan / Act 模式）** | Auto-approve / YOLO mode | Checkpoint 每步快照可回滚 |
| **Codex CLI** | 三档：`read-only` / `auto`（默认） / `full access` | `:workspace` / `:danger-full-access` profile | 沙箱内不需要硬拦（隔离） |

**关键观察 — Hook 的确定性 vs 提示词的概率性**（深刻教训）：

Claude Code 的 Hook 系统是 2026 业界讨论最多的设计点。在 prompt 里写"提交前必须跑 lint"——模型可能跳过（概率性）；写成 PreToolUse Hook（shell 脚本）——保证执行（确定性）。**HumanLayer 总结**："success is silent, failures are verbose"——成功不污染 context，失败把错误注入 agent loop 让模型修。这是把 Coding Agent 从"玩具"变成"生产工具"的关键。

#### 维度 4：Sandbox & Runtime

| Harness | 沙箱路线 | 实现 | 真实 CVE / 风险 |
|---------|---------|------|---------------|
| **Claude Code** | **无沙箱**（靠 permission allowlist + Hook） | host 直接执行；推荐配合 macOS Sandbox Manager | CVE-2025-66479（BashTool 空白 allowlist）/ SOCKS5 null-byte 注入（v2.0.24-v2.1.89） |
| **Cursor** | 本地无沙箱；Cursor 3 起 Agents Window = 云端 VM | 默认本地直接跑 | 与 Claude Code 类似风险面 |
| **Aider** | **无沙箱**（git 是唯一防线） | host 直接执行 | 依赖 git diff review 习惯 |
| **Cline** | **无沙箱**（靠 Plan/Act 显式批准 + checkpoint） | host 直接执行 | 显式批准是首要防线 |
| **Codex CLI** | **OS 原生沙箱** | macOS Seatbelt 框架 / Linux/WSL2 bubblewrap / Windows Restricted Token+ACL | `workspace-write` 默认网络关；`danger-full-access` 等于无 |

**三种沙箱路线的本质权衡**：

| | OS 原生（Codex） | 云端 VM（Devin/Replit/Cursor 3） | 无沙箱（Claude Code/Cline/Aider） |
|---|---|---|---|
| 优势 | 本地零成本；文件直接 share | 极致隔离（VM 级别）；可异步长跑（200 分钟自治） | 零延迟、零基础设施；用户拥有完整 host 能力 |
| 劣势 | 平台差异（macOS ≠ Linux）；复杂规则维护 | 需要 cold start（10-60s）；egress / data exfil 风险 | 必须配 permission + Hook 防线；用户审批疲劳是大问题 |

### 三、Claude Code 扩展生态深度剖析（五机制协同）

Claude Code 的扩展生态是 2026 最完整、最值得深入的 harness 设计范本。社区已总结的发布时间线：

```
2024-11  MCP (Model Context Protocol)            外部工具协议
2025-07  Subagents                                Context 隔离与并行
2025-09  Hooks                                    生命周期确定性触发
2025-10  Plugins                                  分发与打包
2025-10  Skills                                   Progressive Disclosure 知识包
2026-02  Agent Teams                              多 Subagent 编排
```

#### 五种机制的设计对照表

| 维度 | Slash Command | Skill | MCP | Subagent | Hook |
|------|--------------|-------|-----|----------|------|
| **触发** | 用户显式 `/cmd` | 模型自动判断 | 模型 tool_call | 任务委派（Agent tool） | 生命周期事件 |
| **加载成本** | ≈0（模板） | ~30-50 tokens metadata | 1k-50k+ tokens 起步 | 独立 context 隔离 | 注册时几乎为零 |
| **主要用途** | 重复 prompt 模板 | 流程化领域知识 | 外部世界访问 | 并行/隔离子任务 | 强制行为 |
| **文件结构** | 单 `.md` 文件 | 目录 + SKILL.md + 资源 | server 进程 | `.claude/agents/*.md` + frontmatter | `.claude/settings.json` hooks 字段 |
| **加载粒度** | 整体 | 三级渐进 | 全部 schema | 独立 system prompt | 不进 context |
| **确定性** | 用户主动（确定性） | 模型判断（概率性） | 模型决策（概率性） | 父 agent 决策（概率性） | **强制执行（确定性）** |

**记忆口诀**：
> **MCP = plumbing（管道）**
> **Skill = how to（手册）**
> **Slash = manual trigger（按钮）**
> **Subagent = isolation（隔间）**
> **Hook = guardrail（护栏）**

#### Skill 的 Progressive Disclosure 三级架构

Anthropic 把 Skill 拆为三层加载（官方文档明确）：

| 级别 | 加载时机 | Token 成本 | 内容 |
|------|---------|-----------|------|
| **L1 Metadata** | 启动时常驻 | ~50-100 tokens / Skill | YAML frontmatter 的 `name` + `description` |
| **L2 Instructions** | Claude 判断相关时读取 | ~275-8000 tokens | SKILL.md body 主体 |
| **L3 Resources** | bash 按需读取 | 无上限（不进 context） | bundled scripts / reference files |

**经济性证据**：17 个官方 skill 全加载只需 ~1700 token，等价于"知道几十种能力但只为一种付费"。

```yaml
# SKILL.md 必填字段示例
---
name: pdf-processing  # ≤64字符，小写/数字/连字符，禁用 "anthropic"/"claude"
description: Extract text and tables from PDF files. Use when working with PDFs.  # ≤1024字符
---

## Workflow

1. Use `python scripts/extract.py <file>` to extract text
2. ...
```

**核心机制**：Skill 不是把全部内容塞进 system prompt，而是 Claude 在 VM 中通过 bash `read SKILL.md` 显式读取。Script 文件在执行时只把 stdout 注入 context，代码本身不消耗 token。这是 2026 跨厂商采纳的第一个开放标准（OpenAI / Google / GitHub / Cursor 在两个月内集成）。

#### Hook 的 12+ 生命周期事件

Claude Code 官方支持的 hook 事件按 cadence 分类：

```
Per session:    SessionStart / SessionEnd / Setup
Per turn:       UserPromptSubmit / Stop / StopFailure
Per tool call:  PreToolUse / PostToolUse / PostToolBatch
Agent-related:  SubagentStart / SubagentStop / TaskCreated
MCP-related:    PermissionRequest / PermissionDenied / Elicitation
Async:          FileChanged / CwdChanged / PreCompact / PostCompact
```

5 种 handler 类型：`command`（shell） / `http`（POST 端点） / `mcp_tool` / `prompt`（单轮 LLM 评估） / `agent`（带工具的子 agent，实验性）。

**权限决策模型（PreToolUse 独有）**：

```json
// .claude/settings.json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "node /path/to/check-destructive.js"
          }
        ]
      }
    ]
  }
}
```

```javascript
// check-destructive.js — 示例 hook 输出
const input = JSON.parse(require('fs').readFileSync(0, 'utf-8'));
if (input.tool_input.command.match(/rm -rf|git push --force/)) {
  console.log(JSON.stringify({
    permissionDecision: "deny",  // 即使 --dangerously-skip-permissions 也硬拦
    reason: "Destructive command blocked by project policy"
  }));
  process.exit(0);
}
process.exit(0);  // allow
```

**Exit code 约定**（极易踩坑）：
- `0` = 成功（解析 stdout JSON）
- **`2` = blocking error（阻止动作）**
- 其他 = 非阻塞错误
- **注意**：`exit 1` 不阻断，必须 `exit 2` 才阻断

**Permission Decision 四值**：`allow` / `deny`（强拦） / `ask`（强制升级用户审批） / `defer`（走正常权限流程）。

#### Subagent vs MCP server 的边界

这是面试官最爱挖的细节问题：

| 维度 | Subagent | MCP server |
|------|---------|-----------|
| **本质** | Claude 的另一个实例 + 独立 context | 进程 / HTTP 服务 + 工具 schema |
| **解决的问题** | Context 隔离（防 poisoning）；并行探索 | 外部资源访问标准化（N×M 问题） |
| **数据传递** | 父发 prompt → 子返回 string | model `tool_call` → server response |
| **状态** | 子完整 context 不进父 context | server 自己管 state（HTTP）或 stdio |
| **典型用例** | 多文件代码重构、深度调研、并行验证 | 数据库查询、文件系统、GitHub、Sentry |
| **配置** | `.claude/agents/<name>.md` | `.claude/settings.json` 的 `mcpServers` |

**Subagent 的两种语义**：
- **默认 fresh-context**：每次启全新 200k 窗口，子任务跑完只返回最终 string，中间 tool_call 不进父 context
- **`CLAUDE_CODE_FORK_SUBAGENT=1` fork**：继承当前对话历史（基于已有上下文继续探索），代价是 fork 子 context 随父增长，且与 `--print` headless 模式不兼容

#### 实战例子：「提交前自动跑 lint + 写测试时按需加载测试规范」

这是 Scout-A Q3 的经典场景题，五机制如何协同：

```
需求拆解
├── lint 必须每次都跑                  → Hook（确定性）
│   └── PreToolUse on git commit / PostToolUse on Edit
├── 测试规范按需加载                   → Skill（progressive disclosure）
│   └── .claude/skills/testing-conventions/
├── 测试要跨多文件分析                  → Subagent（隔离）
│   └── .claude/agents/test-reviewer.md
├── 查 Sentry 报错                     → MCP（外部世界访问）
│   └── mcpServers: { sentry: { command: "...", args: [...] } }
└── 用户显式入口                       → Slash Command
    └── .claude/commands/run-tests.md → /run-tests
```

**错误用法（常见踩坑）**：
- 把 commit message format 做成 Skill → 应放 CLAUDE.md / AGENTS.md（频繁触发，不是按需）
- 指望 Skill 访问 GitHub → 应用 MCP（Skill 没工具能力，只有 bash 内 stdout）
- 把简单 prompt 模板做成 Skill → 应用 Slash Command（Skill 太重，过度设计）
- 在 prompt 里写"提交前必须跑 lint" → 应该是 Hook（模型可能跳过）

### 四、横向对比小结：选哪个 Harness？

```
团队选型决策树
│
├── 想要"最完整生态 + 深度可扩展"
│   └── Claude Code（五机制协同；锁 Claude 模型）
│
├── 想要"VS Code 原生体验 + 多窗口并发"
│   └── Cursor（Composer 自家模型 + Cursor 3 Agents Window）
│
├── 想要"git-first + 极简 + 多模型"
│   └── Aider（学习曲线最平，repo map 算法精妙）
│
├── 想要"显式 HITL + checkpoint 安全网"
│   └── Cline（Plan/Act 一等公民 + 2026-05 独立 SDK 跨 surface）
│
└── 想要"原生 OS 沙箱 + AGENTS.md 标准"
    └── Codex CLI（Rust 重写，Seatbelt/bubblewrap 隔离）
```

**SWE-bench Verified ≠ 唯一指标**：harness 的产品体验差异远大于跑分差异。Cursor 在交互流畅度上领先，Aider 在 git 集成与可解释性上领先，Claude Code 在生态深度上领先，Codex CLI 在沙箱安全上领先，Cline 在显式可控性上领先。**先问"我的场景看重哪个维度"，再选 harness**。

**2026 工具融合趋势（选型新变量）**：The New Stack 报道 Claude Code / Cursor / Codex 三大工具正通过插件和共享开放标准"merge into one stack"——MCP 让三者接同一批 server，Skill 这种 progressive disclosure 知识包在两个月内被 OpenAI / Google / GitHub / Cursor 集成，AGENTS.md / CLAUDE.md / `.cursor/rules` 也在收敛为相似的"项目指令文件"约定。**这意味着选型的护城河正从「功能差异」转移到「Context / Permission / Sandbox 的工程取舍与生态成熟度」**——能力趋同后，谁的可控性、可审计性、隔离强度更好，谁就更适合生产。另一条并行路线是**人机协作 vs 全自主**的分工：本地 harness（五大）做精细协作，Devin 2.0 这种全自主异步 agent 跑长周期大任务，社区主流做法是组合使用而非二选一。

## 常见误区 / 面试追问

- **误区 1：「Claude Code 五机制可以互相替代」** — 错。MCP / Skill / Slash / Subagent / Hook 各管一个独立的 context 问题，强行替代会过度设计或失效。常见错误：把 commit format 写成 Skill（应在 CLAUDE.md）、指望 Hook 给模型加知识（Hook 不进 context，应该是 Skill）、用 Skill 调 GitHub（应该用 MCP）。

- **误区 2：「Hook 和 Prompt 是一回事，反正都是让模型做什么」** — 这是 Coding Agent 从"玩具"到"生产工具"的核心分水岭。**Hook = 确定性（regardless of model behavior） / Prompt = 概率性（model may skip）**。在 CLAUDE.md 写"提交前必须跑 lint"——模型可能跳过；写成 PreToolUse hook——保证执行。这层分工不理解就做不出可靠的代码 Agent。

- **误区 3：「Cursor 3 的 Agents Window 让 Cursor 变成了 Devin」** — 不完全。Cursor 3 引入云端 sandbox VM，但**仍然以编辑器为主入口**；Devin 则是"全自治长跑"（200 分钟自治再交付）。两者形态不同：Cursor 是"IDE + 异步 agent 加成"，Devin 是"独立 SWE 代理"。

- **误区 4：「OpenAI Codex 模型可以无缝放到任何 harness 里」** — 错。Codex 模型在 post-training 阶段被绑定到 `apply_patch` 工具签名，OpenCode 团队发现脱离这个签名性能会掉。**Post-training coupling 是 2026 行业的硬约束**——模型与 harness 是深度耦合系统。

- **追问 1：「Aider 的 repo map 算法核心是什么？」** → 答：tree-sitter 解析所有源文件成 AST → 提取 `name.definition.*` 和 `name.reference.*` 两类 tag → NetworkX 构建文件依赖图（A 引用 B 定义的符号 → 有向边） → PageRank 排序（chat-mentioned files 50x 权重，well-named identifiers 10x 权重） → token 预算下二分查找最大塞入量。默认 `--map-tokens=1024`，无 chat 文件时膨胀 8x。这是"小窗口模型在大 repo 上能工作"的核心算法。

- **追问 2：「为什么 Claude Code 的 PreToolUse Hook `deny` 决策能强拦 `--dangerously-skip-permissions`？」** → 答：因为 Hook 是项目级别的**配置约束**而非"用户偏好"。Anthropic 的设计哲学是：项目所有者可以设置"用户即使加了 bypass flag 也不能越的红线"，这对企业 / 团队 / 开源项目至关重要。具体到代码，PreToolUse Hook 的 `permissionDecision: "deny"` 在 permission system 层面短路了所有 upstream 判断。

- **追问 3：「Skill 的 Progressive Disclosure 与 RAG 的检索式有什么区别？」** → 答：表面相似，本质不同。
  - **RAG**：query → embedding 检索 → top-k chunks 进 context；触发由"用户 query"驱动，粒度是 chunk
  - **Skill**：模型读到 frontmatter description → 自主判断 → bash 读 SKILL.md；触发由"模型判断"驱动，粒度是"完整流程文档 + scripts"
  - Skill 的关键创新是 **"模型选择加载什么知识"**，而 RAG 是"系统选择灌什么知识给模型"。前者更适合"流程性知识"（工作流、规范、检查清单），后者更适合"事实性知识"（文档、代码、数据）。

- **追问 4：「Cline 2026-05 把 runtime 从 IDE 剥离成 `@cline/sdk`，对行业意味着什么？」** → 答：意味着 Harness 进入"**runtime 独立化 + 多 surface 适配**"阶段。同一份 agent runtime 可跑 IDE / CLI / Web / CI，session 跨 surface 迁移（VSCode 启动的会话可在终端继续）。这预示 2026-2027 Harness 演进方向：**与产品形态解耦，与模型/工具/protocol 协同**。Claude Code 同期也在做类似分层（Claude Agent SDK 与 Claude Code CLI 解耦）。

- **追问 5：「Devin 与 Replit Agent 这种『云端长跑型』为什么没纳入横评？」** → 答：因为它们不是"本地 harness"而是"云端 agent 产品"，比较维度完全不同。Devin/Replit 的核心权衡是"自治长度 vs 失败重试经济学"（Devin 用 ACU 计费、Replit 200 分钟自治），而本题五个 harness 的核心权衡是"context / tool / permission / sandbox 四维选择"。两类产品形态适合两个不同问题：本地协作开发（Claude Code / Cursor / Aider / Cline / Codex CLI） vs 异步 SWE 代理（Devin / Replit）。**补充 Devin 2.0（2026.2）数据**：支持多模型后自主任务完成率约 40%、入门价 $500/mo——这个 40% 正是讨论「全自主 Agent 可靠性」与「为何仍需 HITL」的硬论据，社区主流做法是组合使用（本地 harness 做精细活 + Devin 跑大任务）而非二选一。

- **追问 6：「2026 年 Claude Code / Cursor / Codex 在融合，是不是很快会变成一个工具？」** → 答：能力在趋同，但**四维工程取舍不会合并**。The New Stack 报道的"merge into one stack"指的是**接口层标准化**——MCP 统一工具协议、Skill 统一知识加载、AGENTS.md/CLAUDE.md 统一项目指令；但每个 harness 在 Context（全塞+compact vs 检索式 vs 云端）、Permission（Hook 硬拦 vs 显式批准 vs 沙箱隔离）、Sandbox（无 vs OS 原生 vs 云端 VM）上的**根本赌注不同**，这些是写进架构和模型权重的取舍，不会因插件互通而消失。选型逻辑从「比谁功能多」转向「比谁的可控性、可审计性、隔离强度更适合生产」。

## 参考资料

### 综合横评
- [Requesty — Agentic Coding Tools Compared 2026](https://www.requesty.ai/blog/agentic-coding-tools-compared-2026-claude-code-cursor-codex-aider) — 五大 harness 详细产品对比
- [thoughts.jock.pl — AI Coding Harness Agents 2026](https://thoughts.jock.pl/p/ai-coding-harness-agents-2026) — 行业评论与排名
- [Artificial Analysis — Coding Agents Leaderboard](https://artificialanalysis.ai/agents/coding) — Terminal Bench / SWE-bench 实时跑分
- [OpenAI — Advancing price-performance for developers with GPT-5.6 in Kiro](https://openai.com/index/gpt-5-6-in-kiro/) — Terminal-Bench 2.1 成功任务成本降约 82%（2026-08-24）
- [Claude Agent SDK (TypeScript) Releases](https://github.com/anthropics/claude-agent-sdk-typescript/releases) — v0.3.268：Task/Todo 类工具默认面收窄到旧模型清单，新模型须显式声明（2026-09）
- [htek.dev — All Agent Harnesses: The Live Comparison](https://htek.dev/articles/all-agent-harnesses-live-comparison) — 持续更新的对比表

### 2026 竞争格局与工具融合
- [The New Stack — The AI Coding Tool Stack](https://thenewstack.io/ai-coding-tool-stack/) — Claude Code / Cursor / Codex 通过插件与共享标准 "merge into one stack" 的趋势报道
- [CosmicJS — Claude Code vs Codex vs Cursor](https://www.cosmicjs.com/blog/claude-code-vs-codex-vs-cursor) — 三强产品形态对比
- [Context Studios — AI Coding Agents Showdown 2026](https://www.contextstudios.ai/blog/ai-coding-agents-showdown-claude-code-vs-cursor-vs-codex-2026) — 含 Claude Code Ultrathink 模式解读
- [arcbjorn — State of CLI Coding Agents 2026](https://blog.arcbjorn.com/state-of-cli-coding-agents-2026) — CLI Agent 形态 2026 现状梳理

### Claude Code 五机制深度
- [Anthropic — Equipping Agents for the Real World with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — Progressive Disclosure 官方源头
- [alexop.dev — Understanding Claude Code's Full Stack](https://alexop.dev/posts/understanding-claude-code-full-stack/) — 七层扩展全栈梳理
- [code.claude.com/docs/en/hooks](https://code.claude.com/docs/en/hooks) — Hook 官方文档（12+ 事件、5 种 handler、Exit code 约定）
- [code.claude.com/docs/en/sub-agents](https://code.claude.com/docs/en/sub-agents) — Subagent 官方文档
- [SwirlAI Newsletter — Agent Skills Progressive Disclosure](https://www.newsletter.swirlai.com/p/agent-skills-progressive-disclosure) — 17 个官方 skill 的 token 经济实证
- [Morph — Claude Code Skills vs MCP vs Plugins](https://www.morphllm.com/claude-code-skills-mcp-plugins) — 三件套实战对比

### Aider Repo Map 算法
- [Aider — Building a Better Repo Map with Tree-Sitter](https://aider.chat/2023/10/22/repomap.html) — tree-sitter + PageRank 算法原文
- [DeepWiki — Aider Repository Mapping System](https://deepwiki.com/Aider-AI/aider/4.1-repository-mapping-system)

### Codex CLI 与 Sandbox
- [developers.openai.com/codex/concepts/sandboxing](https://developers.openai.com/codex/concepts/sandboxing) — Seatbelt / bubblewrap / Restricted Token 官方说明
- [developers.openai.com/codex/cli/features](https://developers.openai.com/codex/cli/features) — 三档 approval mode + permission profile
- [developers.openai.com/codex/guides/agents-md](https://developers.openai.com/codex/guides/agents-md) — AGENTS.md 标准

### Cline SDK 与 Plan/Act
- [Cline Blog — Introducing the Cline SDK (2026-05-13)](https://cline.bot/blog/introducing-cline-sdk-the-upgraded-agent-runtime) — Runtime 与 IDE 解耦
- [GitHub — cline/cline](https://github.com/cline/cline) — 5M+ 安装的 VS Code 扩展
- [MarkTechPost — Cline SDK Coverage](https://www.marktechpost.com/2026/05/14/cline-releases-cline-sdk-an-open-source-agent-runtime-now-powering-its-cli-and-kanban-with-ide-extensions-being-migrated/)

### Harness Engineering 理论
- [HumanLayer — Skill Issue: Harness Engineering for Coding Agents](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents) — "It's not a model problem. It's a configuration problem."
- [Addy Osmani — Agent Harness Engineering](https://addyosmani.com/blog/agent-harness-engineering/) — Ralph Loop / Sprint Contracts / Planner-Generator-Evaluator split

### Cursor & Devin & Replit 参照
- [Cognition — Devin's 2025 Performance Review](https://cognition.ai/blog/devin-annual-performance-review-2025)
- [Replit — Introducing Agent 3](https://blog.replit.com/introducing-agent-3-our-most-autonomous-agent-yet) — 200 分钟自治
- [AI Builder Club — Best AI Coding Agent 2026](https://www.aibuilderclub.com/blog/best-ai-coding-agent-2026) — 含 Devin 2.0 多模型、~40% 完成率、$500/mo 定价
- [Totalum — Best AI Coding Agents 2026](https://www.totalum.app/blog/best-ai-coding-agents-2026) — 2026 Coding Agent 评测（含「组合两工具」社区共识）

## 相关阅读

- [027 — MCP（Model Context Protocol）](../03-tool-use/027-model-context-protocol.md)：本题 Tool 维度依赖的协议标准
- [096 — Framework Overview（LangChain / LlamaIndex / Haystack）](096-framework-overview.md)：Framework 层横评，与本题 Harness 横评形成对照
- [098 — 框架 vs 自研](098-framework-vs-custom.md)：自研动机与本题"为什么大厂不用 LangGraph"互为补充
- [099 — OpenAI Agents SDK vs Claude Agent SDK](099-assistants-api-vs-claude-sdk.md)：SDK 层对比，本题更上一层（产品级 harness）
- [109 — 什么是 Agent Harness](../01-agent-architecture/109-what-is-agent-harness.md)：Harness 概念入门，本题的"基础篇"
