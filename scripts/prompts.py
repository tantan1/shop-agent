#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI 编码流程的阶段提示词模板（集中管理 + 版本化）。

为什么存在（对应 22/23 理论）：
  prompt 是 AI 编码流程的 *输入约束*，散落在 agent_orchestrator.py 字符串里时，
  改一处可能影响所有任务却无感知、无回滚、无回归。
  集中到本模块后：
    - 每次改 prompt 走 git diff，可 review、可回滚；
    - 配 PROMPTS_VERSION，CI 跑 benchmark/eval 做改前/改后通过率回归；
    - CHANGELOG 记录每次语义变更，便于追溯"哪版 prompt 导致通过率退化"。

使用方式：
  from prompts import SCOPE_PROMPT, DESIGN_PROMPT, ...
  prompt = SCOPE_PROMPT.format(user_request=state.user_request)
"""
from __future__ import annotations

# ── 版本与变更日志 ──────────────────────────────────────────────────────────
PROMPTS_VERSION = "1.1.0"
"""
CHANGELOG
- 1.1.0 (2026-09-03) 方案 C：把 scope.md 打造成"质量锚点"。
  SCOPE_PROMPT 强制要求验收标准（可测试）、非功能约束、架构边界初判；
  DESIGN_PROMPT 要求逐条追溯 scope 验收标准与非功能约束；REVIEW_PROMPT
  要求严格对照 scope 验收标准/非功能约束判 [BLOCKING]。
- 1.0.0 (2026-09-01) 初始抽取：从 agent_orchestrator.py 把各阶段 prompt 模板集中管理。
  模板语义与抽取前保持一致（scope/design/code/fix/review/test 六阶段），未改措辞。
"""

# ── 阶段模板（占位符用 {name} 形式，调用方 .format() 填充） ──────────────────

# 阶段0：目标澄清 + scope（方案 C：scope.md 是后续所有阶段的质量锚点）
SCOPE_PROMPT = """为以下任务明确目标边界与验收标准（这是后续 design/code/review 的唯一质量锚点，务必写扎实）：

用户需求：{user_request}

请产出 scope.md，至少包含：
1. 目标：一句话说清"做完是什么"。
2. 范围（In scope）：本次要交付的能力清单，逐条可独立验收。
3. 非范围（Out of scope）：明确"这次不做"的相邻诉求，防范围蔓延。
4. 验收标准：每条范围项配 1-2 条可测试、可客观判定的验收条件（输入→预期输出/行为），
   禁止"可用""正确"等主观表述。
5. 非功能约束：性能（延迟/吞吐）、安全（鉴权/数据保护）、兼容性、可观测性（日志/指标）
   等硬约束逐条列出，review 阶段将据此判 [BLOCKING]。
6. 架构边界初判：涉及哪些 app/模块、对外接口边界、复用既有资产的情况。"""

# 阶段1：架构设计（触发：无现成设计文档）
DESIGN_PROMPT = """依据 scope（{scope_ref}）从零设计系统架构。design.md 必须逐条回应 scope，否则 review 会判 [BLOCKING]：

1. 模块划分与职责，产出架构图/目录结构说明。
2. API 契约（入参/出参/错误码）与技术栈清单。
3. 验收追溯：把 scope.md 的「验收标准」逐条映射到本设计的具体决策点
   （指明"哪条设计保证了哪条验收"）。
4. 非功能约束落实：逐条说明 scope 的「非功能约束」如何在设计中满足
   （性能如何达标、安全如何防护、可观测性如何落地）。
5. 风险与权衡：关键取舍、回退方案。

注意：本设计产出后将进入「设计门禁」——需人工 review 确认后才放行编码。"""

# 阶段2：编码实现
CODE_PROMPT = """依据设计文档与 scope 实现代码。
scope: {scope_ref}
design:
{design}
用户需求：{user_request}"""

# 阶段2-fix：审查阻塞后自动回退重做
FIX_PROMPT = """根据审查阻塞问题修复代码：
原代码：{code}
问题：
{issues}"""

# 阶段4：代码审查
REVIEW_PROMPT = """审查以下代码变更，严格对照 scope.md 的验收标准与非功能约束：
- 功能正确性：是否满足 scope 的「验收标准」（逐条核对，不满足即 [BLOCKING]）。
- 非功能：是否满足 scope 的「非功能约束」（性能/安全/可观测性等，违反即 [BLOCKING]）。
- 范围边界：是否超出 scope 的「范围」或动了「非范围」项，越界即 [BLOCKING]。
- 设计一致性：是否与 design.md 的架构/契约一致。

代码：
{code}"""

# 阶段5：测试生成
TEST_PROMPT = """为以下代码生成测试：
{code}"""

# 契约优先提示（第1层免审：确定性工具生成测试）
CONTRACT_FIRST_HINT = (
    "检测到 OpenAPI 契约，建议走 schemathesis 等确定性契约工具生成测试（第1层免审）；"
    "如改测试需声明并分离 PR。"
)
