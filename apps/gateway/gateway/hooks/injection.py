"""① 入向/回流注入第一道闸（01 §2.1 / 10 §1）。

请求进网关即过检测器：user 角色命中即拦（deny），业务无感；
assistant/tool/system 角色命中不阻断，标记为不可信（flagged）+ 告警打点，
由上游做隔离/人在回路确认（信任模型分级处置）。

批次2/10 落地 `RegexInjectionGate`（确定性正则层）：扫描 `messages[].content`
与 `tool_calls[].function.arguments` 等文本，按 `applies_to`（角色白名单）分级处置。

设计约束（scope-10 §3）：
- 规则加载复用公共包 `rules/store.py` 的 `RulesStore`（`injection` 独立实例，
  **不 import `pii/`**）——基线随包 `rules/injection.baseline.yaml` → 快照兜底 →
  中央源仅预留；绝不零规则启动。
- D4（已确认）：`check` 必须 `async`，调用方一律 `await`，与治理链统一。
- 热路径零联网：规则在构造期 `load()`，热路径只读内存副本。
"""
from __future__ import annotations

import hashlib
import json as _json
import re

from ..metrics import register_counter
from ..rules.store import RulesLoadError, RulesStore
from ..types import Verdict

# 注入命中打点（scope-10 §5 降级告警）：{stage, mode} 两维。
# closed 下拦截计数；open 下放行+告警计数（两模式都计，避免 fail-open 掩盖攻击）。
injection_denied_total = register_counter(
    "gateway_injection_denied_total",
    "injection gate denials by stage and fail mode",
    ("stage", "mode"),
)

# 出向（非 user 角色）不可信命中打点：{role, rule} 两维，供监控间接注入面。
injection_flagged_total = register_counter(
    "gateway_injection_flagged_total",
    "injection gate flagged (non-user role) hits by role and rule",
    ("role", "rule"),
)

# 四级角色信任度（按不可信度升序）：system < assistant < tool < user。
# 当前处置只区分 user（deny）与非 user（flag），此常量供未来细化与可读。
_TRUSTED_ROLES = ("system", "assistant", "tool", "user")


class InjectionGate:
    """注入检测接口契约（scope-10 §3 接口契约）。

    所有实现须满足 `check(payload) -> Verdict`：`scan messages[].content` 与
    `tool_calls[].function.arguments` 等文本，按角色分级处置
    （user→deny；其余→flagged），且热路径零联网、可返回 allow。
    未来语义层（本地轻量分类器）作为同接口另一实现即可平替，编排与
    fail_mode 逻辑零改动。
    """

    async def check(self, payload: dict) -> Verdict:
        raise NotImplementedError


class RegexInjectionGate(InjectionGate):
    """确定性正则注入闸（已知攻击模式）。

    预置模式取自社区成熟规则（`rules/injection.baseline.yaml`，不自手搓）。
    每条规则带 `applies_to`（角色白名单），缺省对所有角色生效。
    """

    def __init__(self, name: str = "injection") -> None:
        self._store = RulesStore(name)
        # 构造即加载：基线缺失→RulesLoadError→应用启动失败（绝不零规则启动）。
        rules = self._store.load()
        self._patterns = self._compile(rules.get("patterns", []))

    @staticmethod
    def _compile(patterns: list[dict]) -> list[tuple[str, re.Pattern[str], tuple[str, ...]]]:
        compiled: list[tuple[str, re.Pattern[str], tuple[str, ...]]] = []
        for item in patterns:
            if not isinstance(item, dict):
                continue
            ptype = str(item.get("type", "unknown"))
            regex = item.get("regex")
            if not isinstance(regex, str):
                continue
            # applies_to：角色白名单；缺省/空 = 对所有角色生效。
            applies_to_raw = item.get("applies_to")
            if applies_to_raw is None:
                applies_to: tuple[str, ...] = ()  # 空元组 = 全角色
            elif isinstance(applies_to_raw, list):
                applies_to = tuple(str(r) for r in applies_to_raw)
            else:
                applies_to = (str(applies_to_raw),)
            try:
                compiled.append((ptype, re.compile(regex), applies_to))
            except re.error:
                # 单条坏规则跳过，不让整组失效（基线随包应已校验，运行期兜底）
                continue
        return compiled

    def _patterns_of(self) -> list[tuple[str, re.Pattern[str], tuple[str, ...]]]:
        return self._patterns

    def _applies(self, ptype: str, applies_to: tuple[str, ...], role: str) -> bool:
        """规则是否对该角色生效：空 applies_to = 全角色；否则需角色在白名单内。"""
        if not applies_to:
            return True
        return role in applies_to

    def _scan_text(
        self, text: str, role: str
    ) -> tuple[str | None, list[dict]]:
        """按角色扫描文本。

        返回 (first_deny_hit, flagged_hits)：
        - first_deny_hit：user 角色且命中其适用规则的第一个类型（供 deny）。
        - flagged_hits：非 user 角色命中其适用规则的全部标记（供告警/隔离）。
        绝不含命中原文（E3 纪律）：flag 仅带 rule_id / role / text_hash。
        """
        flagged: list[dict] = []
        deny_hit: str | None = None
        is_user = role == "user"
        for ptype, pat, applies_to in self._patterns_of():
            if not self._applies(ptype, applies_to, role):
                continue
            if pat.search(text):
                if is_user:
                    # user 角色：首个命中即足以 deny，但继续收集其余以打点（可选）。
                    if deny_hit is None:
                        deny_hit = ptype
                else:
                    # 非 user 角色：标记不可信，不阻断（避免自伤正常业务）。
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
                    flagged.append({"rule_id": ptype, "role": role, "text_hash": digest})
        return deny_hit, flagged

    async def check(self, payload: dict) -> Verdict:
        # ① 入向/回流 prompt：扫 messages[].content（按角色分级处置）
        # 信任模型：user 命中 → deny（攻击主入口，拦掉无损）；
        # system/assistant/tool 命中 → flagged（不可信标记 + 告警），
        #   因这些角色内容由平台拼装或外部产物产生，直接 deny 会自伤正常业务，
        #   且真正危险的是「被当指令执行」，应交由上游隔离/人在回路确认。
        all_flagged: list[dict] = []
        for msg in payload.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if not isinstance(role, str):
                continue
            content = msg.get("content")
            deny_hit = None
            if isinstance(content, str):
                deny_hit, flagged = self._scan_text(content, role)
                all_flagged.extend(flagged)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        d, flagged = self._scan_text(part["text"], role)
                        all_flagged.extend(flagged)
                        if d is not None and deny_hit is None:
                            deny_hit = d
            if deny_hit is not None:
                return Verdict.deny(reason=f"injection: {deny_hit}")

        # ③ 工具参数检测：扫 tool_calls[].function.arguments
        # arguments 是结构化参数（可能来自外部/回显系统提示词片段）。
        # 按角色分级：tool 调用参数视作 tool 角色（高风险外部数据）→ 命中则 flagged。
        for tc in payload.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                deny_hit, flagged = self._scan_text(args, "tool")
                all_flagged.extend(flagged)
                if deny_hit is not None:
                    return Verdict.deny(reason=f"injection: {deny_hit}")
            elif isinstance(args, dict):
                deny_hit, flagged = self._scan_text(
                    _json.dumps(args, ensure_ascii=False), "tool"
                )
                all_flagged.extend(flagged)
                if deny_hit is not None:
                    return Verdict.deny(reason=f"injection: {deny_hit}")

        # 出向命中（非 user 角色）：不阻断，标记不可信 + 打点告警。
        if all_flagged:
            for f in all_flagged:
                injection_flagged_total.inc(labels={"role": f["role"], "rule": f["rule_id"]})
            return Verdict.allow(reason="injection-flagged", flagged=all_flagged)

        return Verdict.allow("injection-clean")


# 注入闸实例（模块级，构造即加载规则；加载失败由应用启动捕获→fail-closed）。
# 注意：被 `proxy.py` 以 `await gate.check(...)` 调用（D4 async）。
try:
    gate: InjectionGate = RegexInjectionGate("injection")
except RulesLoadError as _e:
    # 基线缺失：不带着空规则静默放行，让导入失败冒泡至应用启动（fail-closed 纪律）。
    raise
