"""② 治理面钩子（01 §2.1）：脱敏/合规/语义缓存落点。

批次2（06/07/08/10）填充真实实现。本批次仍 passthrough，但**签名已定型**。

D4（已确认）：**所有治理钩子统一 async**。
理由：① B2 顺序要求 gate/guardrails/redact/cache/judge_egress 串成一条链，
同步异步混用会在流式路径（judge_egress 需 await）出错；② 未来任一实现要接
sidecar / 本地服务（Presidio、分类器）就必然变异步，届时改签名会波及所有调用方；
③ 同步实现包在 `async def` 里零成本（不 await 任何东西即可）。
故本批一次定死：**钩子方法一律 `async def`，调用方一律 `await`**。
注意：`async def` 内**禁止**做阻塞 IO（读规则文件走启动期加载 + 后台刷新，
不在热路径同步读盘），否则会卡住事件循环。

D3：钩子内**禁止自吞异常**，一律 `raise GovernanceError(stage, cause=e) from e`
向上抛，由 `controllers/proxy.py` 按 `GATEWAY_FAIL_MODE` 分流（C1）。
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from gateway.config import settings
from gateway.hooks.injection import gate as injection_gate
from gateway.metrics import register_counter
from gateway.pii import PiiEngine
from gateway.types import GovernanceError, GovernanceStage, Verdict, VerdictReview

# C1 治理钩子异常打点（scope-10 §4）：{stage, mode} 两维，closed/open 都计数。
governance_error_total = register_counter(
    "gateway_governance_error_total",
    "governance hook errors by stage and fail mode",
    ("stage", "mode"),
)

# 06 human_review 三通道之一：metrics 计数（E3，{rule, stage} 两维）。
# 注册在 governance 模块级（D2），避免与 08/10 争抢 metrics.py。
guardrails_review_total = register_counter(
    "gateway_guardrails_review_total",
    "human_review flagged texts by rule and stage",
    ("rule", "stage"),
)

# 06 合规模糊边界标记：结构化日志事件名（E3 三通道之二）。
logger = logging.getLogger("gateway.guardrails")


class GovernanceHooks:
    """治理钩子集合。

    2a 落好全部方法骨架（含 2b 三路的空实现），2b 三路只填各自方法体，
    不动类头与他人方法——消除 `hooks/governance.py` 的并行写冲突。
    """

    def __init__(self) -> None:
        # 2a：构造即加载脱敏引擎（规则缺失→RulesLoadError→应用启动失败，
        # 绝不带着空规则静默放行，满足 G2 门② 绝不零规则启动）。
        # pii_enabled=false 时跳过加载，钩子退化为透传（运维显式关闭）。
        self._engine: PiiEngine | None = None
        if settings.pii_enabled:
            self._engine = PiiEngine("pii")
            # 批次3-07：启动后台规则刷新（基线/快照文件变更运行期热生效）。
            # interval<=0 由引擎内部判定不启动。失败静默保留 last-known-good。
            self._engine.start_refresher(settings.pii_refresh_interval_sec)

    def _redact_text(self, text: str) -> str:
        if self._engine is None:
            return text
        masked, _ = self._engine.redact(text)
        return masked

    # --- 07 脱敏（2a 填充） ---

    async def run(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        """非流式：传入完整 body，返回处理后的完整 body。

        stage: "ingress" 前置脱敏 prompt / "egress" 后置兜底掩码模型复读。
        2a 接入 `PiiEngine.redact`（确定性层）；命中即掩码，不跨字段缓冲。

        支持两种形态：
        - 完整 chat completion 响应体 ``{"choices":[{"message":{"content":...}}]}``
          （proxy 出向走此路径）→ 逐条脱敏各 choice 的 ``message.content``；
        - 单条 ``{"content":...}``（兼容单元调用）→ 仅脱敏该 content。
        """
        if self._engine is None:
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("choices"), list):
            new_choices = []
            for ch in payload["choices"]:
                if (
                    isinstance(ch, dict)
                    and isinstance(ch.get("message"), dict)
                    and isinstance(ch["message"].get("content"), str)
                ):
                    new_msg = {**ch["message"], "content": self._redact_text(ch["message"]["content"])}
                    new_choices.append({**ch, "message": new_msg})
                else:
                    new_choices.append(ch)
            return {**payload, "choices": new_choices}
        content = payload.get("content")
        if isinstance(content, str):
            payload = {**payload, "content": self._redact_text(content)}
        return payload

    async def run_stream(self, stage: str, chunk: dict[str, Any]) -> dict[str, Any]:
        """流式：逐块调用，传入单个 SSE chunk，返回（可能改写后的）chunk。

        - 脱敏命中时就地掩码，不跨块缓冲。
        - 无法独立判定的跨块完整 PII：返回原 chunk 并打标，由调用方按 fail 决策处理。
        """
        if self._engine is None:
            return chunk
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            new_choices = []
            for ch in choices:
                if not isinstance(ch, dict):
                    new_choices.append(ch)
                    continue
                delta = ch.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    new_ch = {
                        **ch,
                        "delta": {
                            **delta,
                            "content": self._redact_text(delta["content"]),
                        },
                    }
                    new_choices.append(new_ch)
                else:
                    new_choices.append(ch)
            return {**chunk, "choices": new_choices}
        return chunk

    # --- 06 合规护栏（2b 填充） ---

    @staticmethod
    def _blocklist_entries() -> list[tuple[str, bool]]:
        """解析 GUARDRAILS_BLOCKLIST。

        返回 [(词, fuzzy), ...]；词后缀 ``(fuzzy)`` 标记模糊边界（置信度边缘，
        放行 + human_review，不阻断），用于演示 E3 三通道的 review 落点。
        """
        out: list[tuple[str, bool]] = []
        for raw in (settings.guardrails_blocklist or "").split(","):
            entry = raw.strip()
            if not entry:
                continue
            if entry.lower().endswith("(fuzzy)"):
                out.append((entry[: -len("(fuzzy)")].strip(), True))
            else:
                out.append((entry, False))
        return out

    async def guardrails_check(self, stage: str, text: str) -> Verdict:
        """确定性违禁词/话题检测（06 §3）。

        - 精确命中违禁词 → ``Verdict.deny``（非流式 4xx / 流式截断+终止标记）。
        - 模糊边界命中（blocklist 中 ``(fuzzy)`` 后缀条目）→ 放行 + ``human_review``
          标记（不阻断）。三通道：响应头 ``X-Guardrails-Review``（proxy 注入）+ 结构化
          日志 ``guardrails_review`` 事件（含 rule_id/text_hash，不含原文）+ metrics
          ``gateway_guardrails_review_total{rule, stage}``。标记内容不带原文。
        - C1：钩子内禁止自吞异常，一律 ``raise GovernanceError(stage, cause=e) from e``。

        ``stage`` ∈ {ingress, egress, stream}（对应 ``GovernanceStage``）。
        """
        if not text:
            return Verdict.allow()
        entries = self._blocklist_entries()
        if not entries:
            return Verdict.allow()
        try:
            for word, fuzzy in entries:
                if not word:
                    continue
                if word in text:
                    if fuzzy:
                        # 模糊边界：放行 + 打标（不阻断）
                        rule_id = f"gr-fuzzy:{word}"
                        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                        guardrails_review_total.inc(
                            labels={"rule": rule_id, "stage": stage}
                        )
                        logger.info(
                            "guardrails_review",
                            extra={
                                "event": "guardrails_review",
                                "rule_id": rule_id,
                                "text_hash": text_hash,
                                "stage": stage,
                            },
                        )
                        return Verdict.allow(
                            review=VerdictReview(rule_id=rule_id, text_hash=text_hash)
                        )
                    # 硬违规：命中即拦（reason 只放规则 id，不带原文）
                    return Verdict.deny(reason=f"guardrails blocklist hit: {word}")
        except Exception as e:  # noqa: BLE001
            # C1：绝不自吞，上抛 GovernanceError（reason 不带原文）
            raise GovernanceError(stage, cause=e, reason="guardrails scan failed") from e
        return Verdict.allow()

    # --- 10 注入检测出向（2b 填充） ---

    async def judge_egress(self, response: dict[str, Any]) -> Verdict:
        """②出向注入闸（10 §2）：检测模型输出复读注入 payload / 危险内容。

        必须基于**未脱敏原文**判定（见 scope-10 出向不变量）：proxy 在调用本方法
        前尚未对 response 脱敏，故此处拿到的是模型原始输出。判定放行后，proxy
        才会把 response 送进脱敏外发。

        复用 `RegexInjectionGate`（同一份已知攻击模式字典）对输出原文检测——
        模型若复读 `[SYSTEM] 新指令` / `ignore previous instructions` / `rm -rf`
        等模式即 deny。
        """
        # 把响应规整成 injection gate 可扫的 payload 形状（仅取出站检测关心的字段）。
        scan_payload: dict[str, Any] = {"messages": [], "tool_calls": []}
        choices = response.get("choices")
        if isinstance(choices, list):
            for ch in choices:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    scan_payload["messages"].append({"content": msg["content"]})
                # 模型也可能通过 tool_calls 输出危险指令
                tcs = msg.get("tool_calls") if isinstance(msg, dict) else None
                if isinstance(tcs, list):
                    scan_payload["tool_calls"].extend(tcs)
        verdict = await injection_gate.check(scan_payload)
        if not verdict.allowed:
            return Verdict.deny(reason=verdict.reason)
        return Verdict.allow()

    # --- 08 语义缓存（2b 填充） ---

    async def cache_lookup(self, stage: str, payload: dict[str, Any]) -> str | None:
        """ingress 阶段查缓存，命中返回缓存答案（proxy 据此短路不进模型）。

        提取 prompt（messages[].content 拼接）+ 业务分片（cache-policy 解析），
        调 ``SemanticCache.lookup`` 做词频向量余弦近似命中。命中即返答案；
        未命中返回 None（proxy 继续进模型）。写操作分片（skip_write）由 store
        内部拒绝写入，故此处查也走正常 lookup 即可（写入面已锁死）。
        """
        from gateway.cache.policy import policy as cache_policy
        from gateway.cache.store import cache

        if not settings.cache_enabled:
            return None
        bucket = cache_policy.bucket_of(payload)
        if isinstance(payload, dict):
            msgs = payload.get("messages", []) or []
            prompt = "\n".join(
                m.get("content", "") for m in msgs if isinstance(m, dict) and isinstance(m.get("content"), str)
            )
        else:
            prompt = ""
        if not prompt:
            return None
        try:
            return cache.lookup(prompt, bucket, tenant="default")
        except Exception as e:  # noqa: BLE001
            # C1：缓存查询异常不可阻断主流程（fail-open 语义：查不到就进模型），
            # 但须上抛 GovernanceError 由 proxy 按 fail_mode 分流（closed 下可能转 502）。
            raise GovernanceError(stage, cause=e, reason="cache lookup failed") from e

    async def cache_store(self, stage: str, payload: dict[str, Any], answer: str) -> None:
        """egress 治理链**全部通过后**写缓存（B3 前提）。经 pii_gate 门禁 + TTL。

        调用方（proxy）保证：仅当 judge_egress → guardrails → 脱敏 全部通过后才调用，
        故 answer 已是治理后干净内容（命中直接返回不重跑治理）。
        写操作分片（skip_write）/ PII 命中由 ``SemanticCache.store`` 内部硬约束拦截。
        """
        from gateway.cache.policy import policy as cache_policy
        from gateway.cache.store import cache

        if not settings.cache_enabled:
            return
        bucket = cache_policy.bucket_of(payload)
        if isinstance(payload, dict):
            msgs = payload.get("messages", []) or []
            prompt = "\n".join(
                m.get("content", "") for m in msgs if isinstance(m, dict) and isinstance(m.get("content"), str)
            )
        else:
            prompt = ""
        if not prompt or not answer:
            return
        try:
            cache.store(prompt, answer, bucket, tenant="default")
        except Exception as e:  # noqa: BLE001
            # C1：写入异常（如 PII 门禁内部故障）上抛，由 proxy 按 fail_mode 分流。
            raise GovernanceError(stage, cause=e, reason="cache store failed") from e


hooks = GovernanceHooks()
