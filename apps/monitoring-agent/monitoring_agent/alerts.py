"""告警摄入与入站脱敏（monitoring-agent 的「被叫醒」入口）。

数据接口（04 篇 §5/§6）：
- 指标通道：Prometheus → Alertmanager → webhook ``POST /ingest/alert``
- 日志通道：Loki → Alertmanager → webhook（同源，同一端点）
- LLM 通道应急口：Langfuse webhook ``POST /ingest/event``（不经 Alertmanager，
  不经网关 /v1，不污染业务计量）

安全约束（06/07 跨平面）：告警 payload 可能含用户标识 / 请求片段，推送前若未脱敏，
agent 在入站即做脱敏，**确保 PII 不落到任何落库 / 外发路径**。agent 自身是无状态
分析层（§9），只持久化派生产物（分级结果 / 处置建议），不持久化原始告警流。
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger("monitoring_agent.alerts")

# 入站脱敏：覆盖告警/日志 payload 里常见的 PII 形态。属轻量确定性层（呼应 07 薄引擎），
# 不引入模型依赖；与网关侧 06/07 脱敏引擎同原则、独立实现（边界在告警管道）。
# 注意：确定性正则无法覆盖全部 PII，仅作为入站第一道闸；敏感日志仍应避免带原文。
_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# 仅匹配公网 IPv4（收窄，避免误伤内网/保留段：10/127/172/192/169 开头）。
# 告警脱敏宁枉勿纵：凡首段属上述范围一律保留（不脱），仅脱明确公网段。
_IPV4_PUBLIC = re.compile(
    r"\b(?!10\.|127\.|172\.|192\.|169\.)(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
)
_TRACE_ID = re.compile(r"\b(trace|span)[_]?id\s*[:=]\s*[\"']?[0-9a-fA-F]{12,}[\"']?", re.I)
_PHONE = re.compile(r"\b1[3-9]\d{9}\b")
# 中国大陆身份证（18 位，含可选的 X 校验位）
_IDCARD = re.compile(r"\b\d{17}[\dXx]\b")
# 银行卡（15-19 位连续数字，非被其他编号包含时）
_BANKCARD = re.compile(r"\b(?:62\d{14,17}|[3-6]\d{13,17})\b")
# 常见密钥/令牌（高熵凭证字样）
_SECRET = re.compile(
    r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd|access[_-]?key|authorization
    |bearer)\b\s*[:=]\s*["']?[A-Za-z0-9\-_]{8,}["']?"""
)
_MASK = "***"
_MAX_DEPTH = 12  # 递归深度上限，防恶意超深嵌套


def redact(text: str) -> str:
    """对单段文本做轻量 PII 脱敏，返回脱敏后副本。"""
    if not isinstance(text, str) or not text:
        return text
    text = _EMAIL.sub(_MASK, text)
    text = _PHONE.sub(_MASK, text)
    text = _IDCARD.sub(_MASK, text)
    text = _BANKCARD.sub(_MASK, text)
    text = _SECRET.sub(lambda m: f"{m.group(0).split(':')[0].split('=')[0]}=***", text)
    text = _IPV4_PUBLIC.sub(_MASK, text)
    text = _TRACE_ID.sub(_MASK, text)
    return text


def redact_obj(obj, _depth: int = 0) -> object:
    """递归脱敏任意 JSON 结构（dict / list / str），原结构类型保持。

    带递归深度上限，防御超深嵌套 payload 触发的栈溢出 / 资源耗尽。
    """
    if _depth >= _MAX_DEPTH:
        return _MASK if isinstance(obj, (dict, list)) else obj
    if isinstance(obj, dict):
        return {k: redact_obj(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, _depth + 1) for v in obj]
    if isinstance(obj, str):
        return redact(obj)
    return obj


# ── 统一内部事件模型 ──────────────────────────────────────────────────────
class IngestEvent:
    """从 Alertmanager webhook 或 Langfuse 应急口归一化的内部事件。"""

    def __init__(
        self,
        source: str,
        alerts: list[dict],
        raw: dict,
    ) -> None:
        self.source = source  # "alertmanager" | "langfuse"
        self.alerts = alerts
        self.raw_redacted = raw  # 已脱敏的原始结构（仅用于审计派生，不落库原文）

    def severity(self) -> str:
        """取最高严重度：P0>P1>P2>P3。"""
        rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        best = "P3"
        for a in self.alerts:
            sev = (a.get("labels", {}).get("severity") or "P3").upper()
            if rank.get(sev, 9) < rank.get(best, 9):
                best = sev
        return best

    def summary(self) -> str:
        titles = [a.get("annotations", {}).get("summary") or a.get("labels", {}).get("alertname", "")
                  for a in self.alerts]
        return "; ".join(t for t in titles if t)

    def implicated_services(self) -> list[str]:
        """告警涉及的组件（7.2：RCA 动态查询日志的 service 列表）。

        从告警 labels 提取：优先 ``service_name``（Loki 规则注入），
        兜底 ``service`` / ``job``（Prometheus 规则注入），去空去重。
        """
        services: list[str] = []
        for a in self.alerts:
            labels = a.get("labels", {}) or {}
            for key in ("service_name", "service", "job"):
                val = (labels.get(key) or "").strip()
                if val and val not in services and val != "unknown_service":
                    services.append(val)
        return services


def parse_alertmanager(payload: dict) -> IngestEvent:
    """解析 Alertmanager webhook（标准 schema：{alerts:[...]}）。

    入站即对整个 payload 做脱敏，杜绝 PII 进入后续分析/落库路径。
    """
    raw_redacted = redact_obj(payload)
    alerts = []
    for a in payload.get("alerts", []) or []:
        alerts.append({
            "name": a.get("labels", {}).get("alertname", "unknown"),
            "severity": (a.get("labels", {}).get("severity") or "P3").upper(),
            "summary": a.get("annotations", {}).get("summary", ""),
            "description": a.get("annotations", {}).get("description", ""),
            "labels": a.get("labels", {}),
        })
    return IngestEvent("alertmanager", alerts, raw_redacted)


def parse_langfuse(payload: dict) -> IngestEvent:
    """解析 Langfuse 应急口（异构 schema，归一化为单条 alert 形态）。

    应急口不经 Alertmanager 路由，severity 固定 P2（LLM 质量/成本异常，
    非核心链路中断），可由 payload 覆盖。
    """
    raw_redacted = redact_obj(payload)
    sev = (payload.get("severity") or "P2").upper()
    alerts = [{
        "name": payload.get("event") or payload.get("type") or "langfuse_event",
        "severity": sev,
        "summary": payload.get("summary") or json.dumps(payload, ensure_ascii=False)[:200],
        "description": payload.get("description", ""),
        "labels": {"source": "langfuse", **payload.get("labels", {})},
    }]
    return IngestEvent("langfuse", alerts, raw_redacted)
