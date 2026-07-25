"""Grafana 告警推送（RCA 根因分析结果的对外出口）。

状态说明（v1.3.0）：
- **已弃用**：Grafana 的 ``POST /api/alerts``（unified alerting external alerts）仅显示在
  Alerting 面板，**不触发 Grafana notification policy**（通知只对内部规则生效），因此
  根因推送后无法经 Grafana 转发钉钉/Slack，价值有限。
- **替代方案**：RCA 结果改为经 ``/metrics`` 暴露为 Prometheus 指标（``rca_total`` 事件计数
  + ``rca_last_info`` 最近值 info gauge），由 Prometheus recording/alerting rules + Alertmanager
  承接告警与通知，Grafana 以 Prometheus 数据源做面板展示。见 monitoring/prometheus/rules/rca.yml。
- 本模块保留但默认禁用（GRAFANA_PUSH_ENABLED=false），仅作旁路回退。
"""

from __future__ import annotations

import logging
import os

import httpx

from . import alerts

logger = logging.getLogger("monitoring_agent.grafana")

GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000").rstrip("/")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY", "").strip()  # 服务账号 token（含 alert:write）
_PUSH_TIMEOUT = float(os.getenv("GRAFANA_PUSH_TIMEOUT_SEC", "10"))
# 推送开关：默认关闭（RCA 已改走 Prometheus 指标 + Alertmanager，见模块 docstring）。
# 如需旁路启用，设 GRAFANA_PUSH_ENABLED=true 并配置 GRAFANA_API_KEY。
_ENABLED = os.getenv("GRAFANA_PUSH_ENABLED", "false").lower() not in ("0", "false", "no")


class GrafanaUnavailable(RuntimeError):
    """Grafana 不可达或推送失败（非致命：RCA 降级为仅日志/API 输出）。"""


def push_rca(rca: "object") -> bool:
    """把 RcaResult 推为一条 Grafana 告警。成功返回 True，失败/禁用返回 False。

    ``rca`` 为 :class:`monitoring_agent.rca.RcaResult`，此处用鸭子类型避免循环导入。
    """
    if not _ENABLED:
        logger.info("Grafana 推送已禁用（GRAFANA_PUSH_ENABLED）")
        return False
    if not GRAFANA_API_KEY:
        logger.warning("GRAFANA_API_KEY 未配置，跳过推送")
        return False

    # 出站脱敏：根因 + 处置建议 + 证据文本，PII 不进 Grafana
    root_cause = alerts.redact(rca.root_cause)
    recommendations = [alerts.redact(r) for r in rca.recommendations]
    # 证据里可能有日志/拓扑原文，整体序列化为脱敏字符串
    import json as _json
    evidence_raw = _json.dumps(rca.evidence, ensure_ascii=False, default=str)
    evidence_safe = alerts.redact(evidence_raw)

    payload = {
        "alerts": [
            {
                "alertname": "MonitoringAgentRCA",
                "severity": rca.severity,
                "used_llm": str(rca.used_llm).lower(),
                "affected": ",".join(rca.affected) if rca.affected else "none",
                "root_cause": root_cause,
                "recommendations": " | ".join(recommendations),
                "evidence": evidence_safe[:4000],
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {GRAFANA_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=_PUSH_TIMEOUT) as c:
            r = c.post(f"{GRAFANA_URL}/api/alerts", json=payload, headers=headers)
            r.raise_for_status()
        logger.info("RCA 已推送 Grafana sev=%s", rca.severity)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("RCA 推送 Grafana 失败，降级仅日志/API: %s", exc)
        return False
