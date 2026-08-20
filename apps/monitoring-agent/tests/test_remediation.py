"""remediation.py 单测（不依赖 k8s 集群）。

用 FakeK8sClient 做沙箱验证，monkeypatch k8s_client.set_replicas 做生产执行断言。
"""

from __future__ import annotations

import os
import sys

import pytest

_MON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MON not in sys.path:
    sys.path.insert(0, _MON)

import asyncio

from monitoring_agent.remediation import (
    RemediationError,
    FakeK8sClient,
    Script,
    apply_script,
    generate_script,
    preview_script,
    validate_script,
)
from monitoring_agent import main as main_mod
from monitoring_agent.alerts import parse_alertmanager


# ── generate_script / validate_script ──


def test_generate_script_valid():
    script = generate_script(
        {"action": "scale_up", "target": "redis", "params": {"replicas": 1}}
    )
    assert script.action == "scale_up"
    assert script.target == "redis"
    assert script.params == {"replicas": 1}


def test_generate_script_invalid_action():
    with pytest.raises(RemediationError, match="不在白名单"):
        generate_script({"action": "delete_all", "target": "redis"})


def test_generate_script_invalid_target():
    with pytest.raises(RemediationError, match="不在白名单"):
        generate_script({"action": "scale_up", "target": "mysql"})


def test_validate_script_missing_param():
    script = Script(action="scale_up", target="redis", params={})
    with pytest.raises(RemediationError, match="缺少参数 replicas"):
        validate_script(script)


def test_validate_script_wrong_type():
    script = Script(action="scale_up", target="redis", params={"replicas": "one"})
    with pytest.raises(RemediationError, match="类型应为 int"):
        validate_script(script)


def test_validate_script_replicas_exceed_max():
    script = Script(action="scale_up", target="redis", params={"replicas": 20})
    with pytest.raises(RemediationError, match="超出范围"):
        validate_script(script)


def test_validate_script_negative_replicas():
    script = Script(action="scale_down", target="redis", params={"replicas": -1})
    with pytest.raises(RemediationError, match="超出范围"):
        validate_script(script)


def test_validate_script_ok():
    script = Script(action="scale_up", target="redis", params={"replicas": 2})
    validate_script(script)  # 不抛异常


# ── preview_script (FakeK8sClient) ──


def test_preview_evidence():
    script = generate_script(
        {"action": "scale_up", "target": "redis", "params": {"replicas": 2}}
    )
    fake = FakeK8sClient()
    evidence = preview_script(script, backend=fake)

    assert evidence["plan"]["action"] == "scale_up"
    assert evidence["plan"]["target"] == "redis"
    assert "call_sequence" in evidence
    assert len(evidence["call_sequence"]) == 2
    assert evidence["call_sequence"][0]["step"] == "read_current"
    assert evidence["call_sequence"][1]["step"] == "would_apply"
    assert evidence["call_sequence"][1]["dry_run"] is True


def test_preview_default_backend():
    script = generate_script(
        {"action": "scale_down", "target": "redis", "params": {"replicas": 0}}
    )
    evidence = preview_script(script)
    assert evidence["effective"]["redis"] == 0


# ── apply_script ──


def test_apply_requires_approval():
    script = generate_script(
        {"action": "scale_up", "target": "redis", "params": {"replicas": 1}}
    )
    with pytest.raises(RemediationError, match="未审批"):
        apply_script(script, approved=False)


def test_apply_calls_k8s(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, int]] = []
    script = generate_script(
        {"action": "scale_up", "target": "redis", "params": {"replicas": 1}}
    )

    def fake_set(name: str, replicas: int) -> dict[str, Any]:
        calls.append((name, replicas))
        return {"name": name, "replicas": replicas}

    monkeypatch.setattr(
        "monitoring_agent.k8s_client.set_replicas", fake_set
    )
    result = apply_script(script, approved=True)
    assert result["applied"] is True
    assert calls == [("redis", 1)]


def test_apply_invalid_target(monkeypatch: pytest.MonkeyPatch):
    script = Script(action="scale_up", target="mysql", params={"replicas": 1})
    with pytest.raises(RemediationError, match="不在 k8s 白名单"):
        apply_script(script, approved=True)


# ── 告警去重 + 指标真实计数（⑤ 可观测运营）──


def _make_event(alertname: str, **labels) -> object:
    payload = {
        "alerts": [
            {"labels": {"alertname": alertname, **labels},
             "annotations": {"summary": f"{alertname} fired"}}
        ]
    }
    return parse_alertmanager(payload)


def test_dedup_same_alert_within_window():
    # 同 source+alertname+labels 在窗口内第二次判重
    ev = _make_event("HighP99", service="shop-agent", severity="P2")
    assert main_mod._is_duplicate(ev) is False  # 首次不重
    assert main_mod._is_duplicate(ev) is True   # 窗口内重复


def test_dedup_different_alert_not_duplicated():
    ev1 = _make_event("HighP99", service="shop-agent")
    ev2 = _make_event("High5xx", service="shop-agent")
    assert main_mod._is_duplicate(ev1) is False
    assert main_mod._is_duplicate(ev2) is False  # 不同 alertname 不判重


def test_metrics_real_counters_reflect_bumps():
    # 计数函数递增模块级计数器，metrics 文本应包含非 0 值
    before = main_mod._remediate_plans_total
    main_mod._bump_plan()
    assert main_mod._remediate_plans_total == before + 1

    main_mod._bump_approval("approved")
    main_mod._bump_execution(True)
    main_mod._bump_persisted()

    text = asyncio.run(main_mod.metrics()).body.decode()
    assert f"remediate_plans_total {main_mod._remediate_plans_total}" in text
    assert 'approvals_total{decision="approved"} 1' in text
    assert 'executions_total{result="success"} 1' in text
    assert "rca_persisted_total 1" in text
