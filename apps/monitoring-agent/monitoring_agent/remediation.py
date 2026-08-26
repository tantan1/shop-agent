"""remediation 引擎（脚本生成 → 白名单校验 → 沙箱验证 → 生产执行）。

职责分层：
- 保持纯函数、不落库（持久化由 main.py 接线层调用 store 完成）；
- 脚本 DSL 收敛为类型化动作（scale_up/scale_down）+ 白名单目标 + 参数上限；
- 沙箱默认用 FakeK8sClient 做干跑，产出证据包供审批人核对；
- 当 SANDBOX_ENABLED=1 时，自动切换为 DockerSandboxBackend（真容器隔离）；
- 生产执行仅当 approved=True 时才调用 k8s_client.set_replicas。

设计对齐：docs/monitoring-agent-remediation-safety-design.md §6.2 / §10.3 / §10.6
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("monitoring_agent.remediation")


class RemediationError(Exception):
    """脚本不合规 / 参数非法 / 目标不在白名单 / 未审批禁止执行。"""


# ── 白名单注册表 ──

_ACTION_REGISTRY: dict[str, dict[str, Any]] = {
    "scale_up": {"params": {"replicas": int}, "max_replicas": 10},
    "scale_down": {"params": {"replicas": int}, "max_replicas": 10},
}

_ALLOWED_TARGETS = frozenset({"redis"})


# ── DTO ──


@dataclass
class Script:
    """受控动作 DSL（只允许白名单动作 + 白名单目标 + 参数上限）。"""

    action: str
    target: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "target": self.target, "params": self.params}


# ── 1) 生成脚本 ──


def generate_script(binding: dict[str, Any]) -> Script:
    """由规则/LLM 产出操作脚本（白名单动作 DSL）。

    输入示例（与 rca._rule_attr 输出的 remediation dict 对齐）：
        {"action": "scale_up", "target": "redis", "replicas": 1}
    """
    action = binding.get("action", "")
    target = binding.get("target", "")
    params = binding.get("params", {})

    if action not in _ACTION_REGISTRY:
        raise RemediationError(
            f"动作 {action!r} 不在白名单 {sorted(_ACTION_REGISTRY)}"
        )
    if target not in _ALLOWED_TARGETS:
        raise RemediationError(
            f"目标 {target!r} 不在白名单 {sorted(_ALLOWED_TARGETS)}"
        )

    return Script(action=action, target=target, params=params)


# ── 2) 校验脚本 ──


def validate_script(script: Script) -> None:
    """白名单校验：动作注册、目标白名单、参数类型与上限。"""
    reg = _ACTION_REGISTRY.get(script.action)
    if not reg:
        raise RemediationError(f"动作 {script.action!r} 未注册")

    for key, typ in reg["params"].items():
        if key not in script.params:
            raise RemediationError(f"缺少参数 {key}")
        if not isinstance(script.params[key], typ):
            raise RemediationError(
                f"参数 {key} 类型应为 {typ.__name__}，实际为 {type(script.params[key]).__name__}"
            )

    if "replicas" in script.params:
        rep = int(script.params["replicas"])
        if rep < 0 or rep > reg["max_replicas"]:
            raise RemediationError(
                f"replicas 超出范围 [0, {reg['max_replicas']}]，实际为 {rep}"
            )


# ── 3) 沙箱验证（FakeK8sClient / DockerSandboxBackend） ──


class FakeK8sClient:
    """假 k8s 后端：只读动作返回当前状态，写动作不真发，记录「将执行什么」。
    
    本期默认沙箱（类型化动作，零隔离开销）。
    当 SANDBOX_ENABLED=1 时，自动切换为 DockerSandboxBackend（真容器隔离）。
    """

    def __init__(self) -> None:
        self._state: dict[str, int] = {"redis": 1}
        self.calls: list[dict[str, Any]] = []

    def get_replicas(self, name: str) -> int:
        self.calls.append({"op": "get_replicas", "name": name})
        return self._state.get(name, 0)

    def set_replicas(self, name: str, replicas: int) -> dict[str, Any]:
        self.calls.append(
            {"op": "set_replicas", "name": name, "replicas": replicas, "dry_run": True}
        )
        self._state[name] = replicas
        return {"name": name, "replicas": replicas, "dry_run": True}


def get_default_backend() -> Any:
    """根据环境变量返回默认沙箱后端。

    - SANDBOX_ENABLED=1 且 Docker 可用 → DockerSandboxBackend（真容器隔离）
    - SANDBOX_ENABLED=1 但 Docker 不可用 → 回退到 FakeK8sClient
    - 否则 → FakeK8sClient（内存模拟，本期默认）
    """
    if os.getenv("SANDBOX_ENABLED", "0") != "1":
        return FakeK8sClient()
    try:
        from .sandbox import DockerSandboxBackend
        backend = DockerSandboxBackend()
        if not backend.is_available():
            logger.warning("Docker 沙箱不可用，回退到 FakeK8sClient")
            return FakeK8sClient()
        return backend
    except Exception as exc:
        logger.warning("Docker 沙箱初始化失败，回退到 FakeK8sClient: %s", exc)
        return FakeK8sClient()


def preview_script(
    script: Script, backend: Any = None
) -> dict[str, Any]:
    """在沙箱后端上执行脚本，产出证据包（current/plan/effective/call_sequence）。

    backend 为 None 时使用默认后端（受 SANDBOX_ENABLED 控制）。
    """
    if backend is None:
        backend = get_default_backend()

    calls: list[dict[str, Any]] = []

    if script.action in ("scale_up", "scale_down"):
        current = backend.get_replicas(script.target)
        result = backend.set_replicas(script.target, int(script.params["replicas"]))
        calls.append(
            {"step": "read_current", "target": script.target, "replicas": current}
        )
        calls.append(
            {
                "step": "would_apply",
                "target": script.target,
                "replicas": result["replicas"],
                "dry_run": True,
            }
        )

    evidence = {
        "current": {script.target: backend.get_replicas(script.target)},
        "plan": script.to_dict(),
        "effective": {script.target: int(script.params.get("replicas", 0))},
        "call_sequence": calls,
    }
    return evidence


# ── 4) 生产执行 ──


def apply_script(script: Script, approved: bool = False) -> dict[str, Any]:
    """仅当 approved=True 时才在真 k8s 后端执行。"""
    if not approved:
        raise RemediationError("未审批，禁止执行")

    from .k8s_client import set_replicas, K8sUnavailable, _ALLOWED_TARGETS as K8S_ALLOWED

    if script.target not in K8S_ALLOWED:
        raise RemediationError(
            f"目标 {script.target!r} 不在 k8s 白名单 {sorted(K8S_ALLOWED)}"
        )

    try:
        result = set_replicas(script.target, int(script.params["replicas"]))
        return {"applied": True, **result}
    except K8sUnavailable as exc:
        raise RemediationError(f"K8s 执行失败: {exc}") from exc
