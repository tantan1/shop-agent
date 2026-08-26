"""Docker 沙箱后端（可执行脚本的隔离执行环境）。

职责：
- 实现与 FakeK8sClient 相同的接口（get_replicas / set_replicas）
- 在 Docker 容器内执行脚本（无网络 + 资源上限 + 库白名单 + 只读文件系统）
- 产出证据包（current/plan/effective/call_sequence），与 FakeK8sClient 格式一致

触发条件（设计文档 §10.4）：
1. 允许 LLM 生成任意脚本 且脚本会被执行（不再是类型化动作）
2. 自愈链路要自动触碰生产（无人审批闸门）
3. 需要执行第三方/不可信探测插件

设计对齐：docs/monitoring-agent-remediation-safety-design.md §10.6
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .sandbox_config import SandboxConfig
from .sandbox_runner import SandboxRunner, SandboxExecutionError, DOCKER_AVAILABLE

logger = logging.getLogger("monitoring_agent.sandbox")


class DockerSandboxBackend:
    """Docker 容器沙箱后端（替代 FakeK8sClient）。

    安全约束（分层防御）：
    1. 容器边界：Docker 容器隔离（文件系统/网络/资源/进程全隔离）
    2. 网络 Restricted：--network=none（默认），阻断所有出站网络
    3. 资源上限：timeout 30s + Mem 256m + CPU 0.5 + PIDs 64
    4. 库白名单：AST 扫描 import，非白名单库直接拒绝
    5. 降权运行：容器内 nobody 用户（非 root）
    6. 只读文件系统：除 /tmp 外全部只读

    与 FakeK8sClient 的区别：
    - FakeK8sClient：内存模拟，零隔离，适合类型化动作
    - DockerSandboxBackend：真容器隔离，适合不可信脚本
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._runner = SandboxRunner(self.config)
        self._state: dict[str, int] = {"redis": 1}
        self.calls: list[dict[str, Any]] = []

    def get_replicas(self, name: str) -> int:
        """读取当前副本数（从内存状态读取，不查询真实 k8s）。"""
        self.calls.append({"op": "get_replicas", "name": name})
        return self._state.get(name, 0)

    def set_replicas(self, name: str, replicas: int) -> dict[str, Any]:
        """在沙箱中执行副本数变更（记录调用，不真发）。

        当 Docker 沙箱启用时，在容器内执行恢复脚本模拟 k8s scale 操作。
        内存状态同步更新（用于证据包生成）。
        """
        call_record = {
            "op": "set_replicas",
            "name": name,
            "replicas": replicas,
            "dry_run": True,
            "sandbox": "docker",
        }
        self.calls.append(call_record)

        # 在 Docker 沙箱中执行恢复脚本（真实容器隔离）
        if self.config.SANDBOX_ENABLED and DOCKER_AVAILABLE:
            recovery_script = self._build_recovery_script(name, replicas)
            try:
                exec_result = self._runner.run(recovery_script)
                call_record["execution"] = exec_result
            except Exception as exc:
                call_record["execution_error"] = str(exc)
                logger.warning("Docker 沙箱恢复脚本执行失败: %s", exc)

        self._state[name] = replicas
        return {"name": name, "replicas": replicas, "dry_run": True, "sandbox": "docker"}

    def _build_recovery_script(self, name: str, replicas: int) -> str:
        """构建 Redis 恢复操作的 Docker 执行脚本（模拟 k8s scale）。

        脚本仅使用 stdlib（json），通过 AST 白名单扫描。
        """
        return f'''import json

target = "{name}"
replicas = {replicas}

print("[sandbox] 模拟 k8s scale deployment/" + target + " --replicas=" + str(replicas))
print("[sandbox] 操作类型: 恢复")
print("[sandbox] 目标: " + target)
print("[sandbox] 副本数: " + str(replicas))

result = {{
    "name": target,
    "replicas": replicas,
    "dry_run": True,
    "sandbox": "docker",
    "message": "恢复操作已模拟执行"
}}
print(json.dumps(result))
'''

    def execute_script(self, script: str, timeout: int | None = None) -> dict[str, Any]:
        """在 Docker 沙箱中执行任意 Python 脚本。

        Args:
            script: Python 脚本内容
            timeout: 超时秒数

        Returns:
            执行结果（exit_code/stdout/stderr/duration_ms）

        Raises:
            SandboxExecutionError: 执行失败
        """
        return self._runner.run(script, timeout=timeout)

    def is_available(self) -> bool:
        """检查 Docker 沙箱是否可用。"""
        if not self.config.SANDBOX_ENABLED:
            return False
        if not DOCKER_AVAILABLE:
            return False
        try:
            client = self._runner._get_client()
            client.ping()
            return True
        except Exception as exc:
            logger.warning("Docker 沙箱不可用: %s", exc)
            return False


def is_sandbox_enabled() -> bool:
    """检查 Docker 沙箱是否已启用且可用。"""
    config = SandboxConfig()
    if not config.SANDBOX_ENABLED:
        return False
    backend = DockerSandboxBackend(config)
    return backend.is_available()
