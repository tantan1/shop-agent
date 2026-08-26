"""Docker 沙箱配置（安全约束 + 白名单 + 资源限制）。

设计对齐：docs/monitoring-agent-remediation-safety-design.md §10.6
"""

from __future__ import annotations

import os


class SandboxConfig:
    """沙箱运行配置（可通过环境变量覆盖）。"""

    def __init__(self) -> None:
        # ── Docker 镜像 ──
        self.SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "python:3.11-slim")
        self.SANDBOX_PYTHON_VERSION = os.getenv("SANDBOX_PYTHON_VERSION", "3.11")

        # ── 网络（默认无网络，Datadog "restricted environment" 同款） ──
        self.SANDBOX_NETWORK = os.getenv("SANDBOX_NETWORK", "none")
        self.SANDBOX_DNS = os.getenv("SANDBOX_DNS", "")

        # ── 资源上限（cgroup 杀超） ──
        self.SANDBOX_TIMEOUT_S = int(os.getenv("SANDBOX_TIMEOUT_S", "30"))
        self.SANDBOX_MEM_LIMIT = os.getenv("SANDBOX_MEM_LIMIT", "256m")
        self.SANDBOX_CPU_LIMIT = os.getenv("SANDBOX_CPU_LIMIT", "0.5")
        self.SANDBOX_PIDS_LIMIT = int(os.getenv("SANDBOX_PIDS_LIMIT", "64"))

        # ── 白名单预装库 ──
        self.ALLOWED_LIBRARIES = frozenset({
            "json", "re", "os", "sys", "math", "datetime", "time",
            "collections", "itertools", "functools", "typing",
            "dataclasses", "pathlib", "logging", "hashlib",
            "urllib.parse", "base64", "io", "contextlib",
            "kubernetes",
        })

        # ── 容器内用户（非 root） ──
        self.SANDBOX_USER = os.getenv("SANDBOX_USER", "nobody")

        # ── 是否启用 Docker 沙箱（默认关闭） ──
        self.SANDBOX_ENABLED = os.getenv("SANDBOX_ENABLED", "0") == "1"

        # ── 工作目录 ──
        self.SANDBOX_WORKDIR = "/workspace"
