"""sandbox 模块单测（不依赖真实 Docker）。

通过 monkeypatch 模拟 Docker 客户端，验证：
- SandboxConfig 环境变量读取
- SandboxRunner.validate_script AST 扫描 + 库白名单
- DockerSandboxBackend 接口与 FakeK8sClient 兼容
- is_sandbox_enabled 开关逻辑
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_MON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MON not in sys.path:
    sys.path.insert(0, _MON)

from monitoring_agent.sandbox_config import SandboxConfig
from monitoring_agent.sandbox_runner import SandboxRunner, SandboxExecutionError
from monitoring_agent.sandbox import DockerSandboxBackend, is_sandbox_enabled


# ── SandboxConfig ──


class TestSandboxConfig:
    def test_default_values(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SANDBOX_ENABLED", raising=False)
        config = SandboxConfig()
        assert config.SANDBOX_IMAGE == "python:3.11-slim"
        assert config.SANDBOX_NETWORK == "none"
        assert config.SANDBOX_TIMEOUT_S == 30
        assert config.SANDBOX_MEM_LIMIT == "256m"
        assert config.SANDBOX_CPU_LIMIT == "0.5"
        assert config.SANDBOX_PIDS_LIMIT == 64
        assert config.SANDBOX_ENABLED is False
        assert "json" in config.ALLOWED_LIBRARIES
        assert "kubernetes" in config.ALLOWED_LIBRARIES

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "1")
        monkeypatch.setenv("SANDBOX_IMAGE", "custom:latest")
        monkeypatch.setenv("SANDBOX_TIMEOUT_S", "60")
        config = SandboxConfig()
        assert config.SANDBOX_ENABLED is True
        assert config.SANDBOX_IMAGE == "custom:latest"
        assert config.SANDBOX_TIMEOUT_S == 60


# ── SandboxRunner.validate_script ──


class TestSandboxRunnerValidate:
    @pytest.fixture()
    def runner(self):
        return SandboxRunner(SandboxConfig())

    def test_valid_stdlib_import(self, runner: SandboxRunner):
        script = "import json\nimport re\nprint('ok')"
        runner.validate_script(script)  # 不抛异常

    def test_valid_allowed_lib(self, runner: SandboxRunner):
        script = "import kubernetes\nprint('ok')"
        runner.validate_script(script)  # 不抛异常

    def test_forbidden_lib_raises(self, runner: SandboxRunner):
        script = "import requests\nprint('ok')"
        with pytest.raises(SandboxExecutionError, match="非白名单库"):
            runner.validate_script(script)

    def test_syntax_error_raises(self, runner: SandboxRunner):
        script = "import json\nprint('ok'"
        with pytest.raises(SandboxExecutionError, match="语法错误"):
            runner.validate_script(script)

    def test_disabled_raises(self):
        config = SandboxConfig()
        config.SANDBOX_ENABLED = False
        runner = SandboxRunner(config)
        with pytest.raises(SandboxExecutionError, match="未启用"):
            runner.run("print('ok')")


# ── DockerSandboxBackend ──


class TestDockerSandboxBackend:
    @pytest.fixture(autouse=True)
    def _mock_docker(self, monkeypatch: pytest.MonkeyPatch):
        """mock Docker SDK，不依赖真实 Docker。"""
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "abc123456789"
        mock_container.logs.return_value = b""
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_client.containers.run.return_value = mock_container
        mock_client.containers.get.return_value = mock_container
        mock_client.ping.return_value = True

        import monitoring_agent.sandbox_runner as sr
        monkeypatch.setattr(sr, "docker", MagicMock(from_env=MagicMock(return_value=mock_client)))
        monkeypatch.setattr(sr, "DOCKER_AVAILABLE", True)

    def test_backend_interface_compatible(self, monkeypatch: pytest.MonkeyPatch):
        """DockerSandboxBackend 与 FakeK8sClient 接口兼容。"""
        monkeypatch.setenv("SANDBOX_ENABLED", "1")
        backend = DockerSandboxBackend()
        assert hasattr(backend, "get_replicas")
        assert hasattr(backend, "set_replicas")
        assert hasattr(backend, "execute_script")
        assert hasattr(backend, "is_available")

    def test_get_replicas(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "1")
        backend = DockerSandboxBackend()
        assert backend.get_replicas("redis") == 1

    def test_set_replicas_marks_dry_run(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "1")
        backend = DockerSandboxBackend()
        result = backend.set_replicas("redis", 2)
        assert result["dry_run"] is True
        assert result.get("sandbox") == "docker"
        assert len(backend.calls) == 1
        # Docker 沙箱启用时，set_replicas 会触发恢复脚本执行
        assert "execution" in backend.calls[0]
        assert backend.calls[0]["execution"]["exit_code"] == 0

    def test_is_available_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "1")
        import monitoring_agent.sandbox_runner as sr
        monkeypatch.setattr(sr, "DOCKER_AVAILABLE", True)
        backend = DockerSandboxBackend()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        backend._runner._client = mock_client
        assert backend.is_available() is True

    def test_is_available_false_when_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "0")
        backend = DockerSandboxBackend()
        assert backend.is_available() is False


# ── is_sandbox_enabled ──


class TestIsSandboxEnabled:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "0")
        assert is_sandbox_enabled() is False

    def test_enabled_but_docker_unavailable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "1")
        # Docker SDK 不可用（模拟），应返回 False
        with patch("monitoring_agent.sandbox.DOCKER_AVAILABLE", False):
            assert is_sandbox_enabled() is False


# ── remediation.py get_default_backend ──


class TestGetDefaultBackend:
    def test_default_fake_backend(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "0")
        from monitoring_agent.remediation import get_default_backend, FakeK8sClient
        backend = get_default_backend()
        assert isinstance(backend, FakeK8sClient)

    def test_sandbox_backend_fallback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SANDBOX_ENABLED", "1")
        from monitoring_agent.remediation import get_default_backend, FakeK8sClient

        # 模拟 Docker SDK 不可用，应回退到 FakeK8sClient
        with patch("monitoring_agent.sandbox.DOCKER_AVAILABLE", False):
            backend = get_default_backend()
            assert isinstance(backend, FakeK8sClient)
