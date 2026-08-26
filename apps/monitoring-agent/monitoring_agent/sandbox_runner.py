"""沙箱脚本执行器（在 Docker 容器内执行不可信 Python 脚本）。

职责：
- 将脚本写入临时文件
- 构造 Docker 容器参数（安全约束）
- 启动容器执行脚本
- 捕获 stdout/stderr/exit_code
- 返回执行结果（供证据包使用）

设计对齐：docs/monitoring-agent-remediation-safety-design.md §10.6
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Any

logger = logging.getLogger("monitoring_agent.sandbox_runner")

# 尝试导入 docker 模块（可选依赖）
try:
    import docker
    from docker.errors import DockerException, NotFound, APIError

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None  # type: ignore
    DockerException = Exception  # type: ignore
    NotFound = Exception  # type: ignore
    APIError = Exception  # type: ignore

from .sandbox_config import SandboxConfig


class SandboxExecutionError(Exception):
    """沙箱执行失败（超时/资源超限/脚本错误/库黑名单拦截等）。"""


class SandboxRunner:
    """Docker 容器沙箱执行器。

    安全约束（分层防御）：
    1. 容器边界：Docker 容器隔离（文件系统/网络/资源/进程）
    2. 网络 Restricted：--network=none（或白名单单出口）
    3. 资源上限：timeout + MaxMem + CPU + PIDs
    4. 库白名单：import 前 AST 扫描，非白名单库直接拒绝
    5. 降权运行：容器内非 root 用户
    6. 只读文件系统：除 /tmp 外全部只读
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._client = None

    def _get_client(self) -> Any:
        """惰性初始化 Docker 客户端。"""
        if not DOCKER_AVAILABLE:
            raise SandboxExecutionError(
                "Docker SDK 未安装（pip install docker），无法使用 Docker 沙箱。"
            )
        if self._client is None:
            try:
                self._client = docker.from_env()
            except DockerException as exc:
                raise SandboxExecutionError(f"Docker 客户端初始化失败: {exc}") from exc
        return self._client

    def validate_script(self, script: str) -> None:
        """执行前静态校验（AST 扫描 + 库白名单）。

        不允许的 import 直接拒绝，不进入容器。
        """
        import ast

        try:
            tree = ast.parse(script)
        except SyntaxError as exc:
            raise SandboxExecutionError(f"脚本语法错误: {exc}") from exc

        imported_libs: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    imported_libs.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    imported_libs.add(top)

        # stdlib 模块集合（Python 3.11 标准库）
        stdlib_modules = {
            "abc", "argparse", "array", "ast", "asyncio", "base64", "binascii",
            "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath",
            "cmd", "code", "codecs", "codeop", "collections", "colorsys",
            "compileall", "concurrent", "configparser", "contextlib", "contextvars",
            "copy", "crypt", "csv", "ctypes", "curses", "dataclasses",
            "datetime", "dbm", "decimal", "difflib", "dis", "distutils",
            "doctest", "email", "encodings", "enum", "errno", "faulthandler",
            "fcntl", "filecmp", "fileinput", "fnmatch", "fractions", "ftplib",
            "functools", "gc", "getopt", "getpass", "gettext", "glob",
            "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
            "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect",
            "io", "ipaddress", "itertools", "json", "keyword", "lib2to3",
            "linecache", "locale", "logging", "lzma", "mailbox", "mailcap",
            "marshal", "math", "mimetypes", "mmap", "modulefinder", "msilib",
            "msvcrt", "multiprocessing", "netrc", "nis", "nntplib", "numbers",
            "operator", "optparse", "os", "ossaudiodev", "parser", "pathlib",
            "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
            "plistlib", "poplib", "posix", "posixpath", "pprint", "profile",
            "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc",
            "queue", "quopri", "random", "re", "readline", "reprlib",
            "resource", "rlcompleter", "runpy", "sched", "secrets", "select",
            "selectors", "shelve", "shlex", "shutil", "signal", "site",
            "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
            "sqlite3", "sre_compile", "sre_constants", "sre_parse", "ssl",
            "stat", "statistics", "string", "stringprep", "struct", "subprocess",
            "sunau", "symbol", "symtable", "sys", "sysconfig", "syslog",
            "tabnanny", "tarfile", "telnetlib", "tempfile", "termios",
            "test", "textwrap", "threading", "time", "timeit", "tkinter",
            "token", "tokenize", "tomllib", "trace", "traceback", "tracemalloc",
            "tty", "turtle", "turtledemo", "types", "typing", "typing_extensions",
            "unicodedata", "unittest", "urllib", "uu", "uuid", "venv",
            "warnings", "wave", "weakref", "webbrowser", "winreg", "winsound",
            "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp", "zipfile",
            "zipimport", "zlib", "zoneinfo",
        }

        forbidden = []
        for lib in imported_libs:
            if lib in stdlib_modules:
                continue
            if lib in self.config.ALLOWED_LIBRARIES:
                continue
            forbidden.append(lib)

        if forbidden:
            raise SandboxExecutionError(
                f"脚本导入非白名单库（安全拦截）: {forbidden}。"
                f"允许库: stdlib + {sorted(self.config.ALLOWED_LIBRARIES)}"
            )

    def run(self, script: str, timeout: int | None = None) -> dict[str, Any]:
        """在 Docker 沙箱中执行脚本。

        Args:
            script: Python 脚本内容
            timeout: 超时秒数（None 则用配置默认值）

        Returns:
            {
                "exit_code": int,
                "stdout": str,
                "stderr": str,
                "duration_ms": float,
                "container_id": str,
                "dry_run": True,
            }

        Raises:
            SandboxExecutionError: 执行失败（超时/资源超限/库黑名单等）
        """
        if not self.config.SANDBOX_ENABLED:
            raise SandboxExecutionError("Docker 沙箱未启用（SANDBOX_ENABLED=0）")

        self.validate_script(script)

        client = self._get_client()
        timeout_s = timeout or self.config.SANDBOX_TIMEOUT_S

        # 构造 Docker 容器参数
        container_config = self._build_container_config(script)

        container_id = None
        try:
            container = client.containers.run(
                image=self.config.SANDBOX_IMAGE,
                command=["python", "-c", script],
                detach=True,
                remove=False,  # 先不删除，以便获取日志
                **container_config,
            )
            container_id = container.id[:12]
            logger.info("沙箱容器已启动 container_id=%s image=%s", container_id, self.config.SANDBOX_IMAGE)

            # 等待执行完成（带超时）
            start_time = time.time()
            result = container.wait(timeout=timeout_s)
            duration_ms = (time.time() - start_time) * 1000

            # 获取日志
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")
            exit_code = result.get("StatusCode", -1) if isinstance(result, dict) else result

            logger.info(
                "沙箱执行完成 container_id=%s exit_code=%s duration_ms=%.0f",
                container_id, exit_code, duration_ms,
            )

            return {
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": round(duration_ms, 2),
                "container_id": container_id,
                "dry_run": True,
            }

        except NotFound as exc:
            raise SandboxExecutionError(f"沙箱镜像不存在: {self.config.SANDBOX_IMAGE}") from exc
        except APIError as exc:
            raise SandboxExecutionError(f"Docker API 错误: {exc}") from exc
        except Exception as exc:
            raise SandboxExecutionError(f"沙箱执行异常: {exc}") from exc
        finally:
            # 清理容器
            if container_id:
                try:
                    container = client.containers.get(container_id)
                    container.remove(force=True)
                    logger.debug("沙箱容器已清理 container_id=%s", container_id)
                except Exception as exc:
                    logger.warning("沙箱容器清理失败 container_id=%s err=%s", container_id, exc)

    def _build_container_config(self, script: str) -> dict[str, Any]:
        """构造 Docker 容器运行参数（安全约束）。"""
        config: dict[str, Any] = {
            "working_dir": self.config.SANDBOX_WORKDIR,
            "user": self.config.SANDBOX_USER,
            "mem_limit": self.config.SANDBOX_MEM_LIMIT,
            "cpu_quota": int(float(self.config.SANDBOX_CPU_LIMIT) * 100000),
            "cpu_period": 100000,
            "pids_limit": self.config.SANDBOX_PIDS_LIMIT,
            "read_only": True,  # 只读文件系统
            "security_opt": ["no-new-privileges"],  # 禁止提权
            "tmpfs": {  # 临时文件系统（/tmp 可写）
                "/tmp": "rw,noexec,nosuid,size=64m",
                self.config.SANDBOX_WORKDIR: "rw,noexec,nosuid,size=32m",
            },
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
            "log_config": {"type": "json-file", "config": {"max-size": "1m", "max-file": "1"}},
        }

        # 网络隔离
        if self.config.SANDBOX_NETWORK == "none":
            config["network_mode"] = "none"
        elif self.config.SANDBOX_NETWORK:
            config["network_mode"] = self.config.SANDBOX_NETWORK

        # DNS 配置（仅网络模式不为 none 时生效）
        if self.config.SANDBOX_DNS and self.config.SANDBOX_NETWORK != "none":
            config["dns"] = [self.config.SANDBOX_DNS]

        return config
