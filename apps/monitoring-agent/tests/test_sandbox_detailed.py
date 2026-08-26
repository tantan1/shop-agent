"""Docker 沙箱 Redis 恢复流程 - 详细参数与过程输出演示。

展示每个步骤的输入参数、过程输出、结果。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SANDBOX_ENABLED"] = "1"

from monitoring_agent.sandbox_config import SandboxConfig
from monitoring_agent.sandbox_runner import SandboxRunner
from monitoring_agent.sandbox import DockerSandboxBackend
from monitoring_agent.remediation import preview_script, generate_script, validate_script


def dump(title, data):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(data)


def main():
    # ──────────────────────────────────────────────
    # Step 0: 环境与配置
    # ──────────────────────────────────────────────
    config = SandboxConfig()
    dump("Step 0: SandboxConfig 参数", {
        "SANDBOX_ENABLED": config.SANDBOX_ENABLED,
        "SANDBOX_IMAGE": config.SANDBOX_IMAGE,
        "SANDBOX_NETWORK": config.SANDBOX_NETWORK,
        "SANDBOX_TIMEOUT_S": config.SANDBOX_TIMEOUT_S,
        "SANDBOX_MEM_LIMIT": config.SANDBOX_MEM_LIMIT,
        "SANDBOX_CPU_LIMIT": config.SANDBOX_CPU_LIMIT,
        "SANDBOX_PIDS_LIMIT": config.SANDBOX_PIDS_LIMIT,
        "SANDBOX_USER": config.SANDBOX_USER,
        "SANDBOX_WORKDIR": config.SANDBOX_WORKDIR,
        "ALLOWED_LIBRARIES": sorted(config.ALLOWED_LIBRARIES),
    })

    # ──────────────────────────────────────────────
    # Step 1: 初始化后端
    # ──────────────────────────────────────────────
    backend = DockerSandboxBackend(config)
    backend._state["redis"] = 0  # 模拟故障：replicas=0
    dump("Step 1: 后端初始化", {
        "backend_type": type(backend).__name__,
        "initial_state": backend._state,
        "is_available": backend.is_available(),
    })

    # ──────────────────────────────────────────────
    # Step 2: 生成恢复脚本
    # ──────────────────────────────────────────────
    binding = {"action": "scale_up", "target": "redis", "params": {"replicas": 1}}
    script = generate_script(binding)
    validate_script(script)
    dump("Step 2: 生成 Script DTO", {
        "input_binding": binding,
        "script.to_dict()": script.to_dict(),
        "action_registry": "scale_up / scale_down",
        "allowed_targets": ["redis"],
    })

    # ──────────────────────────────────────────────
    # Step 3: 沙箱验证（preview）
    # ──────────────────────────────────────────────
    evidence = preview_script(script, backend=backend)
    dump("Step 3: 沙箱验证证据包", evidence)

    # ──────────────────────────────────────────────
    # Step 4: 检查 Docker 执行结果
    # ──────────────────────────────────────────────
    set_replicas_call = None
    for call in reversed(backend.calls):
        if call.get("op") == "set_replicas":
            set_replicas_call = call
            break

    if set_replicas_call and "execution" in set_replicas_call:
        exec_result = set_replicas_call["execution"]
        dump("Step 4: Docker 容器执行结果", {
            "container_id": exec_result.get("container_id"),
            "exit_code": exec_result.get("exit_code"),
            "duration_ms": exec_result.get("duration_ms"),
            "stdout": exec_result.get("stdout", "").strip(),
            "stderr": exec_result.get("stderr", "").strip() or "(empty)",
            "dry_run": exec_result.get("dry_run"),
        })
    else:
        dump("Step 4: Docker 执行结果", "未检测到 Docker 执行")

    # ──────────────────────────────────────────────
    # Step 5: 构造真实的 Docker 运行参数
    # ──────────────────────────────────────────────
    runner = SandboxRunner(config)
    container_config = runner._build_container_config("print('test')")
    # 清理不可序列化的值
    serializable_config = {
        "image": config.SANDBOX_IMAGE,
        "command": ["python", "-c", "..."],
        "detach": True,
        "remove": False,
        "working_dir": container_config.get("working_dir"),
        "user": container_config.get("user"),
        "mem_limit": container_config.get("mem_limit"),
        "cpu_quota": container_config.get("cpu_quota"),
        "cpu_period": container_config.get("cpu_period"),
        "pids_limit": container_config.get("pids_limit"),
        "read_only": container_config.get("read_only"),
        "security_opt": container_config.get("security_opt"),
        "tmpfs": container_config.get("tmpfs"),
        "network_mode": container_config.get("network_mode"),
        "environment": container_config.get("environment"),
    }
    dump("Step 5: 实际产生的 Docker 容器参数", serializable_config)

    # ──────────────────────────────────────────────
    # Step 6: 完整流程总结
    # ──────────────────────────────────────────────
    execution_ok = (
        set_replicas_call is not None
        and set_replicas_call.get("execution", {}).get("exit_code") == 0
    )
    dump("Step 6: 完整流程总结", {
        "故障状态": "Redis replicas = 0",
        "恢复操作": "scale_up replicas = 1",
        "沙箱模式": "DockerSandboxBackend",
        "Docker 执行": "成功" if execution_ok else "失败",
        "证据包字段": list(evidence.keys()),
        "call_sequence 长度": len(evidence.get("call_sequence", [])),
        "安全约束": [
            "network=none",
            "memory=256m",
            "cpus=0.5",
            "pids-limit=64",
            "read-only",
            "security-opt=no-new-privileges",
            "user=nobody",
            "tmpfs /tmp 和 /workspace (noexec)",
        ],
    })


if __name__ == "__main__":
    main()
