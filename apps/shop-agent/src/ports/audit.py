"""审计日志端口（stub 实现：本地 JSONL 文件）。

A 维度修复：敏感操作（越权拦截、密钥读取、权限变更）需留不可篡改审计条目。
当前 stub 写入本地 JSONL；生产可替换为 WORM 存储 / 独立审计服务。
接口设计：调用方只依赖 `audit.log(...)`，不感知存储实现。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

# stub 落盘路径（可用环境变量覆盖）
_AUDIT_PATH = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
_lock = threading.Lock()


def log(
    event: str,
    *,
    principal: Optional[str] = None,
    action: Optional[str] = None,
    resource: Optional[str] = None,
    decision: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> None:
    """记录一条审计事件到本地 JSONL（stub）。

    Args:
        event: 事件类型，如 "authz.denied" / "secret.read" / "role.changed"
        principal: 操作主体（调用方标识，不含完整密钥）
        action: 动作
        resource: 资源
        decision: 决策（allow/deny）
        detail: 附加结构化信息
    """
    record = {
        "ts": time.time(),
        "event": event,
        "principal": principal,
        "action": action,
        "resource": resource,
        "decision": decision,
        "detail": detail or {},
    }
    try:
        os.makedirs(os.path.dirname(_AUDIT_PATH) or ".", exist_ok=True)
        with _lock:
            with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        # stub 阶段审计写入失败不阻塞主流程
        pass
