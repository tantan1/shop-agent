"""
安全端口（Ports & Adapters）—— 外部安全能力以端口接口预留，具体实现先用 stub。

当前提供文件配置 stub 级别的基础能力：
  - sanitize_input: 去除/转义可能用于注入的元字符（Prompt Injection 防护的入口点）
  - verify_integrity: 占位校验（后续可接签名/哈希校验）

说明：真正的鉴权/审计由 auth 模块与网关承担；本端口聚焦"输入可信化"与
后续可插拔的安全增强点。外部商业安全 SDK 不应被直接依赖。
"""

from __future__ import annotations

import re
from typing import Optional

# 常见注入元字符/控制序列（最小化集合，仅作 stub 级别防护）
_DANGEROUS_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_input(text: str, *, strip_control: bool = True) -> str:
    """对进入 LLM/工具上下文的用户输入做基础净化（stub 实现）。

    - 默认移除不可见控制字符（防隐藏指令/分隔符注入）
    - 不在此做语义级注入判定（由网关 injection 钩子与模型侧承担）

    Args:
        text: 原始输入
        strip_control: 是否移除控制字符
    Returns:
        净化后的文本
    """
    if not text:
        return text
    if strip_control:
        text = _DANGEROUS_CHARS.sub("", text)
    return text


def verify_integrity(payload: str, signature: Optional[str] = None) -> bool:
    """完整性校验占位（stub）。

    后续可替换为 HMAC/签名校验，验证外部传入内容未被篡改。
    当前 stub：未提供签名即视为通过（开发态），生产应改为强制校验。
    """
    if signature is None:
        return True
    # TODO(port): 接入真实签名校验（如 HMAC-SHA256），失败时返回 False
    return True
