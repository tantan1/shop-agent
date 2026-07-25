"""统一脱敏组件（写日志 / Langfuse 等 SDK 出站前共用）。

目标：
- 单点维护一套脱敏规则，日志（structlog processor）与 Langfuse（SDK 自带 mask 钩子）
  等所有对外出口复用同一实现，避免各模块各自造轮子。
- 与 gateway / monitoring-agent 的脱敏同原则：确定性正则、保留可读前缀/后缀、
  不引入模型依赖；此处为 shop-agent 侧的统一实现（三服务各自持有实现，边界一致）。

能力：
- :func:`redact`    —— 单段文本脱敏（PII + 敏感凭证）
- :func:`redact_obj` —— 递归脱敏任意 JSON 结构（dict / list / str），带深度上限
- :func:`redact_dict` —— 按键名脱敏（api_key / password / token / secret ...）
- :func:`redact_processor` —— structlog processor：对整个 event_dict 脱敏
- :func:`mask_for_langfuse` —— 适配 Langfuse v4 SDK 的 ``mask`` 钩子签名
  ``(*, data, **kwargs)``，递归脱敏 input / output / metadata。

掩码策略：手机/邮箱等保留末 4 位可读（可读但不可定位），密钥整段替换。
"""

from __future__ import annotations

import re
from typing import Any

# ── 常量 ─────────────────────────────────────────────────────────────
_MASK = "***"
# 递归深度上限，防恶意超深嵌套触发栈溢出 / 资源耗尽
_MAX_DEPTH = 12

# ── PII 正则（与 monitoring-agent / gateway 同原则，独立实现） ────────
_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# 中国大陆手机号：保留末 4 位（用 (?<!\d) / (?!\d) 替代 \b，避免中文/字母前匹配失败）
_PHONE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
# 身份证（18 位，含可选 X）：保留前 4 后 4（用 (?<!\d) / (?!\d) 替代 \b）
_IDCARD = re.compile(r"(?<!\d)(\d{4})\d{10}(\d{3}[\dXx])(?!\d)")
# 银行卡（15-19 位）：保留后 4（用 (?<!\d) / (?!\d) 替代 \b）
_BANKCARD = re.compile(r"(?<!\d)(62\d{14,17}|[3-6]\d{13,17})(?!\d)")
# 公网 IPv4（不误伤内网/保留段）
_IPV4_PUBLIC = re.compile(
    r"\b(?!10\.|127\.|172\.|192\.|169\.)(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
)
# trace_id / span_id（可读性无用，整段脱敏）
_TRACE_ID = re.compile(r"\b(trace|span)[_]?id\s*[:=]\s*[\"']?[0-9a-fA-F]{12,}[\"']?", re.I)
# 密钥/令牌赋值形态：key=value
_SECRET = re.compile(
    r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd|access[_-]?key
    |authorization|bearer|private[_-]?key|refresh[_-]?token)\b\s*[:=]\s*
    ["']?[A-Za-z0-9\-_.+/]{8,}["']?"""
)

# 需要整值脱敏（而非仅扫描文本）的字段名（大小写不敏感）
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "password",
    "passwd",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "authorization_header",
    "private_key",
    "client_secret",
    "cookie",
    "set-cookie",
    "x-api-key",
    "aws_access_key_id",
    "aws_secret_access_key",
    "s3_endpoint",
    "redis_auth",
    "pgvector_password",
    "fixed_api_key",
    "openai_api_key",
    "azure_openai_api_key",
    "webhook_token",
    "langfuse_public_key",
    "langfuse_secret_key",
    "auth",
    "proxy_password",
    "ftp_password",
    "smtp_password",
    "ldap_password",
}


def _mask_phone(m: re.Match) -> str:
    return f"{m.group(1)}****{m.group(3)}"


def _mask_idcard(m: re.Match) -> str:
    return f"{m.group(1)}**********{m.group(2)}"


def _mask_bankcard(m: re.Match) -> str:
    g = m.group(0)
    return "*" * (len(g) - 4) + g[-4:]


def _mask_secret(m: re.Match) -> str:
    """保留 key= 前缀，值整段替换。"""
    head = m.group(0).split(":", 1)[0].split("=", 1)[0]
    sep = "=" if "=" in m.group(0) else ":"
    return f"{head}{sep}{_MASK}"


def redact(text: str) -> str:
    """对单段文本做轻量 PII 脱敏，返回脱敏后副本。"""
    if not isinstance(text, str) or not text:
        return text
    text = _EMAIL.sub(_MASK, text)
    text = _PHONE.sub(_mask_phone, text)
    text = _IDCARD.sub(_mask_idcard, text)
    text = _BANKCARD.sub(_mask_bankcard, text)
    text = _SECRET.sub(_mask_secret, text)
    text = _IPV4_PUBLIC.sub(_MASK, text)
    text = _TRACE_ID.sub(_MASK, text)
    return text


def _redact_scalar(value: Any) -> Any:
    """标量脱敏：str 走文本脱敏；bytes 尝试解码；其余原样。"""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, bytes):
        try:
            return redact(value.decode("utf-8", errors="replace"))
        except Exception:
            return _MASK
    return value


def redact_obj(obj: Any, _depth: int = 0) -> Any:
    """递归脱敏任意 JSON 结构（dict / list / str），原结构类型保持。

    深度超限即整体掩码，防御超深嵌套 payload 的栈溢出 / 资源耗尽。
    """
    if _depth >= _MAX_DEPTH:
        return _MASK if isinstance(obj, (dict, list)) else _redact_scalar(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v, _depth + 1) for v in obj]
    return _redact_scalar(obj)


def redact_dict(obj: Any, _depth: int = 0) -> Any:
    """按键名脱敏（敏感字段整值替换），非敏感字段仅做文本 PII 扫描。

    用于日志/Langfuse 中「键值结构」类数据：既保证 api_key/password 等
    凭证字段整值不落盘，也顺带扫掉字符串值里的 PII。
    """
    if _depth >= _MAX_DEPTH:
        return _MASK if isinstance(obj, (dict, list)) else _redact_scalar(obj)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k).lower().replace("-", "_").replace(" ", "_")
            if key in _SENSITIVE_KEYS:
                out[k] = _MASK
            else:
                out[k] = redact_dict(v, _depth + 1)
        return out
    if isinstance(obj, list):
        return [redact_dict(v, _depth + 1) for v in obj]
    return _redact_scalar(obj)


def redact_processor(logger: Any, method_name: str, event_dict: dict) -> dict:
    """structlog processor：对整个事件 dict 脱敏（键名 + 值扫描）。

    用法（见 src/shared/logger.py）：
        processors=[..., redact_processor, JSONRenderer(...)]
    放在 JSONRenderer 之前即可。返回新 dict，不修改原 event_dict。
    """
    return redact_dict(event_dict)


def mask_for_langfuse(*, data: Any, **kwargs: Any) -> Any:
    """适配 Langfuse v4 SDK 的 ``mask`` 钩子（MaskFunction 协议）。

    覆盖 input / output / metadata 等通过 Langfuse SDK API 写入的数据：
    键名脱敏 + 值内 PII 扫描，返回可 JSON 序列化的脱敏副本。
    """
    return redact_dict(data)
