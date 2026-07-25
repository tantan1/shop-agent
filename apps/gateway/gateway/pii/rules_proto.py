"""脱敏接口契约（可替换性基准，G2 门①）。

所有脱敏实现——确定性层 ``PiiEngine``、未来语义层 Presidio 封装——须满足
同一 ``redact`` 签名，调用方仅依赖本协议，替换后端时网关 hook / proxy 零改动。

    redact(text: str) -> tuple[str, list[str]]

出参为 (脱敏后文本, 命中类型列表)。命中类型取自规则声明（如 "phone"/"email"），
供审计与 human_review 标记使用，不含命中原文。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Redactor(Protocol):
    def redact(self, text: str) -> tuple[str, list[str]]:
        """返回 (脱敏后文本, 命中类型列表)。热路径零联网、可返回空命中。"""
        ...
