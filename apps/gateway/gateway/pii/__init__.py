"""脱敏引擎包（07 批次2 落地，确定性层）。

对外暴露：
- ``Redactor``：脱敏实现须满足的统一接口协议（可替换性基准）。
- ``PiiEngine``：确定性层实现（本批唯一实现，消费 rules 公共包）。
"""
from __future__ import annotations

from .engine import PiiEngine
from .rules_proto import Redactor

__all__ = ["Redactor", "PiiEngine"]
