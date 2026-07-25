"""
ReAct Agent 工具函数。
"""
from __future__ import annotations


def _looks_like_ack(text: str) -> bool:
    """判断最终答复是否只是「空泛致谢语」（模型未吸收工具结果）。"""
    t = (text or "").strip()
    if not t or len(t) >= 60:
        return False
    return any(
        p in t for p in ("我来", "帮您", "为您", "请稍等", "稍等", "这就", "这就为您", "已收到")
    )
