"""PII 脱敏端口（Ports & Adapters）。

设计约束：外部能力只暴露端口接口，实现用文件 / 标准库 stub，
不硬依赖任何外部 SDK。

默认实现委托 `src.shared.redact`（本地正则脱敏，无外部依赖）。
生产环境如需更强能力（如可逆 tokenization 替换），在适配层替换即可，
调用方无需改动。
"""
from __future__ import annotations

from src.shared.redact import redact as _redact_impl


def redact(text: str) -> str:
    """对用户自由文本做 PII 脱敏，返回脱敏副本。

    在把用户内容写入发给 LLM 的 prompt / 落库前调用，避免明文 PII
    进入模型上下文。结构化的业务参数（order_id、已抽取的 phone 等）由
    确定性抽取流程单独处理，不在此处脱敏，以保留业务能力。
    """
    if not text:
        return text
    return _redact_impl(text)
