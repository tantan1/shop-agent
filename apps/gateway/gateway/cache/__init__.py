"""08 语义缓存包（批次2/2b）。

业务维度分片 + PII 写入门禁 + TTL/写操作不命中。embedding 本批用本地词频向量 +
余弦相似度（零依赖，可命中近似 prompt），真实语义 embedding 留 embed 钩子。
"""
from __future__ import annotations

from .pii_gate import PiiGate
from .policy import CachePolicy
from .store import SemanticCache

__all__ = ["SemanticCache", "CachePolicy", "PiiGate"]
