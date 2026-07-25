"""通用组合模式：Agent 树基础设施。

将 Agent 抽象为统一的 `AgentNode` 接口，支持 Leaf（单个 Agent）和
Composite（组合多个 Agent）。领域无关，可供 DisputeCoordinator 等复用。
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.shared.logger import APILogger

logger = APILogger("agent_composite")


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class AgentContext:
    """Agent 执行上下文（领域无关）。"""
    request_message: str = ""
    facts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    emotion_level: Any = None
    order_id: Optional[str] = None
    conversation_id: str = ""
    domain: str = "ecommerce"


# ── 抽象基类 ─────────────────────────────────────────────────────────


class AgentNode(ABC):
    """Agent 抽象基类（组合模式）。"""

    name: str = ""

    @abstractmethod
    async def execute(self, ctx: AgentContext) -> Any:
        """执行 Agent 逻辑，返回结果。"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ── 组合节点 ─────────────────────────────────────────────────────────


class ParallelComposite(AgentNode):
    """并行组合 Agent（同时执行多个子 Agent，合并结果）。"""

    name = "parallel_composite"

    def __init__(
        self,
        children: List[AgentNode],
        merger: Optional[Any] = None,
        result_keys: Optional[List[str]] = None,
    ):
        self.children = children
        self.merger = merger or ResultMerger()
        self.result_keys = result_keys or [child.name for child in children]

    async def execute(self, ctx: AgentContext) -> Any:
        results = await asyncio.gather(
            *[child.execute(ctx) for child in self.children],
            return_exceptions=True,
        )

        perspectives: List[Any] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"并行 Agent 执行失败: {str(r)[:100]}")
                continue
            perspectives.append(r)

        for result, key in zip(perspectives, self.result_keys, strict=True):
            ctx.metadata[f"{key}_result"] = result

        return self.merger.merge(perspectives, ctx)


class SerialComposite(AgentNode):
    """串行组合 Agent（按顺序执行子 Agent，后一个依赖前一个的输出）。"""

    name = "serial_composite"

    def __init__(self, children: List[AgentNode]):
        self.children = children

    async def execute(self, ctx: AgentContext) -> Any:
        last_result: Optional[Any] = None

        for child in self.children:
            try:
                result = await child.execute(ctx)
                last_result = result
                ctx.metadata[f"{child.name}_result"] = result
            except Exception as e:
                logger.warning(f"串行 Agent {child.name} 失败: {str(e)[:100]}")

        return last_result


# ── 结果合并器 ────────────────────────────────────────────────────────


class ResultMerger:
    """Agent 结果合并器。"""

    @staticmethod
    def merge(results: List[Any], ctx: AgentContext) -> Any:
        if not results:
            return None

        if len(results) == 1:
            return results[0]

        if hasattr(results[0], "confidence"):
            best = max(results, key=lambda r: getattr(r, "confidence", 0))
            return best

        return results[0]
