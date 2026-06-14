"""
SchemaDrivenExtractor — MCP 时代的参数抽取器

核心设计：结构层从 MCP inputSchema 动态驱动，语义层只存"字段类型 → 正则"映射。
新增字段 / 字段改名都不需要改 extract() 循环，只需在 alias 层加一行。

三层解耦：
  _PATTERNS       —— 语义类型 → 正则（一种语义一个正则，绝不重复）
  _FIELD_ALIASES  —— MCP 字段名 → 语义类型（字段改名时只加 alias）
  mcp_schema      —— 结构层（有哪些字段、叫什么），MCP Server 动态提供
"""

from __future__ import annotations

import re
from typing import Dict, Any, Pattern


class SchemaDrivenExtractor:
    """Schema-driven 参数提取器：正则的 field 列表由 MCP schema 驱动，不硬编码。"""

    # ── 语义层：一个语义类型 → 一个正则（零重复） ──
    _PATTERNS: Dict[str, Pattern] = {
        "order": re.compile(
            r"(?:订单号|订单编号|订单ID|订单\s*状态|我的订单|订单|#)"
            r"\s*"
            r"([A-Za-z]{0,4}\d{8,})"
            r"|(?<!\d)([A-Z]{2,4}\d{10,})(?!\d)"
        ),
        "phone": re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)"),
        "tracking": re.compile(
            r"(?:快递单号|运单号|物流单号|快递号|快递|物流|包裹)\s*[:：]?\s*"
            r"([A-Za-z]?[A-Za-z0-9]{9,})"
            r"|(?<!\d)(SF|YT|JD|EMS|FA|DB|ZA|STO|YUNDA|ZT)\s*(\d{8,})(?!\d)"
        ),
        "order_status": re.compile(r"待付款|已发货|派送中|已签收"),
        "return_reason": re.compile(r"质量|坏了|破损|不想要|发错|不符合|不适应"),
        "coupon_type": re.compile(r"满减|折扣|打折|运费券|免邮|包邮"),
    }

    # ── alias 层：MCP 字段名 → 语义类型 ──
    _FIELD_ALIASES: Dict[str, str] = {
        # 订单相关（多个字段名共享同一正则）
        "order_id":         "order",
        "order_num":        "order",        # 改名了只需一行 alias
        "order_number":     "order",
        # 手机
        "phone":            "phone",
        "mobile":           "phone",
        # 物流
        "tracking_number":  "tracking",
        "tracking_no":      "tracking",
        # 状态 / 原因 / 优惠券类型
        "status":           "order_status",
        "status_filter":    "order_status",
        "reason":           "return_reason",
        "return_reason":    "return_reason",
        "coupon_type":      "coupon_type",
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def extract(cls, message: str, mcp_schema: dict | None = None) -> Dict[str, Any]:
        """从用户消息中提取参数，字段列表由 mcp_schema["properties"] 驱动。

        Args:
            message:     用户原始消息
            mcp_schema:  MCP tool 的 inputSchema（含 "properties" 键），
                         为 None 时回退到全部已注册 alias 字段。

        Returns:
            参数字典，如 {"order_id": "WB202405270001", "phone": "13800138000"}。
            未匹配的字段不出现在结果中。
        """
        result: Dict[str, Any] = {}

        # 结构层：从 schema 动态获取字段列表
        field_names: list[str]
        if mcp_schema and "properties" in mcp_schema:
            field_names = list(mcp_schema["properties"].keys())
        else:
            # 无 schema 时回退到全部 alias 字段
            field_names = list(cls._FIELD_ALIASES.keys())

        for field_name in field_names:
            value = cls._extract_field(message, field_name)
            if value is not None:
                result[field_name] = value

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @classmethod
    def _extract_field(cls, message: str, field_name: str) -> str | None:
        """提取单个字段的值。"""
        sem_type = cls._FIELD_ALIASES.get(field_name)
        if sem_type is None:
            return None

        pattern = cls._PATTERNS.get(sem_type)
        if pattern is None:
            return None

        m = pattern.search(message)
        if not m:
            return None

        # 处理多 capture group：拼接所有非 None 的 group
        # 场景：快递单选 "SF" + "1234567890" → "SF1234567890"
        groups = [g for g in m.groups() if g is not None]
        if groups:
            return "".join(groups)

        return m.group(0)

    # ------------------------------------------------------------------
    # 注册接口（允许运行时扩展 alias / pattern）
    # ------------------------------------------------------------------

    @classmethod
    def register_alias(cls, field_name: str, sem_type: str) -> None:
        """注册新的字段名 → 语义类型 alias（字段改名时使用）。"""
        cls._FIELD_ALIASES[field_name] = sem_type

    @classmethod
    def register_pattern(cls, sem_type: str, pattern: Pattern | str) -> None:
        """注册新的语义类型 → 正则（出现新的字段语义时使用）。"""
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        cls._PATTERNS[sem_type] = pattern
