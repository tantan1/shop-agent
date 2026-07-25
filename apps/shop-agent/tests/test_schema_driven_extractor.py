"""
SchemaDrivenExtractor 测试

覆盖场景：
  - 基础正则提取（order_id, phone, tracking_number）
  - alias 层：多个字段名映射到同一语义类型
  - MCP schema 动态驱动字段列表
  - 无 schema 时回退全部 alias 字段
  - 运行时注册新 alias / pattern
"""

import pytest
import re
from src.modules.chat.core.schema_driven_extractor import SchemaDrivenExtractor


class TestSchemaDrivenExtractor:

    # ── 基础提取（通过 alias 层） ──

    def test_extract_order_id_by_alias(self):
        """通过 alias 'order_id' → 'order' 语义类型提取订单号"""
        result = SchemaDrivenExtractor.extract(
            "帮我查一下订单号 WB202405270001",
            {"properties": {"order_id": {"type": "string"}}},
        )
        assert result["order_id"] == "WB202405270001"

    def test_extract_order_num_by_alias(self):
        """字段改名 order_id → order_num，alias 已注册，正则不重复"""
        result = SchemaDrivenExtractor.extract(
            "我的订单 GD202405250016",
            {"properties": {"order_num": {"type": "string"}}},
        )
        assert result["order_num"] == "GD202405250016"

    def test_extract_phone(self):
        result = SchemaDrivenExtractor.extract(
            "手机号 13800138000 帮我查订单",
            {"properties": {"phone": {"type": "string"}}},
        )
        assert result["phone"] == "13800138000"

    def test_extract_tracking_number(self):
        result = SchemaDrivenExtractor.extract(
            "查物流 快递单号: SF1234567890",
            {"properties": {"tracking_number": {"type": "string"}}},
        )
        assert result["tracking_number"] == "SF1234567890"

    # ── alias 同语义多字段名 ──

    def test_order_number_alias_shares_same_regex(self):
        """order_id 和 order_number 指向同一语义类型，共用同一正则"""
        msg = "我的订单 WB202403150066 到哪了"

        r1 = SchemaDrivenExtractor.extract(
            msg, {"properties": {"order_id": {"type": "string"}}}
        )
        r2 = SchemaDrivenExtractor.extract(
            msg, {"properties": {"order_number": {"type": "string"}}}
        )

        assert r1.get("order_id") == "WB202403150066"
        assert r2.get("order_number") == "WB202403150066"

    def test_mobile_phone_alias(self):
        """mobile 和 phone 指向同一语义类型"""
        msg = "电话是 13900139000"
        r1 = SchemaDrivenExtractor.extract(
            msg, {"properties": {"phone": {"type": "string"}}}
        )
        r2 = SchemaDrivenExtractor.extract(
            msg, {"properties": {"mobile": {"type": "string"}}}
        )
        assert r1.get("phone") == "13900139000"
        assert r2.get("mobile") == "13900139000"

    # ── Schema 驱动字段列表 ──

    def test_schema_drives_field_list(self):
        """只有 mcp_schema 里声明的字段才会被提取，多余字段不提取"""
        schema = {
            "properties": {
                "order_id": {"type": "string"},
                "phone": {"type": "string"},
            }
        }
        msg = "查订单 WB202405050088 快递 SF1234567890"
        result = SchemaDrivenExtractor.extract(msg, schema)

        # order_id 和 phone 在 schema 里 → 被提取
        assert "order_id" in result
        # tracking_number 不在 schema 里 → 不会被提取
        assert "tracking_number" not in result
        assert "tracking_no" not in result

    def test_empty_schema_returns_empty(self):
        """空 schema 返回空结果"""
        result = SchemaDrivenExtractor.extract(
            "订单 WB202405270001", {"properties": {}}
        )
        assert result == {}

    def test_no_schema_fallback_all_aliases(self):
        """无 schema 时回退到全部已注册 alias 字段"""
        msg = "订单 WB202405270001 手机 13800138000 快递 SF1234567890"
        result = SchemaDrivenExtractor.extract(msg, None)
        # 应该提取到所有匹配的字段
        assert len(result) > 0
        # order_id 应该被提取
        order_fields = [k for k in result if k.startswith("order_")]
        assert len(order_fields) >= 1

    # ── 多字段同时提取 ──

    def test_extract_multiple_fields_from_schema(self):
        schema = {
            "properties": {
                "order_id": {"type": "string"},
                "phone": {"type": "string"},
                "tracking_number": {"type": "string"},
            }
        }
        msg = "查订单 WB202405270001 手机号 13800138000 快递 SF1234567890"
        result = SchemaDrivenExtractor.extract(msg, schema)

        assert result.get("order_id") == "WB202405270001"
        assert result.get("phone") == "13800138000"
        assert "SF1234567890" in str(result.get("tracking_number", ""))

    # ── 无匹配 ──

    def test_no_match_returns_empty(self):
        result = SchemaDrivenExtractor.extract(
            "今天天气真好",
            {"properties": {"order_id": {"type": "string"}}},
        )
        assert result == {}

    def test_unknown_field_name_skipped(self):
        """schema 里声明了但 alias 层不认识的字段 → 跳过"""
        result = SchemaDrivenExtractor.extract(
            "一些文本",
            {"properties": {"unknown_field": {"type": "string"}}},
        )
        assert "unknown_field" not in result

    # ── 运行时注册 ──

    def test_register_alias_dynamic(self):
        """运行时注册新 alias，立即生效"""
        SchemaDrivenExtractor.register_alias("custom_id", "order")
        result = SchemaDrivenExtractor.extract(
            "订单 GD202405010099",
            {"properties": {"custom_id": {"type": "string"}}},
        )
        assert result.get("custom_id") == "GD202405010099"

    def test_register_pattern_dynamic(self):
        """运行时注册新语义类型"""
        SchemaDrivenExtractor.register_pattern("sku", re.compile(r"SKU[:\s]*(\w+)"))
        SchemaDrivenExtractor.register_alias("product_sku", "sku")

        result = SchemaDrivenExtractor.extract(
            "帮我查 SKU: ABC12345 的库存",
            {"properties": {"product_sku": {"type": "string"}}},
        )
        assert result.get("product_sku") == "ABC12345"

    # ── 参数值过滤空字符串 ──

    def test_empty_params(self):
        """只传 schema 字段名，无参数时返回空"""
        result = SchemaDrivenExtractor.extract(
            "你好", {"properties": {"order_id": {"type": "string"}}}
        )
        assert result == {}
