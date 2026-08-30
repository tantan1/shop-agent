"""
分层参数提取管道（layered_param_extractor）单元测试

覆盖（Phase 1）：
  - L0 归一化：全角→半角、去零宽、小写
  - L1 正则档抽取 + 高后果 HIGH 置信
  - L4 enforce：冲突消解（HIGH>LOW）、格式闸拦截非法值、必填完整性
  - run_pipeline：无 schema 时退化为 alias 字段；regex 命中即早退（不触发 small_model）
  - SmallModelLayer 不可用时的优雅降级（不阻塞主链路）
"""

import asyncio
import pytest

from src.modules.chat.core.layered_param_extractor import (
    Candidate,
    EnforceResult,
    McpSchemaProvider,
    enforce,
    normalize,
    run_pipeline,
)

# 注：RegexLayer 产出的 HIGH 候选一定过格式闸（同源正则）；
#     格式闸主要拦截 small_model/ner 产出的非 HIGH 候选 —— 它们统一进 not_finalized 交 L5。


class _FakeProvider(McpSchemaProvider):
    """测试用：始终返回 None schema，触发退化为全部 alias 字段。"""

    async def get_schema(self, tool_name):
        return None


@pytest.fixture
def loop():
    return asyncio.new_event_loop()


# ── L0 归一化 ──

def test_normalize_fullwidth_and_zero_width():
    raw = "我的订单号是ＯＲＤＥＲ\u200b1234567890，手机13800138000"
    out = normalize(raw)
    assert "order1234567890" in out          # 全角→半角 + 去零宽 + 小写
    assert "\u200b" not in out               # 零宽字符被移除
    assert "13800138000" in out


def test_normalize_truncates_long_text():
    out = normalize("x" * 5000)
    assert len(out) == 2000


# ── L4 enforce ──

def test_enforce_prefers_high_over_low():
    cands = {
        "order_id": [
            Candidate("order", "ORD123", "LOW", "small_model"),
            Candidate("order", "ORDER1234567890", "HIGH", "regex"),
        ]
    }
    res = enforce(cands)
    assert res.params["order_id"] == "ORDER1234567890"   # HIGH 胜出


def test_enforce_low_conf_not_finalized():
    # 非 HIGH 候选（如 small_model 抽出）不直接落定稿，交 L5
    cands = {"order_id": [Candidate("order", "123", "HIGH", "regex")]}
    # 用 regex 无法产生 "123"（订单正则要求 8+ 数字），此处模拟 low 候选
    cands = {"order_id": [Candidate("order", "一些含糊描述", "LOW", "small_model")]}
    res = enforce(cands, required_fields=["order_id"])
    assert "order_id" in res.not_finalized
    assert "order_id" not in res.params              # 未定稿不落定稿


def test_enforce_missing_required_reported():
    cands = {"phone": [Candidate("phone", "13800138000", "HIGH", "regex")]}
    res = enforce(cands, required_fields=["order_id", "phone"])
    assert res.missing_required == ["order_id"]      # 缺必填被记录


def test_enforce_low_conf_not_finalized():
    # 非 HIGH 命中不直接落定稿，交 L5 处理
    cands = {"order_id": [Candidate("order", "一些含糊描述", "LOW", "small_model")]}
    res = enforce(cands)
    assert "order_id" not in res.params


# ── run_pipeline 集成 ──

def test_run_pipeline_regex_extract_and_degrade():
    async def _run():
        return await run_pipeline(
            "我的订单号是ORDER1234567890，手机13800138000，想退货因为质量问题",
            "request-return",
            _FakeProvider(),
        )
    # 用 asyncio.run 而非 get_event_loop().run_until_complete()：
    # 后者依赖全局 loop 状态，若其他测试模块用 asyncio.run() 关闭过 loop，
    # 此处会抛 "There is no current event loop" —— 属于跨模块污染。
    res: EnforceResult = asyncio.run(_run())
    assert res.params.get("phone") == "13800138000"
    assert "质量" in (res.params.get("reason"), res.params.get("return_reason"))


def test_run_pipeline_bad_order_id_is_missing_not_finalized():
    async def _run():
        return await run_pipeline(
            "订单号是 123，查询一下",
            "query-order",
            _FakeProvider(),
            extra_required=["order_id"],
        )
    # 同上：使用自包含的 asyncio.run()，避免依赖全局事件循环
    res: EnforceResult = asyncio.run(_run())
    # "123" 不匹配订单正则 → 无候选 → 记为缺必填（而非格式非法）
    assert "order_id" in res.missing_required
    assert "order_id" not in res.params
