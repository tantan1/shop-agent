"""
针对 5 个远程业务 API 的意图识别 + 参数抽取 + Mock API 端到端测试

使用方式:
    REMOTE_API_BASE_URL=http://localhost:8000/api/v1/mockapi  pytest tests/test_mockapi.py -v
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

from src.modules.chat.schemas import IntentResult, ChatRequest
from src.modules.chat.core.intent_recognizer import IntentRecognizer
from src.modules.chat.core.param_extractor import LocalParamExtractor
from src.modules.chat.core.tool_registry import ToolService


# =============================================================================
# 测试 1: 意图识别 —— 5 种远程 API 意图均能正确识别
# =============================================================================

class TestIntentRecognizer:
    """意图识别器单元测试 —— 5个远程API意图 + 复杂性门控"""

    @pytest.fixture(autouse=True)
    def clean_faiss_class_state(self):
        yield
        from src.modules.chat.core.intent_recognizer import IntentRecognizer as IR
        IR._faiss_index = None
        IR._intent_actions = []
        IR._intent_dim = 0

    @pytest.mark.asyncio
    async def test_recognize_query_order(self):
        """查订单意图识别"""
        recognizer = await self._build_recognizer_for_action("query-order", score=0.92)
        result = await recognizer.recognize("帮我查一下我的订单到哪了")
        assert result.intent == "call_remote_api"
        assert result.action == "query-order"
        assert result.similarity_score == 0.92
        assert result.complexity == "simple"  # 明确查询，分数高，无否定词

    @pytest.mark.asyncio
    async def test_recognize_check_shipping(self):
        """查物流意图识别"""
        recognizer = await self._build_recognizer_for_action("check-shipping", score=0.88)
        result = await recognizer.recognize("快递现在在什么地方")
        assert result.intent == "call_remote_api"
        assert result.action == "check-shipping"

    @pytest.mark.asyncio
    async def test_recognize_request_return(self):
        """退货退款意图识别 —— 强信号：ALWAYS_AGENT_ACTIONS，必为 multi_step"""
        recognizer = await self._build_recognizer_for_action("request-return", score=0.90)
        result = await recognizer.recognize("我要退货退款，这个商品不满意")
        assert result.intent == "call_remote_api"
        assert result.action == "request-return"
        assert result.complexity == "multi_step"  # request-return 永远 multi_step

    @pytest.mark.asyncio
    async def test_recognize_check_balance(self):
        """查余额意图识别"""
        recognizer = await self._build_recognizer_for_action("check-balance", score=0.87)
        result = await recognizer.recognize("我账户里还有多少余额")
        assert result.intent == "call_remote_api"
        assert result.action == "check-balance"

    @pytest.mark.asyncio
    async def test_recognize_coupon_inquiry(self):
        """优惠券查询意图识别"""
        recognizer = await self._build_recognizer_for_action("coupon-inquiry", score=0.85)
        result = await recognizer.recognize("我有什么优惠券可以用")
        assert result.intent == "call_remote_api"
        assert result.action == "coupon-inquiry"

    @pytest.mark.asyncio
    async def test_complexity_gating_multi_step_by_keywords(self):
        """含推理关键词 + 边界分数的意图 → multi_step"""
        recognizer = await self._build_recognizer_for_action("check-shipping", score=0.80)
        # 含"为什么"(触发词) + 低分(0.80 < 0.85)
        result = await recognizer.recognize("为什么我的快递还没到，怎么办")
        assert result.complexity == "multi_step"

    @pytest.mark.asyncio
    async def test_recognize_negation_patterns(self):
        """否定模式：政策/流程类问题走 RAG，不走远程 API"""
        recognizer = await self._build_recognizer_for_action("request-return", score=0.96)
        test_cases = [
            "退货政策是什么",
            "退款流程怎么走",
            "怎么退货啊",
            "退货条件有哪些",
        ]
        for msg in test_cases:
            result = await recognizer.recognize(msg)
            assert result.intent == "rag_answer", f"msg='{msg}' should be rag_answer"

    @pytest.mark.asyncio
    async def test_recognize_below_threshold(self):
        """FAISS 匹配分低于阈值 → 回退 RAG"""
        recognizer = await self._build_recognizer_for_action("query-order", score=0.60)
        result = await recognizer.recognize("今天天气怎么样")
        assert result.intent == "rag_answer"

    @staticmethod
    async def _build_recognizer_for_action(
        action: str, score: float
    ) -> IntentRecognizer:
        """
        构造带 FAISS mock 的意图识别器，精确控制匹配到的 action 和分数。

        通过 mock _faiss_index.search 返回值，模拟真实 BGE embedding 的语义匹配结果。
        """
        from src.modules.chat.core.intent_recognizer import INTENT_EXAMPLES

        # 构建 actions_flat（与 _init_faiss_intent_index 一致）
        actions_flat = []
        for a, phrases in INTENT_EXAMPLES.items():
            for _ in phrases:
                actions_flat.append(a)

        # action_idx 是 action 在 keys 中的位置；FAISS 索引是扁平化的，每个 action 有 N 条示例
        action_keys = list(INTENT_EXAMPLES.keys())
        action_idx = action_keys.index(action)
        examples_per_action = len(next(iter(INTENT_EXAMPLES.values())))
        flat_idx = action_idx * examples_per_action   # e.g. check_shipping=1*4=4

        # mock FAISS 索引
        mock_index = MagicMock()
        scores_arr = np.array([[score, score * 0.8]], dtype=np.float32)
        indices_arr = np.array([[flat_idx, 0]], dtype=np.int64)
        mock_index.search.return_value = (scores_arr, indices_arr)
        mock_index.ntotal = len(actions_flat)

        # mock embedding service
        mock_emb = MagicMock()
        mock_emb.get_embeddings.return_value = mock_emb
        mock_emb.aembed_query = AsyncMock(return_value=_random_norm_vector(768))
        mock_emb.aembed_documents = AsyncMock(return_value=[_random_norm_vector(768)] * len(actions_flat))

        mock_es = MagicMock()
        mock_es.get_embeddings.return_value = mock_emb

        recognizer = IntentRecognizer(embedding_service=mock_es)
        IntentRecognizer._faiss_index = mock_index
        IntentRecognizer._intent_actions = actions_flat
        IntentRecognizer._intent_dim = 768
        return recognizer


# =============================================================================
# 测试 2: 本地参数抽取 —— 5 种意图
# =============================================================================

class TestParamExtractor:
    """本地正则参数抽取单元测试"""

    def test_extract_query_order_with_id(self):
        params = LocalParamExtractor.extract(
            "帮我查订单 WB202405270001 到哪了", "query-order"
        )
        assert params.get("order_id") == "WB202405270001"

    def test_extract_query_order_with_status(self):
        params = LocalParamExtractor.extract(
            "看看已发货的订单", "query-order"
        )
        assert params.get("status_filter") == "已发货"

    def test_extract_check_shipping_with_tracking(self):
        params = LocalParamExtractor.extract(
            "快递单号 SF1234567890 现在到哪了", "check-shipping"
        )
        assert params.get("tracking_number") == "SF1234567890"

    def test_extract_check_shipping_with_order_id(self):
        params = LocalParamExtractor.extract(
            "查下 JD202405270016 这个订单的物流", "check-shipping"
        )
        assert "order_id" in params or "tracking_number" in params

    def test_extract_request_return_with_reason(self):
        params = LocalParamExtractor.extract(
            "WB202405270001 这个订单质量有问题，我要退货", "request-return"
        )
        assert params.get("order_id") == "WB202405270001"
        assert params.get("reason") == "质量问题"

    def test_extract_check_balance(self):
        params = LocalParamExtractor.extract("查余额", "check-balance")
        assert params == {}

    def test_extract_coupon_inquiry(self):
        params = LocalParamExtractor.extract("有没有满减券", "coupon-inquiry")
        assert params.get("coupon_type") == "满减券"


# =============================================================================
# 测试 3: Tool Registry —— Mock 远程 API 调用
# =============================================================================

class TestToolService:
    """ToolService —— 5种意图的tool调用"""

    @pytest.mark.asyncio
    async def test_dispatch_query_order_with_id(self):
        """指定 order_id 查询"""
        service = ToolService()
        result = await service.dispatch("query-order", {"order_id": "WB202405270001"})
        assert "WB202405270001" in result
        assert "查询" in result  # json.dumps 中的 "note": "已按 order_id=... 查询"

    @pytest.mark.asyncio
    async def test_dispatch_query_order_no_params(self):
        """不传参数=返回最近订单"""
        service = ToolService()
        result = await service.dispatch("query-order", {})
        data = json.loads(result)
        assert "orders" in data
        assert len(data["orders"]) >= 2

    @pytest.mark.asyncio
    async def test_dispatch_check_shipping(self):
        service = ToolService()
        result = await service.dispatch("check-shipping", {"tracking_number": "SF1234567890"})
        data = json.loads(result)
        assert "tracking" in data

    @pytest.mark.asyncio
    async def test_dispatch_request_return(self):
        service = ToolService()
        result = await service.dispatch("request-return", {
            "order_id": "WB202405270001",
            "reason": "质量问题"
        })
        # 返回为结构化 JSON（退货单号 + 待审核状态），而非含"退货"字样的文本
        data = json.loads(result)
        assert data.get("return_id"), "应返回退货单号"
        assert data.get("order_id") == "WB202405270001"
        assert data.get("status") == "待审核"

    @pytest.mark.asyncio
    async def test_dispatch_check_balance(self):
        service = ToolService()
        result = await service.dispatch("check-balance", {})
        data = json.loads(result)
        assert data.get("balance") == 520.00
        assert data.get("points") == 1280

    @pytest.mark.asyncio
    async def test_dispatch_coupon_inquiry(self):
        service = ToolService()
        result = await service.dispatch("coupon-inquiry", {})
        data = json.loads(result)
        assert "coupons" in data
        assert any("满200减30" in c["name"] for c in data["coupons"])

    @pytest.mark.asyncio
    async def test_dispatch_unknown_action_fallback(self):
        """未注册的 action 应该有兜底处理"""
        service = ToolService()
        result = await service.dispatch("unknown_action", {})
        assert "未配置" in result

    @pytest.mark.asyncio
    async def test_dispatch_all_five_actions(self):
        """所有 5 种远程 API 意图都能正常 dispatch"""
        service = ToolService()
        actions = ["query-order", "check-shipping", "request-return", "check-balance", "coupon-inquiry"]
        for action in actions:
            result = await service.dispatch(action, {})
            assert result, f"action={action} returned empty result"
            assert len(result) > 5, f"action={action} result too short: {result[:50]}"


# =============================================================================
# 辅助函数
# =============================================================================

def _random_norm_vector(dim: int) -> np.ndarray:
    """生成随机归一化向量（模拟 BGE embedding）"""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec
