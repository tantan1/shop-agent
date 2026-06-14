"""
本地参数抽取器 —— 纯正则 + 关键词匹配，零 API 调用，毫秒级延迟
替代 Qwen LLM 的 structured output 方案
"""
from typing import List, Dict, Any, Optional
import re


class LocalParamExtractor:
    """本地参数抽取器 —— 正则+关键词，无需 LLM"""

    # ── 订单号：关键词组 + 任意中间内容 + ID数字 ──
    _ORDER_ID_RE = re.compile(
        r'(?:订单号|订单编号|订单ID|订单状态|订单\s*状态|我的订单|订单|#)'
        r'[^\d]*'  # 允许关键词和数字之间夹任意非数字内容（如"状态"）
        r'([A-Za-z]{0,4}\d{8,})'
        r'|(?<!\d)([A-Z]{2,4}\d{10,})(?!\d)'
    )
    _ORDER_ID_FALLBACK_RE = re.compile(
        r'(?<!\d)(\d{16,})(?!\d)'  # 纯数字长串（10+位可能是订单号/交易号）
    )

    # ── 手机号 ──
    _PHONE_FULL_RE = re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)')
    _PHONE_LAST4_RE = re.compile(
        r'(?:手机号|手机|电话|号码)(?:后四位|末尾|最后\s*四位)?[\s:：为是]+\**(\d{4})'
    )

    # ── 快递单号 ──
    _TRACKING_RE = re.compile(
        r'(?:快递单号|运单号|物流单号|快递号|快递|物流|包裹)\s*[:：]?\s*'
        r'([A-Za-z]?[A-Za-z0-9]{9,})'
        r'|(?<!\d)(SF|YT|JD|EMS|FA|DB|ZA|STO|YUNDA|ZT)\s*(\d{8,})(?!\d)'
    )

    # ── 订单状态关键词 ──
    _STATUS_MAP: Dict[str, str] = {
        "待付款": r"待付款|没付款|未付款|未支付|还没付|没付钱",
        "已发货": r"已发货|发货了|发出了|发货没|发货状态",
        "派送中": r"派送中|配送中|运输中|在路上|途中|送.*路上",
        "已签收": r"已签收|签收了|收到了|到货了",
    }

    # ── 退货原因关键词 ──
    _RETURN_REASON_MAP: Dict[str, str] = {
        "质量问题": r"质量|坏了|破损|有瑕疵|有问题|不好使|不工作|次品",
        "不想要": r"不想要|不需要|买错|不想要了|后悔|冲动",
        "发错货": r"发错|发错货|错发|不是我要|不对",
        "与描述不符": r"不符合|不符|不一[样致]|差太多|假货|虚假",
    }

    # ── 优惠券类型关键词 ──
    _COUPON_TYPE_MAP: Dict[str, str] = {
        "满减券": r"满减|满.*减",
        "折扣券": r"折扣|打折|几折",
        "运费券": r"运费券|免邮|包邮|运费.*券",
    }

    # ========================================================================
    # 公共工具方法
    # ========================================================================

    @classmethod
    def _extract_order_id(cls, message: str) -> Optional[str]:
        """从消息中提取订单号"""
        m = cls._ORDER_ID_RE.search(message)
        if m:
            return m.group(1) or m.group(2)
        m = cls._ORDER_ID_FALLBACK_RE.search(message)
        if m:
            return m.group(1)
        return None

    @classmethod
    def _extract_phone(cls, message: str) -> Optional[str]:
        """从消息中提取手机号"""
        m = cls._PHONE_FULL_RE.search(message)
        if m:
            return m.group(1)
        m = cls._PHONE_LAST4_RE.search(message)
        if m:
            return m.group(1)
        return None

    @classmethod
    def _extract_tracking_number(cls, message: str) -> Optional[str]:
        """从消息中提取快递单号"""
        m = cls._TRACKING_RE.search(message)
        if m:
            return m.group(1) or m.group(2) or m.group(3)
        return None

    @classmethod
    def _match_keyword(cls, message: str, mapping: Dict[str, str]) -> Optional[str]:
        """用正则关键词匹配，返回第一个命中的标签"""
        for label, pattern in mapping.items():
            if re.search(pattern, message):
                return label
        return None

    # ========================================================================
    # 各意图的参数抽取方法
    # ========================================================================

    @classmethod
    def _extract_query_order(cls, message: str) -> Dict[str, Any]:
        """查订单参数抽取"""
        result: Dict[str, Any] = {}
        oid = cls._extract_order_id(message)
        if oid:
            result["order_id"] = oid
        phone = cls._extract_phone(message)
        if phone:
            result["phone"] = phone
        status = cls._match_keyword(message, cls._STATUS_MAP)
        if status:
            result["status_filter"] = status
        return result

    @classmethod
    def _extract_check_shipping(cls, message: str) -> Dict[str, Any]:
        """查物流参数抽取"""
        result: Dict[str, Any] = {}
        tn = cls._extract_tracking_number(message)
        if tn:
            result["tracking_number"] = tn
        oid = cls._extract_order_id(message)
        if oid:
            result["order_id"] = oid
        return result

    @classmethod
    def _extract_request_return(cls, message: str) -> Dict[str, Any]:
        """退货退款参数抽取"""
        result: Dict[str, Any] = {}
        oid = cls._extract_order_id(message)
        if oid:
            result["order_id"] = oid
        reason = cls._match_keyword(message, cls._RETURN_REASON_MAP)
        if reason:
            result["reason"] = reason
        return result

    @classmethod
    def _extract_check_balance(cls, message: str) -> Dict[str, Any]:
        """查余额参数抽取（暂无参数）"""
        return {}

    @classmethod
    def _extract_coupon_inquiry(cls, message: str) -> Dict[str, Any]:
        """查优惠券参数抽取"""
        result: Dict[str, Any] = {}
        ctype = cls._match_keyword(message, cls._COUPON_TYPE_MAP)
        if ctype:
            result["coupon_type"] = ctype
        return result

    # ========================================================================
    # 语义类型 → 抽取方法映射（供 Schema 驱动抽取使用）
    # ========================================================================

    _SEMANTIC_MAP: Dict[str, classmethod] = {}

    @classmethod
    def _ensure_semantic_map(cls):
        """懒加载语义类型映射表"""
        if cls._SEMANTIC_MAP:
            return
        cls._SEMANTIC_MAP.update({
            "order_id": cls._extract_order_id,
            "phone": cls._extract_phone,
            "tracking_number": cls._extract_tracking_number,
            "order_status": lambda m: cls._match_keyword(m, cls._STATUS_MAP),
            "return_reason": lambda m: cls._match_keyword(m, cls._RETURN_REASON_MAP),
            "coupon_type": lambda m: cls._match_keyword(m, cls._COUPON_TYPE_MAP),
        })

    # ========================================================================
    # 统一入口
    # ========================================================================

    _EXTRACTORS: Dict[str, classmethod] = {}

    @classmethod
    def _ensure_extractors(cls):
        """懒加载抽取器注册表（旧 action-based 路径，向后兼容）"""
        if cls._EXTRACTORS:
            return
        cls._EXTRACTORS.update({
            "query-order":     cls._extract_query_order,
            "check-shipping":  cls._extract_check_shipping,
            "request-return":  cls._extract_request_return,
            "check-balance":   cls._extract_check_balance,
            "coupon-inquiry":  cls._extract_coupon_inquiry,
        })

    @classmethod
    def extract(
        cls,
        message: str,
        action: str = "",
        params_schema: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Any]:
        """
        从用户消息中提取结构化参数。

        优先使用 params_schema（从 SKILL.md frontmatter 解析）驱动抽取，
        根据每个参数的 semantic 标签匹配对应正则/关键词提取器。
        无 params_schema 时回退到 action-based 抽取（兼容旧调用）。
        """
        # 新方式：Schema 驱动
        if params_schema:
            cls._ensure_semantic_map()
            result: Dict[str, Any] = {}
            for param_name, param_def in params_schema.items():
                if not isinstance(param_def, dict):
                    continue
                semantic = param_def.get("semantic")
                if not semantic:
                    continue
                extractor = cls._SEMANTIC_MAP.get(semantic)
                if extractor is None:
                    continue
                value = extractor(message)
                if value:
                    result[param_name] = value
            return result

        # 旧方式：action-based（向后兼容）
        if action:
            cls._ensure_extractors()
            fn = cls._EXTRACTORS.get(action)
            if fn is not None:
                return fn(message)
        return {}
