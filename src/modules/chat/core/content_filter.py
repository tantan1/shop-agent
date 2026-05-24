"""
内容安全过滤服务 (Content Safety Filter Service)

职责：
- 输入过滤：规则引擎轻量拦截（仅阻断明显恶意/非法内容、Prompt Injection）
- 输出过滤：规则引擎强制审查（最终合规底线，扫描 LLM 输出内容）
- 零 LLM 调用成本，纯规则引擎 + 正则匹配

设计原则（纵深防御）：
  输入过滤（轻量拦截，节省资源）→ LLM 安全审查（Step2，风险标记）→ LLM 生成 → 输出过滤（硬阻断，合规底线）
"""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# 敏感词库
# ═══════════════════════════════════════════════════════════════════════

# 输入敏感词 —— 仅拦截明显恶意/非法内容（极保守，避免误杀）
INPUT_BLOCK_KEYWORDS: List[str] = [
    # 明确违法
    "制作毒品", "买卖枪支", "贩毒", "洗钱方法",
    "制作炸弹", "炸药配方", "暗网链接",
    # 明确色情
    "儿童色情", "幼女图片",
]

# 输出敏感词 —— 按领域分类（合规底线，硬阻断）
OUTPUT_BLOCK_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "medical": {
        # block: 包含这些词代表 LLM 输出了诊断/用药建议
        "block": [
            "根据你的症状，建议服用", "你的诊断是", "确诊为", "开药给你",
            "推荐剂量", "自行购买", "自行服用", "我建议你吃",
            "不用去医院", "在家治疗", "偏方治疗", "祖传秘方",
        ],
        # replace: 包含这些词则替换为 ***（降级而非完全拦截）
        "replace": [],
    },
    "ecommerce": {
        "block": [
            "加微信", "加我QQ", "扫码付款", "私下交易",
            "刷单", "刷好评", "刷销量", "假货工厂",
            "点击链接购买", "复制链接", "转到其他平台",
            "银行卡号", "身份证号", "信用卡信息",
        ],
        "replace": [
            "联系方式", "微信号", "手机号",
        ],
    },
    "customer_service": {
        "block": [
            "加微信", "加QQ", "私下联系", "给个好评返现",
            "色情", "暴力威胁", "人肉搜索", "政治敏感",
        ],
        "replace": [],
    },
    "general": {
        "block": [
            # 违法/暴力
            "制作武器", "购买毒品", "暗网交易",
            # 色情
            "色情内容", "成人服务", "性交易",
            # 诈骗
            "刷单返利", "投资稳赚", "博彩平台",
        ],
        "replace": [],
    },
}

# ═══════════════════════════════════════════════════════════════════════
# Prompt Injection 检测模式
# ═══════════════════════════════════════════════════════════════════════

INJECTION_PATTERNS: List[re.Pattern] = [
    # 经典 jailbreak：忽略之前的指令
    re.compile(
        r"(?i)(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|above|prior|the)\s+(?:instructions|directions|prompts|rules|content|messages)",
        re.IGNORECASE,
    ),
    # "你是 DAN" / 角色扮演绕过
    re.compile(
        r"(?i)(?:you\s+are\s+now|pretend\s+you\s+are|act\s+as)\s+(?:DAN|jailbreak|an?\s+unfiltered|evil)",
        re.IGNORECASE,
    ),
    # System prompt 泄露尝试
    re.compile(
        r"(?i)(?:repeat|show|print|output|tell\s+me|what\s+is)\s+(?:the\s+words\s+above|your\s+(?:system\s+)?prompt|your\s+instructions|everything\s+above)",
        re.IGNORECASE,
    ),
    # 分隔符注入（连续 3+ 个反引号或引号包裹指令）
    re.compile(r"`{3,}\s*(?:system|assistant|user)\s*[:：]", re.IGNORECASE),
    # "From now on" / "starting now" 角色替换
    re.compile(
        r"(?i)(?:from\s+now\s+on|starting\s+now|henceforth)\s+(?:you\s+(?:are|will))?",
        re.IGNORECASE,
    ),
    # "输出以下内容" 型绕过
    re.compile(
        r"(?i)(?:输出以下内容|直接输出|必须按照|\bMUST\b).{0,20}(?:忽略|不要|禁止).{0,20}(?:规则|限制|安全)",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════

class FilterResult(BaseModel):
    """过滤结果"""
    is_safe: bool = Field(default=True, description="内容是否安全")
    risk_level: str = Field(default="low", description="风险等级: low | medium | high | critical")
    risk_categories: List[str] = Field(default_factory=list, description="命中风险类别")
    filtered_text: Optional[str] = Field(default=None, description="过滤后文本（None=完全拦截）")
    reason: str = Field(default="", description="拦截/过滤原因")


# ═══════════════════════════════════════════════════════════════════════
# 过滤服务
# ═══════════════════════════════════════════════════════════════════════

class ContentFilterService:
    """内容安全过滤服务（单例，纯规则引擎，零 LLM 成本）"""

    _instance: Optional["ContentFilterService"] = None

    @classmethod
    def get_instance(cls) -> "ContentFilterService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # 输入过滤（轻量拦截，仅阻塞明显恶意内容）
    # ------------------------------------------------------------------

    def filter_input(self, text: str, domain: str = "general") -> FilterResult:
        """
        输入安全过滤 —— 轻量级，仅拦截：
        1. Prompt Injection 攻击
        2. 明确非法内容（制作毒品、儿童色情等）

        Args:
            text: 用户输入文本
            domain: 业务领域（暂未按领域区分输入敏感词）

        Returns:
            FilterResult
        """
        risk_categories: List[str] = []

        # 1. Prompt Injection 检测
        for pattern in INJECTION_PATTERNS:
            if pattern.search(text):
                risk_categories.append("prompt_injection")
                break  # 一个命中就足够

        # 2. 明确非法关键词检测
        for kw in INPUT_BLOCK_KEYWORDS:
            if kw in text:
                risk_categories.append("illegal_content")
                break

        if risk_categories:
            return FilterResult(
                is_safe=False,
                risk_level="critical",
                risk_categories=risk_categories,
                filtered_text=None,  # None = 完全拦截
                reason=f"输入包含不安全内容: {', '.join(risk_categories)}",
            )

        # 3. 长度/格式异常检测（辅助规则）
        if self._detect_length_anomaly(text):
            return FilterResult(
                is_safe=False,
                risk_level="medium",
                risk_categories=["anomaly_pattern"],
                filtered_text=None,
                reason="输入格式异常",
            )

        return FilterResult(is_safe=True)

    # ------------------------------------------------------------------
    # 输出过滤（硬阻断，合规底线）
    # ------------------------------------------------------------------

    def filter_output(self, text: str, domain: str = "general") -> FilterResult:
        """
        输出安全过滤 —— 强制合规底线，扫描 LLM 生成的最终回答

        策略：
        - 'block' 关键词命中 → 完全拦截，返回 None
        - 'replace' 关键词命中 → 替换为 ***，返回过滤后文本
        - 均未命中 → 通过

        Args:
            text: LLM 输出文本
            domain: 业务领域 (medical/ecommerce/customer_service/general)

        Returns:
            FilterResult
        """
        domain_kw = OUTPUT_BLOCK_KEYWORDS.get(domain, OUTPUT_BLOCK_KEYWORDS["general"])
        block_keywords = domain_kw.get("block", [])
        replace_keywords = domain_kw.get("replace", [])
        hit_block: List[str] = []
        hit_replace: List[str] = []

        # 扫描 block 关键词
        for kw in block_keywords:
            if kw in text:
                hit_block.append(kw)

        # 扫描 replace 关键词
        for kw in replace_keywords:
            if kw in text:
                hit_replace.append(kw)

        if hit_block:
            return FilterResult(
                is_safe=False,
                risk_level="high",
                risk_categories=[f"blocked:{','.join(hit_block)}"],
                filtered_text=None,
                reason=f"输出包含违规内容: {', '.join(hit_block)}",
            )

        if hit_replace:
            # 敏感词替换为 ***
            filtered = text
            for kw in hit_replace:
                filtered = filtered.replace(kw, "***")
            return FilterResult(
                is_safe=False,
                risk_level="medium",
                risk_categories=[f"replaced:{','.join(hit_replace)}"],
                filtered_text=filtered,
                reason=f"输出包含敏感词，已脱敏处理: {', '.join(hit_replace)}",
            )

        return FilterResult(is_safe=True)

    # ------------------------------------------------------------------
    # 异常检测辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_length_anomaly(text: str) -> bool:
        """检测输入长度/内容异常（如大量重复字符）"""
        if len(text) < 3:
            return False

        # 同一字符连续出现超过 50 次
        char_counts: Dict[str, int] = {}
        for ch in text:
            char_counts[ch] = char_counts.get(ch, 0) + 1
        if char_counts and max(char_counts.values()) > max(len(text) * 0.9, 50):
            return True

        return False
