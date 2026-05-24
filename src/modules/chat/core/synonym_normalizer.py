"""
输入同义词归一化服务 (Synonym Normalization Service)

职责：
- L1 静态同义词表：将用户变体表达归一化为标准意图术语（覆盖电商核心场景）
- L2 文本标准化：全角→半角、繁体→简体、多余空格/标点清理
- L3 LLM 归一化：调用 LLM 将任意自然语言表达改写为标准查询（默认关闭，按需开启）

设计原则：
  - L1+L2 默认开启，零 LLM 成本，零延迟
  - L3 按配置开关控制，提供兜底覆盖长尾表达
"""

import re
from typing import Optional, Dict, List

from src.core.config import config as global_config
from src.shared.logger import APILogger

logger = APILogger("synonym_normalizer")

# ═══════════════════════════════════════════════════════════════════════
# L1: 同义词映射表（高频电商场景）
# 将变体表达 → 归一化为标准术语，帮助 FAISS 意图识别命中
# ═══════════════════════════════════════════════════════════════════════

SYNONYM_MAP: Dict[str, List[str]] = {
    # ── 退货/退款 ──
    "我要退货": [
        "我想退货", "我要退了这个", "把这个退了", "帮我退了", "我想退掉",
        "可以退货吗", "怎么退货", "退货怎么操作", "申请退货",
        "我要退款", "我想退款", "把钱退给我", "可以退款吗", "怎么退款",
        "退款申请", "我要退钱", "退钱给我", "给我退款", "帮我把钱退了",
        "想退", "不想要了", "我想退单", "取消订单退", "订单退了",
        "能退吗", "退了吧", "退掉这个", "这个退了",
    ],
    "查询订单": [
        "查订单", "我的订单", "订单在哪", "看看我买的", "我买了什么",
        "订单状态", "订单到哪了", "订单进度", "查一下订单",
        "我的东西到哪了", "买的什么时候到", "订单什么时候发货",
        "显示我的订单", "列出我的订单", "查看购买记录", "购买记录",
        "历史订单", "最近订单", "订单列表", "我下的单",
        "帮我看下订单", "订单详情", "看下订单", "订单在哪看",
    ],
    "查询物流": [
        "物流信息", "快递到哪了", "查快递", "物流进度", "什么时候到",
        "快递信息", "送哪了", "派送了吗", "还能到吗", "配送进度",
        "发货了吗", "什么时候发货", "还没发货", "催一下发货",
        "快递单号", "看下物流", "运单号", "快递还没到",
        "包裹在哪", "东西到哪了", "怎么还没到", "已经发货了吗",
    ],
    "查询余额": [
        "余额", "账户余额", "钱包余额", "还有多少钱", "查一下余额",
        "余额查询", "账户有多少钱", "我的余额", "剩余金额",
        "卡里余额", "钱还有多少", "账户资产",
    ],
    "优惠券": [
        "优惠券", "有什么券", "领优惠券", "券在哪", "打折券",
        "满减券", "红包", "代金券", "抵扣券", "查看优惠券",
        "有没有券", "领取红包", "可以用的券", "怎么领券",
        "我还有什么券", "优惠券在哪看", "能用什么优惠",
    ],
    # ── 客服/咨询 ──
    "售后咨询": [
        "客服", "人工客服", "联系客服", "找人工", "转人工",
        "投诉", "建议", "售后", "反馈", "举报",
        "有问题", "遇到问题", "帮我解决", "怎么办",
    ],
    # ── 商品咨询 ──
    "商品咨询": [
        "这个商品", "好不好用", "质量怎么样", "多少钱", "有货吗",
        "什么时候有货", "补货", "上架", "怎么买",
        "有没有这个", "有没有卖", "想买", "怎么下单",
        "如何购买", "购买方式", "支付方式",
    ],
}

# 构建快速查找字典: variant → canonical
_VARIANT_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _variants in SYNONYM_MAP.items():
    for _v in _variants:
        _VARIANT_TO_CANONICAL[_v] = _canonical


# ═══════════════════════════════════════════════════════════════════════
# L1: 正则模式归一化（更灵活的词级替换）
# ═══════════════════════════════════════════════════════════════════════

REGEX_REPLACE_RULES: List[tuple] = [
    # (正则模式, 替换文本, 领域)
    # 退货意图模式
    (r"(?:我想?|我要|帮我|帮我|给我|申请|可以|能不能)(?:退掉?[货单]|退[货单]|退货|退款|退钱)(?:.*)", "我要退货", "all"),
    (r"(?:想|要|想?要)(?:退|取消|不要)(?:了|掉)(?:这个|那个|它|订单)?", "我要退货", "all"),
    (r"(?:能不能|可以|怎么|如何)(?:退[货单]|退款)", "我要退货", "all"),
    (r"(?:不想要了|不满意|不喜欢|想退单|取消订单)", "我要退货", "all"),
    # 查单模式
    (r"(?:查|看|帮我查|帮我看看|帮我查一下|查看|显示)(?:我的)?(?:一下)?(?:订单|购买记录|下的单)", "查询订单", "all"),
    (r"(?:订单|快递|东西|包裹)(?:到哪了|到哪|在哪|怎么还没到|什么时候到)", "查询物流", "all"),
    (r"(?:什么时候|还能)(?:发货|送到|到货)", "查询物流", "all"),
    # 余额模式
    (r"(?:查|看)(?:一下)?(?:我的|账户)?(?:余额|钱包|账户|有多少钱|资产)", "查询余额", "all"),
    # 优惠券模式
    (r"(?:查|看|领|领取|有什么|有没有)(?:一下)?(?:我的)?(?:优惠券|券|红包|满减券|折扣券)", "优惠券", "all"),
    (r"(?:怎么|如何)(?:领|领取)(?:券|优惠券|红包)", "优惠券", "all"),
    # 客服模式
    (r"(?:转|找|联系|切换)(?:人工|客服|人工客服)", "售后咨询", "all"),
]


# ═══════════════════════════════════════════════════════════════════════
# L2: 文本标准化工具
# ═══════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """
    文本预处理标准化（毫秒级，纯规则）

    处理：
    1. 全角字母/数字 → 半角
    2. 全角标点 → 半角（但中文标点保留）
    3. 多余空白字符 → 单个空格
    4. 首尾空白去除
    5. 连续重复标点归一化（如"？？？" → "？"）
    """
    if not text:
        return ""

    # 1. 全角英文字母 → 半角 (Ａ-Ｚ→A-Z, ａ-ｚ→a-z)
    result = []
    for ch in text:
        code = ord(ch)
        if 0xFF21 <= code <= 0xFF3A:  # 全角大写 A-Z
            result.append(chr(code - 0xFF21 + 0x41))
        elif 0xFF41 <= code <= 0xFF5A:  # 全角小写 a-z
            result.append(chr(code - 0xFF41 + 0x61))
        elif 0xFF10 <= code <= 0xFF19:  # 全角数字 0-9
            result.append(chr(code - 0xFF10 + 0x30))
        else:
            result.append(ch)
    text = "".join(result)

    # 2. 多余空白 → 单个空格（保留必要分隔）
    text = re.sub(r"\s+", " ", text)

    # 3. 连续重复标点归一化（如"？？？"→"？", "！！!"→"！"）
    text = re.sub(r"([？！?！])\1{2,}", r"\1", text)
    text = re.sub(r"([。，、；：""''…])\1{2,}", r"\1", text)

    # 4. 首尾空白去除
    text = text.strip()

    return text


# ═══════════════════════════════════════════════════════════════════════
# L1+L2 归一化服务
# ═══════════════════════════════════════════════════════════════════════

class SynonymNormalizer:
    """
    输入同义词归一化服务

    用途：在意图识别前，将用户多样化的口语表达归一化为标准术语，
          提高 FAISS 向量匹配的命中率和准确性。

    使用方式：
        normalizer = SynonymNormalizer()
        normalized = normalizer.normalize("帮我把这个退了吧")
        # → "我要退货"
    """

    # 最大输入长度（超过此长度的查询跳过归一化，直接返回原文本）
    MAX_QUERY_LENGTH: int = 80

    def __init__(self):
        self._synonym_map = SYNONYM_MAP
        self._variant_lookup = _VARIANT_TO_CANONICAL
        self._regex_rules = REGEX_REPLACE_RULES

    def normalize(self, text: str, domain: str = "ecommerce") -> str:
        """
        对用户输入执行同义词归一化

        流程：
        L1a: 精确匹配同义词表 (O(1) 查找)
        L1b: 正则模糊匹配 (按顺序，首命中即返回)
        L2:  文本标准化（全角半角、空白清理）

        Args:
            text: 用户原始输入
            domain: 业务领域

        Returns:
            归一化后的文本（如果无需归一化则返回原文本）
        """
        if not text:
            return text

        original = text

        # ── L2: 文本标准化（始终执行） ──
        text = normalize_text(text)

        # 过长查询不归一化（可能是多轮或复杂问题）
        if len(text) > self.MAX_QUERY_LENGTH:
            return text

        # ── L1a: 精确匹配同义词表 ──
        canonical = self._variant_lookup.get(text)
        if canonical:
            logger.info(f"同义词归一化 [精确匹配]  {original!r} → {canonical!r}")
            return canonical

        # ── L1b: 正则模式匹配 ──
        for pattern, replacement, scope in self._regex_rules:
            if scope not in ("all", domain):
                continue
            m = re.match(pattern, text)
            if m:
                logger.info(f"同义词归一化 [正则匹配]  {original!r} → {replacement!r}  pattern={pattern}")
                return replacement

        # 返回标准化后的文本（若 L2 改变了内容）
        if text != original:
            logger.info(f"同义词归一化 [L2标准化]  {original!r} → {text!r}")
        return text


# ═══════════════════════════════════════════════════════════════════════
# L3: LLM 归一化（可选，默认关闭）
# ═══════════════════════════════════════════════════════════════════════

LLM_NORMALIZE_SYSTEM_PROMPT = """你是一个查询归一化助手。你的任务是将用户的口语化表达改写成标准查询短语。

规则：
1. 保留用户的核心意图（退货、查单、查物流、查余额、用优惠券等）
2. 去掉语气词和礼貌用语（"请"、"麻烦"、"帮我"、"谢谢"等）
3. 改写为简短的标准查询短语，不超过20个字
4. 如果输入已经是标准查询，直接原文返回
5. 只输出改写后的查询短语，不要任何解释

标准短语示例：
- "我要退货"
- "查询订单"
- "查询物流"
- "查询余额"
- "优惠券"
- "售后咨询"
- "商品咨询"
"""

LLM_NORMALIZE_USER_TEMPLATE = "用户输入：{message}\n\n请将上述输入改写为标准查询短语（只输出短语本身，不要加引号或解释）："


async def normalize_via_llm(
    message: str,
    llm_service,
    domain: str = "ecommerce",
) -> str:
    """
    L3: 使用 LLM 将用户输入归一化为标准查询短语

    Args:
        message: 用户原始输入
        llm_service: LLMService 实例
        domain: 业务领域

    Returns:
        归一化后的查询文本
    """
    if not message or not llm_service:
        return message

    try:
        messages = [
            {"role": "system", "content": LLM_NORMALIZE_SYSTEM_PROMPT},
            {"role": "user", "content": LLM_NORMALIZE_USER_TEMPLATE.format(message=message)},
        ]

        result = await llm_service.chat_qwen(messages, temperature=0.0)
        # 去掉可能的引号（英文单引号、双引号、中文双引号）
        normalized = result.strip().strip('\u201c\u201d"\'')

        if normalized and normalized != message:
            logger.info(f"同义词归一化 [LLM]  {message!r} → {normalized!r}")
            return normalized

        return message
    except Exception as e:
        logger.warning(f"LLM 同义词归一化失败: {str(e)[:150]}，回退到原文本")
        return message


# ═══════════════════════════════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════════════════════════════

class InputNormalizer:
    """
    输入归一化统一入口

    整合 L1（同义词表）+ L2（文本标准化）+ L3（LLM，可选）

    使用方式：
        from src.modules.chat.core.synonym_normalizer import InputNormalizer

        normalizer = InputNormalizer(llm_service=llm_service)
        normalized = await normalizer.normalize("帮我把这个退了吧", domain="ecommerce")
    """

    def __init__(self, llm_service=None):
        """
        Args:
            llm_service: LLMService 实例（L3 需要，不传则仅使用 L1+L2）
        """
        self._synonym_normalizer = SynonymNormalizer()
        self._llm_service = llm_service
        # L3 开关，从全局配置读取
        self._llm_normalize_enabled = getattr(
            global_config, "SYNONYM_NORMALIZE_LLM_ENABLED", False
        )

    async def normalize(self, message: str, domain: str = "ecommerce") -> str:
        """
        对用户输入执行归一化

        流程：
        1. L1+L2: 同义词表 + 文本标准化（始终执行）
        2. L3:   LLM 归一化（仅当 _llm_normalize_enabled=True 时执行）

        Args:
            message: 用户原始输入
            domain: 业务领域

        Returns:
            归一化后的文本
        """
        if not message:
            return message

        # Step 1: L1 + L2
        normalized = self._synonym_normalizer.normalize(message, domain)

        # Step 2: L3（可选）
        if self._llm_normalize_enabled and self._llm_service and len(message) <= 80:
            normalized = await normalize_via_llm(
                normalized, self._llm_service, domain
            )

        return normalized
