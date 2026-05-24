"""
情绪检测服务 (Sentiment Detection Service)

职责：
- L1 规则关键词：极低成本关键词匹配（<1ms），覆盖明确情绪信号
- L2 本地模型分类：复用 Qwen3-1.7B 做零样本情绪分类（~30ms）
- L3 云端 LLM 兜底：仅在 L1+L2 无结论时触发（~300ms）
- Session 情绪跟踪器：维护滑动窗口 + 趋势方向，支持预测性升级

设计原则（级联分类器）：
  L1 规则 → 短小明确文本，直接出结果
  L2 本地模型 → L1 未命中且消息足够长时的补充
  L3 云端 LLM → 边界情况兜底（极少触发）
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional, Dict

from src.shared.logger import APILogger

if TYPE_CHECKING:
    from src.modules.chat.core.local_model_service import LocalModelService
    from src.modules.chat.core.llm_service import LLMService

logger = APILogger("sentiment_service")


# ═══════════════════════════════════════════════════════════════════════
# 情绪等级定义
# ═══════════════════════════════════════════════════════════════════════

class EmotionLevel(IntEnum):
    """情绪等级（值越大越危险，>=4 触发升级）"""
    GRATEFUL = 0       # 感激 — "太感谢了"
    SATISFIED = 1      # 满意 — "好的谢谢"
    NEUTRAL = 2        # 中立 — "帮我查下订单"
    ANXIOUS = 3        # 焦虑 — "怎么还没发货"
    DISAPPOINTED = 4   # 失望 — "等了三天了"
    ANGRY = 5          # 愤怒 — "你们是骗子吗"
    EMERGENCY = 6      # 舆情风险 — "我要打12315"


# ── 升级阈值 ──
ESCALATE_THRESHOLD = EmotionLevel.DISAPPOINTED   # >=4 建议升级
EMERGENCY_THRESHOLD = EmotionLevel.EMERGENCY      # =6 强制升级


@dataclass
class EmotionResult:
    """单条消息的情绪检测结果"""
    level: EmotionLevel
    confidence: float                    # 0.0~1.0
    escalate: bool                       # 是否建议升级到人工
    is_emergency: bool                   # 是否强制升级
    keywords: List[str] = field(default_factory=list)
    source: str = "L0:none"              # 来源标识


# ═══════════════════════════════════════════════════════════════════════
# L1: 规则关键词表（零成本，<1ms）
# ═══════════════════════════════════════════════════════════════════════

EMOTION_KEYWORDS: Dict[EmotionLevel, List[str]] = {
    EmotionLevel.EMERGENCY: [
        # 法律/监管威胁
        "12315", "消费者协会", "工商局", "消协", "举报你们",
        "起诉", "走法律程序", "我要曝光", "上热搜", "媒体曝光",
        "报警", "诈骗", "欺诈", "虚假宣传", "虚假广告",
        # 人身威胁
        "人身安全", "威胁生命", "生命危险",
    ],
    EmotionLevel.ANGRY: [
        # 强烈情绪
        "骗子", "骗钱", "垃圾", "死全家", "日了狗", "操",
        "倒闭", "黑心", "奸商", "太过分了", "不可原谅",
        "再不处理我就", "一遍又一遍", "反复忽悠",
        # 升级暗示
        "找你们领导", "投诉到底", "一直推脱", "拖了这么久",
    ],
    EmotionLevel.DISAPPOINTED: [
        "等了", "还没到", "又坏了", "怎么又", "失望",
        "上次就说", "说好的", "承诺", "不靠谱", "有问题",
        "不回复", "不处理", "客服态度", "无法接受",
    ],
    EmotionLevel.ANXIOUS: [
        "什么时候", "多久能", "还能到吗",
        "不会丢了吧", "怕", "担心", "着急", "急用",
        "催一下", "加急", "尽快", "麻烦快一点",
    ],
    EmotionLevel.SATISFIED: [
        "好的谢谢", "ok", "好的", "行", "可以",
        "明白了", "懂了", "了解了", "明白了谢谢",
    ],
    EmotionLevel.GRATEFUL: [
        "太感谢了", "谢谢你们", "很满意", "好评",
        "推荐给你们", "非常棒", "服务很好", "很到位",
        "帮忙解决了", "解决了", "感谢", "麻烦了",
    ],
}

# 否定前缀模式：匹配 "不是骗子" "没有骗" 等，反转情绪
# 允许否定词和关键词之间有 0-3 个任意字符（如 "不" + "是" + "骗子"）
_NEGATION_PATTERNS = re.compile(
    r"(不|没|非|别|无|本不是|没有|并非).{0,3}("
    r"骗子|骗钱|垃圾|曝光|举报|投诉|诈骗"
    r")"
)


def _has_negation(text: str) -> bool:
    """检测文本中是否包含"我不是要说xx"类的否定表达。
    例如 "这个不是骗子平台，挺好的" → 不应标为 ANGRY。
    """
    return bool(_NEGATION_PATTERNS.search(text))


# ═══════════════════════════════════════════════════════════════════════
# L2: 本地模型分类 Prompt
# ═══════════════════════════════════════════════════════════════════════

_EMOTION_CLASSIFY_PROMPT = """你是一个用户情绪分类器。请判断用户消息中传达的情绪。

可选情绪等级（按严重程度排序）：
- EMERGENCY: 涉及法律威胁、曝光、报警等
- ANGRY: 强烈不满、辱骂、质疑诚信
- DISAPPOINTED: 失望、投诉、抱怨产品或服务
- ANXIOUS: 焦急等待、担心延误
- NEUTRAL: 普通咨询、不包含明显情绪
- SATISFIED: 表示理解或认可
- GRATEFUL: 表示感谢、满意

请严格按以下JSON格式输出，不要包含任何额外文字：
{"emotion": "EMERGENCY|ANGRY|DISAPPOINTED|ANXIOUS|NEUTRAL|SATISFIED|GRATEFUL", "evidence": "触发该情绪的简短关键词（中文）"}

用户消息："""


# ═══════════════════════════════════════════════════════════════════════
# L3: 云端 LLM 兜底 Prompt（仅在边界模糊时使用）
# ═══════════════════════════════════════════════════════════════════════

_EMOTION_CLOUD_PROMPT = """判断以下用户消息的情绪等级。输出严格JSON：

{"level": 数字0-6, "confidence": 0-1之间浮点数, "escalate": true/false, "reason": "简短原因"}

等级对应关系：0=GRATEFUL 感激, 1=SATISFIED 满意, 2=NEUTRAL 中立, 3=ANXIOUS 焦虑, 4=DISAPPOINTED 失望, 5=ANGRY 愤怒, 6=EMERGENCY 舆情风险

用户消息："""


# ═══════════════════════════════════════════════════════════════════════
# Session 情绪跟踪器
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SessionEmotionTracker:
    """维护单个会话的情绪窗口。

    设计：
    - 滑动窗口（最近 N 条消息）→ 捕捉情绪趋势
    - peak：会话历史最高情绪等级
    - trend：上升/下降/平稳
    - escalate_triggered_at：首次触发升级的时间
    """

    window: deque = field(default_factory=lambda: deque(maxlen=5))
    peak: EmotionLevel = EmotionLevel.NEUTRAL
    trend: str = "stable"          # "escalating" | "de-escalating" | "stable"
    escalate_triggered_at: Optional[float] = None  # time.monotonic()

    def update(self, result: EmotionResult) -> None:
        """添加一条新消息的情绪结果并更新趋势。"""
        self.window.append(result)
        self.peak = max(self.peak, result.level)

        # 趋势计算：最近 3 条
        window_list = list(self.window)
        if len(window_list) >= 3:
            recent = window_list[-3:]
            levels = [r.level.value for r in recent]
            if levels[-1] > levels[0]:
                self.trend = "escalating"
            elif levels[-1] < levels[0]:
                self.trend = "de-escalating"
            else:
                self.trend = "stable"

    @property
    def should_escalate(self) -> bool:
        """是否应该触发人工升级：
        - 单次 >= DISAPPOINTED 且趋势 escalating
        - 或者 peak >= ANGRY
        - 或者 emergency
        """
        return (
            self.peak >= EmotionLevel.ANGRY
            or (
                list(self.window)
                and list(self.window)[-1].level >= EmotionLevel.DISAPPOINTED
                and self.trend == "escalating"
            )
        )

    @property
    def mode(self) -> str:
        """会话级情绪模式，用于驱动 prompt 模板选择。"""
        if self.peak == EmotionLevel.EMERGENCY:
            return "emergency"
        if self.peak == EmotionLevel.ANGRY:
            return "angry"
        if self.peak == EmotionLevel.DISAPPOINTED:
            return "disappointed"
        if self.peak == EmotionLevel.ANXIOUS:
            return "anxious"
        return "normal"


# ═══════════════════════════════════════════════════════════════════════
# 情绪驱动的 System Prompt 模板
# ═══════════════════════════════════════════════════════════════════════

EMOTION_TONE_PROMPTS: Dict[str, str] = {
    "emergency": (
        "\n## 情绪感知\n"
        "用户情绪极为激动，可能涉及舆情风险。\n"
        "回复要求：\n"
        "1. 开头必须真诚道歉，承认问题\n"
        "2. 立即提供明确的升级渠道（人工客服/电话）\n"
        "3. 不要试图在对话中完全解决问题\n"
        "4. 不要使用'但是''不过'等转折词\n"
    ),
    "angry": (
        "\n## 情绪感知\n"
        "用户当前非常不满。\n"
        "回复要求：\n"
        "1. 先表示理解和歉意，再给方案\n"
        "2. 给出明确的行动步骤和时间承诺\n"
        "3. 避免推卸责任或解释过多流程细节\n"
        "4. 不要反问用户（如'您为什么不先看看说明书'）\n"
    ),
    "disappointed": (
        "\n## 情绪感知\n"
        "用户对服务体验感到失望。\n"
        "回复要求：\n"
        "1. 共情用户的等待/不便\n"
        "2. 主动让步（优惠券/加急/优先处理）若场景合适\n"
        "3. 用具体时间代替模糊承诺（如'今天18:00前'而非'尽快'）\n"
    ),
    "anxious": (
        "\n## 情绪感知\n"
        "用户较为焦急，希望快速得到结果。\n"
        "回复要求：\n"
        "1. 回复简洁高效，不绕圈子\n"
        "2. 优先给出核心信息（状态/时间节点）\n"
        "3. 结尾可以安抚一句（如'请放心，正在加急处理'）\n"
    ),
    "normal": "",  # 正常模式，不注入额外 prompt
}


# ═══════════════════════════════════════════════════════════════════════
# 情绪检测服务主体
# ═══════════════════════════════════════════════════════════════════════

class SentimentService:
    """三级级联情绪检测器。

    使用方式：
        svc = SentimentService(local_model=local_svc, llm=llm_svc)
        result = await svc.detect("怎么还没发货，等了三天了")
        # → EmotionResult(level=DISAPPOINTED, escalate=True, source="L1:rule")
    """

    _instance: Optional["SentimentService"] = None

    def __init__(
        self,
        *,
        local_model: "LocalModelService | None" = None,
        llm: "LLMService | None" = None,
    ):
        self._local_model = local_model
        self._llm = llm
        self._sessions: Dict[str, SessionEmotionTracker] = {}

    @classmethod
    def get_instance(
        cls,
        local_model: "LocalModelService | None" = None,
        llm: "LLMService | None" = None,
    ) -> "SentimentService":
        """获取单例（首次调用需要传入服务引用）。

        Lazy: 首次 detect 前必须有 local_model/llm 引用。
        如果在应用启动时设置，则后面 detect 无需再传。
        """
        if cls._instance is None:
            cls._instance = cls(local_model=local_model, llm=llm)
        else:
            # 如果后续提供了引用且当前为空，补充注入
            if local_model and not cls._instance._local_model:
                cls._instance._local_model = local_model
            if llm and not cls._instance._llm:
                cls._instance._llm = llm
        return cls._instance

    # ── 主入口 ──────────────────────────────────────────────────────────

    async def detect(
        self,
        text: str,
        *,
        session_id: str | None = None,
        skip_cloud: bool = True,
    ) -> EmotionResult:
        """检测单条消息的情绪。

        Args:
            text: 用户消息文本
            session_id: 会话 ID（用于更新情绪跟踪器）
            skip_cloud: 是否跳过 L3 云端检测（默认跳过，避免延迟）

        Returns:
            EmotionResult
        """
        if not text or len(text.strip()) < 2:
            result = EmotionResult(
                level=EmotionLevel.NEUTRAL,
                confidence=1.0,
                escalate=False,
                is_emergency=False,
                source="L0:short",
            )
        else:
            result = await self._detect_cascade(text, skip_cloud=skip_cloud)

        # 更新会话跟踪器
        if session_id:
            tracker = self._get_tracker(session_id)
            tracker.update(result)

        return result

    def get_session(self, session_id: str) -> Optional[SessionEmotionTracker]:
        """获取指定会话的情绪跟踪器（用于查询历史趋势）。"""
        return self._sessions.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """清除指定会话的情绪跟踪器。"""
        self._sessions.pop(session_id, None)

    # ── 三级级联 ──────────────────────────────────────────────────────

    async def _detect_cascade(
        self, text: str, *, skip_cloud: bool = True
    ) -> EmotionResult:
        """L1 → L2 → L3 级联情绪分类器。"""

        # ── L1: 规则关键词（<1ms）──
        l1_result = self._l1_classify(text)
        if l1_result:
            return l1_result

        # 短文本且 L1 未命中 → 大概率是 NEUTRAL
        if len(text) <= 5:
            return EmotionResult(
                level=EmotionLevel.NEUTRAL,
                confidence=0.7,
                escalate=False,
                is_emergency=False,
                source="L1:fallback_short",
            )

        # ── L2: 本地 Qwen3 分类（~30ms）──
        l2_result = await self._l2_local_classify(text)
        if l2_result:
            return l2_result

        # ── L3: 云端 LLM 兜底（~300ms，极少触发）──
        if not skip_cloud:
            l3_result = await self._l3_cloud_classify(text)
            if l3_result:
                return l3_result

        # 全未命中 → NEUTRAL
        return EmotionResult(
            level=EmotionLevel.NEUTRAL,
            confidence=0.5,
            escalate=False,
            is_emergency=False,
            source="L3:unclassified",
        )

    # ── L1 实现 ──────────────────────────────────────────────────────

    def _l1_classify(self, text: str) -> Optional[EmotionResult]:
        """规则关键词匹配（O(N*M) 但在电商场景下 M<100 且文本 <100 字，足够快）。"""
        text_lower = text.lower()
        has_neg = _has_negation(text)

        # 从高到底依次检查，优先匹配最高情绪等级
        for level in [EmotionLevel.EMERGENCY, EmotionLevel.ANGRY,
                      EmotionLevel.DISAPPOINTED, EmotionLevel.ANXIOUS,
                      EmotionLevel.SATISFIED, EmotionLevel.GRATEFUL]:
            keywords = EMOTION_KEYWORDS.get(level, [])
            hit_kw = [kw for kw in keywords if kw in text_lower]

            if hit_kw:
                # 否定反转：如果文本用否定前缀，降级
                if has_neg and level >= EmotionLevel.DISAPPOINTED:
                    # 例如 "不是骗子" → 不应标 ANGRY
                    continue

                return EmotionResult(
                    level=level,
                    confidence=min(0.95, 0.6 + 0.1 * len(hit_kw)),
                    escalate=level >= ESCALATE_THRESHOLD,
                    is_emergency=level == EmotionLevel.EMERGENCY,
                    keywords=hit_kw,
                    source=f"L1:rule({len(hit_kw)})",
                )

        return None

    # ── L2 实现 ──────────────────────────────────────────────────────

    async def _l2_local_classify(self, text: str) -> Optional[EmotionResult]:
        """用本地 Qwen3 模型做零样本情绪分类。"""
        if self._local_model is None:
            return None

        try:
            ok = self._local_model._ensure_loaded()
            if not ok or self._local_model._model is None:
                return None

            # 构建输入
            from src.modules.chat.core.local_model_service import _perf_time

            user_msg = f"{_EMOTION_CLASSIFY_PROMPT}{text}"

            if hasattr(self._local_model._tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "你是一个电商客服情绪分类器。只输出JSON，不要解释。"},
                    {"role": "user", "content": user_msg},
                ]
                try:
                    prompt = self._local_model._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except Exception:
                    prompt = (
                        "<|im_start|>system\n你是一个电商客服情绪分类器。只输出JSON。<|im_end|>\n"
                        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
            else:
                prompt = user_msg

            # 推理
            import torch as _torch

            t_start = _perf_time.monotonic()
            input_ids = self._local_model._tokenizer.encode(prompt)
            input_tensor = _torch.tensor([input_ids]).to(self._local_model._model.device)

            with _torch.no_grad():
                generated_ids = self._local_model._model.generate(
                    input_tensor,
                    max_new_tokens=32,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self._local_model._tokenizer.encode("<|endoftext|>")[0] if hasattr(self._local_model._tokenizer, "pad_token_id") else None,
                )

            raw_output = self._local_model._tokenizer.decode(
                generated_ids[0][len(input_ids):], skip_special_tokens=True,
            ).strip()
            elapsed = (_perf_time.monotonic() - t_start) * 1000

            # 解析 JSON
            level, evidence = self._parse_local_emotion_output(raw_output)
            if level is not None:
                return EmotionResult(
                    level=level,
                    confidence=0.8,
                    escalate=level >= ESCALATE_THRESHOLD,
                    is_emergency=level == EmotionLevel.EMERGENCY,
                    keywords=[evidence] if evidence else [],
                    source=f"L2:local({elapsed:.0f}ms)",
                )

            return None

        except Exception:
            logger.debug("L2 本地情绪分类失败，降级", exc_info=True)
            return None

    def _parse_local_emotion_output(self, raw: str) -> tuple:
        """解析 L2 本地模型的 JSON 输出。"""
        import json as _json

        # 尝试提取 JSON
        raw = raw.strip()
        # 去掉 markdown 代码块
        for prefix in ["```json", "```"]:
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

        # 尝试解析
        try:
            data = _json.loads(raw)
        except _json.JSONDecodeError:
            # 尝试匹配裸字符串
            for name, level in _EMOTION_NAME_MAP.items():
                if name in raw:
                    return level, raw
            return None, None

        emotion_str = data.get("emotion", "").strip().upper()
        level = _EMOTION_NAME_MAP.get(emotion_str)
        evidence = data.get("evidence", "")
        return level, evidence

    # ── L3 实现 ──────────────────────────────────────────────────────

    async def _l3_cloud_classify(self, text: str) -> Optional[EmotionResult]:
        """用云端 LLM 兜底（极少触发）。"""
        if self._llm is None:
            return None

        try:
            import json as _json

            prompt = f"{_EMOTION_CLOUD_PROMPT}{text}"
            response = await self._llm.chat_qwen_with_prompt(
                prompt=prompt,
                system_prompt="你是一个电商客服情绪分类器。只输出JSON。",
            )
            data = _json.loads(response) if isinstance(response, str) else response

            if not isinstance(data, dict):
                return None

            raw_level = data.get("level", 2)
            level_value = int(raw_level) if isinstance(raw_level, (int, float)) else 2
            level_value = max(0, min(6, level_value))
            level = EmotionLevel(level_value)
            confidence = float(data.get("confidence", 0.6))

            return EmotionResult(
                level=level,
                confidence=min(1.0, max(0.0, confidence)),
                escalate=level >= ESCALATE_THRESHOLD or data.get("escalate", False),
                is_emergency=level == EmotionLevel.EMERGENCY,
                keywords=[],
                source=f"L3:cloud({data.get('reason', '')[:30]})",
            )

        except Exception:
            logger.debug("L3 云端情绪分类失败", exc_info=True)
            return None

    # ── 会话跟踪器 ──────────────────────────────────────────────────────

    def _get_tracker(self, session_id: str) -> SessionEmotionTracker:
        """获取或创建会话级情绪跟踪器。"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionEmotionTracker()
        return self._sessions[session_id]


# ── 情绪名称映射（L2/L3 解析用）──

_EMOTION_NAME_MAP: Dict[str, EmotionLevel] = {
    "EMERGENCY": EmotionLevel.EMERGENCY,
    "ANGRY": EmotionLevel.ANGRY,
    "DISAPPOINTED": EmotionLevel.DISAPPOINTED,
    "ANXIOUS": EmotionLevel.ANXIOUS,
    "NEUTRAL": EmotionLevel.NEUTRAL,
    "SATISFIED": EmotionLevel.SATISFIED,
    "GRATEFUL": EmotionLevel.GRATEFUL,
}
