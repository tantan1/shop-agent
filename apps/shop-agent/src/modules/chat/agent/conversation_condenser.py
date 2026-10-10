"""多轮对话融合（指代消解 / 实体消歧）。

设计依据：docs/architecture/multi-turn-condensation-design.md
核心原则（红线 §4）：只改变喂给检索 / 选型的 query 字符串；跨轮纠正、标注捕获、
参数抽取、回复生成一律锁原文。强实体走确定性 Redis 槽位，软指代走 LLM，歧义反问，
失败 fallback 原文。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

from src.core.config import config as core_config
from src.modules.chat.core.llm_service import LLMService
from src.modules.chat.core.redis_cache_service import RedisCacheService
from src.ports.pii import redact as redact_pii
from src.shared.logger import APILogger

logger = APILogger("conversation_condenser")

# 门控阈值：短句（字符数，语言无关）；超过则仅 cross_turn_signal 触发
_GATE_MAX_CHARS = 200
_N_TURNS = 3
_ENTITY_SLOT_KEY_PREFIX = "chat:entity_slots:"
_STRONG_ENTITY_KEYS = ("order_id", "phone", "tracking_no", "refund_id")

# 跨轮信号（仅作门控补充，不用于语义判定）
_CROSS_TURN_RE = re.compile(
    r"(它|这个|那个|这单|那单|刚才|之前|上次|前面|上述|其|该|同上|这东西|那东西)"
)

# 融合 LLM 调用的超时上限（秒）。
# 该调用在检索前同步阻塞，网关抖动/503 时会把整轮拖到数秒甚至数十秒；
# 设计上「失败 fallback 原文」，因此超时直接放弃融合、用原句检索，保证响应时延可控。
_CONDENSE_TIMEOUT_S = float(getattr(core_config, "CONDENSE_TIMEOUT_S", 3.0))

# 强实体正则（确定性回填用，不参与语义判定）
_ORDER_ID_RE = re.compile(
    r"(?:订单[号单]?|order[ _]?id|order)\s*[:：]?\s*([A-Za-z0-9\-]{4,})", re.IGNORECASE
)
_PHONE_RE = re.compile(r"(?<!\d)(\d{7,11})(?!\d)")


def _empty_result() -> Dict[str, Any]:
    return {
        "standalone_query": None,  # 由调用方回填 raw
        "correction": {"is_correction": False, "kind": None, "corrected_intent": None},
        "ambiguous": False,
    }


def _gate(text: str, has_history: bool) -> bool:
    """触发门控：有历史 且 (当前句短 或 含跨轮信号)。"""
    if not has_history:
        return False
    if len(text) <= _GATE_MAX_CHARS:
        return True
    return bool(_CROSS_TURN_RE.search(text))


def _load_recent_history(conv_id: str, redis: Optional[RedisCacheService]) -> str:
    """加载最近对话历史（与 step4 同源：get_chat_messages → 格式化 → redact PII）。

    说明：此处刻意不做 LLM 摘要，避免融合调用叠加昂贵摘要，保持门控后仅 1 次轻量调用。
    历史超长时调用方可在 fold_into_step1 路径另行摘要。
    """
    if not conv_id or redis is None or not redis.is_available:
        return ""
    try:
        msgs = redis.get_chat_messages(conv_id, max_turns=_N_TURNS)
    except Exception:
        return ""
    if not msgs:
        return ""
    lines = [
        f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
        for m in msgs
        if isinstance(m, dict)
    ]
    return redact_pii("\n".join(lines))


def _load_entity_slots(
    conv_id: str, redis: Optional[RedisCacheService] = None
) -> Dict[str, Any]:
    """读取强实体槽位；异常 / 缺失 → 静默返回 {}。"""
    if redis is None:
        redis = RedisCacheService.get_instance()
    if not conv_id or redis is None or not redis.is_available:
        return {}
    try:
        raw = redis.get_json(_ENTITY_SLOT_KEY_PREFIX + conv_id)
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _extract_strong_entities(params: Dict[str, Any]) -> Dict[str, Any]:
    """从意图参数中抽取强实体（确定性回填用）。"""
    if not isinstance(params, dict):
        return {}
    return {k: params[k] for k in _STRONG_ENTITY_KEYS if params.get(k)}


def _save_entity_slots(
    conv_id: str,
    params: Dict[str, Any],
    redis: Optional[RedisCacheService] = None,
    ttl: Optional[int] = None,
) -> None:
    """意图抽取后写强实体；best-effort；无实体不写空 key。"""
    if redis is None:
        redis = RedisCacheService.get_instance()
    if not conv_id or redis is None or not redis.is_available:
        return
    slots = _extract_strong_entities(params)
    if not slots:
        return
    ttl = ttl or getattr(core_config, "ENTITY_SLOTS_TTL_SECONDS", 1800)
    try:
        redis.set_json(_ENTITY_SLOT_KEY_PREFIX + conv_id, slots, ex=ttl)
    except Exception as e:
        logger.warning("实体槽位写入失败（已忽略）", conversation_id=conv_id, error=str(e)[:120])


async def _condense_llm(raw: str, history: str, llm: LLMService) -> Dict[str, Any]:
    """调用便宜档 LLM，产出 standalone_query + correction。失败 → fallback 原文。"""
    sys_prompt = (
        "你是多轮对话融合助手。给定可选的对话历史与用户当前句，仅输出一个 JSON 对象：\n"
        '{"standalone_query": "将代词/省略展开为自包含查询，保留关键实体；若无需改写则原样返回当前句",\n'
        '"is_correction": false 或 true（用户是否在纠正/推翻上一轮，如"不对/应该是/搞错了"），\n'
        '"kind": null 或 "rephrase"/"negation",\n'
        '"corrected_intent": "简短纠正意图，非纠正则为空字符串",\n'
        '"ambiguous": false 或 true（存在多个可能指代、需反问时为 true）}\n'
        "只输出 JSON，不要任何额外说明。歧义时 ambiguous=true 且 standalone_query 可为空。"
    )
    user_prompt = (
        f"对话历史:\n{history}\n\n当前句:\n{raw}" if history else f"当前句:\n{raw}"
    )
    try:
        resp = await llm.chat_step1(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        cleaned = resp.strip().strip("`")
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        # 本地 vLLM 的 Qwen3 会忽略 enable_thinking=False 仍吐出 <think>…</think>，
        # 直接 json.loads 会因 think 前缀必然失败 → 每轮融合白做还白等数秒。先剥离再解析。
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        if "<think>" in cleaned:  # 未闭合的 think 块：只保留其后内容
            cleaned = cleaned.split("<think>")[0].strip()
        # 兜底：从首个 { 开始截取，规避模型输出的前导说明文字
        brace = cleaned.find("{")
        if brace > 0:
            cleaned = cleaned[brace:]
        data = json.loads(cleaned)
        return {
            "standalone_query": (data.get("standalone_query") or "").strip() or raw,
            "correction": {
                "is_correction": bool(data.get("is_correction", False)),
                "kind": data.get("kind"),
                "corrected_intent": data.get("corrected_intent") or None,
            },
            "ambiguous": bool(data.get("ambiguous", False)),
        }
    except Exception as e:
        logger.warning("condensation LLM 调用/解析失败，fallback 原文", error=str(e)[:120])
        return _empty_result()


async def condense_question(
    raw: str,
    conv_id: str,
    fold_into_step1: bool = False,
    llm_service: Optional[LLMService] = None,
    redis_cache_service: Optional[RedisCacheService] = None,
) -> Dict[str, Any]:
    """多轮融合主入口（门控）。

    Args:
        raw: 用户当前原始输入（ctx.request.message）。
        conv_id: 会话 ID。
        fold_into_step1: 预留——为 True 时表示调用方已在 step1 改写中自行折入历史，
            本函数仍会做门控融合（调用方可忽略其 retrieval_query 产物）。
        llm_service / redis_cache_service: 可选注入，便于测试与复用单例。

    Returns:
        {
          "standalone_query": str,   # 失败/无历史/未触发 → raw
          "correction": {is_correction, kind, corrected_intent},
          "ambiguous": bool,         # True 时由上层反问
        }
    """
    llm = llm_service or LLMService.get_instance()
    redis = redis_cache_service or RedisCacheService.get_instance()
    result = _empty_result()
    result["standalone_query"] = raw

    has_history = False
    if conv_id and redis.is_available:
        try:
            has_history = bool(redis.get_chat_messages(conv_id, max_turns=1))
        except Exception:
            has_history = False

    if not _gate(raw, has_history):
        return result

    history = _load_recent_history(conv_id, redis) if has_history else ""
    try:
        out = await asyncio.wait_for(
            _condense_llm(raw, history, llm), timeout=_CONDENSE_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        # 超时：放弃本轮融合，直接用原句检索（结果可接受，且时延可控）
        logger.warning(
            "condensation 超时，fallback 原文",
            timeout=_CONDENSE_TIMEOUT_S,
            conversation_id=conv_id,
        )
        return result
    if out.get("ambiguous"):
        result["ambiguous"] = True
        return result
    result["standalone_query"] = out.get("standalone_query") or raw
    result["correction"] = out.get("correction", result["correction"])
    return result
