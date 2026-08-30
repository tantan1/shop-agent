"""
分层参数提取管道（Phase 1）—— 正则 / 本地小模型 / 校验补全

设计文档：docs/architecture/param-extraction-layered-design.md
实施计划：docs/architecture/param-extraction-implementation-plan.md

核心原则（§0）：
  - L1-L3 只做「语义发现」产出 Candidate，L4 是唯一定稿者（确定性决策）。
  - 本地小模型（small_model）≠ 云端大模型，流程内不引入云端大模型常驻兜底（§8）。
  - 权限在系统边界校验（Rust 订单服务 §7.1），Agent 侧不预查归属。

设计对应：
  L0  normalize        —— 输入归一化
  L1  RegexLayer       —— 正则档（高后果，HIGH 置信）
  L2  NerLayer         —— （Phase 2 预留，未实装）
  L3  SmallModelLayer  —— 本地小模型兜底档（LOW 置信，仅前档全 miss 时触发）
  L4  enforce          —— 冲突消解 + 格式闸 + 必填完整性，唯一定稿者
  L5  （调用方接入）    —— 缺必填走 HITL 反问，不查系统代填

复用：正则引擎直接复用 SchemaDrivenExtractor（其 _PATTERNS / _FIELD_ALIASES），
     不重复实现正则；small_model 档复用 local_model_service.extract_params（含单飞+熔断）。
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# L0 归一化
# ============================================================================

_MAX_TEXT_LEN = 2000


def normalize(text: str) -> str:
    """L0 输入归一化：全角→半角、去零宽字符、小写化、截断。

    让下游各档吃统一格式，避免每个 extractor 各处理一遍。
    """
    if not text:
        return ""
    # 全角 → 半角
    text = unicodedata.normalize("NFKC", text)
    # 去除零宽字符（zero-width space / joiner / BOM 等）
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u2060\ufff9-\ufffb]", "", text)
    # 折叠多余空白
    text = re.sub(r"\s+", " ", text).strip()
    # 小写化（中文无影响，英文关键词匹配更稳健）
    text = text.lower()
    # 截断，避免超长文本拖慢正则/模型
    if len(text) > _MAX_TEXT_LEN:
        text = text[:_MAX_TEXT_LEN]
    return text


# ============================================================================
# Candidate 协议（L1-L3 统一产出）
# ============================================================================

@dataclass
class Candidate:
    """单档抽取产出的候选（不是定稿）。

    source: "regex" / "ner" / "small_model"
    confidence: "HIGH" / "MEDIUM" / "LOW"
    """

    field: str
    value: str
    confidence: str  # "HIGH" | "MEDIUM" | "LOW"
    source: str      # "regex" | "ner" | "small_model"
    raw: str = ""    # 原始命中片段（审计用）

    def __lt__(self, other: "Candidate") -> bool:
        # 用于冲突消解时的排序（高置信 > 低置信）
        order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        return order.get(self.confidence, 0) < order.get(other.confidence, 0)


# 置信度等级 → 数值，供 enforce 比较
_CONF_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


# ============================================================================
# BaseLayer —— 各档位统一接口
# ============================================================================

class BaseLayer(ABC):
    """档位抽象：每个档位只回答「我能不能发现这个字段」，不负责定稿。"""

    name: str = "base"

    @abstractmethod
    async def extract(self, text: str, sem_type: str) -> List[Candidate]:
        """从文本中抽取某语义类型的候选列表（可能为空）。"""
        raise NotImplementedError


# ============================================================================
# L1 RegexLayer —— 复用 SchemaDrivenExtractor 正则引擎
# ============================================================================

class RegexLayer(BaseLayer):
    """正则档：高后果字段，零依赖零延迟，命中即 HIGH 置信候选。

    复用 SchemaDrivenExtractor 的 _PATTERNS / _FIELD_ALIASES，不重复实现正则。
    """

    name = "regex"

    def __init__(self) -> None:
        from .schema_driven_extractor import SchemaDrivenExtractor

        self._ext = SchemaDrivenExtractor
        # 字段名 → 语义类型
        self._aliases = SchemaDrivenExtractor._FIELD_ALIASES
        # 语义类型 → 正则（兼作 L4 格式闸）
        self._patterns = SchemaDrivenExtractor._PATTERNS

    async def extract(self, text: str, sem_type: str) -> List[Candidate]:
        pattern = self._patterns.get(sem_type)
        if pattern is None:
            return []
        m = pattern.search(text)
        if not m:
            return []
        groups = [g for g in m.groups() if g is not None]
        value = "".join(groups) if groups else m.group(0)
        return [
            Candidate(
                field=sem_type,
                value=value,
                confidence="HIGH",
                source="regex",
                raw=m.group(0),
            )
        ]


# ============================================================================
# L3 SmallModelLayer —— 本地小模型兜底档（复用 local_model_service）
# ============================================================================

class SmallModelLayer(BaseLayer):
    """本地小模型兜底档：前档全 miss 时才触发，产出 LOW 置信候选。

    复用 local_model_service.extract_params（含单飞 + 熔断）。小模型只做语义发现，
    不定稿；高后果字段即使被抽出也须过 L4 硬强制。
    """

    name = "small_model"

    def __init__(self) -> None:
        self._service = None
        self._available = False
        try:
            from .local_model_service import local_model_service

            self._service = local_model_service
            self._available = True
        except Exception as e:  # pragma: no cover - 模型未配置时优雅降级
            logger.warning(f"SmallModelLayer 初始化失败，将跳过该档: {e}")
            self._available = False

    async def extract(self, text: str, sem_type: str) -> List[Candidate]:
        if not self._available or self._service is None:
            return []
        try:
            # 轻量通用提示：让小模型把该语义类型的值从文本中抽出
            prompt = (
                f"从用户消息中抽取「{sem_type}」对应的值。"
                f"只返回该值本身，不要解释。若无则返回空字符串。\n"
                f"消息：{text}"
            )
            # 复用现有 structured output 接口，用宽松 schema
            from pydantic import BaseModel

            class _Slot(BaseModel):
                value: str = ""

            data = await self._service.extract_params(
                extraction_prompt=prompt,
                message=text,
                output_schema=_Slot,
            )
            value = (data or {}).get("value", "")
            if value:
                return [
                    Candidate(
                        field=sem_type,
                        value=str(value),
                        confidence="LOW",
                        source="small_model",
                        raw=str(value),
                    )
                ]
        except Exception as e:  # 熔断/超时/解析失败 → 该档跳过，不阻塞主链路
            logger.warning(f"SmallModelLayer 抽取失败已跳过: {e}")
        return []


# ============================================================================
# SchemaProvider —— 隔离「schema 从哪来」（§3.1）
# ============================================================================

class SchemaProvider(ABC):
    """schema 来源对调用方透明的抽象接口。"""

    @abstractmethod
    async def get_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """返回某工具的 inputSchema（含 "properties"）。来源对调用方透明。"""
        raise NotImplementedError


class McpSchemaProvider(SchemaProvider):
    """现状实现：从 mcp_client 的 tools/list 缓存获取 schema。

    返回 model_visible_schema（已剥离 order_id 等高后果字段），保证模型/参数抽取层
    不会把高后果字段当作「由模型生成」的参数（Phase 2 断言 A）。真实值由
    ToolService._try_mcp_dispatch 以 hardcode 形式注入 tools/call。
    """

    async def get_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        try:
            from .mcp_client import mcp_manager

            info = mcp_manager.get_tool_info(tool_name)
            if info is not None:
                # 优先返回已剥离高后果字段的「模型可见 schema」
                mv = getattr(info, "model_visible_schema", None)
                if mv:
                    return mv
                return getattr(info, "input_schema", None) or {}
        except Exception as e:
            logger.warning(f"McpSchemaProvider 获取 schema 失败: {e}")
        return None


# ============================================================================
# 档位注册表（扩展点，§3）
# ============================================================================
# 语义类型 → 档位序列。新增档位（如 NerLayer）只需在此注册 + 加入 _LAYER_REGISTRY，
# 不动 run_pipeline 调度循环。高后果字段档位序列不含 small_model（避免每次必调）。
_SEMANTIC_LAYER_PLAN: Dict[str, List[str]] = {
    "order": ["regex", "small_model"],
    "phone": ["regex"],                # 高后果，只正则
    "tracking": ["regex", "small_model"],
    "order_status": ["regex", "small_model"],
    "return_reason": ["regex", "small_model"],
    "coupon_type": ["regex", "small_model"],
}

_LAYER_REGISTRY: Dict[str, BaseLayer] = {}


def _get_layer(layer_name: str) -> BaseLayer:
    """单例获取档位实例（懒加载，避免循环导入）。"""
    if layer_name not in _LAYER_REGISTRY:
        if layer_name == "regex":
            _LAYER_REGISTRY[layer_name] = RegexLayer()
        elif layer_name == "small_model":
            _LAYER_REGISTRY[layer_name] = SmallModelLayer()
        # "ner" 档在 Phase 2 注册
        else:
            raise ValueError(f"未知档位: {layer_name}")
    return _LAYER_REGISTRY[layer_name]


def register_layer(layer_name: str, layer: BaseLayer) -> None:
    """运行时注册新档位（如 Phase 2 的 NerLayer）。"""
    _LAYER_REGISTRY[layer_name] = layer


# ============================================================================
# L4 enforce —— 唯一定稿者（冲突消解 + 格式闸 + 必填完整性）
# ============================================================================

@dataclass
class EnforceResult:
    params: Dict[str, Any] = field(default_factory=dict)          # 定稿参数
    missing_required: List[str] = field(default_factory=list)     # 缺必填字段
    not_finalized: List[str] = field(default_factory=list)       # 未定稿字段（非 HIGH / 格式非法 → 交 L5）
    candidates_by_field: Dict[str, List[Candidate]] = field(default_factory=dict)


def _format_ok(sem_type: str, value: str) -> bool:
    """格式闸：用正则引擎的模式校验业务格式合法性（非仅结构）。"""
    try:
        from .schema_driven_extractor import SchemaDrivenExtractor

        pattern = SchemaDrivenExtractor._PATTERNS.get(sem_type)
        if pattern is None:
            return True  # 无模式定义则不拦（交由查系统定稿）
        return pattern.search(value) is not None
    except Exception:
        return True


def enforce(
    candidates: Dict[str, List[Candidate]],
    required_fields: Optional[List[str]] = None,
) -> EnforceResult:
    """L4 定稿：冲突消解 + 格式闸 + 必填完整性。

    规则（§4.1）：
      - 同字段多档命中 → 取最高置信（HIGH>MEDIUM>LOW），同置信取最长匹配
      - 格式闸：定稿值须过对应语义类型正则，不通过→不写 Slot + 记入 format_rejected
      - 必填完整性：缺必填记入 missing_required（交由 L5 HITL 反问）
    """
    required_fields = required_fields or []
    result = EnforceResult()

    for field_name, cands in candidates.items():
        if not cands:
            continue
        # 冲突消解：最高置信优先，同置信取最长匹配
        best = max(
            cands,
            key=lambda c: (_CONF_RANK.get(c.confidence, 0), len(c.value)),
        )
        # 高后果字段：非 HIGH 命中一律不写 Slot（走 L5 补全 / HITL）
        #   —— 当前以 sem_type 命中正则档(HIGH)为可信；small_model(LOW)仅作候选发现
        if best.confidence != "HIGH":
            # 非高置信：不直接落定稿，交 L5 处理（查系统 / HITL 反问）
            result.not_finalized.append(field_name)
            result.candidates_by_field[field_name] = cands
            continue

        # 格式闸：定稿值须过对应语义类型正则，不通过→不写 Slot + 记入 not_finalized
        if not _format_ok(best.field, best.value):
            result.not_finalized.append(field_name)
            result.candidates_by_field[field_name] = cands
            continue

        result.params[field_name] = best.value
        result.candidates_by_field[field_name] = cands

    # 必填完整性
    for req in required_fields:
        if req not in result.params:
            result.missing_required.append(req)

    return result


# ============================================================================
# run_pipeline —— 管道调度（上层不感知 schema 来源 / 档位实现）
# ============================================================================

async def run_pipeline(
    text: str,
    tool_name: str,
    schema_provider: SchemaProvider,
    extra_required: Optional[List[str]] = None,
) -> EnforceResult:
    """L0→L1/L3→L4 主管道。

    Args:
        text:            用户原始消息
        tool_name:       工具名（用于向 SchemaProvider 取 schema）
        schema_provider: schema 来源（McpSchemaProvider 等）
        extra_required:  额外必填字段（如业务强制要求 order_id）

    Returns:
        EnforceResult（params / missing_required / format_rejected）
    """
    # L0 归一化
    clean = normalize(text)

    # 结构层：从 provider 动态取 schema（失败降级见 SchemaProvider）
    mcp_schema = await schema_provider.get_schema(tool_name)
    if mcp_schema and "properties" in mcp_schema:
        field_names = list(mcp_schema["properties"].keys())
    else:
        # 降级：退化为全部已注册 alias 字段（兼容无 schema 场景）
        from .schema_driven_extractor import SchemaDrivenExtractor

        field_names = list(SchemaDrivenExtractor._FIELD_ALIASES.keys())

    # 字段名 → 语义类型（用于档位映射）
    from .schema_driven_extractor import SchemaDrivenExtractor

    aliases = SchemaDrivenExtractor._FIELD_ALIASES

    candidates: Dict[str, List[Candidate]] = {}
    for field_name in field_names:
        sem_type = aliases.get(field_name)
        if sem_type is None:
            continue
        layers = _SEMANTIC_LAYER_PLAN.get(sem_type, ["regex"])
        for layer_name in layers:
            try:
                layer = _get_layer(layer_name)
            except ValueError:
                continue
            cands = await layer.extract(clean, sem_type)
            if cands:
                candidates.setdefault(field_name, []).extend(cands)
                # 正则档高置信命中，跳过后续档（早退，避免 small_model 每次必调）
                if layer_name == "regex":
                    break

    # L4 定稿
    required = list(extra_required or [])
    # 从 schema 的 required 取必填
    if mcp_schema and isinstance(mcp_schema.get("required"), list):
        required.extend(mcp_schema["required"])

    return enforce(candidates, required)
