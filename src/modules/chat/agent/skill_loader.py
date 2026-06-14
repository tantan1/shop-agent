"""
Skill Loader —— 从 skills/ 目录加载 SKILL.md 文件，自动构建工具描述和意图映射。

Skills 目录结构：
    skills/
    ├── query-order/
    │   └── SKILL.md          # YAML frontmatter + Markdown 正文
    ├── check-shipping/
    │   └── SKILL.md
    └── ...

SKILL.md 格式（name/allowed-tools 使用连字符 + 空格分隔字符串）：
    ---
    name: query-order
    description: 查询用户的订单列表...
    allowed-tools: query-order check-shipping
    tags:
      - 订单
      - 查询
    priority: 10
    ---

    # 订单查询 Skill
    ...

Loaded data feeds into P0 (INTENT_TOOL_MAP) and P1 (embedding descriptions).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Type, Union as UnionType

import yaml
from pydantic import BaseModel

from src.shared.logger import APILogger

logger = APILogger("skill_loader")

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_tool_string(raw: object) -> List[str]:
    """解析 `allowed-tools` 值（空格分隔字符串）。

    此函数处理两种兼容格式：
    - 字符串："query-order check-shipping" → ["query-order", "check-shipping"]
    - 列表（旧格式兼容）：["query-order"] → ["query-order"]
    """
    if isinstance(raw, str):
        if "," in raw:
            logger.warning(f"allowed-tools 使用了逗号分隔，建议改用空格分隔: {raw}")
        return [t.strip(",").strip() for t in raw.split() if t.strip(",").strip()]
    if isinstance(raw, list):
        return [str(t).strip() for t in raw]
    return []


@dataclass
class SkillDef:
    """单个 Skill 的完整定义（从 SKILL.md 解析出）。"""
    name: str
    display_name: str
    description: str
    tags: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    priority: int = 10
    body: str = ""  # frontmatter 后面的 Markdown 正文
    params: Dict[str, Dict] = field(default_factory=dict)  # 工具参数定义，单一来源


@dataclass
class SkillRegistry:
    """Skills 注册表 —— 所有已加载 skill 的集合。"""

    skills: List[SkillDef]

    @property
    def tool_descriptions(self) -> Dict[str, str]:
        """生成工具描述字典（key 为连字符格式）。"""
        return {s.name: s.description for s in self.skills}

    @property
    def intent_tool_map(self) -> Dict[str, Set[str]]:
        """根据 allowed_tools 自动生成 INTENT_TOOL_MAP（key/value 均为连字符格式）。

        每个 skill 的 name 作为 intent key，allowed_tools 作为候选工具集。
        unknown 回退到所有工具名。
        """
        mapping: Dict[str, Set[str]] = {}
        all_names: Set[str] = set()
        for s in self.skills:
            mapping[s.name] = set(s.allowed_tools)
            all_names.add(s.name)
        mapping["unknown"] = all_names
        return mapping


# 在模块加载时（同步上下文）预计算 skills 默认根路径，
# 避免 LangGraph dev ASGI 事件循环中触发 blockbuster 阻塞检测。
_DEFAULT_SKILLS_ROOT: Path | None = None


def _get_default_skills_root() -> Path:
    """自动推断 skills/ 目录（项目根目录下的 skills/），使用惰性缓存。"""
    global _DEFAULT_SKILLS_ROOT
    if _DEFAULT_SKILLS_ROOT is not None:
        return _DEFAULT_SKILLS_ROOT
    # react_agent.py → agent/ → chat/ → modules/ → src/ → shop-agent/
    candidate = Path(__file__).resolve().parents[4] / "skills"
    if candidate.is_dir():
        _DEFAULT_SKILLS_ROOT = candidate
    else:
        _DEFAULT_SKILLS_ROOT = Path.cwd() / "skills"
    return _DEFAULT_SKILLS_ROOT


class SkillLoader:
    """从 skills/ 目录批量加载 SKILL.md 文件。"""

    def __init__(self, skills_root: str | None = None):
        self._root = Path(skills_root) if skills_root else _get_default_skills_root()
        self._registry: SkillRegistry | None = None

    def load(self) -> SkillRegistry:
        """加载所有 SKILL.md，返回 SkillRegistry。"""
        if self._registry is not None:
            return self._registry

        skills: List[SkillDef] = []
        if not self._root.exists():
            logger.warning(f"Skills 目录不存在: {self._root}")
            return SkillRegistry(skills=skills)

        has_skill_files = False
        parse_errors = 0
        for item in sorted(self._root.iterdir()):
            if not item.is_dir():
                continue
            md_file = item / "SKILL.md"
            if not md_file.exists():
                continue
            has_skill_files = True

            try:
                skill = self._parse_skill(md_file)
                skills.append(skill)
                logger.info(f"加载 Skill: {skill.name}", file=str(md_file))
            except Exception as e:
                parse_errors += 1
                logger.error(f"解析 SKILL.md 失败: {md_file}", error=str(e))

        self._registry = SkillRegistry(skills=skills)
        if not skills and has_skill_files:
            logger.warning(f"所有 {parse_errors} 个 SKILL.md 文件加载失败，P0/P1 工具过滤将不可用")

        logger.info(
            "Skills 加载完成",
            total=len(skills),
            names=[s.name for s in skills],
        )
        return self._registry

    @staticmethod
    def _params_from_pydantic(model: Type[BaseModel]) -> Dict[str, Dict]:
        """从 Pydantic 模型生成 params 字典（类型/必填/描述/semantic）。

        params 的元数据来源：
          - type:     从 field_info.annotation 推断 Python 类型 → JSON 类型
          - required: 从 field_info.is_required() 推断
          - description:  从 Field(description=...) 读取
          - semantic: 从 Field(json_schema_extra={"semantic": ...}) 读取
        """
        result: Dict[str, Dict] = {}
        _PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}
        # Optional[str] 的 annotation 会被 Pydantic 展开为 Union[str, None]；
        # 这里直接用 model_fields 拿到已解析的 FieldInfo 来推断。
        for fname, finfo in model.model_fields.items():
            if not finfo.annotation:
                continue
            ann = finfo.annotation
            # 处理 Optional[T] / Union[T, None] → 取 T
            origin = getattr(ann, "__origin__", None)
            args = getattr(ann, "__args__", ())
            if origin is UnionType and args:
                # 取第一个非 NoneType 的成员
                inner = next((a for a in args if a is not type(None)), str)
                py_type = inner
            else:
                py_type = ann

            json_type = _PY_TO_JSON.get(py_type, "string")
            entry: Dict = {
                "type": json_type,
                "required": finfo.is_required(),
                "description": (finfo.description or ""),
            }
            extra = finfo.json_schema_extra or {}
            if extra.get("semantic"):
                entry["semantic"] = extra["semantic"]
            result[fname] = entry
        return result

    @staticmethod
    def _parse_skill(path: Path) -> SkillDef:
        """解析单个 SKILL.md 文件的 YAML frontmatter + Markdown 正文。

        params 不再从 YAML 读取，改为从 Pydantic 模型（schemas.INTENT_PARAM_SCHEMAS）生成。
        原因：Pydantic 是 params 的单一数据源，SKILL.md 中的 params 块仅供人类浏览，代码忽略。
        """
        raw = path.read_text(encoding="utf-8")

        # 提取 YAML frontmatter
        m = FRONTMATTER_RE.match(raw)
        if not m:
            raise ValueError(f"SKILL.md 缺少 YAML frontmatter: {path}")

        try:
            meta = yaml.safe_load(m.group(1))
        except yaml.YAMLError as e:
            raise ValueError(f"SKILL.md YAML frontmatter 解析失败: {path}") from e

        # 提取正文（frontmatter 之后的内容）
        body = raw[m.end():].strip()

        raw_name = str(meta.get("name") or "").strip()
        dir_name = path.parent.name

        # name 为空时回退到目录名，防止下游出现空键
        if not raw_name:
            logger.warning(f"SKILL.md 缺少 name 字段，回退到目录名: {dir_name}")
            name = dir_name
        else:
            name = raw_name
            # 校验 name 与目录名一致
            if name != dir_name:
                logger.warning(
                    f"SKILL.md 的 name '{name}' 与目录名 '{dir_name}' 不一致，"
                    f"可能导致工具路由异常"
                )

        # ── params：从 Pydantic 模型生成（单一数据源）──
        params_raw: Dict[str, Dict] = {}
        try:
            from src.modules.chat.schemas import INTENT_PARAM_SCHEMAS
            model = INTENT_PARAM_SCHEMAS.get(name)
            if model is not None:
                params_raw = SkillLoader._params_from_pydantic(model)
        except Exception as e:
            logger.warning(f"无法从 Pydantic 模型生成 params (name={name}): {e}")

        return SkillDef(
            name=name,
            display_name=str(meta.get("display_name") or raw_name or name).strip(),
            description=str(meta.get("description") or "").strip(),
            tags=[str(t).strip() for t in meta.get("tags", [])],
            allowed_tools=_parse_tool_string(meta.get("allowed-tools", "")),
            priority=int(meta.get("priority", 10)),
            body=body,  # 正文供 _build_system_prompt 按意图内联注入
            params=params_raw,  # 参数定义（来源：Pydantic 模型）
        )


# 在模块导入时（同步上下文）预先计算 skills 根路径，
# 彻底避免 LangGraph ASGI 事件循环中触发 blockbuster 阻塞检测。
_get_default_skills_root()

# ── 单例 ──

_loader: SkillLoader | None = None
_registry: SkillRegistry | None = None
_lock = threading.Lock()


def get_skill_registry() -> SkillRegistry:
    """获取全局 SkillRegistry 单例（线程安全）。"""
    global _loader, _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _loader = SkillLoader()
                _registry = _loader.load()
    return _registry
