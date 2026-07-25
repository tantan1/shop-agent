"""
ReAct Agent 规则加载器 —— 仿 Claude Code rules/ 机制，支持 YAML frontmatter 作用域过滤。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel

from src.shared.logger import APILogger

logger = APILogger("react_agent_rules")

_RULES_DIR = Path(__file__).resolve().parents[4] / "agent-rules"


class Rule(BaseModel):
    """单条 Agent 规则的结构化表示。"""
    name: str
    body: str
    intents: set[str] = set()
    tools: set[str] = set()
    always: bool = True


_RULES: list[Rule] | None = None


def _load_rules() -> list[Rule]:
    """扫描 agent-rules/*.md，解析 YAML frontmatter + Markdown 正文。"""
    global _RULES
    if _RULES is not None:
        return _RULES

    _RULES = []
    if not _RULES_DIR.is_dir():
        logger.info("agent-rules/ 目录不存在，跳过规则加载")
        return _RULES

    for md_file in sorted(_RULES_DIR.glob("*.md")):
        rule = _parse_rule_file(md_file)
        if rule:
            _RULES.append(rule)

    global_rules = [r for r in _RULES if r.always]
    scoped_rules = [r for r in _RULES if not r.always]
    logger.info(
        f"已加载 {len(_RULES)} 条 Agent 规则",
        global_count=len(global_rules),
        scoped_count=len(scoped_rules),
        global_names=[r.name for r in global_rules],
        scoped_names=[r.name for r in scoped_rules],
    )
    return _RULES


def _parse_rule_file(md_file) -> Rule | None:
    """解析单个规则文件。"""
    try:
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            return None

        frontmatter: dict = {}
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError:
                    logger.warning(f"规则文件 YAML frontmatter 解析失败: {md_file}")
                body = parts[2].strip()

        lines = body.split("\n")
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        body = "\n".join(lines).strip()

        if not body:
            return None

        return Rule(
            name=md_file.stem,
            body=body,
            intents=set(frontmatter.get("intents", []) or []),
            tools=set(frontmatter.get("tools", []) or []),
            always=frontmatter.get("always", not frontmatter),
        )
    except Exception:
        logger.warning(f"读取规则文件失败: {md_file}")
        return None


def _filter_rules(
    rules: list[Rule],
    intent: str | None,
    tool_names: set[str],
) -> str:
    """根据当前意图和选中工具过滤规则，返回拼接后的规则文本。"""
    blocks: list[str] = []
    for rule in rules:
        if rule.always:
            blocks.append(rule.body)
            continue

        intent_match = (not rule.intents) or (intent in rule.intents)
        tool_match = (not rule.tools) or bool(rule.tools & tool_names)

        if intent_match and tool_match:
            blocks.append(rule.body)

    return "\n\n---\n\n".join(blocks) if blocks else ""
