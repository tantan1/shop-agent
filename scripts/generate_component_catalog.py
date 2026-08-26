#!/usr/bin/env python3
"""静态分析扫描 src/ 下所有公共类/函数，自动生成组件清单 markdown。

对应 docs/ai-coding-series/04-规则即代码.md 的「规则维护：自动化而非手动」方案：
手写组件清单两周后就过时，本脚本用 AST 扫描代码、输出 component-catalog.mdc，
由 CodeBuddy Automation 定时（每周一）同步，保证清单永远和代码一致。

判定「公共组件」的规则：
  - 排除 __init__.py、测试文件（test_*.py / *_test.py）
  - 排除私有符号（以单下划线开头）
  - 排除魔术方法（__xxx__）
  - 只收 class / def（含类内方法）
  - 必须有 docstring 才视为「可复用组件」（无文档的过滤掉，避免噪音）

用法：
  python scripts/generate_component_catalog.py [--src apps/shop-agent/src] [--out .codebuddy/rules/component-catalog.mdc]
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# 被视为「业务可复用组件」的模块白名单前缀（避免把纯内部 util 也列进来造成噪音）
# 空列表表示不排除任何模块（全部公共符号都收录）。
MODULE_INCLUDE_PREFIXES: list[str] = []


@dataclass
class Component:
    kind: str            # "class" | "function" | "method"
    name: str
    module: str          # 点分模块路径，如 src.core.redis_cache_service
    signature: str       # 函数签名 / 类定义头
    docstring: str       # 首行摘要
    line: int            # 起始行号
    calls: list[str] = field(default_factory=list)  # 调用了哪些符号（调用图边）


def _is_public(name: str) -> bool:
    if name.startswith("_"):
        return False
    return True


def _is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _module_path(src_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _signature_from_node(node: ast.AST) -> str:
    try:
        return ast.unparse(node).split("\n")[0][:160]
    except Exception:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            return f"class {node.name}({bases})"[:160]
        return f"<{type(node).__name__}: {node.name}>"


def _extract_calls(node: ast.AST) -> list[str]:
    """从函数体提取被调用的符号名（调用图边）。"""
    calls: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Attribute):
                # obj.method → 记录 "obj.method"
                calls.append(ast.unparse(func))
            elif isinstance(func, ast.Name):
                calls.append(func.id)
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def scan_file(src_root: Path, file_path: Path) -> list[Component]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    module = _module_path(src_root, file_path)
    components: list[Component] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            doc = ast.get_docstring(node) or ""
            if not doc:
                continue
            components.append(Component(
                kind="class", name=node.name, module=module,
                signature=_signature_from_node(node),
                docstring=doc.splitlines()[0].strip(),
                line=node.lineno,
                calls=_extract_calls(node),
            ))
            # 类内公共方法
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and _is_public(sub.name):
                    mdoc = ast.get_docstring(sub) or ""
                    if not mdoc:
                        continue
                    components.append(Component(
                        kind="method", name=f"{node.name}.{sub.name}",
                        module=module,
                        signature=_signature_from_node(sub),
                        docstring=mdoc.splitlines()[0].strip(),
                        line=sub.lineno,
                        calls=_extract_calls(sub),
                    ))
        elif isinstance(node, ast.FunctionDef) and _is_public(node.name):
            doc = ast.get_docstring(node) or ""
            if not doc:
                continue
            components.append(Component(
                kind="function", name=node.name, module=module,
                signature=_signature_from_node(node),
                docstring=doc.splitlines()[0].strip(),
                line=node.lineno,
                calls=_extract_calls(node),
            ))
    return components


def render_md(components: list[Component], src_root: Path) -> str:
    by_module: dict[str, list[Component]] = {}
    for c in components:
        by_module.setdefault(c.module, []).append(c)

    lines: list[str] = []
    lines.append("---")
    lines.append("description: 自动生成的组件清单——项目内可复用公共类/函数。由 scripts/generate_component_catalog.py 扫描生成，勿手动编辑。")
    lines.append("alwaysApply: false")
    lines.append("---")
    lines.append("")
    lines.append("# 组件清单（自动同步）")
    lines.append("")
    lines.append(f"> 本文件由 `scripts/generate_component_catalog.py` 基于 `ast` 静态分析自动生成，")
    lines.append(f"> 生成时间 {_dt.date.today().isoformat()}，扫描根 `{src_root}`。")
    lines.append("> **AI 编码前先检索此清单**：要加某能力时，优先复用下方已有组件，禁止重复实现。")
    lines.append("")
    lines.append("## 使用约定")
    lines.append("")
    lines.append("- 列表仅含带 docstring 的**公共**类/函数/方法（排除私有 `_`、魔术方法、测试文件）。")
    lines.append("- 新增能力前，先 grep / 搜此清单确认无现成实现；若有，直接 import 复用。")
    lines.append("- 本清单由 Automation 每周一自动重建，无需人工维护。")
    lines.append("")

    # 按模块分组输出
    for module in sorted(by_module):
        lines.append(f"## `{module}`")
        lines.append("")
        lines.append("| 组件 | 类型 | 摘要 | 定义位置 |")
        lines.append("|------|------|------|----------|")
        for c in sorted(by_module[module], key=lambda x: x.name):
            loc = f"{module.replace('.', '/')}.py:{c.line}"
            doc = c.docstring.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{c.name}` | {c.kind} | {doc} | {loc} |")
        lines.append("")

    # 调用图摘要（影响面分析辅助）
    edges = [(c.name, callee) for c in components for callee in c.calls if c.calls]
    if edges:
        lines.append("## 调用关系摘要（影响面参考）")
        lines.append("")
        lines.append("> 下列为静态调用边，供「改某组件前评估影响面」参考。完整图请用 impact_analysis 工具。")
        lines.append("")
        for c in components:
            if c.calls:
                callees = ", ".join(c.calls[:8])
                if len(c.calls) > 8:
                    callees += f" …(+{len(c.calls) - 8})"
                lines.append(f"- `{c.name}` → {callees}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    default_src = here.parent / "apps" / "shop-agent" / "src"
    default_out = here.parent / ".codebuddy" / "rules" / "component-catalog.mdc"

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=default_src, help="待扫描的 src 根目录")
    ap.add_argument("--out", type=Path, default=default_out, help="输出的 mdc 路径")
    args = ap.parse_args()

    src_root = args.src.resolve()
    if not src_root.exists():
        raise SystemExit(f"src 根不存在: {src_root}")

    py_files: list[Path] = []
    for p in src_root.rglob("*.py"):
        if _is_test_file(p):
            continue
        if p.name == "__init__.py":
            continue
        py_files.append(p)

    all_components: list[Component] = []
    for pf in py_files:
        all_components.extend(scan_file(src_root, pf))

    md = render_md(all_components, src_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")

    print(f"[generate_component_catalog] 扫描 {len(py_files)} 个 py 文件，"
          f"收录 {len(all_components)} 个公共组件，输出 {args.out}")


if __name__ == "__main__":
    main()
