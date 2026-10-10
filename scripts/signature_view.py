#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""签名视图工具（规格驱动测试专用）。

仅输出 Python 模块的**公开签名**（函数/类/方法名、参数、类型注解、返回注解、
显式抛出的异常），**不输出任何函数体实现**。

用途：test-generator 在"规格驱动"模式下需要"寻址"被测单元（知道有哪些
函数、参数怎么传、返回什么类型），但**不应读取实现体**（否则会写出配合代码、
只证自洽的假绿测试）。本工具让"只看签名、不看实现"从意愿变成结构上的默认路径：
编排主代理把本工具的输出（而非实现文件原文）喂给 test-generator 即可。

用法：
  python scripts/signature_view.py path/to/module.py
  python scripts/signature_view.py path/to/module.py::ClassName     # 只看某类及其方法
  python scripts/signature_view.py path/to/module.py::function_name # 只看某函数

注意：本工具只做静态 AST 提取，不导入模块、不执行代码，因此对含副作用/
循环导入的模块也安全。
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _args_with_defaults(args_list: list, defaults_list: list) -> list[str]:
    n_defaults = len(defaults_list)
    out: list[str] = []
    for i, a in enumerate(args_list):
        ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
        default = ""
        offset = i - (len(args_list) - n_defaults)
        if 0 <= offset < n_defaults:
            default = f"={ast.unparse(defaults_list[offset])}"
        out.append(f"{a.arg}{ann}{default}")
    return out


def _fmt_args(node: ast.arguments) -> str:
    parts = _args_with_defaults(getattr(node, "posonlyargs", []), getattr(node, "posonlyargs_defaults", []))
    parts += _args_with_defaults(node.args, node.defaults)
    if node.vararg:
        parts.append(f"*{node.vararg.arg}")
    parts += _fmt_args_kw(node)
    if node.kwarg:
        parts.append(f"**{node.kwarg.arg}")
    return "(" + ", ".join(parts) + ")"


def _fmt_args_kw(node: ast.arguments) -> list[str]:
    out: list[str] = []
    for a, dft in zip(node.kwonlyargs, node.kw_defaults):
        ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
        default = ""
        if dft is not None:
            default = f"={ast.unparse(dft)}"
        out.append(f"{a.arg}{ann}{default}")
    return out


def _signature(func: ast.AST) -> str:
    ret = f" -> {ast.unparse(func.returns)}" if getattr(func, "returns", None) else ""
    async_pref = "async " if isinstance(func, ast.AsyncFunctionDef) else ""
    return f"{async_pref}def {func.name}{_fmt_args(func.args)}{ret}"


def _raises(func: ast.AST) -> list[str]:
    """提取函数体里显式 `raise XxxError` 的异常名（仅作签名提示，非穷尽）。"""
    out: list[str] = []
    for n in ast.walk(func):
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Name):
            out.append(n.exc.id)
    return out


def collect(tree: ast.Module, target: str | None = None) -> list[str]:
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if target and node.name != target:
                continue
            line = "  " + _signature(node)
            raises = _raises(node)
            if raises:
                line += f"  # raises: {', '.join(dict.fromkeys(raises))}"
            out.append(line)
        elif isinstance(node, ast.ClassDef):
            if target and node.name != target:
                # 若 target 是方法名，仍要扫描所有类找方法
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == target:
                        out.append(f"class {node.name}:")
                        out.append("    " + _signature(sub))
                continue
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            head = f"class {node.name}({bases})" if bases else f"class {node.name}"
            out.append(head + ":")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    line = "    " + _signature(sub)
                    raises = _raises(sub)
                    if raises:
                        line += f"  # raises: {', '.join(dict.fromkeys(raises))}"
                    out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="仅输出模块公开签名（不含实现体）")
    ap.add_argument("module", help="模块路径，可选 ::ClassName 或 ::function_name 过滤")
    args = ap.parse_args()

    raw, _, target = args.module.partition("::")
    target = target or None
    path = Path(raw)
    if not path.exists():
        print(f"[ERROR] 模块不存在: {path}", file=sys.stderr)
        return 2
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"[ERROR] 语法错误: {e}", file=sys.stderr)
        return 2

    sigs = collect(tree, target)
    if not sigs:
        print(f"# 未找到匹配签名（target={target}）", file=sys.stderr)
        return 1
    print(f"# 签名视图: {path}" + (f"  (filter={target})" if target else ""))
    print("# 仅含签名，不含实现体 —— 供 test-generator 寻址被测符号")
    print()
    for s in sigs:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
