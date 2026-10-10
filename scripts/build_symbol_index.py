#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Tree-sitter 为 Python 项目构建「签名级符号索引」。

只抽取 类/函数/方法 的 name、parameters(含类型注解)、return type、decorators，
**刻意不读取函数体(block)** —— 这正是「签名 + Spec 生成测试」所需的契约表面输入。

用法:
  python build_symbol_index.py --selftest            # 内置单元测试
  python build_symbol_index.py --root . --out idx.json   # 全项目扫描
  python build_symbol_index.py --root apps/monitoring-agent   # 指定子目录
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import tree_sitter
    import tree_sitter_python
except ImportError:
    sys.exit("请先安装依赖: pip install tree-sitter tree-sitter-python")

PY = tree_sitter.Language(tree_sitter_python.language())
try:
    _PARSER = tree_sitter.Parser(PY)
except TypeError:  # 兼容旧版 API
    _PARSER = tree_sitter.Parser()
    _PARSER.set_language(PY)

_FUNC_TYPES = ("function_definition", "async_function_definition")
_EXCLUDES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "venv_cuda", "venv_mineru", "env",
    ".tox", "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache", "volumes",
}


def _unwrap(node):
    """剥掉 decorated_definition 外壳,返回 (内部定义节点, [decorator 名])。"""
    if node.type == "decorated_definition":
        defs = [c for c in node.children
                if c.type in ("function_definition", "async_function_definition", "class_definition")]
        decos = []
        for d in node.children:
            if d.type == "decorator":
                named = [c for c in d.children if c.is_named]
                if named:
                    decos.append(named[0].text.decode())
        if defs:
            return defs[0], decos
    return node, []


def _param_text(pr):
    """单个 parameter 节点 -> 'name: type' / 'name' / None(匿名标点)。

    tree-sitter-python 的 parameter 子节点是位置式的(identifier + type),
    不是命名字段,需按类型遍历。
    """
    if not pr.is_named:           # 跳过 '(' ',' ')' 等匿名节点
        return None
    ident = None
    typ = None
    for c in pr.children:
        if c.type == "identifier":
            ident = c
        elif c.type == "type":
            typ = c
    if ident is None:
        return None
    name = ident.text.decode()
    return f"{name}: {typ.text.decode()}" if typ else name


def _signature(func):
    """取函数/方法签名(不 descend 进 body)。"""
    params = func.child_by_field_name("parameters")
    ret = func.child_by_field_name("return_type")
    param_list = []
    if params:
        for c in params.children:
            p = _param_text(c)
            if p:
                param_list.append(p)
    return {
        "name": func.child_by_field_name("name").text.decode(),
        "params": param_list,
        "returns": ret.text.decode() if ret else None,
    }


def index_source(code: bytes) -> dict:
    """解析一段源码,返回 {classes: {name: [methods]}, functions: [...]}。"""
    tree = _PARSER.parse(code)
    out = {"classes": {}, "functions": []}

    def scan_scope(children, into_classes, into_funcs, class_name=None):
        for child in children:
            node, decos = _unwrap(child)
            if node.type in _FUNC_TYPES:
                sig = _signature(node)
                sig["decorators"] = decos
                if class_name:
                    into_classes[class_name].append(sig)
                else:
                    into_funcs.append(sig)
            elif node.type == "class_definition":
                cname = node.child_by_field_name("name").text.decode()
                into_classes[cname] = []
                body = node.child_by_field_name("body")
                if body:
                    scan_scope(body.children, into_classes, into_funcs, cname)

    scan_scope(tree.root_node.children, out["classes"], out["functions"])
    return out


def build_index(root: Path, out_path: Path) -> dict:
    files = [p for p in root.rglob("*.py")
             if not any(part in _EXCLUDES for part in p.parts)
             and p.suffix == ".py"]
    index = {}
    total_syms = 0
    t0 = time.time()
    for f in files:
        try:
            data = index_source(f.read_bytes())
        except Exception as exc:  # 单个文件解析失败不影响整体
            print(f"  [warn] 解析失败 {f}: {exc}", file=sys.stderr)
            continue
        if data["classes"] or data["functions"]:
            rel = str(f.relative_to(root))
            index[rel] = data
            total_syms += len(data["functions"]) + sum(len(v) for v in data["classes"].values())
    summary = {
        "root": str(root),
        "files_scanned": len(files),
        "files_with_symbols": len(index),
        "symbols_extracted": total_syms,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_path.write_text(json.dumps({"summary": summary, "files": index},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def selftest() -> None:
    sample = b"""
class Foo:
    @property
    def x(self) -> int: ...
    def bar(self, a: int, b: str = "x") -> bool: ...
async def baz(n: int) -> None: ...
@decorator
def qux() -> str: ...
"""
    idx = index_source(sample)
    assert "Foo" in idx["classes"], idx
    methods = idx["classes"]["Foo"]
    assert any(m["name"] == "x" and "property" in m["decorators"] for m in methods), methods
    assert any(m["name"] == "bar" and "a: int" in m["params"] and m["returns"] == "bool" for m in methods), methods
    assert any(f["name"] == "baz" for f in idx["functions"]), idx
    assert any(f["name"] == "qux" and "decorator" in f["decorators"] for f in idx["functions"]), idx
    print("selftest: OK")


def main():
    ap = argparse.ArgumentParser(description="Tree-sitter 签名级符号索引构建器")
    ap.add_argument("--root", default=".", help="项目根目录(默认当前目录)")
    ap.add_argument("--out", default="symbol_index.json", help="输出 JSON 路径")
    ap.add_argument("--selftest", action="store_true", help="运行内置测试后退出")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    root = Path(args.root).resolve()
    if not root.exists():
        sys.exit(f"root 不存在: {root}")
    summary = build_index(root, Path(args.out).resolve())
    print("构建完成:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
