#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 Tree-sitter 为 Python 项目构建「项目内调用图」。

与 build_symbol_index.py 同一思路: LLM 已知第三方库语义, 所以我们
**只保留项目内部的调用边**, 把 requests.get / sqlalchemy.Session.query
这类第三方调用直接剪掉 —— 它们记下来只是噪声。

产物:
  - 每个函数/方法的「调用了谁」(calls)
  - 全局 caller/callee 邻接表(已裁剪到项目内符号)
  - 三个查询能力: get_callers / get_callees / impact_radius

用法:
  python build_call_graph.py --selftest               # 内置单元测试
  python build_call_graph.py --root . --out cg.json    # 全项目扫描
  python build_call_graph.py --root apps/monitoring-agent
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


def _callee_name(call_node) -> str | None:
    """从 call 节点取出被调用者的名字。

    foo()            -> 'foo'        (identifier)
    obj.method()     -> 'method'     (attribute 的最后一个分量)
    mod.sub.func()   -> 'func'       (同上)
    """
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "identifier":
        return fn.text.decode()
    if fn.type == "attribute":
        attr = fn.child_by_field_name("attribute")
        if attr:
            return attr.text.decode()
    return None


def extract_file(code: bytes):
    """解析一段源码, 返回 (symbols, calls_by_func, classes)。

    symbols:     该函数/方法/嵌套函数的全限定名列表
                  (类内方法 -> 'Class.method', 顶层 -> 'func', 嵌套 -> 'outer.inner')
    calls_by_func: qname -> {被调用的原始名字集合}
                  调用归属到**最内层**函数(靠 func_stack 实现嵌套正确归属)
    classes:     本文件内定义的类名集合
    """
    tree = _PARSER.parse(code)
    symbols: list[str] = []
    calls_by_func: dict[str, set[str]] = {}
    classes: set[str] = set()
    func_stack: list[str] = []

    def visit(node, class_ctx: list[str]) -> None:
        t = node.type
        if t == "class_definition":
            cname = node.child_by_field_name("name").text.decode()
            classes.add(cname)
            body = node.child_by_field_name("body")
            if body:
                for c in body.children:
                    visit(c, class_ctx + [cname])
            return
        if t in _FUNC_TYPES:
            fname = node.child_by_field_name("name").text.decode()
            qname = ".".join(class_ctx + [fname])
            symbols.append(qname)
            func_stack.append(qname)
            body = node.child_by_field_name("body")
            if body:
                for c in body.children:
                    visit(c, class_ctx)
            func_stack.pop()
            return
        if t == "call":
            if func_stack:
                callee = _callee_name(node)
                if callee:
                    calls_by_func.setdefault(func_stack[-1], set()).add(callee)
            # 继续下钻, 捕获参数表达式里嵌套的调用(仍归属当前调用方)
            for c in node.children:
                visit(c, class_ctx)
            return
        for c in node.children:
            visit(c, class_ctx)

    visit(tree.root_node, [])
    return symbols, calls_by_func, classes


def resolve_edges(all_symbols: set[str], class_names: set[str], calls_by_func: dict[str, set[str]]):
    """把每个原始被调用名解析成项目内符号, 得到裁剪后的调用边。

    解析规则(语法级 best-effort):
      - 直接命中: callee 在符号集 -> 边
      - 方法命中: 任一 'Class.callee' 在符号集 -> 边 (覆盖 self.method / cls.method)
    解析不到的(多为第三方库调用)直接丢弃。
    """
    edges: list[tuple[str, str]] = []
    for caller, callees in calls_by_func.items():
        for callee in callees:
            if callee in all_symbols:
                edges.append((caller, callee))
                continue
            hit = False
            for cls in class_names:
                cand = f"{cls}.{callee}"
                if cand in all_symbols:
                    edges.append((caller, cand))
                    hit = True
                    break
            # 解析不到 -> 视为外部调用, 剪枝
    return edges


def build_graph(root: Path, out_path: Path) -> dict:
    files = [p for p in root.rglob("*.py")
             if not any(part in _EXCLUDES for part in p.parts)
             and p.suffix == ".py"]

    all_symbols: set[str] = set()
    class_names: set[str] = set()
    per_file: dict[str, dict] = {}
    total_call_sites = 0          # 所有 call 节点(含外部)
    resolved_internal = 0         # 裁剪后保留的项目内边(含重复)

    t0 = time.time()
    for f in files:
        try:
            syms, calls, classes = extract_file(f.read_bytes())
        except Exception as exc:  # 单文件解析失败不影响整体
            print(f"  [warn] 解析失败 {f}: {exc}", file=sys.stderr)
            continue
        if not syms and not calls:
            continue
        rel = str(f.relative_to(root))
        all_symbols.update(syms)
        class_names.update(classes)
        file_entry = {"functions": [], "classes": {}}
        for s in syms:
            if "." in s:
                cls, meth = s.split(".", 1)
                file_entry["classes"].setdefault(cls, []).append(meth)
            else:
                file_entry["functions"].append(s)
        # 记录每个函数的调用(原始名, 供展示)
        calldict = {k: sorted(v) for k, v in calls.items() if k in syms}
        total_call_sites += sum(len(v) for v in calls.values())
        file_entry["calls"] = calldict
        per_file[rel] = file_entry

    # 全局解析 + 裁剪
    edges = resolve_edges(all_symbols, class_names, calls_global(per_file))
    resolved_internal = len(edges)

    # 邻接表
    callers: dict[str, list[str]] = {}
    callees: dict[str, list[str]] = {}
    for a, b in edges:
        callers.setdefault(b, []).append(a)
        callees.setdefault(a, []).append(b)

    summary = {
        "root": str(root),
        "files_scanned": len(files),
        "files_with_symbols": len(per_file),
        "symbols_extracted": len(all_symbols),
        "total_call_sites": total_call_sites,
        "project_internal_edges": resolved_internal,
        "external_pruned": total_call_sites - resolved_internal,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out = {
        "summary": summary,
        "edges": [[a, b] for a, b in edges],
        "callers": {k: sorted(set(v)) for k, v in callers.items()},
        "callees": {k: sorted(set(v)) for k, v in callees.items()},
        "files": per_file,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def calls_global(per_file: dict) -> dict[str, set[str]]:
    """从 per_file 的 calls 字段恢复 caller->callees 原始映射。"""
    merged: dict[str, set[str]] = {}
    for entry in per_file.values():
        for caller, callees in entry.get("calls", {}).items():
            merged.setdefault(caller, set()).update(callees)
    return merged


# ---------- 查询能力 ----------
def get_callers(graph: dict, symbol: str) -> list[str]:
    return graph["callers"].get(symbol, [])


def get_callees(graph: dict, symbol: str) -> list[str]:
    return graph["callees"].get(symbol, [])


def impact_radius(graph: dict, symbol: str) -> list[str]:
    """从 symbol 出发, BFS 沿调用边逆向(谁会受它影响), 返回受影响符号列表。"""
    seen = set()
    queue = [symbol]
    while queue:
        cur = queue.pop()
        for up in graph["callers"].get(cur, []):
            if up not in seen:
                seen.add(up)
                queue.append(up)
    seen.discard(symbol)
    return sorted(seen)


def selftest() -> None:
    sample = b"""
class Foo:
    def a(self):
        self.b()
        helper()
    def b(self): ...
def helper():
    external_lib.doThing()
"""
    syms, calls, _classes = extract_file(sample)
    assert "Foo.a" in syms and "Foo.b" in syms and "helper" in syms, syms
    assert "b" in calls["Foo.a"], calls
    assert "helper" in calls["Foo.a"], calls
    assert "external_lib" not in calls["helper"], calls  # 第三方不入栈

    edges = resolve_edges(set(syms), {"Foo"}, calls)
    assert ("Foo.a", "Foo.b") in edges, edges
    assert ("Foo.a", "helper") in edges, edges
    assert all("external" not in a and "external" not in b for a, b in edges), edges

    g = {
        "callers": {"Foo.b": ["Foo.a"], "helper": ["Foo.a"]},
        "callees": {"Foo.a": ["Foo.b", "helper"]},
    }
    assert get_callers(g, "Foo.b") == ["Foo.a"]
    assert impact_radius(g, "helper") == ["Foo.a"]
    print("selftest: OK")


def main():
    ap = argparse.ArgumentParser(description="Tree-sitter 项目内调用图构建器")
    ap.add_argument("--root", default=".", help="项目根目录(默认当前目录)")
    ap.add_argument("--out", default="call_graph.json", help="输出 JSON 路径")
    ap.add_argument("--selftest", action="store_true", help="运行内置测试后退出")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    root = Path(args.root).resolve()
    if not root.exists():
        sys.exit(f"root 不存在: {root}")
    summary = build_graph(root, Path(args.out).resolve())
    print("构建完成:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
