#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变异测试（mutation testing）门禁 —— 兑现文章 23 的核心论点：

    "所有路径都要过机器验证：注入一个缺陷，测试必须失败。"

本脚本不依赖 mutmut（对 async/FastAPI 项目 setup 成本高），改用轻量自写变异注入器：
  - 对目标源码注入一组确定性变异（布尔翻转 / 比较符翻转 / 算术边界 / 短路逻辑）
  - 每个变异生成临时副本，跑该模块的 pytest
  - 若测试仍全过 → 判"未捕获变异"（即测试无法感知此缺陷 = 假绿证据）
  - 若至少一个测试失败 → 判"已捕获"（变异被测试感知，凭证有效）

输出：
  - 人类可读报告（含每个未捕获变异的位置，便于补测试）
  - --json 导出结构化结果（供 CI 门禁：未捕获数 > 阈值则失败）
  - 退出码：未捕获数 > threshold 时为 1（阻断），否则 0

用法：
  python scripts/mutation_check.py --module apps/gateway/gateway/limiter.py \
      --tests "apps/gateway/tests/test_2b03_cost.py::test_proxy_returns_429_on_rate_limit" --threshold 0
"""
from __future__ import annotations

import argparse
import ast
import atexit
import json
import signal
import subprocess
import sys
import tempfile
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── 变异算子 ────────────────────────────────────────────────────────────────
# 每个算子接收源码字符串，返回 (变异后源码, 说明) 列表。
# 设计原则：只做"语义明显改变但语法仍合法"的局部改写，避免破坏模块导入。

BOOL_FLIPS = [("True", "False"), ("False", "True")]
CMP_FLIPS = [
    ("==", "!="), ("!=", "=="),
    ("<=", ">"), (">=", "<"),
    ("<", ">="), (">", "<="),
]
# 算术边界：±1 偏移（仅在数字字面量上）
ARITH = re.compile(r"\b(\d+)\b")


def _flip_tokens(src: str, pairs, label) -> list:
    out = []
    for a, b in pairs:
        # 只在代码区替换：跳过字符串字面量与注释，避免产生无行为变化的伪变异
        new = _replace_outside_strings_and_comments(src, a, b)
        if new != src:
            out.append((new, f"{label}: {a} -> {b}"))
    return out


def _iter_code_spans(src: str):
    """按行切分源码，标注每行可变异的"代码区"与需跳过的"注释/文档串区"。

    需跳过两类文本（改动它们不产生任何行为变化，会生成伪变异）：
      1) `#` 之后的注释
      2) 三引号文档字符串（docstring）——跨行，用状态机跟踪
    返回 [(code_part, skipped_part, eol), ...]
    """
    in_docstring = None
    spans = []
    for line in src.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        eol = line[len(stripped):]

        if in_docstring is not None:
            # 处于文档串内部：整行跳过，并检查是否结束
            idx = stripped.find(in_docstring)
            if idx != -1:
                in_docstring = None
            spans.append(("", stripped, eol))
            continue

        code, comment = _split_code_comment(stripped)
        # 检测本行是否开启了三引号文档串（只看代码区，避免把注释里的引号算入）
        quote = None
        i = 0
        n = len(code)
        while i < n:
            c = code[i]
            if c in ("'", '"'):
                if code[i:i + 3] in ("'''", '"""'):
                    quote = code[i:i + 3]
                    i += 3
                    continue
                i += 1
                continue
            i += 1
        if quote is not None:
            # 同行内是否闭合？未闭合则进入跨行文档串
            if code.count(quote) >= 2:
                pass  # 同行闭合，属普通字符串，保留代码区
            else:
                in_docstring = quote
                spans.append(("", stripped, eol))
                continue
        spans.append((code, comment, eol))
    return spans


def _split_code_comment(line: str) -> tuple:
    """把一行拆成 (代码部分, 注释部分)。

    注释里的 True/False/== 等词被替换后不产生任何行为变化，
    若不剥离会生成"伪变异"，导致门禁恒定阻断（假阳性）。
    """
    in_str = None
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in ("'", '"'):
            in_str = c
            i += 1
            continue
        if c == "#":
            return line[:i], line[i:]
        i += 1
    return line, ""


def _replace_outside_strings_and_comments(src: str, old: str, new: str) -> str:
    """在源码的"代码区"替换 old->new（跳过字符串字面量与注释）。"""
    out_lines = []
    for code, skipped, eol in _iter_code_spans(src):
        # 只对代码区替换，注释与文档串原样保留
        out_lines.append(_replace_outside_strings(code, old, new) + skipped + eol)
    return "".join(out_lines)


def _replace_outside_strings(src: str, old: str, new: str) -> str:
    """在源码中非字符串字面量区域替换 old->new（避免误改字符串）。"""
    result = []
    i = 0
    n = len(src)
    in_str = None
    while i < n:
        c = src[i]
        if in_str:
            result.append(c)
            if c == in_str and (i + 1 >= n or src[i + 1] != in_str):
                in_str = None
            i += 1
            continue
        if c in ("'", '"'):
            # 简单处理：进入字符串（不处理转义嵌套，够用）
            in_str = c
            result.append(c)
            i += 1
            continue
        if src[i:i + len(old)] == old:
            # 检查词边界，避免把 '==' 里的 '=' 当独立替换
            before = src[i - 1] if i > 0 else " "
            after = src[i + len(old)] if i + len(old) < n else " "
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                result.append(new)
                i += len(old)
                continue
        result.append(c)
        i += 1
    return "".join(result)


def _arith_offset(src: str) -> list:
    """对数字字面量做 ±1 偏移（跳过字符串/注释）。"""
    out = []
    for delta in (1, -1):
        def repl(m, d=delta):
            v = int(m.group(1))
            return str(v + d)
        new = ARITH.sub(lambda m: _replace_in_str_context(src, m, repl), src) if False else None
        # 在代码区替换（跳过字符串与注释）
        new = _arith_replace_skipping_comments(src, delta)
        if new != src:
            out.append((new, f"算术边界: ±1 (delta={delta})"))
    return out


def _arith_replace_skipping_comments(src: str, delta: int) -> str:
    """对整份源码做算术 ±1，但跳过注释（注释里的数字无行为影响）。"""
    out_lines = []
    for code, skipped, eol in _iter_code_spans(src):
        out_lines.append(_arith_replace(code, delta) + skipped + eol)
    return "".join(out_lines)


def _arith_replace(src: str, delta: int) -> str:
    result = []
    i = 0
    n = len(src)
    in_str = None
    while i < n:
        c = src[i]
        if in_str:
            result.append(c)
            if c == in_str:
                in_str = None
            i += 1
            continue
        if c in ("'", '"'):
            in_str = c
            result.append(c)
            i += 1
            continue
        m = ARITH.match(src[i:])
        if m and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            v = int(m.group(1))
            result.append(str(v + delta))
            i += len(m.group(1))
            continue
        result.append(c)
        i += 1
    return "".join(result)


def _replace_in_str_context(src, m, repl):
    return src  # 占位，实际用 _arith_replace


# ── 伪变异（噪声）过滤 ──────────────────────────────────────────────────────
# 有些替换只改动"对行为无影响"的位置，例如：
#   logger.debug(..., exc_info=True) → False   # 仅影响日志是否带堆栈
#   json.dumps(..., ensure_ascii=False) → True # 仅影响非 ASCII 是否转义
# 这些变异必然存活（测试无法也不应感知），会把捕获率永远压低、门禁恒定阻断。
# 这里按"改动行是否全部命中噪声模式"来丢弃，且丢弃数量会写入报告与 JSON，
# 保证过滤行为本身可审计、可证伪（不是静默掩盖）。
NOISE_PATTERNS = (
    "exc_info=",
    "ensure_ascii=",
)


def _diff_changed_lines(orig: str, mutant: str) -> list:
    """返回变异体相对原码发生改动的行（新行内容）。"""
    import difflib

    changed = []
    for line in difflib.unified_diff(
        orig.splitlines(), mutant.splitlines(), lineterm="", n=0
    ):
        if line.startswith("+") and not line.startswith("+++"):
            changed.append(line[1:])
    return changed


def _is_noise_mutant(orig: str, mutant: str) -> bool:
    """该变异的所有改动是否都落在无行为影响的位置。"""
    changed = _diff_changed_lines(orig, mutant)
    if not changed:
        return True
    return all(any(p in ln for p in NOISE_PATTERNS) for ln in changed)


def generate_mutants(src: str, filter_noise: bool = True) -> tuple:
    """返回 (mutants, skipped_noise_count)。

    mutants: [(mutant_src, description), ...]
    filter_noise=True 时丢弃仅改动噪声位置的伪变异，并统计丢弃数。
    """
    raw = []
    raw += _flip_tokens(src, BOOL_FLIPS, "布尔翻转")
    raw += _flip_tokens(src, CMP_FLIPS, "比较翻转")
    raw += _arith_offset(src)

    if not filter_noise:
        return raw, 0

    kept, skipped = [], 0
    for mutant, desc in raw:
        if _is_noise_mutant(src, mutant):
            skipped += 1
            continue
        kept.append((mutant, desc))
    return kept, skipped


# ── 运行单个变异 ────────────────────────────────────────────────────────────
# 安全防护：变异是"就地覆盖原文件"，若进程被中断（超时/取消/异常退出），
# finally 不会执行，源码将永久残留变异体。因此额外做：
#   1) 启动前把原始内容备份到同目录 .bak 文件
#   2) 注册 atexit + 信号处理器，保证任意退出路径都还原
_BACKUP_STATE: dict = {}


def _install_safety_net(module_path: Path, orig_src: str) -> None:
    """备份原始内容并注册还原钩子（防进程中断导致源码残留变异）。"""
    if _BACKUP_STATE:
        return
    bak = module_path.with_suffix(module_path.suffix + ".mutbak")
    bak.write_text(orig_src, encoding="utf-8")
    _BACKUP_STATE.update({"module": module_path, "src": orig_src, "bak": bak})

    def _restore(*_args):
        st = _BACKUP_STATE
        try:
            if st and st["module"].exists():
                # 优先用 .mutbak 备份文件还原（最权威），回退到内存快照
                restore_src = st["src"]
                bak = st.get("bak")
                if bak and bak.exists():
                    try:
                        restore_src = bak.read_text(encoding="utf-8")
                    except Exception:
                        pass
                if st["module"].read_text(encoding="utf-8") != restore_src:
                    st["module"].write_text(restore_src, encoding="utf-8")
            if st.get("bak") and st["bak"].exists():
                try:
                    st["bak"].unlink()
                except Exception:
                    pass
        except Exception:
            pass

    atexit.register(_restore)
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, lambda s, f: (_restore(), sys.exit(130)))
            except Exception:
                pass


def run_mutant(module_path: Path, mutant_src: str, tests_glob: str, timeout: int = 120) -> tuple:
    """把变异写入原模块跑 pytest。
    返回 (captured, timed_out)：
      captured=True  测试感知到缺陷（pytest 退出码非 0）
      timed_out=True 变异导致测试进程挂死（如重新引入无限循环），属明显异常行为
    """
    orig = module_path.read_text(encoding="utf-8")
    _install_safety_net(module_path, orig)
    try:
        module_path.write_text(mutant_src, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", tests_glob, "-q", "--no-header", "-p", "no:cacheprovider"],
                cwd=REPO_ROOT,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            # pytest 退出码非 0 = 有失败/错误 = 变异被捕获
            captured = proc.returncode != 0
            return captured, False
        except subprocess.TimeoutExpired as e:
            # 变异使测试挂死（如无限循环）—— 属明显异常行为，计为已捕获但单独标注
            # 关键：超时不要遗留挂死的子进程，否则它会继续持有已变异的模块
            try:
                if e.cmd and hasattr(e, "process") and e.process is not None:
                    e.process.kill()
            except Exception:
                pass
            return True, True
    finally:
        module_path.write_text(orig, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="轻量变异测试门禁（兑现文章23'测试必须失败'）")
    ap.add_argument("--module", required=True, help="待测模块绝对/相对路径")
    ap.add_argument("--tests", required=True, help="对应 pytest 路径/表达式")
    ap.add_argument("--threshold", type=int, default=0, help="未捕获变异允许上限，超过则阻断")
    ap.add_argument("--json", help="导出结果 JSON")
    ap.add_argument(
        "--keep-noise", action="store_true",
        help="保留伪变异（不过滤），用于审计过滤器是否掩盖了真实缺陷",
    )
    ap.add_argument(
        "--timeout", type=int, default=60,
        help="每个变异体运行 pytest 的超时秒数；超时视为'已捕获'(变异导致挂死)",
    )
    args = ap.parse_args()

    module_path = (REPO_ROOT / args.module).resolve() if not Path(args.module).is_absolute() else Path(args.module)
    if not module_path.exists():
        print(f"[ERROR] 模块不存在: {module_path}", file=sys.stderr)
        sys.exit(2)

    src = module_path.read_text(encoding="utf-8")
    # 语法预检
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"[ERROR] 源码语法错误: {e}", file=sys.stderr)
        sys.exit(2)

    mutants, skipped_noise = generate_mutants(src, filter_noise=not args.keep_noise)
    print(f"\n=== 变异测试门禁 ===")
    print(f"模块: {module_path}")
    print(f"测试: {args.tests}")
    print(f"生成变异数: {len(mutants)}")
    if skipped_noise:
        print(f"伪变异(已过滤): {skipped_noise}  <- 仅改动无行为影响处("
              f"{'/'.join(NOISE_PATTERNS)})，测试无法也不应感知；可用 --keep-noise 审计")
    print()

    killed = 0
    survived = []
    timed_out = []
    for idx, (mutant_src, desc) in enumerate(mutants, 1):
        # 先验证变异本身语法合法
        try:
            ast.parse(mutant_src)
        except SyntaxError:
            continue
        captured, is_timeout = run_mutant(module_path, mutant_src, args.tests, timeout=args.timeout)
        if captured:
            killed += 1
            status = "捕获(超时)" if is_timeout else "捕获"
            if is_timeout:
                timed_out.append(desc)
        else:
            survived.append(desc)
            status = "未捕获(假绿)"
        print(f"  [{idx:02d}] {status:10s} | {desc}")

    total = killed + len(survived)
    rate = (killed / total * 100) if total else 0.0
    print(f"\n--- 汇总 ---")
    print(f"  变异总数: {total}")
    print(f"  已捕获:   {killed}")
    print(f"  未捕获:   {len(survived)}  <- 这些缺陷注入后测试仍全绿")
    print(f"  捕获率:   {rate:.0f}%")
    if timed_out:
        print(f"  超时捕获: {len(timed_out)}  <- 变异使测试挂死(如重新引入无限循环)，计为已捕获")
        for s in timed_out:
            print(f"    - [超时] {s}")
    if survived:
        print(f"\n  未捕获变异清单（需补测试让'测试必须失败'）:")
        for s in survived:
            print(f"    - {s}")

    blocked = len(survived) > args.threshold
    print(f"\n  门禁(threshold={args.threshold}): {'[BLOCKED] 阻断' if blocked else '[PASS] 通过'}")

    if args.json:
        payload = {
            "module": str(module_path),
            "total": total,
            "killed": killed,
            "survived": survived,
            "timed_out": timed_out,
            "capture_rate": round(rate, 1),
            "blocked": blocked,
            "noise_filtered": skipped_noise,
            "keep_noise": bool(args.keep_noise),
        }
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  结果已导出: {args.json}")

    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()
