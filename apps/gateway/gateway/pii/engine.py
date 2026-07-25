"""确定性脱敏引擎（07 批次2 落地，确定性层）。

消费规则公共包 ``rules/store.RulesStore`` 的 ``pii`` 实例。热路径只调
``redact()``（纯内存正则，零联网、无 IO）。启动期 ``PiiEngine`` 构造即加载规则，
加载失败抛 ``RulesLoadError``（由调用方转 ``GovernanceError`` 上抛，绝不带着
空规则放行——G2 门② 绝不零规则启动）。

掩码策略（规则声明，见 pii/pii.baseline.yaml）：
- ``keep_prefix``：保留前缀 N 位，其余打星（如手机 ``138****8000``）
- ``keep_suffix``：保留后缀 N 位
- ``full``：整段替换为 ``<TYPE>`` 占位符

确定性层命中即掩码，**保留前后缀**以满足「可读但不可定位」的脱敏目标。
"""
from __future__ import annotations

import re
import threading
from typing import Any

from gateway.rules.store import RulesLoadError, RulesStore


class PiiEngine:
    """确定性脱敏引擎（满足 Redactor 协议）。"""

    def __init__(self, ruleset_name: str = "pii") -> None:
        self._store = RulesStore(ruleset_name)
        self._lock = threading.Lock()
        self._compiled: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        rules = self._store.load()
        compiled: list[dict[str, Any]] = []
        for entry in rules.get("patterns", []):
            pattern = entry.get("regex")
            if not pattern:
                continue
            try:
                rx = re.compile(pattern)
            except re.error:
                # 单条规则编译失败不影响其余规则（但基线应经测试，不应有坏正则）
                continue
            compiled.append(
                {
                    "type": entry.get("type", "unknown"),
                    "rx": rx,
                    "mask": entry.get("mask", "full"),
                    "keep": int(entry.get("keep", 0) or 0),
                    "prefix": int(entry.get("prefix", 0) or 0),
                    "suffix": int(entry.get("suffix", 0) or 0),
                }
            )
        with self._lock:
            self._compiled = compiled

    def refresh(self) -> None:
        """运行期热更新（07 §2）：重新从 RulesStore 加载并原子重编译。

        失败（如基线临时缺失）时保留当前内存编译结果（last-known-good），
        不抛错、不影响热路径。依赖 RulesStore.refresh 的版本保留语义。
        """
        rules = self._store.refresh()
        compiled: list[dict[str, Any]] = []
        for entry in rules.get("patterns", []):
            pattern = entry.get("regex")
            if not pattern:
                continue
            try:
                rx = re.compile(pattern)
            except re.error:
                continue
            compiled.append(
                {
                    "type": entry.get("type", "unknown"),
                    "rx": rx,
                    "mask": entry.get("mask", "full"),
                    "keep": int(entry.get("keep", 0) or 0),
                    "prefix": int(entry.get("prefix", 0) or 0),
                    "suffix": int(entry.get("suffix", 0) or 0),
                }
            )
        with self._lock:
            self._compiled = compiled

    def start_refresher(self, interval: float) -> None:
        """启动后台规则刷新线程（委托 RulesStore，引擎仅重编译）。"""
        self._store.start_refresher(interval)

    def stop_refresher(self) -> None:
        self._store.stop_refresher()

    def version(self) -> str:
        return self._store.version()

    def redact(self, text: str) -> tuple[str, list[str]]:
        """返回 (脱敏后文本, 命中类型列表)。"""
        if not text:
            return text, []
        with self._lock:
            rules = list(self._compiled)
        hit_types: list[str] = []
        out = text
        for rule in rules:
            rx = rule["rx"]
            for m in rx.finditer(out):
                t = rule["type"]
                if t not in hit_types:
                    hit_types.append(t)
                out = out[: m.start()] + self._mask(rule, m.group(0)) + out[m.end():]
        return out, hit_types

    @staticmethod
    def _mask(rule: dict[str, Any], matched: str) -> str:
        mask = rule["mask"]
        keep = rule["keep"]
        if mask == "keep_prefix" and keep > 0 and len(matched) > keep:
            return matched[:keep] + "*" * (len(matched) - keep)
        if mask == "keep_suffix" and keep > 0 and len(matched) > keep:
            return "*" * (len(matched) - keep) + matched[-keep:]
        if mask == "both":
            pre, suf = rule["prefix"], rule["suffix"]
            if pre + suf < len(matched) and pre >= 0 and suf >= 0:
                return matched[:pre] + "*" * (len(matched) - pre - suf) + matched[-suf:]
        return f"<{rule['type'].upper()}>"
