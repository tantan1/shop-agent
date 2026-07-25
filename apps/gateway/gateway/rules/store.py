"""规则存储层（三级来源，G2 门②）。

RulesStore 按「规则集名」参数化（如 ``"pii"`` / ``"injection"``），每个实例
独立持有一套规则，互不影响。三级来源（加载优先级从高到低，即先试高优，
失败回退低优）：

1. **磁盘快照** ``<name>.snapshot.json``：运行期刷新成功即落盘，作为 last-known-good。
2. **构建基线** ``<name>.baseline.yaml``：随包分发，生产默认规则（非 example）。
3. **中央源** ``<name>_RULES_URL`` env 指定：仅异步刷新预留，本批不实现拉取。

**绝不零规则启动**（G2 硬纪律）：基线随包存在，结构上排除 fail-open 静默。
若基线文件也缺失（打包错误），``load()`` 抛 ``RulesLoadError``，由调用方决定
如何 fail（默认接入方会触发 fail-closed，绝不带着空规则放行）。

热路径零联网：``load()`` 在启动期/后台调用，``get_rules()`` 返回已加载的
内存副本，热路径只读数不触网。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from gateway.config import settings


class RulesLoadError(Exception):
    """规则加载失败（基线缺失/解析错误/中央源不可达）。

    调用方应将其转为 GovernanceError 上抛，避免带着空规则静默放行。
    """


# 包内规则目录（相对本文件）：rules/ 公共包与 pii/ 包同级（gateway/ 下）
_RULES_DIR = Path(__file__).resolve().parent
_PII_BASELINE = Path(__file__).resolve().parent.parent / "pii" / "pii.baseline.yaml"


def _baseline_path(name: str) -> Path:
    """规则集 name 的随包基线文件。

    - ``pii``     → ``pii/pii.baseline.yaml``
    - ``injection`` → ``rules/injection.baseline.yaml``
    其余 name 默认落在 ``rules/<name>.baseline.yaml``。
    """
    if name == "pii":
        return _PII_BASELINE
    return _RULES_DIR / f"{name}.baseline.yaml"


def _snapshot_path(name: str) -> Path:
    return _RULES_DIR / f"{name}.snapshot.json"


class RulesStore:
    """参数化规则存储（按规则集名隔离）。

    用法::

        store = RulesStore("pii")
        store.load()                       # 启动期调用
        rules = store.get_rules()          # 热路径只读
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._rules: dict[str, Any] | None = None
        self._version: str = ""
        self._refresher: threading.Thread | None = None
        self._stop = threading.Event()

    # --- 公共 API（热路径只读） ---

    def get_rules(self) -> dict[str, Any]:
        """返回已加载的规则（内存副本）。未 load 或加载失败抛 RulesLoadError。"""
        with self._lock:
            if self._rules is None:
                raise RulesLoadError(f"ruleset {self.name}: not loaded")
            return self._rules

    def is_loaded(self) -> bool:
        with self._lock:
            return self._rules is not None

    def version(self) -> str:
        """当前生效规则版本（基线内容哈希，短签）。未加载返回空串。"""
        with self._lock:
            return self._version

    def stop_refresher(self) -> None:
        """停止后台刷新线程（进程退出时调用）。"""
        self._stop.set()
        if self._refresher is not None:
            self._refresher.join(timeout=2.0)
            self._refresher = None

    # --- 加载（启动期/后台调用，非热路径） ---

    def load(self) -> dict[str, Any]:
        """三级来源加载：基线（随包权威）→ 快照（兜底）→ 中央源（预留）。

        启动期**基线优先**（随包基线代表运维意图，打包更新立即生效）；
        快照仅作基线缺失时的 last-known-good 兜底（运行期中央源刷新成功后亦可写）。
        基线缺失且快照也无 → 抛 RulesLoadError（绝不零规则启动）。

        版本化（07 §5）：每次成功提交计算基线内容哈希作为版本号；
        若加载失败且已有生效版本，保留旧版本（last-known-good 真正生效），
        不抛错——只在「从未成功加载过」时才抛 RulesLoadError。
        """
        # 1) 构建基线（随包默认，启动期权威）
        baseline = self._try_read_baseline()
        if baseline is not None:
            self._commit(baseline)
            return baseline

        # 2) 磁盘快照（基线缺失时的 last-known-good 兜底）
        snap = self._try_read_snapshot()
        if snap is not None:
            self._commit(snap)
            return snap

        # 3) 中央源（本批仅结构预留，不实现拉取）
        # url = self._central_url()
        # if url: ... 本批不实现

        # 若有已生效版本（运行期刷新失败），保留 last-known-good，不抛
        with self._lock:
            if self._rules is not None:
                return self._rules

        # 基线缺失 = 打包错误，且从未成功加载过 → 绝不空跑
        raise RulesLoadError(
            f"ruleset {self.name}: no baseline found at {_baseline_path(self.name)}"
        )

    def refresh(self) -> dict[str, Any]:
        """运行期热刷新（07 §2 异步刷新闭环）。

        重新走三级来源加载；成功则原子提交新版本，失败则保留当前内存版本
        （last-known-good），不破坏热路径、不抛错。返回当前生效规则。
        无已加载版本时退化等同于 ``load()``（保底绝不零规则启动）。
        """
        try:
            rules = self._load_once()
        except RulesLoadError:
            with self._lock:
                if self._rules is not None:
                    return self._rules  # 刷新失败，保留旧版本
            raise  # 从未加载过，向上传递 fail-closed
        self._commit(rules)
        return rules

    def _load_once(self) -> dict[str, Any]:
        """单次加载尝试（不含 last-known-good 兜底判定），供 load/refresh 共用。"""
        baseline = self._try_read_baseline()
        if baseline is not None:
            return baseline
        snap = self._try_read_snapshot()
        if snap is not None:
            return snap
        raise RulesLoadError(
            f"ruleset {self.name}: no baseline found at {_baseline_path(self.name)}"
        )

    def start_refresher(self, interval: float) -> None:
        """启动后台刷新线程（07 §2）。

        interval<=0 不启动。线程定时调用 ``refresh()``（失败静默保留旧版本），
        热路径（``get_rules``）零影响。仅基线/快照文件变更后下一轮自动生效，
        无需重启网关。中央源拉取仍是结构预留，不触网。
        """
        if interval <= 0:
            return
        if self._refresher is not None and self._refresher.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                self._stop.wait(interval)
                if self._stop.is_set():
                    break
                try:
                    self.refresh()
                except RulesLoadError:
                    # 从未加载过才会在 refresh 内抛；保底保留，不致命
                    continue

        self._refresher = threading.Thread(target=_loop, name=f"rules-refresh-{self.name}", daemon=True)
        self._refresher.start()

    # --- 内部 ---

    def _commit(self, rules: dict[str, Any]) -> None:
        version = _stable_hash(rules)
        with self._lock:
            self._rules = rules
            self._version = version
        # 落盘快照作为 last-known-good：基线缺失/刷新失败时兜底（07 §2）
        self._write_snapshot(rules)

    def _try_read_snapshot(self) -> dict[str, Any] | None:
        path = _snapshot_path(self.name)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except (json.JSONDecodeError, OSError):
            return None
        return None

    def _try_read_baseline(self) -> dict[str, Any] | None:
        if self.name == "pii" and settings.pii_rules_path:
            path = Path(settings.pii_rules_path)
            # settings 默认值是相对 cwd 的占位；若文件不存在（如从仓库根运行），
            # 回退到包内绝对基线（与 _baseline_path 对称），避免误判基线缺失。
            if not path.exists():
                path = _baseline_path(self.name)
        elif self.name == "injection" and settings.injection_rules_path:
            path = Path(settings.injection_rules_path)
            if not path.exists():
                path = _baseline_path(self.name)
        else:
            path = _baseline_path(self.name)
        if not path.exists():
            return None
        try:
            import yaml  # 延迟导入：仅在加载期用，不在热路径
        except ImportError:
            # 无 yaml 依赖时退回极简解析（基线文件用最简结构）
            return self._read_baseline_minimal(path)
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001 - 基线解析失败回退
            return None
        return None

    @staticmethod
    def _read_baseline_minimal(path: Path) -> dict[str, Any] | None:
        """无 PyYAML 时的保底解析：支持 ``- item`` 嵌套 dict 列表。

        基线文件结构极简（patterns: 列表，每项是 dict），避免强依赖 yaml。
        """
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        out: dict[str, Any] = {}
        current_list: list[dict[str, str]] | None = None
        current_item: dict[str, str] | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip())
            if line.endswith(":") and indent == 0:
                key = line[:-1].strip()
                out[key] = []
                current_list = out[key]
                current_item = None
            elif line.startswith("- ") and current_list is not None:
                current_item = {}
                current_list.append(current_item)
                kv = line[2:].strip()
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    current_item[k.strip()] = v.strip().strip("'\"")
            elif ":" in line and current_item is not None:
                k, v = line.split(":", 1)
                current_item[k.strip()] = v.strip().strip("'\"")
            elif ":" in line and current_list is None:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip("'\"")
        return out if out else None

    def _write_snapshot(self, rules: dict[str, Any]) -> None:
        path = _snapshot_path(self.name)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(rules, f, ensure_ascii=False, indent=2)
        except OSError:
            # 快照写失败不致命（仅失去 last-known-good），不影响本次加载
            pass


def _stable_hash(rules: dict[str, Any]) -> str:
    """规则内容稳定哈希（版本号）。排序键保证同内容同版本，便于版本比对。"""
    payload = json.dumps(rules, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
