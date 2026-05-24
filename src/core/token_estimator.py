"""
Qwen3 Token 预估器 — 基于 HuggingFace tokenizers Rust 引擎。

零 torch 依赖，仅需 tokenizer.json 文件（~10MB），微秒级估算。
Qwen3 全系列（0.6B/1.7B/3.6B/flash/plus）共用同一套 tokenizer。

用法：
    from src.core.token_estimator import get_token_estimator

    estimator = get_token_estimator()
    tokens = estimator.estimate("你好，我想买一台笔记本电脑")
    # => ~12 tokens
"""

import os
import threading
from functools import lru_cache
from typing import List, Dict, Optional, Callable

from src.shared.logger import APILogger

logger = APILogger("token_estimator")

# ── 模块级状态：由 TokenEstimator 加载后注入 ──
# LRU 缓存是模块级函数（脱离实例，避免 self 引用导致内存泄漏），
# 实际的 encode 逻辑通过 _encode_fn 闭包变量注入。
#
# 为什么用模块级函数：
#   1. lru_cache 挂类方法时 self 无法被 hash，且缓存持有 self → 内存泄漏
#   2. system prompt / 聊天历史在多轮对话中重复出现，缓存命中率可 >80%
#   3. Qwen3 tokenizer 是确定性纯函数，相同输入 × 相同输出

_encode_fn: Optional[Callable] = None
"""由 TokenEstimator._ensure_loaded() 注入。签名为 (text: str) -> tokenizers.Encoding。"""


@lru_cache(maxsize=8192)
def _encode_text_cached(text: str) -> int:
    """纯函数：BPE 编码 text 并返回 token 数。不持有实例引用。"""
    if _encode_fn is None:
        return 0
    return len(_encode_fn(text).ids)


class TokenEstimator:
    """基于 HuggingFace tokenizers 的 Qwen3 轻量 Token 预估器。

    只加载 tokenizer.json（BPE 词表），不碰任何模型权重、config、torch。
    精度 = AutoTokenizer（100% Qwen 原生），体积 ≈ tiktoken。

    特性：
    - 线程安全（双检锁）
    - 懒加载（首次调用时自动加载）
    - 加载失败自动降级到字符估算
    - LRU 缓存（相同文本不重复编码，system prompt / 聊天历史命中率 80%+）
    """

    _instance: Optional["TokenEstimator"] = None
    _lock = threading.Lock()

    def __init__(self, tokenizer_path: str = "./models/Qwen3-1.7B/tokenizer.json"):
        self._tokenizer_path = tokenizer_path
        self._tokenizer = None   # tokenizers.Tokenizer 实例
        self._loaded = False
        self._load_error = None
        self._init_lock = threading.Lock()

    # ── 单例 ──────────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "TokenEstimator":
        """获取全局单例。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        from src.core.config import config
                        path = config.TOKENIZER_PATH
                    except Exception:
                        path = "./models/Qwen3-1.7B/tokenizer.json"
                    cls._instance = cls(tokenizer_path=path)
        return cls._instance

    # ── 加载 ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """确保 tokenizer 已加载（线程安全，懒加载）。"""
        global _encode_fn, _encode_text_cached
        if self._loaded:
            return True
        with self._init_lock:
            if self._loaded:  # 双重检查
                return True
            try:
                from tokenizers import Tokenizer

                path = self._tokenizer_path
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"tokenizer.json 不存在: {path}")

                tk = Tokenizer.from_file(path)
                self._tokenizer = tk
                self._loaded = True

                # ── 注入模块级 encode 函数，清空 LRU ──
                _encode_fn = tk.encode
                _encode_text_cached.cache_clear()

                # 验证
                test_ids = tk.encode("你好").ids
                logger.info(
                    "Token 预估器加载成功",
                    path=path,
                    vocab_size=tk.get_vocab_size(),
                    test="你好",
                    test_tokens=len(test_ids),
                    cache="LRU:8192",
                )
                return True

            except FileNotFoundError as e:
                self._load_error = str(e)
                logger.warning(
                    "Token 预估器: tokenizer.json 未找到，降级为字符估算",
                    path=self._tokenizer_path,
                )
                return False
            except ImportError:
                self._load_error = "tokenizers 库未安装 (pip install tokenizers)"
                logger.warning("Token 预估器: tokenizers 未安装，降级为字符估算")
                return False
            except Exception as e:
                self._load_error = str(e)
                logger.warning(
                    "Token 预估器加载失败，降级为字符估算",
                    error=str(e),
                )
                return False

    # ── 主方法 ────────────────────────────────────────────────────────

    def estimate(self, text: str) -> int:
        """预估文本的 token 数（含 LRU 缓存）。

        相同文本第二次调用时直接返回缓存结果，不重新做 BPE 编码。

        Args:
            text: 输入文本（中文/英文/混合）

        Returns:
            token 数量
        """
        if not text:
            return 0

        # 尝试用真实 tokenizer（带 LRU 缓存）
        if self._ensure_loaded() and self._tokenizer is not None:
            try:
                return _encode_text_cached(text)
            except Exception:
                pass  # 异常时降级

        # 降级：字符估算（中文 ≈ 1.5 token/字，英文 ≈ 0.3 token/字）
        return _estimate_by_chars(text)

    def estimate_messages(self, messages: List[Dict[str, str]]) -> int:
        """预估消息列表的总 token 数（含 role 前缀开销）。

        每条消息的 role 和 content 分别走 estimate() → 享受 LRU 缓存。
        例如 system prompt 在 100 次请求中只编码 1 次。

        Args:
            messages: [{"role": "user", "content": "..."}, ...]

        Returns:
            预估 token 数（含 role 开销，约 +4 token/条）
        """
        total = 0
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            total += self.estimate(role) + self.estimate(content) + 4
        return total

    def estimate_system_prompt(self, system: str, user_content: str) -> int:
        """预估 system + user 的总 token 数。"""
        return (self.estimate(system) + self.estimate(user_content) + 12)

    # ── 缓存统计 ──────────────────────────────────────────────────────

    def cache_info(self) -> Dict:
        """返回 LRU 缓存的命中率等统计信息。"""
        info = _encode_text_cached.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "maxsize": info.maxsize,
            "currsize": info.currsize,
            "hit_rate": f"{info.hits / max(info.hits + info.misses, 1) * 100:.1f}%",
        }

    def clear_cache(self):
        """手动清空 LRU 缓存（通常不需要，tokenizer 重载时自动清）。"""
        _encode_text_cached.cache_clear()

    # ── 状态查询 ──────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return 0

    # ── 输入截断 ──────────────────────────────────────────────────────

    def truncate_to_tokens(
        self, text: str, max_tokens: int, strategy: str = "keep_both_ends"
    ) -> tuple[str, int, int]:
        """将文本截断到 max_tokens 以内，返回 (截断后文本, 原始token数, 截断后token数)。

        三种截断策略（电商客服场景推荐 keep_both_ends）：
        - keep_both_ends: 保留首 60% + 尾 20%，中间插入省略标记
        - keep_start_only: 仅保留开头（RAG/意图识别兼容性最好）
        - keep_end_only: 仅保留结尾

        Args:
            text: 原始文本
            max_tokens: 最大 token 数（含省略标记的开销）
            strategy: 截断策略

        Returns:
            (截断后文本, 原始token数, 截断后token数)
        """
        if not text:
            return text, 0, 0

        original_tokens = self.estimate(text)
        if original_tokens <= max_tokens:
            return text, original_tokens, original_tokens

        # 省略标记的 token 开销（预留）
        OMISSION_MARKER = "…[内容过长，已自动截断核心部分]…"
        omission_tokens = self.estimate(OMISSION_MARKER)
        usable_tokens = max_tokens - omission_tokens

        if usable_tokens <= 0:
            # 极端情况：max_tokens 太小，只能放省略标记
            logger.warning(
                "max_tokens 过小无法保留有效内容",
                max_tokens=max_tokens,
                original_tokens=original_tokens,
            )
            return OMISSION_MARKER, original_tokens, omission_tokens

        # 用字符比例近似定位截断点，再用 tokenizer 精确校准
        # 中文约 0.65 token/字，英文约 3 token/字，综合取 1.0 token/字（保守估计）
        _chars_per_token = 1.0

        if strategy == "keep_start_only":
            rough_chars = int(usable_tokens * _chars_per_token)
            head = _truncate_text_to_max_tokens(
                text, usable_tokens, self.estimate
            )
            result = head + OMISSION_MARKER
        elif strategy == "keep_end_only":
            tail_chars = int(usable_tokens * _chars_per_token)
            tail = text[-tail_chars:] if tail_chars < len(text) else text
            tail = _truncate_text_to_max_tokens(tail, usable_tokens, self.estimate)
            result = OMISSION_MARKER + tail
        else:  # keep_both_ends（默认）
            head_ratio = 0.60
            tail_ratio = 0.20
            head_token_budget = int(usable_tokens * head_ratio)
            tail_token_budget = int(usable_tokens * tail_ratio)

            # 截取头部（按字符数粗切，再用 tokenizer 精确校准）
            head_char_target = int(head_token_budget * _chars_per_token)
            head_text = text[:head_char_target] if head_char_target < len(text) else text
            head_text = _truncate_text_to_max_tokens(head_text, head_token_budget, self.estimate)

            # 截取尾部
            tail_char_target = int(tail_token_budget * _chars_per_token)
            tail_text = text[-tail_char_target:] if tail_char_target < len(text) else text
            tail_text = _truncate_text_to_max_tokens(tail_text, tail_token_budget, self.estimate)

            result = head_text + OMISSION_MARKER + tail_text

        truncated_tokens = self.estimate(result)
        logger.info(
            "用户输入过长，已自动截断",
            strategy=strategy,
            original_tokens=original_tokens,
            truncated_tokens=truncated_tokens,
            max_tokens=max_tokens,
            original_preview=text[:40],
            truncated_preview=result[:40],
        )
        return result, original_tokens, truncated_tokens


# ── 辅助函数 ──────────────────────────────────────────────────────────

def _truncate_text_to_max_tokens(
    text: str, max_tokens: int, estimate_fn
) -> str:
    """逐句截断文本直到 token 数 ≤ max_tokens。

    使用句子边界（。！？\n）分割，从末尾逐句删除，确保截断在语义边界上。
    如果逐句仍然超出，回退到字符级截断。
    """
    if estimate_fn(text) <= max_tokens:
        return text

    # 按句子边界分割
    import re
    sentences = re.split(r'(?<=[。！？\n])\s*', text)
    # 从末尾逐句删除
    for i in range(len(sentences) - 1, -1, -1):
        truncated = ''.join(sentences[:i])
        if estimate_fn(truncated) <= max_tokens:
            return truncated

    # 逐句无法满足，回退到字符级
    chars_per_token = 0.65  # 中文保守估计
    target_chars = int(max_tokens * chars_per_token)
    if target_chars < len(text):
        return text[:target_chars]
    return text


# ── 字符降级估算（提取为纯函数供 document_service 等复用）───────────

def _estimate_by_chars(text: str) -> int:
    """字符级 token 估算（去掉了 document_service 中的重复实现）。
    中文 ≈ 1.5 token/字，英文/ASCII ≈ 0.3 token/字。
    """
    if not text:
        return 0
    chinese = 0
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            chinese += 1
    other = len(text) - chinese
    return int(chinese * 1.5 + other * 0.3)


# ── 全局便捷函数 ──

def get_token_estimator() -> TokenEstimator:
    """获取全局 TokenEstimator 单例。"""
    return TokenEstimator.get_instance()


def estimate_tokens(text: str) -> int:
    """快捷函数：预估文本 token 数。"""
    return get_token_estimator().estimate(text)
