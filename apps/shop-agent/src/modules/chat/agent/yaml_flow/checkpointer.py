"""Redis 持久化 checkpoint（基于 LangGraph InMemorySaver 的统一存储语义）。

LangGraph 的 ``MemorySaver`` 是纯进程内存储：多副本 / 进程重启后，
human_approval 的 interrupt/resume 执行现场（checkpoint 与 pending writes）即丢失，
跨请求 resume 会失败。

本模块提供 ``RedisCheckpointSaver``：在 ``InMemorySaver``（LangGraph 默认实现，
语义经官方多年打磨）之上叠加一层 Redis 镜像——每次有状态变更（put / put_writes）
都把该线程的 checkpoint 快照写入 Redis，读取前先从 Redis 惰性灌入内存。这样：

- 同一 ``thread_id`` 的挂起/恢复可以在多进程 / 进程重启后跨请求续跑（这是
  Redis 化要解决的核心问题）；
- checkpoint 的版本协商、delta channel 历史、pending writes、blob 去重等
  版本敏感逻辑全部复用 ``InMemorySaver``，不重造轮子、不随 langgraph 升级漂移。

降级策略（与仓库"Redis 不可用 → 主链路照常"一致）：连接失败或不可用时不抛异常，
退化为纯内存（等价于 MemorySaver），只在日志记录。

存储格式：以 ``chat:checkpoint:{thread_id}`` 为 key，pickle 一个
thread 局部的快照（storage / blobs / writes）。pickle 而非 JSON，是因为这些
结构里混有 serde 的二进制 bytes，JSON 无损的二进制序列化会引入不必要的复杂度。
所有快照与本仓库 langgraph 版本强绑定；升级 langgraph 时应清空旧 key。
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import redis
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langchain_core.runnables import RunnableConfig

from src.modules.chat.config import ChatConfig, chat_config
from src.shared.logger import APILogger

logger = APILogger("yaml_flow_checkpointer")

# Redis key 前缀，避免与缓存服务的 chat: 前缀、其它业务键冲突
CHECKPOINT_KEY_PREFIX = "chat:checkpoint:"


class RedisCheckpointSaver(InMemorySaver):
    """以 ``InMemorySaver`` 为执行语义、Redis 为持久后端的 checkpoint saver。

    写入：``put`` / ``put_writes`` 更新内存后写穿到 Redis（pickle 快照）。
    读取：``get_tuple`` 等读取前，先把线程对应的 Redis 快照惰性灌入内存，再走
    ``InMemorySaver`` 的官方读取逻辑。
    删除：删除内存的同时清除 Redis 线程 key。
    降级：Redis 不可用 → 本次读/写退化为纯内存，不阻塞中断恢复（主链路降级）。
    """

    def __init__(
        self,
        *,
        serde: Optional[SerializerProtocol] = None,
        key_prefix: str = CHECKPOINT_KEY_PREFIX,
        redis_config: Optional[ChatConfig] = None,
    ) -> None:
        super().__init__(serde=serde)
        self._key_prefix = key_prefix
        self._cfg = redis_config or chat_config
        self._client: Optional[redis.Redis] = None
        self._redis_ok: Optional[bool] = None  # None=未探测

    # ── Redis 连接管理 ───────────────────────────────────────────────────
    def _client_available(self) -> bool:
        """确保 Redis 客户端可用；探测失败则置为不可用（后续不再重试）。"""
        if self._redis_ok is False:
            return False
        if self._client is None:
            cfg = self._cfg
            try:
                self._client = redis.Redis(
                    host=cfg.redis_host or "localhost",
                    port=cfg.redis_port or 6379,
                    password=os.environ.get("REDIS_AUTH")
                    or getattr(cfg, "redis_password", "")
                    or None,
                    db=0,
                    decode_responses=False,  # checkpoint 数据为二进制
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                self._client.ping()
                self._redis_ok = True
                logger.info("RedisCheckpointSaver: Redis 连接成功，启用持久 checkpoint")
            except redis.RedisError as e:
                logger.warning(
                    f"RedisCheckpointSaver: Redis 不可用，降级为内存 checkpoint: {e}"
                )
                self._redis_ok = False
                self._client = None
        return self._redis_ok is True

    def _thread_key(self, thread_id: str) -> str:
        return f"{self._key_prefix}{thread_id}"

    def _persist(self, thread_id: str) -> None:
        """把该线程的存储快照 pickle 后写入 Redis（写穿）。"""
        if not self._client_available():
            return
        try:
            snapshot = {
                "storage": self.storage.get(thread_id, {}),
                "writes": {k: v for k, v in self.writes.items() if k[0] == thread_id},
                "blobs": {k: v for k, v in self.blobs.items() if k[0] == thread_id},
            }
            self._client.set(self._thread_key(thread_id), pickle.dumps(snapshot))
        except redis.RedisError as e:
            logger.warning(f"RedisCheckpointSaver: {thread_id} 写入失败: {e}")

    def _ensure_loaded(self, thread_id: str) -> None:
        """读取前把该线程的 Redis 快照灌入内存（仅首次，避免重放）。"""
        if not self._client_available():
            return
        # 内存里已有该线程的 checkpoint 数据则不重放
        if thread_id in self.storage:
            return
        try:
            raw = self._client.get(self._thread_key(thread_id))
            if not raw:
                return
            snap: Dict[str, Any] = pickle.loads(raw)
            for ns, cp_map in snap.get("storage", {}).items():
                self.storage[thread_id][ns].update(cp_map)
            for k, v in snap.get("writes", {}).items():
                self.writes[k] = v
            for k, v in snap.get("blobs", {}).items():
                self.blobs[k] = v
        except (redis.RedisError, pickle.UnpicklingError, ValueError) as e:
            logger.warning(f"RedisCheckpointSaver: {thread_id} 加载失败，重建快照: {e}")

    def _clear_redis(self, thread_id: str) -> None:
        if not self._client_available():
            return
        try:
            self._client.delete(self._thread_key(thread_id))
        except redis.RedisError as e:
            logger.warning(f"RedisCheckpointSaver: 删除 {thread_id} 失败: {e}")

    # ── 写穿 · 镜像 InMemorySaver 官方写方法 ─────────────────────────────
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        result = super().put(config, checkpoint, metadata, new_versions)
        self._persist(thread_id)
        return result

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Any,
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        super().put_writes(config, writes, task_id, task_path)
        self._persist(thread_id)

    # ── 读 · 先灌入再走 InMemorySaver 官方读 ────────────────────────────
    def get_tuple(self, config: RunnableConfig):
        thread_id = config["configurable"]["thread_id"]
        self._ensure_loaded(thread_id)
        return super().get_tuple(config)

    def get_delta_channel_history(self, *, config: RunnableConfig, channels):
        thread_id = config["configurable"]["thread_id"]
        self._ensure_loaded(thread_id)
        return super().get_delta_channel_history(config=config, channels=channels)

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        self._clear_redis(thread_id)