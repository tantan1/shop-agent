"""
A2A Task Service —— 异步任务管理（A2A 协议核心）。

生命周期：
  1. POST /a2a/tasks/send    → 创建任务，返回 task_id（status=pending）
  2. 后台异步执行 Agent 对话
  3. GET  /a2a/tasks/{id}     → 轮询任务状态/结果
  4. POST /a2a/tasks/{id}/cancel → 取消任务

存储：Redis（跨节点可见）+ 内存兜底；运行中的 asyncio.Task 句柄进程本地（不可跨节点序列化）。
Webhook：任务完成后自动回调已注册的 URL
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import aiohttp

from src.modules.chat.core.redis_cache_service import get_redis_cache_service
from src.modules.chat.schemas import (
    A2ATaskListResponse,
    A2ATaskStatusResponse,
)
from src.shared.logger import APILogger

logger = APILogger("a2a_task")

# ── 任务状态枚举 ──
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_CANCELLED = "cancelled"

# ── 最大保留任务数（内存兜底安全）──
_MAX_TASKS = 10000

# ── Redis 键前缀 / 任务状态 TTL ──
_TASK_KEY_PREFIX = "a2a:task:"
_TASK_INDEX_KEY = "a2a:tasks:index"  # ZSET：task_id -> 创建时间 epoch，用于按时间倒序列举
_TASK_TTL = 7 * 24 * 3600  # 任务状态保留 7 天


class A2ATaskService:
    """A2A 异步任务管理器（单例）。

    任务状态外置 Redis，使多实例可跨节点查询任务进度；
    运行中的 asyncio.Task 句柄仍保留在内存（进程本地，无法序列化跨节点）。
    """

    _instance: Optional["A2ATaskService"] = None

    def __init__(self) -> None:
        # 内存兜底（Redis 不可用时）+ 进程本地执行句柄（asyncio.Task 不可跨节点序列化）
        self._tasks: Dict[str, A2ATaskStatusResponse] = {}
        self._running_futures: Dict[str, asyncio.Task] = {}
        self._max_tasks = _MAX_TASKS
        self._task_ttl = _TASK_TTL

    @classmethod
    def get_instance(cls) -> "A2ATaskService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Redis 接入（任务状态外置，跨节点可见）──

    def _redis(self):
        """懒获取 Redis 服务；不可用时返回 None（降级内存）。"""
        try:
            svc = get_redis_cache_service()
            return svc if svc and svc.is_available else None
        except Exception:
            return None

    def _task_key(self, task_id: str) -> str:
        return f"{_TASK_KEY_PREFIX}{task_id}"

    def _persist_task(self, task: A2ATaskStatusResponse) -> None:
        """持久化任务状态到 Redis（不可用时降级内存）。"""
        redis = self._redis()
        if redis is None:
            self._tasks[task.task_id] = task
            return
        try:
            redis.set_json(
                self._task_key(task.task_id),
                task.model_dump(mode="json"),
                ex=self._task_ttl,
            )
            try:
                score = datetime.fromisoformat(task.created_at).timestamp()
            except Exception:
                score = 0.0
            redis.zadd_score(_TASK_INDEX_KEY, {task.task_id: score})
        except Exception as e:
            logger.warning("A2A 任务持久化失败，降级内存", task_id=task.task_id, error=str(e))
            self._tasks[task.task_id] = task

    def _load_task(self, task_id: str) -> Optional[A2ATaskStatusResponse]:
        redis = self._redis()
        if redis is not None:
            data = redis.get_json(self._task_key(task_id))
            if data is not None:
                try:
                    return A2ATaskStatusResponse.model_validate(data)
                except Exception as e:
                    logger.warning("A2A 任务反序列化失败", task_id=task_id, error=str(e))
        return self._tasks.get(task_id)

    # ── CRUD ─────────────────────────────────────────────────────────────

    def create_task(
        self,
        message: str,
        domain: str = "ecommerce",
        conversation_id: Optional[str] = None,
        skill_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
    ) -> A2ATaskStatusResponse:
        """创建任务并返回状态对象（status=pending）。"""
        task_id = f"task_{uuid.uuid4().hex[:16]}"
        now = self._now_iso()

        task = A2ATaskStatusResponse(
            task_id=task_id,
            status=TASK_PENDING,
            created_at=now,
            conversation_id=conversation_id or f"conv_{task_id}",
            domain=domain,
        )

        # 淘汰最旧的任务（内存兜底安全）
        if len(self._tasks) >= self._max_tasks:
            oldest_key = next(iter(self._tasks))
            self._tasks.pop(oldest_key, None)
            self._running_futures.pop(oldest_key, None)

        self._persist_task(task)

        logger.info("A2A 任务已创建", task_id=task_id, domain=domain)
        return task

    def get_task(self, task_id: str) -> Optional[A2ATaskStatusResponse]:
        """获取任务状态（优先 Redis，降级内存）。"""
        return self._load_task(task_id)

    def cancel_task(self, task_id: str) -> Tuple[bool, str]:
        """取消任务。

        Returns:
            (是否成功, 说明消息)
        """
        task = self._load_task(task_id)
        if task is None:
            return False, f"任务 {task_id} 不存在"

        if task.status in (TASK_COMPLETED, TASK_FAILED):
            return False, f"任务 {task_id} 已终态 ({task.status})，无法取消"

        if task.status == TASK_CANCELLED:
            return False, f"任务 {task_id} 已被取消"

        # 取消正在运行的 asyncio Task（仅本进程内有效，跨节点取消需共享调度器）
        future = self._running_futures.pop(task_id, None)
        if future and not future.done():
            future.cancel()

        task.status = TASK_CANCELLED
        task.completed_at = self._now_iso()
        self._persist_task(task)
        logger.info("A2A 任务已取消", task_id=task_id)
        return True, f"任务 {task_id} 已取消"

    def list_tasks(self, limit: int = 50, offset: int = 0) -> A2ATaskListResponse:
        """列出任务（最新优先）。优先 Redis 索引，降级内存。"""
        redis = self._redis()
        if redis is not None:
            ids = redis.zrevrange(_TASK_INDEX_KEY, offset, offset + limit - 1)
            tasks: list[A2ATaskStatusResponse] = []
            stale: list[str] = []
            for tid in ids:
                data = redis.get_json(self._task_key(tid))
                if data is None:
                    stale.append(tid)
                    continue
                try:
                    tasks.append(A2ATaskStatusResponse.model_validate(data))
                except Exception:
                    stale.append(tid)
            if stale:
                redis.zrem(_TASK_INDEX_KEY, *stale)
            total = redis.zcard(_TASK_INDEX_KEY)
            return A2ATaskListResponse(total=total, tasks=tasks)

        all_tasks = sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )
        page = all_tasks[offset : offset + limit]
        return A2ATaskListResponse(
            total=len(all_tasks),
            tasks=page,
        )

    # ── 异步执行 ─────────────────────────────────────────────────────────

    async def run_task(
        self,
        task: A2ATaskStatusResponse,
        message: str,
        domain: str,
        conversation_id: Optional[str],
        skill_id: Optional[str],
        context: Optional[Dict[str, Any]],
        callback_url: Optional[str],
    ) -> None:
        """后台执行 Agent 对话，完成后更新状态 + 回调 webhook。"""
        task_id = task.task_id

        try:
            task.status = TASK_RUNNING
            task.started_at = self._now_iso()
            self._persist_task(task)

            # ── 调用现有的 Agent 对话服务 ──
            result_message = await self._execute_agent_chat(
                message=message,
                domain=domain,
                conversation_id=conversation_id,
                skill_id=skill_id,
                context=context,
            )

            task.status = TASK_COMPLETED
            task.result = result_message
            task.completed_at = self._now_iso()
            self._persist_task(task)
            logger.info("A2A 任务完成", task_id=task_id)

            # ── 回调 Webhook ──
            if callback_url:
                await self._fire_webhook(
                    url=callback_url,
                    event="task.completed",
                    payload=task.model_dump(),
                )

        except asyncio.CancelledError:
            task.status = TASK_CANCELLED
            task.completed_at = self._now_iso()
            self._persist_task(task)
            logger.info("A2A 任务被取消", task_id=task_id)

        except Exception as e:
            task.status = TASK_FAILED
            task.error = str(e)[:500]
            task.completed_at = self._now_iso()
            self._persist_task(task)
            logger.error("A2A 任务失败", task_id=task_id, error=str(e))

            if callback_url:
                await self._fire_webhook(
                    url=callback_url,
                    event="task.failed",
                    payload=task.model_dump(),
                )

        finally:
            self._running_futures.pop(task_id, None)

    async def _execute_agent_chat(
        self,
        message: str,
        domain: str,
        conversation_id: Optional[str],
        skill_id: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """实际执行 Agent 对话（复用 ChatAgentService）。"""
        from src.modules.chat.schemas import ChatRequest
        from src.modules.chat.services import ChatAgentService

        # 构建 ChatRequest
        request = ChatRequest(
            message=message,
            conversation_id=conversation_id,
            stream=False,  # A2A 任务统一用非流式
            domain=domain,
        )

        # 创建 session（绕过 FastAPI Depends 体系）
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        from src.core.config import config as core_config

        db_url = getattr(core_config, "DATABASE_URL", None) or "sqlite+aiosqlite:///./shop_agent.db"
        engine = create_async_engine(db_url, echo=False)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as db:
            service = ChatAgentService(db)
            response = await service.chat_with_agent(request)
            return response.message

    # ── Webhook 回调 ──────────────────────────────────────────────────────

    async def _fire_webhook(
        self, url: str, event: str, payload: dict, secret: Optional[str] = None
    ) -> None:
        """向指定 URL 发送 Webhook 回调。"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json", "X-A2A-Event": event}
                if secret:
                    # 简单 HMAC-SHA256 签名（接收方可验证来源）
                    import hashlib
                    import hmac

                    body = json.dumps(payload, ensure_ascii=False)
                    signature = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                    headers["X-A2A-Signature"] = f"sha256={signature}"
                else:
                    body = json.dumps(payload, ensure_ascii=False)

                async with session.post(
                    url, data=body, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status >= 400:
                        logger.warning(
                            "Webhook 回调失败",
                            url=url,
                            event=event,
                            status=resp.status,
                        )
        except Exception as e:
            logger.warning("Webhook 回调异常", url=url, event=event, error=str(e))


def get_a2a_task_service() -> A2ATaskService:
    """获取 A2A 任务服务单例。"""
    return A2ATaskService.get_instance()
