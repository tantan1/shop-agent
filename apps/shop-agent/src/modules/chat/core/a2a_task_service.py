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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import aiohttp

from src.modules.chat.core.redis_cache_service import get_redis_cache_service
from src.modules.chat.schemas import (
    A2AArtifact,
    A2ATaskListResponse,
    A2ATaskStatusResponse,
)
from src.shared.logger import APILogger

if TYPE_CHECKING:
    from src.modules.chat.schemas import ChatResponse

logger = APILogger("a2a_task")

# ── 任务状态枚举 ──
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_CANCELLED = "cancelled"
# 任务暂停等待外部输入：信息不足需澄清，或高后果动作（退款）需人工审批。
# 非终态 —— 对端可通过 POST /a2a/tasks/{task_id}/input 恢复。
TASK_INPUT_REQUIRED = "input-required"

# 可恢复 / 终态集合（状态机守卫）
_RESUMABLE_STATUSES = (TASK_INPUT_REQUIRED,)
_TERMINAL_STATUSES = (TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED)

def _build_artifacts(chat_response: "ChatResponse") -> List[A2AArtifact]:
    """把 ChatResponse 转成 A2A artifacts，让对端 Agent 拿到结构化结果而非纯文本。

    两类产物：
      - result (text/plain)        ：面向终端用户的自然语言回复
      - trace  (application/json)  ：面向对端 Agent 的执行轨迹（步骤 / 引用文档 / 安全标记）

    此前所有信息被压成一个 result 字符串，对端若要结构化字段只能反向解析文本。
    """
    artifacts: List[A2AArtifact] = []

    message = getattr(chat_response, "message", "") or ""
    if message:
        artifacts.append(
            A2AArtifact(name="result", mime_type="text/plain", parts=[{"kind": "text", "text": message}])
        )

    trace = {
        "status": getattr(chat_response, "status", "completed"),
        "steps": getattr(chat_response, "steps", []) or [],
        "documents_used": getattr(chat_response, "documents_used", []) or [],
        "safety_passed": getattr(chat_response, "safety_passed", True),
        "domain": getattr(chat_response, "domain", "ecommerce"),
    }
    artifacts.append(
        A2AArtifact(name="trace", mime_type="application/json", parts=[{"kind": "data", "data": trace}])
    )
    return artifacts


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
            # 回显 skill_id：GET /a2a/tasks 列表时能看出「这是什么任务」，
            # 此前该字段在入口被接收后即丢弃，任务列表无从分类。
            skill_id=skill_id,
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

        input-required 视为可取消（等待人工确认 = 尚未产生副作用）；
        completed / failed / cancelled 为终态，不可取消。

        Returns:
            (是否成功, 说明消息)
        """
        task = self._load_task(task_id)
        if task is None:
            return False, f"任务 {task_id} 不存在"

        if task.status in _TERMINAL_STATUSES:
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
            chat_response = await self._execute_agent_chat(
                message=message,
                domain=domain,
                conversation_id=conversation_id,
                skill_id=skill_id,
                context=context,
            )

            # ── HITL 分流：等待人工确认时任务进入 input-required，而非 completed ──
            # 修复前：_execute_agent_chat 只返回 message 字符串，
            # ChatResponse.status / interrupt_data 被丢弃 —— 退款这类需审批的动作
            # 在 A2A 通道里会被误报为「已完成」，审批链路断裂。
            if getattr(chat_response, "status", "completed") == "waiting_for_confirmation":
                task.status = TASK_INPUT_REQUIRED
                task.result = chat_response.message
                task.message = "任务已暂停，等待人工确认后再继续执行。"
                task.interrupt_data = chat_response.interrupt_data
                task.artifacts = _build_artifacts(chat_response)
                self._persist_task(task)
                logger.info("A2A 任务暂停等待人工确认", task_id=task_id)

                await self._notify_event(
                    event="task.input_required",
                    payload=task.model_dump(),
                    callback_url=callback_url,
                )
                return

            task.status = TASK_COMPLETED
            task.result = chat_response.message
            task.artifacts = _build_artifacts(chat_response)
            task.completed_at = self._now_iso()
            self._persist_task(task)
            logger.info("A2A 任务完成", task_id=task_id)

            # ── 回调 Webhook：订阅表 + 任务自带 callback_url ──
            await self._notify_event(
                event="task.completed",
                payload=task.model_dump(),
                callback_url=callback_url,
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

            # ── 回调 Webhook：订阅表 + 任务自带 callback_url ──
            await self._notify_event(
                event="task.failed",
                payload=task.model_dump(),
                callback_url=callback_url,
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
    ) -> "ChatResponse":
        """实际执行 Agent 对话（复用 ChatAgentService）。

        返回完整 ChatResponse（而非仅 message 字符串）—— 调用方需要
        status / interrupt_data 才能识别「等待人工确认」这类非终态。

        修复前：skill_id 与 context 在入口被接收后从未透传（构造 ChatRequest 时
        直接省略），等于「指定 skill」功能完全空转，所有任务一律走全局意图识别。
        """
        from src.modules.chat.schemas import ChatRequest
        from src.modules.chat.services import ChatAgentService

        request = ChatRequest(
            message=message,
            conversation_id=conversation_id,
            stream=False,  # A2A 任务统一用非流式
            domain=domain,
            # ── 硬路由：跳过意图识别，把 action 锁定为指定 skill ──
            skill_id=skill_id,
            # ── 结构化入参：入口已按 skill 参数契约校验，此处作为确定性参数注入，
            #    与项目「参数硬强制注入」同思路 —— 对端给的参数不再让模型重新抽取 ──
            context=context,
        )

        # 复用全局 engine（绕过 FastAPI Depends 体系）
        #
        # 修复前：每次任务都 create_async_engine() 且从不 dispose()，
        # 长跑会累积连接池直至耗尽。现改为复用 src.shared.database 的全局 engine，
        # 与项目「单品单例服务模式」一致。
        from src.shared.database import get_async_session

        async with get_async_session() as db:
            service = ChatAgentService(db)
            return await service.chat_with_agent(request)

    # ── 人工确认恢复（input-required → 继续执行）────────────────────────

    async def resume_task(
        self,
        task_id: str,
        confirm: bool = True,
        remark: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[A2ATaskStatusResponse]]:
        """恢复处于 input-required 的任务（人工审批回执）。

        Returns:
            (是否成功, 说明消息, 最新任务状态)
        """
        task = self._load_task(task_id)
        if task is None:
            return False, f"任务 {task_id} 不存在", None

        if task.status not in _RESUMABLE_STATUSES:
            return (
                False,
                f"任务 {task_id} 当前状态为 {task.status}，不可恢复"
                f"（仅 {TASK_INPUT_REQUIRED} 状态可恢复）",
                task,
            )

        from src.modules.chat.agent.react_agent import ReActAgent
        from src.modules.chat.agent.postgres_approval_store import PostgresApprovalStore
        from src.shared.database import get_async_session

        try:
            from src.modules.chat.core.tool_registry import ToolService

            async with get_async_session() as db:
                response = await ReActAgent.resume_execution(
                    thread_id=task.conversation_id or "",
                    confirm=confirm,
                    tool_service=ToolService(),
                    approval_store=PostgresApprovalStore(db),
                )
        except Exception as e:
            logger.error("A2A 任务恢复执行失败", task_id=task_id, error=str(e))
            return False, f"恢复执行失败：{str(e)[:200]}", task

        if response is None:
            task.status = TASK_FAILED
            task.error = "未找到对应的审批中断记录，可能已过期或被其他节点处理"
            task.completed_at = self._now_iso()
            self._persist_task(task)
            return False, task.error, task

        # 拒绝审批 → 终态；批准 → 以恢复执行的结果为准
        task.result = response.message
        task.artifacts = _build_artifacts(response)
        if getattr(response, "status", "completed") == "waiting_for_confirmation":
            task.status = TASK_INPUT_REQUIRED
            task.interrupt_data = response.interrupt_data
        else:
            task.status = TASK_COMPLETED
            task.completed_at = self._now_iso()
            task.interrupt_data = None
        task.message = None
        self._persist_task(task)
        logger.info(
            "A2A 任务已恢复执行", task_id=task_id, confirm=confirm, status=task.status, remark=remark
        )

        await self._notify_event(
            event="task.completed" if task.status == TASK_COMPLETED else "task.input_required",
            payload=task.model_dump(),
            callback_url=callback_url,
        )
        return True, f"任务 {task_id} 已恢复执行", task

    # ── Webhook 回调 ──────────────────────────────────────────────────────

    async def _notify_event(
        self, event: str, payload: dict, callback_url: Optional[str] = None
    ) -> int:
        """向「订阅表中的订阅者 + 任务自带 callback_url」广播事件。

        修复前：任务完成只回调 `callback_url`，从不查询订阅表 —— 导致
        `POST /a2a/webhooks` 注册的订阅永远不会收到通知（订阅功能空转）。

        现在：两者都是通知目标。订阅者携带自己的 secret（若有）用于 HMAC 签名；
        任务自带的 callback_url 无 secret，走无签名分支（保持向后兼容）。

        Returns:
            已投递的通知条数
        """
        from src.modules.chat.core.a2a_webhook_service import get_a2a_webhook_service

        targets: list[tuple[str, Optional[str]]] = []  # (url, secret)

        # 1. 订阅表中的订阅者（携带各自 secret）
        try:
            subs = get_a2a_webhook_service().get_subscribers(event)
            for sub in subs:
                targets.append((sub.url, sub.secret))
        except Exception as e:
            # 注意：不可用 event= 作 kwarg —— structlog 的 event 是保留参数，
            # 传入会与位置参数冲突抛 TypeError
            logger.warning(
                "查询 Webhook 订阅表失败，仅回调任务自带 URL", a2a_event=event, error=str(e)
            )

        # 2. 任务自带的 callback_url（无 secret，向后兼容）
        if callback_url:
            targets.append((callback_url, None))

        if not targets:
            return 0

        for url, secret in targets:
            await self._fire_webhook(url=url, event=event, payload=payload, secret=secret)

        logger.info(
            "A2A 事件已广播",
            a2a_event=event,
            targets=len(targets),
            signed=sum(1 for _, s in targets if s),
        )
        return len(targets)

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
                            a2a_event=event,  # 不用 event=，structlog 保留参数
                            status=resp.status,
                        )
        except Exception as e:
            logger.warning(
                "Webhook 回调异常", url=url, a2a_event=event, error=str(e)
            )  # 不用 event=，structlog 保留参数


def get_a2a_task_service() -> A2ATaskService:
    """获取 A2A 任务服务单例。"""
    return A2ATaskService.get_instance()
