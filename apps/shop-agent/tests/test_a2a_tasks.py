"""A2A 协议测试 —— 异步任务生命周期 + Webhook 订阅/回调链路。

背景：修复前存在两个缺陷（详见 docs 或 git log）：
  1. 任务完成时只回调任务自带的 callback_url，从不查询订阅表
     → POST /a2a/webhooks 注册的订阅永远不会收到通知（订阅功能空转）
  2. _fire_webhook 的 secret 从未被传入 → X-A2A-Signature 头从不生成
     （WebhookSubscriptionResponse 也没有保存 secret）

第二批修复（单入口路由能力）同样在此锁定：
  3. skill_id 在入口被接收后从未透传 → 「指定 skill」功能空转
  4. 任务状态不回显 skill_id → 列表里看不出任务类型
  5. 无 input-required 态 → 退款等需人工确认的任务被误报为 completed
  6. 产物只有一个 result 字符串 → 对端只能反向解析文本（现补 artifacts）
  7. Agent Card examples 恒为空 → 对端只能靠 description 猜路由

运行：
    cd apps/shop-agent && python -m pytest tests/test_a2a_tasks.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import pytest

from src.modules.chat.core.a2a_task_service import A2ATaskService
from src.modules.chat.core.a2a_webhook_service import A2AWebhookService
from src.modules.chat.schemas import ChatResponse, WebhookSubscriptionResponse


def _make_chat_response(
    message: str = "已完成",
    status: str = "completed",
    **kwargs: Any,
) -> ChatResponse:
    """构造 ChatResponse 桩 —— _execute_agent_chat 现已返回完整响应对象。"""
    return ChatResponse(
        message=message,
        conversation_id="conv_test",
        status=status,
        **kwargs,
    )


# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def task_service() -> A2ATaskService:
    """每个用例使用独立实例，避免单例串扰。"""
    svc = A2ATaskService()
    svc._tasks.clear()
    svc._running_futures.clear()
    return svc


@pytest.fixture
def webhook_service() -> A2AWebhookService:
    """每个用例使用独立实例，避免订阅表串扰。"""
    svc = A2AWebhookService()
    svc._subscriptions.clear()
    return svc


@pytest.fixture
def no_redis(task_service: A2ATaskService):
    """禁用 Redis，强制走内存兜底路径（测试不依赖外部 Redis）。"""
    with patch.object(task_service, "_redis", return_value=None):
        yield


# ────────────────────────────────────────────────────────────────
# 1. 任务生命周期
# ────────────────────────────────────────────────────────────────


class TestTaskLifecycle:
    """任务创建 → 执行 → 完成/失败/取消 的状态流转。"""

    def test_create_task_initial_state(self, task_service: A2ATaskService, no_redis):
        """创建任务后为 pending 状态，且带 task_id / conversation_id。"""
        task = task_service.create_task(message="查询订单物流", domain="ecommerce")

        assert task.task_id.startswith("task_")
        assert task.status == "pending"
        assert task.created_at
        assert task.conversation_id is not None
        assert task.domain == "ecommerce"

    def test_get_task_roundtrip(self, task_service: A2ATaskService, no_redis):
        """创建的任务可被 get_task 读回。"""
        created = task_service.create_task(message="hello")
        fetched = task_service.get_task(created.task_id)

        assert fetched is not None
        assert fetched.task_id == created.task_id

    def test_get_nonexistent_task_returns_none(self, task_service: A2ATaskService, no_redis):
        assert task_service.get_task("task_does_not_exist") is None

    def test_cancel_pending_task(self, task_service: A2ATaskService, no_redis):
        """pending 任务可被取消。"""
        task = task_service.create_task(message="hello")
        ok, msg = task_service.cancel_task(task.task_id)

        assert ok is True
        assert task_service.get_task(task.task_id).status == "cancelled"

    def test_cancel_terminal_task_rejected(self, task_service: A2ATaskService, no_redis):
        """已终态（completed）的任务不可取消 —— 状态机守卫。"""
        task = task_service.create_task(message="hello")
        task.status = "completed"
        task_service._persist_task(task)

        ok, msg = task_service.cancel_task(task.task_id)

        assert ok is False
        assert "终态" in msg

    def test_cancel_nonexistent_task(self, task_service: A2ATaskService, no_redis):
        ok, msg = task_service.cancel_task("task_nope")
        assert ok is False
        assert "不存在" in msg

    def test_list_tasks_pagination(self, task_service: A2ATaskService, no_redis):
        """列表支持分页，total 反映总数。"""
        for i in range(5):
            task_service.create_task(message=f"msg-{i}")

        page1 = task_service.list_tasks(limit=2, offset=0)
        page2 = task_service.list_tasks(limit=2, offset=2)

        assert page1.total == 5
        assert len(page1.tasks) == 2
        assert len(page2.tasks) == 2

    @pytest.mark.asyncio
    async def test_run_task_success_transitions(self, task_service: A2ATaskService, no_redis):
        """成功执行后状态转 completed 且填充 result。"""
        task = task_service.create_task(message="查询订单")

        with patch.object(
            A2ATaskService,
            "_execute_agent_chat",
            new=AsyncMock(return_value=_make_chat_response("物流已到达")),
        ):
            await task_service.run_task(
                task=task,
                message="查询订单",
                domain="ecommerce",
                conversation_id=None,
                skill_id=None,
                context=None,
                callback_url=None,
            )

        final = task_service.get_task(task.task_id)
        assert final.status == "completed"
        assert final.result == "物流已到达"
        assert final.completed_at is not None

    @pytest.mark.asyncio
    async def test_run_task_failure_transitions(self, task_service: A2ATaskService, no_redis):
        """执行异常后状态转 failed 且记录 error。"""
        task = task_service.create_task(message="查询订单")

        with patch.object(
            A2ATaskService, "_execute_agent_chat", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            await task_service.run_task(
                task=task,
                message="查询订单",
                domain="ecommerce",
                conversation_id=None,
                skill_id=None,
                context=None,
                callback_url=None,
            )

        final = task_service.get_task(task.task_id)
        assert final.status == "failed"
        assert "boom" in final.error


# ────────────────────────────────────────────────────────────────
# 2. Webhook 订阅服务
# ────────────────────────────────────────────────────────────────


class TestWebhookSubscription:
    """订阅 CRUD 与事件匹配。"""

    def test_subscribe_default_events(self, webhook_service: A2AWebhookService):
        sub = webhook_service.subscribe(url="https://example.com/hook")

        assert sub.subscription_id.startswith("wh_")
        assert sub.events == ["task.completed", "task.failed"]
        assert sub.expires_at is not None

    def test_subscribe_persists_secret_internally(self, webhook_service: A2AWebhookService):
        """Bug 2 修复点：secret 必须被保存，否则回调时无从签名。"""
        sub = webhook_service.subscribe(
            url="https://example.com/hook",
            secret="my-secret-key",
        )

        # 内部可读
        stored = webhook_service._subscriptions[sub.subscription_id]
        assert stored.secret == "my-secret-key"
        assert sub.has_signature is True

    def test_secret_not_exposed_in_response_dump(self, webhook_service: A2AWebhookService):
        """secret 属于敏感信息，不得出现在对外响应中。"""
        sub = webhook_service.subscribe(url="https://example.com/hook", secret="top-secret")
        dumped = sub.model_dump()

        assert "secret" not in dumped
        assert "top-secret" not in json.dumps(dumped)

    def test_unsubscribe(self, webhook_service: A2AWebhookService):
        sub = webhook_service.subscribe(url="https://example.com/hook")

        assert webhook_service.unsubscribe(sub.subscription_id) is True
        assert webhook_service.unsubscribe(sub.subscription_id) is False

    def test_get_subscribers_filters_by_event(self, webhook_service: A2AWebhookService):
        webhook_service.subscribe(url="https://a.com/hook", events=["task.completed"])
        webhook_service.subscribe(url="https://b.com/hook", events=["task.failed"])

        completed = webhook_service.get_subscribers("task.completed")
        failed = webhook_service.get_subscribers("task.failed")

        assert len(completed) == 1 and completed[0].url == "https://a.com/hook"
        assert len(failed) == 1 and failed[0].url == "https://b.com/hook"

    def test_expired_subscription_excluded(self, webhook_service: A2AWebhookService):
        """过期订阅不参与事件分发。"""
        sub = webhook_service.subscribe(url="https://a.com/hook", ttl_seconds=60)
        # 手动把过期时间改到过去
        webhook_service._subscriptions[sub.subscription_id].expires_at = "2020-01-01T00:00:00+00:00"

        assert webhook_service.get_subscribers("task.completed") == []


# ────────────────────────────────────────────────────────────────
# 3. 事件广播链路（Bug 1 核心修复）
# ────────────────────────────────────────────────────────────────


class TestEventBroadcast:
    """锁定「订阅表 + callback_url」双通道通知。"""

    @staticmethod
    def _make_capture() -> Tuple[Any, List[Dict[str, Any]]]:
        """构造一个记录所有 _fire_webhook 调用的 mock。

        注意：patch.object 替换的是类属性，经 `self._fire_webhook(...)` 调用时
        会绑定 self 作为首参，故 spy 必须声明 self。
        """
        calls: List[Dict[str, Any]] = []

        async def _capture(
            self, url: str, event: str, payload: dict, secret: Optional[str] = None
        ):
            calls.append({"url": url, "event": event, "payload": payload, "secret": secret})

        return _capture, calls

    @pytest.mark.asyncio
    async def test_notify_reaches_subscriber(
        self, task_service: A2ATaskService, webhook_service: A2AWebhookService
    ):
        """Bug 1 核心断言：订阅者必须收到通知（修复前此处为空）。"""
        webhook_service.subscribe(url="https://subscriber.com/hook", events=["task.completed"])
        capture, calls = self._make_capture()

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            return_value=webhook_service,
        ), patch.object(A2ATaskService, "_fire_webhook", new=capture):
            count = await task_service._notify_event(
                event="task.completed", payload={"task_id": "t1"}, callback_url=None
            )

        assert count == 1
        assert calls[0]["url"] == "https://subscriber.com/hook"

    @pytest.mark.asyncio
    async def test_notify_includes_task_own_callback_url(
        self, task_service: A2ATaskService, webhook_service: A2AWebhookService
    ):
        """任务自带的 callback_url 仍被通知 —— 向后兼容。"""
        capture, calls = self._make_capture()

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            return_value=webhook_service,
        ), patch.object(A2ATaskService, "_fire_webhook", new=capture):
            await task_service._notify_event(
                event="task.completed",
                payload={},
                callback_url="https://requester.com/hook",
            )

        assert [c["url"] for c in calls] == ["https://requester.com/hook"]

    @pytest.mark.asyncio
    async def test_notify_both_channels(
        self, task_service: A2ATaskService, webhook_service: A2AWebhookService
    ):
        """订阅者 + 任务自带 URL 都要收到，且不重复。"""
        webhook_service.subscribe(url="https://subscriber.com/hook", events=["task.completed"])
        capture, calls = self._make_capture()

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            return_value=webhook_service,
        ), patch.object(A2ATaskService, "_fire_webhook", new=capture):
            count = await task_service._notify_event(
                event="task.completed",
                payload={},
                callback_url="https://requester.com/hook",
            )

        urls = {c["url"] for c in calls}
        assert count == 2
        assert urls == {"https://subscriber.com/hook", "https://requester.com/hook"}

    @pytest.mark.asyncio
    async def test_notify_no_targets_returns_zero(
        self, task_service: A2ATaskService, webhook_service: A2AWebhookService
    ):
        """无订阅者且无 callback_url 时，不发送任何通知。"""
        capture, calls = self._make_capture()

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            return_value=webhook_service,
        ), patch.object(A2ATaskService, "_fire_webhook", new=capture):
            count = await task_service._notify_event(event="task.completed", payload={})

        assert count == 0
        assert calls == []

    @pytest.mark.asyncio
    async def test_subscriber_secret_is_passed_to_signature(
        self, task_service: A2ATaskService, webhook_service: A2AWebhookService
    ):
        """Bug 2 核心断言：订阅者的 secret 必须传到 _fire_webhook 用于签名。"""
        webhook_service.subscribe(
            url="https://subscriber.com/hook",
            events=["task.completed"],
            secret="sub-secret",
        )
        capture, calls = self._make_capture()

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            return_value=webhook_service,
        ), patch.object(A2ATaskService, "_fire_webhook", new=capture):
            await task_service._notify_event(
                event="task.completed", payload={}, callback_url="https://requester.com/hook"
            )

        by_url = {c["url"]: c["secret"] for c in calls}
        assert by_url["https://subscriber.com/hook"] == "sub-secret"
        assert by_url["https://requester.com/hook"] is None  # 任务自带 URL 无密钥

    @pytest.mark.asyncio
    async def test_subscription_table_failure_degrades_to_callback_url(
        self, task_service: A2ATaskService
    ):
        """订阅表异常时降级：仍回调任务自带 URL，不因订阅表故障丢失通知。"""
        capture, calls = self._make_capture()

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            side_effect=RuntimeError("subscription store down"),
        ), patch.object(A2ATaskService, "_fire_webhook", new=capture):
            count = await task_service._notify_event(
                event="task.completed",
                payload={},
                callback_url="https://requester.com/hook",
            )

        assert count == 1
        assert calls[0]["url"] == "https://requester.com/hook"


# ────────────────────────────────────────────────────────────────
# 4. HMAC 签名（Bug 2 端到端）
# ────────────────────────────────────────────────────────────────


class TestWebhookSignature:
    """验证 X-A2A-Signature 头真实生成且可被接收方校验。"""

    @staticmethod
    def _expected_signature(secret: str, payload: dict) -> str:
        body = json.dumps(payload, ensure_ascii=False)
        return "sha256=" + hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()

    @pytest.mark.asyncio
    async def test_signature_header_generated_when_secret_provided(
        self, task_service: A2ATaskService
    ):
        """有 secret 时必须生成签名头（修复前该分支永远不会进入）。"""
        captured: Dict[str, Any] = {}

        class _FakeResp:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def post(self, url, data=None, headers=None, timeout=None):
                captured.update(url=url, data=data, headers=headers or {})
                return _FakeResp()

        payload = {"task_id": "t1", "status": "completed"}

        with patch("aiohttp.ClientSession", return_value=_FakeSession()):
            await task_service._fire_webhook(
                url="https://subscriber.com/hook",
                event="task.completed",
                payload=payload,
                secret="my-secret",
            )

        assert "X-A2A-Signature" in captured["headers"]
        assert captured["headers"]["X-A2A-Signature"] == self._expected_signature(
            "my-secret", payload
        )
        assert captured["headers"]["X-A2A-Event"] == "task.completed"

    @pytest.mark.asyncio
    async def test_no_signature_header_without_secret(self, task_service: A2ATaskService):
        """无 secret 时不生成签名头（向后兼容旧行为）。"""
        captured: Dict[str, Any] = {}

        class _FakeResp:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def post(self, url, data=None, headers=None, timeout=None):
                captured.update(headers=headers or {})
                return _FakeResp()

        with patch("aiohttp.ClientSession", return_value=_FakeSession()):
            await task_service._fire_webhook(
                url="https://requester.com/hook",
                event="task.completed",
                payload={"task_id": "t1"},
                secret=None,
            )

        assert "X-A2A-Signature" not in captured["headers"]
        assert captured["headers"]["X-A2A-Event"] == "task.completed"

    @pytest.mark.asyncio
    async def test_webhook_failure_does_not_raise(self, task_service: A2ATaskService):
        """回调失败（4xx/网络异常）不得抛出 —— 通知失败不能影响任务本身。"""
        with patch("aiohttp.ClientSession", side_effect=RuntimeError("network down")):
            # 不应抛异常
            await task_service._fire_webhook(
                url="https://down.com/hook",
                event="task.completed",
                payload={},
                secret="s",
            )


# ────────────────────────────────────────────────────────────────
# 5. 端到端：任务完成触发订阅回调
# ────────────────────────────────────────────────────────────────


class TestEndToEndNotification:
    """run_task 完成时，订阅者确实收到通知（这是修复前断裂的链路）。"""

    @pytest.mark.asyncio
    async def test_task_completion_notifies_subscriber(
        self, task_service: A2ATaskService, webhook_service: A2AWebhookService, no_redis
    ):
        webhook_service.subscribe(
            url="https://subscriber.com/hook",
            events=["task.completed"],
            secret="sub-secret",
        )

        sent: List[Dict[str, Any]] = []

        async def _spy(self, url, event, payload, secret=None):
            sent.append({"url": url, "event": event, "secret": secret})

        task = task_service.create_task(message="查询订单")

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            return_value=webhook_service,
        ), patch.object(
            A2ATaskService,
            "_execute_agent_chat",
            new=AsyncMock(return_value=_make_chat_response("已完成")),
        ), patch.object(A2ATaskService, "_fire_webhook", new=_spy):
            await task_service.run_task(
                task=task,
                message="查询订单",
                domain="ecommerce",
                conversation_id=None,
                skill_id=None,
                context=None,
                callback_url=None,
            )

        assert len(sent) == 1
        assert sent[0]["url"] == "https://subscriber.com/hook"
        assert sent[0]["event"] == "task.completed"
        assert sent[0]["secret"] == "sub-secret"

    @pytest.mark.asyncio
    async def test_task_failure_notifies_subscriber(
        self, task_service: A2ATaskService, webhook_service: A2AWebhookService, no_redis
    ):
        """失败事件同样通知订阅者。"""
        webhook_service.subscribe(url="https://subscriber.com/hook", events=["task.failed"])

        sent: List[Dict[str, Any]] = []

        async def _spy(self, url, event, payload, secret=None):
            sent.append({"event": event})

        task = task_service.create_task(message="查询订单")

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            return_value=webhook_service,
        ), patch.object(
            A2ATaskService,
            "_execute_agent_chat",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ), patch.object(A2ATaskService, "_fire_webhook", new=_spy):
            await task_service.run_task(
                task=task,
                message="查询订单",
                domain="ecommerce",
                conversation_id=None,
                skill_id=None,
                context=None,
                callback_url=None,
            )

        assert [s["event"] for s in sent] == ["task.failed"]


# ────────────────────────────────────────────────────────────────
# 6. skill_id 硬路由（Bug 3 / Bug 4）
# ────────────────────────────────────────────────────────────────


def _fake_registry() -> SimpleNamespace:
    """构造 SkillRegistry 桩，避免测试依赖真实 SKILL.md 与 langgraph 导入链。"""
    return SimpleNamespace(
        skills=[
            SimpleNamespace(
                name="query-order",
                params={
                    "order_id": {"type": "string", "required": False},
                    "phone": {"type": "string", "required": False},
                },
            ),
            SimpleNamespace(name="legacy-skill", params={}),  # 无参数契约
        ]
    )


class TestSkillIdRouting:
    """锁定「skill_id 真被透传 + 任务回显」这两条（修复前 skill_id 是死参数）。"""

    def test_create_task_echoes_skill_id(self, task_service: A2ATaskService, no_redis):
        """任务对象必须带 skill_id —— 否则列表里分不清任务类型。"""
        task = task_service.create_task(message="查订单", skill_id="query-order")

        assert task.skill_id == "query-order"
        assert task_service.get_task(task.task_id).skill_id == "query-order"

    @pytest.mark.asyncio
    async def test_skill_id_is_passed_to_chat_request(
        self, task_service: A2ATaskService, no_redis
    ):
        """核心断言：skill_id 必须进入 ChatRequest（修复前被直接丢弃）。"""
        task = task_service.create_task(message="查订单", skill_id="query-order")
        captured: Dict[str, Any] = {}

        async def _capture_request(self, message, domain, conversation_id, skill_id, context):
            captured.update(skill_id=skill_id, context=context)
            return _make_chat_response("订单状态：已发货")

        with patch.object(A2ATaskService, "_execute_agent_chat", new=_capture_request):
            await task_service.run_task(
                task=task,
                message="查订单",
                domain="ecommerce",
                conversation_id=None,
                skill_id="query-order",
                context={"order_id": "WB202405270001"},
                callback_url=None,
            )

        assert captured["skill_id"] == "query-order"
        assert captured["context"] == {"order_id": "WB202405270001"}
        assert task_service.get_task(task.task_id).status == "completed"

    def test_unknown_skill_id_rejected(self):
        """未知 skill_id 必须 400 —— 不能静默退化成意图识别。"""
        from src.modules.chat.a2a_routers import _validate_skill_routing

        with patch(
            "src.modules.chat.agent.skill_loader.get_skill_registry",
            return_value=_fake_registry(),
        ):
            result = _validate_skill_routing("does-not-exist", None)

        assert result is not None
        status_code, detail = result
        assert status_code == 400
        assert "does-not-exist" in detail

    def test_undeclared_context_field_rejected(self):
        """context 里出现 skill 未声明的字段 → 422，不做静默丢弃。

        静默丢弃会让对端以为参数生效了，实际没有 —— 这是最难排查的一类故障。
        """
        from src.modules.chat.a2a_routers import _validate_skill_routing

        with patch(
            "src.modules.chat.agent.skill_loader.get_skill_registry",
            return_value=_fake_registry(),
        ):
            result = _validate_skill_routing("query-order", {"order_id": "WB001", "bogus": 1})

        assert result is not None
        status_code, detail = result
        assert status_code == 422
        assert "bogus" in detail

    def test_declared_context_field_accepted(self):
        """契约内字段放行。"""
        from src.modules.chat.a2a_routers import _validate_skill_routing

        with patch(
            "src.modules.chat.agent.skill_loader.get_skill_registry",
            return_value=_fake_registry(),
        ):
            assert _validate_skill_routing("query-order", {"order_id": "WB001"}) is None
            assert _validate_skill_routing("query-order", None) is None
            # 未指定 skill_id 时不校验 context（走意图识别，参数由抽取流水线产出）
            assert _validate_skill_routing(None, {"anything": 1}) is None

    def test_skill_without_param_contract_skips_context_check(self):
        """skill 没有参数契约时不挡门 —— 无据可依就放行，避免全部调用被拒。"""
        from src.modules.chat.a2a_routers import _validate_skill_routing

        with patch(
            "src.modules.chat.agent.skill_loader.get_skill_registry",
            return_value=_fake_registry(),
        ):
            assert _validate_skill_routing("legacy-skill", {"whatever": 1}) is None


# ────────────────────────────────────────────────────────────────
# 7. artifacts 结构化产物（Bug 6）
# ────────────────────────────────────────────────────────────────


class TestArtifacts:
    """对端 Agent 应能拿到结构化结果，而不是被迫解析自然语言。"""

    @pytest.mark.asyncio
    async def test_completed_task_exposes_result_and_trace(
        self, task_service: A2ATaskService, no_redis
    ):
        task = task_service.create_task(message="查订单")
        chat_resp = _make_chat_response(
            "订单已发货",
            steps=[{"step_name": "工具调用", "step_order": 1, "status": "success"}],
            documents_used=["kb_001"],
        )

        with patch.object(
            A2ATaskService, "_execute_agent_chat", new=AsyncMock(return_value=chat_resp)
        ):
            await task_service.run_task(
                task=task,
                message="查订单",
                domain="ecommerce",
                conversation_id=None,
                skill_id=None,
                context=None,
                callback_url=None,
            )

        final = task_service.get_task(task.task_id)
        by_name = {a.name: a for a in final.artifacts}

        assert set(by_name) == {"result", "trace"}
        assert by_name["result"].mime_type == "text/plain"
        assert by_name["result"].parts[0]["text"] == "订单已发货"
        assert by_name["trace"].mime_type == "application/json"
        trace = by_name["trace"].parts[0]["data"]
        assert trace["documents_used"] == ["kb_001"]
        assert len(trace["steps"]) == 1

    def test_empty_message_produces_no_result_artifact(self):
        """空回复不产生 result 产物（但 trace 始终存在）。"""
        from src.modules.chat.core.a2a_task_service import _build_artifacts

        artifacts = _build_artifacts(_make_chat_response(""))

        assert [a.name for a in artifacts] == ["trace"]


# ────────────────────────────────────────────────────────────────
# 8. input-required 状态机（Bug 5：HITL 链路闭环）
# ────────────────────────────────────────────────────────────────


class TestInputRequired:
    """退款等需人工审批的任务必须停在 input-required，而非误报 completed。"""

    def _run_task_kwargs(self) -> Dict[str, Any]:
        return dict(
            message="申请退款",
            domain="ecommerce",
            conversation_id=None,
            skill_id="request-return",
            context=None,
            callback_url=None,
        )

    @pytest.mark.asyncio
    async def test_waiting_for_confirmation_freezes_task(
        self, task_service: A2ATaskService, no_redis
    ):
        task = task_service.create_task(message="申请退款", skill_id="request-return")
        pending_approval = _make_chat_response(
            "退款申请已生成，等待人工审批",
            status="waiting_for_confirmation",
            interrupt_data={"action": "request_return", "order_id": "WB001", "amount": 99.0},
        )

        with patch.object(
            A2ATaskService, "_execute_agent_chat", new=AsyncMock(return_value=pending_approval)
        ):
            await task_service.run_task(task=task, **self._run_task_kwargs())

        final = task_service.get_task(task.task_id)
        assert final.status == "input-required"
        assert final.completed_at is None  # 非终态
        assert final.interrupt_data["order_id"] == "WB001"
        assert "等待人工确认" in final.message

    @pytest.mark.asyncio
    async def test_input_required_is_not_completed_notification(
        self, task_service: A2ATaskService, webhook_service: A2AWebhookService, no_redis
    ):
        """暂停时广播的是 task.input_required，绝不能是 task.completed
        —— 后者会让对端以为钱已经退了。"""
        sent: List[Dict[str, Any]] = []

        async def _spy(self, url, event, payload, secret=None):
            sent.append({"event": event})

        # 注册一个只关心 input-required 的订阅者，否则没有通知目标可断言
        webhook_service.subscribe(
            url="https://approver.com/hook", events=["task.input_required"]
        )

        task = task_service.create_task(message="申请退款")
        pending_approval = _make_chat_response("等待审批", status="waiting_for_confirmation")

        with patch(
            "src.modules.chat.core.a2a_webhook_service.get_a2a_webhook_service",
            return_value=webhook_service,
        ), patch.object(
            A2ATaskService, "_execute_agent_chat", new=AsyncMock(return_value=pending_approval)
        ), patch.object(A2ATaskService, "_fire_webhook", new=_spy):
            await task_service.run_task(task=task, **self._run_task_kwargs())

        assert [s["event"] for s in sent] == ["task.input_required"]

    def test_input_required_task_is_cancellable(self, task_service: A2ATaskService, no_redis):
        """等待审批 = 尚未产生副作用，应允许取消。"""
        task = task_service.create_task(message="申请退款")
        task.status = "input-required"
        task_service._persist_task(task)

        ok, msg = task_service.cancel_task(task.task_id)

        assert ok is True
        assert task_service.get_task(task.task_id).status == "cancelled"

    @pytest.mark.asyncio
    async def test_resume_rejects_non_resumable_task(
        self, task_service: A2ATaskService, no_redis
    ):
        """终态任务不可恢复 —— 状态机守卫。"""
        task = task_service.create_task(message="申请退款")
        task.status = "completed"
        task_service._persist_task(task)

        ok, msg, _ = await task_service.resume_task(task.task_id)

        assert ok is False
        assert "不可恢复" in msg

    @pytest.mark.asyncio
    async def test_resume_unknown_task(self, task_service: A2ATaskService, no_redis):
        ok, msg, task = await task_service.resume_task("task_nope")

        assert ok is False
        assert "不存在" in msg
        assert task is None

    @pytest.mark.asyncio
    async def test_resume_approved_completes_task(
        self, task_service: A2ATaskService, no_redis
    ):
        """人工批准后任务转 completed，且 artifacts 被刷新。"""
        task = task_service.create_task(message="申请退款", skill_id="request-return")
        task.status = "input-required"
        task.result = "等待审批"
        task_service._persist_task(task)

        approved = _make_chat_response("退款已受理，1-3 个工作日到账", status="completed")

        class _FakeSession:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                return False

        with patch(
            "src.modules.chat.agent.react_agent.ReActAgent.resume_execution",
            new=AsyncMock(return_value=approved),
        ), patch(
            "src.shared.database.get_async_session", return_value=_FakeSession()
        ), patch(
            "src.modules.chat.agent.postgres_approval_store.PostgresApprovalStore",
            return_value=object(),
        ), patch(
            "src.modules.chat.core.tool_registry.ToolService",
            return_value=object(),
        ):
            ok, msg, final = await task_service.resume_task(task.task_id, confirm=True)

        assert ok is True
        assert final.status == "completed"
        assert final.result == "退款已受理，1-3 个工作日到账"
        assert final.completed_at is not None
        assert {a.name for a in final.artifacts} == {"result", "trace"}

    @pytest.mark.asyncio
    async def test_resume_without_interrupt_record_fails(
        self, task_service: A2ATaskService, no_redis
    ):
        """审批记录已过期/被其他节点消费时，任务置 failed 并给出明确原因。"""
        task = task_service.create_task(message="申请退款")
        task.status = "input-required"
        task_service._persist_task(task)

        class _FakeSession:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *args):
                return False

        with patch(
            "src.modules.chat.agent.react_agent.ReActAgent.resume_execution",
            new=AsyncMock(return_value=None),
        ), patch(
            "src.shared.database.get_async_session", return_value=_FakeSession()
        ), patch(
            "src.modules.chat.agent.postgres_approval_store.PostgresApprovalStore",
            return_value=object(),
        ), patch(
            "src.modules.chat.core.tool_registry.ToolService",
            return_value=object(),
        ):
            ok, msg, final = await task_service.resume_task(task.task_id, confirm=True)

        assert ok is False
        assert final.status == "failed"
        assert "审批中断记录" in msg


# ────────────────────────────────────────────────────────────────
# 9. Agent Card examples（Bug 7：对端路由判断依据）
# ────────────────────────────────────────────────────────────────


class TestAgentCardExamples:
    """examples 是对端判断「该不该路由给这个 skill」的依据，不能恒为空。"""

    def test_skill_md_examples_are_parsed(self):
        from src.modules.chat.agent.skill_loader import get_skill_registry

        registry = get_skill_registry()
        by_name = {s.name: s for s in registry.skills}

        assert "query-order" in by_name
        assert by_name["query-order"].examples, "SKILL.md 的 examples 未被解析"

    def test_agent_card_exposes_examples(self):
        from src.modules.chat.core import agent_card

        agent_card._cached_card = None  # 清缓存，确保重新构建
        card = agent_card.build_agent_card()

        assert card.skills, "Agent Card 未暴露任何 skill"
        for skill in card.skills:
            assert skill.examples, f"skill '{skill.id}' 的 examples 为空，对端无法据此路由"
