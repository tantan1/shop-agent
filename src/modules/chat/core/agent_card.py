"""
Agent Card Builder —— 从 SkillRegistry + Config 动态生成 A2A Agent Card。

职责：
  GET /.well-known/agent-card.json  → 返回标准 Agent Card JSON
  GET /api/v1/chatagent/agent/card → 同上（带 API 前缀版本）

数据来源：
  - agent 元信息  → config (AGENT_NAME, AGENT_DESCRIPTION, AGENT_URL 等)
  - skills 列表   → SkillRegistry (skills/ 目录下的 SKILL.md)
  - capabilities  → 硬编码 + config 动态检测
  - endpoints     → 从 FastAPI app 路由自动发现
"""
from __future__ import annotations

from datetime import datetime
from typing import List, TYPE_CHECKING

from src.core.config import config
from src.shared.logger import APILogger

if TYPE_CHECKING:
    from src.modules.chat.schemas import (
        AgentCard, AgentSkill, AgentCapabilities,
        AgentAuth, AgentRateLimitInfo, AgentEndpoint,
    )

logger = APILogger("agent_card")

# ── Agent Card 默认值（config 无对应字段时的 fallback） ──
_AGENT_NAME = "Shop-Agent Orchestrator"
_AGENT_DESC = (
    "电商客服多 Agent 系统，支持订单查询、物流追踪、退货退款、余额查询、优惠券查询等场景。"
    "内置意图识别流水线（P0/P1/P2）、ReAct 推理编排、纠纷协调多 Agent 协作、人在回路审批。"
)
_AGENT_URL_DEFAULT = "http://localhost:8000"

# ── 内置缓存：首次 build 后缓存结果，避免每次请求都触发 skill_loader → langgraph 导入链 ──
_cached_card: "AgentCard | None" = None

# ── 端点声明（A2A 协议增强）──
_A2A_ENDPOINTS: List[dict] = [
    {"method": "GET",  "path": "/.well-known/agent-card.json", "identifier": "discovery",        "description": "A2A Agent 能力发现",          "requires_auth": False},
    {"method": "GET",  "path": "/a2a/health",                  "identifier": "a2a_health",        "description": "A2A 专用健康检查",            "requires_auth": False},
    {"method": "POST", "path": "/a2a/tasks/send",              "identifier": "task_send",          "description": "提交异步 Agent 任务",          "requires_auth": True},
    {"method": "GET",  "path": "/a2a/tasks/{task_id}",         "identifier": "task_get",           "description": "查询任务状态/结果",            "requires_auth": True},
    {"method": "POST", "path": "/a2a/tasks/{task_id}/cancel",  "identifier": "task_cancel",        "description": "取消进行中的任务",             "requires_auth": True},
    {"method": "GET",  "path": "/a2a/tasks",                   "identifier": "task_list",          "description": "列出所有任务",                 "requires_auth": True},
    {"method": "POST", "path": "/a2a/webhooks",                "identifier": "webhook_subscribe",  "description": "注册 Webhook 回调订阅",       "requires_auth": True},
    {"method": "DELETE","path": "/a2a/webhooks/{subscription_id}","identifier": "webhook_unsubscribe","description": "取消 Webhook 订阅",         "requires_auth": True},
    {"method": "GET",  "path": "/a2a/conversations",           "identifier": "conv_list",          "description": "列出对话（多 Agent 上下文共享）","requires_auth": True},
    {"method": "GET",  "path": "/a2a/conversations/{conversation_id}/messages","identifier": "conv_messages","description": "获取对话历史消息",   "requires_auth": True},
    {"method": "POST", "path": "/mcp",                         "identifier": "mcp_jsonrpc",       "description": "MCP JSON-RPC (tools/list, tools/call)", "requires_auth": False},
]


def _resolve_agent_url() -> str:
    """推断 Agent 公开访问地址。"""
    host = getattr(config, "FASTMCP_HOST", "127.0.0.1") or "127.0.0.1"
    port = getattr(config, "FASTMCP_PORT", 8000) or 8000
    return f"http://{host}:{port}"


def build_agent_card() -> "AgentCard":
    """从 SkillRegistry + Config 动态生成增强版 Agent Card（带内存缓存）。

    核心原则：Agent Card 的数据源是 SkillRegistry（与 MCP tools/list 同源），
    保证 /agent/card 和 /mcp 的 tools/list 返回的工具描述一致。

    增强字段（A2A 推荐）：
      - authentication:  认证方式声明（api_key_header）
      - rate_limit:      速率限制信息
      - endpoints:       所有可用端点列表

    首次调用触发 skill_loader → langgraph 导入链（~5s），结果缓存后后续调用 <1ms。
    """
    global _cached_card
    if _cached_card is not None:
        return _cached_card

    from src.modules.chat.schemas import (
        AgentCard, AgentSkill, AgentCapabilities,
        AgentAuth, AgentRateLimitInfo, AgentEndpoint,
    )

    # ── 1. 元信息 ──
    name = getattr(config, "AGENT_NAME", "") or _AGENT_NAME
    description = getattr(config, "AGENT_DESCRIPTION", "") or _AGENT_DESC
    url = _resolve_agent_url()

    # ── 2. Skills（从 SkillRegistry 动态获取，与 MCP tools/list 同源） ──
    skills: List[AgentSkill] = []
    try:
        from src.modules.chat.agent.skill_loader import get_skill_registry
        registry = get_skill_registry()
        for skill in registry.skills:
            skills.append(AgentSkill(
                id=skill.name,
                name=skill.display_name or skill.name,
                description=skill.description,
                tags=skill.tags,
                examples=[],  # SKILL.md 暂无 example 字段，可后续扩展
            ))
    except Exception as e:
        logger.warning(f"SkillRegistry 加载失败，Agent Card skills 为空: {e}")

    # ── 3. Capabilities（动态检测 + 硬编码） ──
    mcp_enabled = getattr(config, "MCP_ENABLED", False)
    capabilities = AgentCapabilities(
        streaming=True,
        pushNotifications=False,  # webhook 订阅接口已提供，但 push 需主动注册
        asyncTasks=True,          # A2A Tasks API 已实现
    )

    # ── 4. Authentication 声明 ──
    auth = AgentAuth(
        type="api_key_header",
        header_name="X-API-Key",
    )

    # ── 5. Rate Limit 信息 ──
    rate_limit = AgentRateLimitInfo(
        requests_per_minute=60,
        burst=10,
    )

    # ── 6. Endpoints 列表 ──
    endpoints: List[AgentEndpoint] = []
    for ep in _A2A_ENDPOINTS:
        endpoints.append(AgentEndpoint(
            method=ep["method"],
            path=ep["path"],
            identifier=ep["identifier"],
            description=ep["description"],
            requires_auth=ep["requires_auth"],
        ))

    # ── 7. 构建 AgentCard ──
    _cached_card = AgentCard(
        name=name,
        description=description,
        url=url,
        capabilities=capabilities,
        skills=skills,
        authentication=auth,
        rate_limit=rate_limit,
        endpoints=endpoints,
        documentation_url=f"{url}/docs" if getattr(config, "DEBUG_MODE", False) else None,
    )
    return _cached_card


def warmup_agent_card() -> None:
    """预热 Agent Card 缓存 —— 应在 lifespan startup 中调用。

    把首次 build（~5s 的 skill_loader + langgraph 导入链）从第一个 HTTP 请求
    移到服务启动阶段，让所有请求都能享受 <1ms 的缓存命中。
    """
    t0 = __import__("time").perf_counter()
    card = build_agent_card()
    elapsed = __import__("time").perf_counter() - t0
    logger.info(
        f"Agent Card 预热完成 ({elapsed*1000:.0f}ms)",
        skills_count=len(card.skills),
        skills=[s.id for s in card.skills],
        endpoints_count=len(card.endpoints) if card.endpoints else 0,
    )
