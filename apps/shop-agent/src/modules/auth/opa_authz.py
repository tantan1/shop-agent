"""OPA 授权客户端（PEP → PDP 调用）。

职责（仅做"授权"决策，认证已由 keycloak_auth 完成）：
  - 把 Principal(来自 Keycloak JWT) + 资源 + 动作 组装成 OPA input
  - 调 OPA /v1/data/<package> 取 allow 决策
  - 提供 FastAPI 依赖 require_permission(tool, action) 直接挂路由

设计要点：
  - 默认 fail-closed：OPA 不可达/超时 → 拒绝（403），避免越权放行
  - OPA 部署在私有网络（docker-compose 内网），策略来自本地/内网 bundle，不连公网 Git
  - 与现有 core.permissions.check_tool_permission 可并存，按路由逐步迁移
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

import httpx
from fastapi import Depends, HTTPException, status

from src.core.config import config
from src.modules.auth.keycloak_auth import Principal, get_current_principal
from src.ports import audit
from src.shared.logger import APILogger

logger = APILogger("opa_authz")

_OPA_URL = getattr(config, "OPA_URL", os.getenv("OPA_URL", "http://opa:8181"))
_OPA_PACKAGE = getattr(config, "OPA_PACKAGE", os.getenv("OPA_PACKAGE", "shopagent.authz"))

# 工具归属部门（示例：用于 ABAC 同部门判定；生产应从配置/DB 读取）
TOOL_OWNER_DEPT: Dict[str, str] = {
    "request-return": "aftersale",
    "query-order": "order",
    "check-balance": "finance",
}


class OPAClient:
    """OPA 决策点客户端（thin wrapper）。"""

    def __init__(self, url: str = _OPA_URL, package: str = _OPA_PACKAGE, timeout: float = 0.5) -> None:
        self.url = url.rstrip("/")
        self.package = package
        self.timeout = timeout

    def _build_input(
        self,
        principal: Principal,
        resource_id: str,
        action: str,
        environment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "user": {
                "sub": principal.sub,
                "roles": sorted(principal.roles),
                "dept": principal.attributes.get("dept"),
                "tenant": principal.attributes.get("tenant"),
                "client_id": principal.client_id,
                "is_service": principal.is_service,
            },
            "resource": {
                "id": resource_id,
                "owner_dept": TOOL_OWNER_DEPT.get(resource_id),
            },
            "action": action,
            "environment": environment or {},
        }

    async def is_allowed(
        self,
        principal: Principal,
        resource_id: str,
        action: str,
        environment: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """返回 True/False；OPA 异常按 fail-closed 返回 False。"""
        payload = {"input": self._build_input(principal, resource_id, action, environment)}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.url}/v1/data/{self.package}", json=payload
                )
                resp.raise_for_status()
                result = resp.json().get("result", {})
                return bool(result.get("allow", False))
        except Exception as e:  # noqa: BLE001
            audit.log(
                "authz.opa_error",
                principal=principal.sub,
                action=action,
                decision="deny",  # fail-closed
                detail={"error": str(e)},
            )
            logger.warning(f"OPA 决策失败（fail-closed 拒绝）: {e}")
            return False


opa = OPAClient()


def require_permission(resource_id: str, action: str = "execute"):
    """依赖工厂：在路由上声明"需要该资源/动作的 OPA 授权"。

    用法：
        @router.post("/tools/request-return")
        async def handler(_: Principal = Depends(require_permission("request-return", "execute"))):
            ...
    """

    async def _dep(
        p: Principal = Depends(get_current_principal),
    ) -> Principal:
        allowed = await opa.is_allowed(p, resource_id, action)
        if not allowed:
            audit.log(
                "authz.denied",
                principal=p.sub,
                action=f"{resource_id}:{action}",
                decision="deny",
                detail={"roles": sorted(p.roles), "dept": p.attributes.get("dept")},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"无权执行 {resource_id}:{action}",
            )
        return p

    return _dep
