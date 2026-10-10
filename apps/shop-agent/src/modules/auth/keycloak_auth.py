"""Keycloak OIDC / JWT 验证（私有部署，无外部 SaaS 依赖）。

职责（仅做"认证"，授权交给 OPA）：
  - 从 Keycloak 拉取 JWKS（带缓存）验证 JWT 签名、audience、issuer、有效期
  - 解析 sub / preferred_username / realm+client roles / 自定义属性(dept, tenant)
  - 输出 Principal，供 opa_authz 做细粒度授权判定

与现有 API-Key 体系（src.core.permissions）并行：
  - 新路径：前端/其他应用拿 Keycloak 签发的 JWT，走本模块
  - 旧路径：X-API-Key / FIXED_API_KEY 仍可用（向后兼容，见 auth/dependencies.py）
  - 两者通过不同 FastAPI 依赖注入，互不污染
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional

import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from jose.jwk import construct

from src.core.config import config
from src.ports import audit
from src.shared.logger import APILogger

logger = APILogger("keycloak_auth")

# ── 配置读取（从 config 或环境变量，私有部署不依赖公网）─────────────
_KC_SERVER = getattr(config, "KEYCLOAK_SERVER_URL", os.getenv("KEYCLOAK_SERVER_URL", "http://keycloak:8080"))
_KC_REALM = getattr(config, "KEYCLOAK_REALM", os.getenv("KEYCLOAK_REALM", "shop-agent"))
_KC_CLIENT = getattr(config, "KEYCLOAK_CLIENT_ID", os.getenv("KEYCLOAK_CLIENT_ID", "shop-agent-api"))
_KC_AUDIENCE = getattr(config, "KEYCLOAK_AUDIENCE", _KC_CLIENT)
_KC_ISSUER = getattr(
    config, "KEYCLOAK_ISSUER",
    os.getenv("KEYCLOAK_ISSUER", f"{_KC_SERVER}/realms/{_KC_REALM}"),
)


@dataclass(frozen=True)
class Principal:
    """从 JWT 解析出的主体（类比 core.permissions.ClientInfo，但面向 OIDC 声明）。"""

    sub: str
    username: str
    roles: FrozenSet[str]
    attributes: Dict[str, str] = field(default_factory=dict)  # dept / tenant 等 ABAC 属性
    client_id: Optional[str] = None  # M2M: azp
    is_service: bool = False  # 是否为 service account（client_credentials）

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles


class KeycloakOIDC:
    """轻量 Keycloak JWT 验证器（不引入 keycloak-python 重客户端，仅用 JWKS）。"""

    def __init__(
        self,
        server_url: str = _KC_SERVER,
        realm: str = _KC_REALM,
        audience: str = _KC_AUDIENCE,
        issuer: str = _KC_ISSUER,
        jwks_ttl: int = 3600,
        timeout: float = 3.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.realm = realm
        self.audience = audience
        self.issuer = issuer
        self._jwks_ttl = jwks_ttl
        self._timeout = timeout
        self._jwks: list = []
        self._jwks_fetched = 0.0

    async def _get_jwks(self) -> list:
        now = time.time()
        if self._jwks and (now - self._jwks_fetched) < self._jwks_ttl:
            return self._jwks
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(
                f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/certs"
            )
            resp.raise_for_status()
            self._jwks = resp.json().get("keys", [])
            self._jwks_fetched = now
            return self._jwks

    def _key_for(self, kid: Optional[str]):
        for k in self._jwks:
            if k.get("kid") == kid:
                return construct(k, "RS256")
        return None

    async def verify_token(self, token: str) -> Principal:
        """验证 JWT 并返回 Principal；任何失败抛 JWTError（由依赖层转 401）。"""
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        jwks = await self._get_jwks()
        key = self._key_for(kid)
        if key is None:
            raise JWTError("no matching JWK for kid")
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=self.audience,
            issuer=self.issuer,
        )
        return self._to_principal(claims)

    @staticmethod
    def _to_principal(claims: dict) -> Principal:
        roles: set = set(claims.get("realm_access", {}).get("roles", []))
        roles.update(
            claims.get("resource_access", {})
            .get(_KC_CLIENT, {})
            .get("roles", [])
        )
        attributes = {
            k: claims[k] for k in ("dept", "tenant") if claims.get(k) is not None
        }
        azp = claims.get("azp") or ""
        is_service = bool(azp.startswith("service-account") or claims.get("typ") == "Service")
        return Principal(
            sub=claims.get("sub", ""),
            username=claims.get("preferred_username", ""),
            roles=frozenset(roles),
            attributes=attributes,
            client_id=azp or claims.get("client_id"),
            is_service=is_service,
        )


# 模块级单例（私有部署固定配置）
keycloak = KeycloakOIDC()

_bearer = HTTPBearer(auto_error=False)


async def get_current_principal(
    cred: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Principal:
    """FastAPI 依赖：从 Bearer JWT 解析当前主体。

    用法（在路由上叠加，与现有 get_current_client 二选一或组合）：
        @router.post("/tools/xxx")
        async def handler(p: Principal = Depends(get_current_principal),
                          _: Principal = Depends(require_role("operator"))):
            ...
    """
    if cred is None or not cred.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Bearer Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return await keycloak.verify_token(cred.credentials)
    except JWTError as e:
        audit.log(
            "authz.jwt_failed",
            principal="?",
            action="verify_token",
            decision="deny",
            detail={"error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def require_role(role: str):
    """依赖工厂：要求主体具备指定角色，否则 403。"""

    async def _dep(p: Principal = Depends(get_current_principal)) -> Principal:
        if role not in p.roles:
            audit.log(
                "authz.role_denied",
                principal=p.sub,
                action="require_role",
                decision="deny",
                detail={"required": role, "had": sorted(p.roles)},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要角色: {role}",
            )
        return p

    return _dep
