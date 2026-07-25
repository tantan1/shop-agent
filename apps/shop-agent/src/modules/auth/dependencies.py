from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import config
from src.core.permissions import (
    ClientInfo,
    Role,
    lookup_client,
    register_legacy_client,
    set_current_client,
)
from src.shared.logger import APILogger

# 创建Bearer token安全方案（不自动报错：允许无 Authorization 头，交给 verify 处理 X-API-Key）
security = HTTPBearer(
    auto_error=False,
    scheme_name="Bearer Token",
    description="请在Authorization头中提供Bearer token格式的API密钥",
)

logger = APILogger("auth_dependencies")

# ── 向后兼容：将 .env 的旧 FIXED_API_KEY 注册为 admin ──
register_legacy_client(config.FIXED_API_KEY)


def _resolve_api_key(
    credentials: Optional[HTTPAuthorizationCredentials],
    x_api_key: Optional[str],
) -> Optional[str]:
    """决定使用哪个 API Key：优先 Bearer，其次 X-API-Key。

    demo 页用 X-API-Key 头发送，外部调用用 Authorization: Bearer。
    """
    if credentials is not None and credentials.credentials:
        return credentials.credentials
    if x_api_key is not None and x_api_key.strip():
        return x_api_key.strip()
    return None


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),  # noqa: B008
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """验证API密钥的依赖注入函数

    从Authorization头中提取Bearer token，先在 mock 调用方表查找，
    未命中则回退到 FIXED_API_KEY 校验（向后兼容）。

    Args:
        credentials: HTTP Bearer认证凭据

    Raises:
        HTTPException: 认证失败时抛出401错误
    """
    try:
        api_key = _resolve_api_key(credentials, x_api_key)

        if not api_key:
            logger.log_api_call(
                api_key="None",
                endpoint="auth_verification",
                success=False,
                error="缺少 API Key",
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的API密钥",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 查找 mock 调用方 或 回退旧 FIXED_API_KEY
        client = lookup_client(api_key)
        if client is not None:
            logger.log_api_call(
                api_key=api_key[:8] + "****",
                endpoint="auth_verification",
                success=True,
                key_name=client.client_name,
            )
            return  # 验证成功

        # 回退：旧版 FIXED_API_KEY（兼容）
        if api_key == config.FIXED_API_KEY:
            logger.log_api_call(
                api_key=api_key[:8] + "****",
                endpoint="auth_verification",
                success=True,
                key_name="固定API密钥（旧版兼容）",
            )
            return  # 验证成功

        logger.log_api_call(
            api_key=api_key[:8] + "****" if api_key else "None",
            endpoint="auth_verification",
            success=False,
            error="无效的API密钥",
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API密钥",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except HTTPException:
        raise
    except Exception as e:
        api_key_for_log = _resolve_api_key(credentials, x_api_key) or ""
        logger.log_api_call(
            api_key=(api_key_for_log[:8] + "****") if api_key_for_log else "None",
            endpoint="auth_verification",
            success=False,
            error=f"系统错误: {str(e)}",
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="认证服务内部错误",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# ============ 调用方认证相关函数 ============


async def get_current_client(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),  # noqa: B008
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> ClientInfo:
    """
    获取当前调用方信息的依赖函数
    验证API密钥并返回带角色的调用方信息

    Args:
        credentials: HTTP Bearer认证凭据

    Returns:
        ClientInfo: 包含角色、调用方ID等信息的上下文

    Raises:
        HTTPException: 认证失败时抛出401错误
    """
    await verify_api_key(credentials, x_api_key)

    api_key = _resolve_api_key(credentials, x_api_key)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API密钥",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 优先从 mock 表查找
    client = lookup_client(api_key)
    if client is not None:
        set_current_client(client)
        return client

    # 回退：旧版 FIXED_API_KEY → admin（兼容）
    client = ClientInfo(
        client_id="legacy-admin",
        role=Role.ADMIN,
        client_name="旧版调用方（FIXED_API_KEY）",
        api_key_prefix=api_key[:8] + "****",
    )
    set_current_client(client)
    return client


async def require_admin(client: ClientInfo = Depends(get_current_client)) -> ClientInfo:  # noqa: B008
    """
    要求管理员权限的依赖函数

    Args:
        client: 当前调用方信息 (由 get_current_client 提供)

    Returns:
        ClientInfo: 调用方信息

    Raises:
        HTTPException: 权限不足时抛出403错误
    """
    if not client.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return client
