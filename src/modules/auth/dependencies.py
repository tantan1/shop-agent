
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.core.config import config
from src.shared.logger import APILogger

# 创建Bearer token安全方案
security = HTTPBearer(
    scheme_name="Bearer Token",
    description="请在Authorization头中提供Bearer token格式的API密钥"
)

logger = APILogger("auth_dependencies")


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> None:
    """验证固定API密钥的依赖注入函数

    从Authorization头中提取Bearer token并验证其是否为固定的API密钥
    如果验证失败则抛出HTTP异常

    Args:
        credentials: HTTP Bearer认证凭据

    Raises:
        HTTPException: 认证失败时抛出401错误
    """
    try:
        # 提取API密钥
        api_key = credentials.credentials

        # 验证是否为固定的API密钥
        if api_key != config.FIXED_API_KEY:
            logger.log_api_call(
                api_key=api_key[:8] + "****" if api_key else "None",
                endpoint="auth_verification",
                success=False,
                error="无效的API密钥"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="无效的API密钥",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 记录成功的API调用
        logger.log_api_call(
            api_key=api_key[:8] + "****",
            endpoint="auth_verification",
            success=True,
            key_name="内部系统"
        )

        # 验证成功，不返回任何内容

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        # 记录系统错误
        logger.log_api_call(
            api_key=credentials.credentials[:8] + "****" if credentials and credentials.credentials else "None",
            endpoint="auth_verification",
            success=False,
            error=f"系统错误: {str(e)}"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="认证服务内部错误",
            headers={"WWW-Authenticate": "Bearer"},
        )