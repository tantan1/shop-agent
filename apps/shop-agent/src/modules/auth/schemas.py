"""认证接口 Schema（请求/响应）。

统一从 `models` 导出，便于路由层引用，避免散落定义。
"""
from src.modules.auth.models import ApiKeyAuth  # noqa: F401

__all__ = ["ApiKeyAuth"]
