"""HTTP 端点层（controller）。

职责：仅做 HTTP 入口装配 + 把请求转交给 router/hooks/metrics 等模块，
不含任何业务分支判断（路由/治理/计量逻辑在各自模块）。main.py 用 include_router 挂载。
"""
from __future__ import annotations

from .health import router as health_router
from .metrics_endpoint import router as metrics_router
from .proxy import router as proxy_router

__all__ = ["health_router", "metrics_router", "proxy_router"]
