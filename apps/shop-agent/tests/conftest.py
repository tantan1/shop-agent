"""pytest 共享配置：让纠纷协调器单测脱离外部依赖，可离线、确定性运行。

- FIXED_API_KEY 必须在导入 src（其会实例化 Settings）之前注入，否则收集阶段即报校验错误。
- ORDER_SERVICE_URL 置空，使 resolve() 中的 get_after_sale_evidence 直接返回诚实错误 JSON，
  而非发起真实的网络请求（避免测试变慢、依赖 docker 环境、产生偶发抖动）。
"""
import os

os.environ.setdefault("FIXED_API_KEY", "test-key-for-pytest")
os.environ["ORDER_SERVICE_URL"] = ""

import pytest  # noqa: E402

from src.core.config import config  # noqa: E402


@pytest.fixture(autouse=True)
def _no_order_service():
    """运行每个测试前将订单服务 URL 置空，结束后还原。"""
    prev = config.ORDER_SERVICE_URL
    config.ORDER_SERVICE_URL = ""
    yield
    config.ORDER_SERVICE_URL = prev
