"""可选预标注:对未标样本跑一次模型选型,把 llm_suggested_tool 预填到前端(§2.4)。

默认走 HTTP 网关(不硬依赖 shop-agent 内部模块);未配置 PRELABEL_URL 时返回 None,
前端照常工作(只是没有预填提示)。
"""
import json
from typing import List, Optional

import requests

from . import config


def prelabel(query_text: str, candidate_tools: List[str], available_tools: List[str]) -> Optional[str]:
    """返回模型建议工具名;失败/未配置返回 None。"""
    if not config.PRELABEL_URL:
        return None
    try:
        resp = requests.post(
            config.PRELABEL_URL,
            json={
                "query": query_text,
                "candidate_tools": candidate_tools,
                "available_tools": available_tools,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("llm_suggested_tool") or data.get("suggested_tool")
    except Exception:
        return None
