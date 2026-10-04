#!/usr/bin/env python3
"""
触发工具选择管线异常率告警的测试脚本

通过注入各类错误验证告警指标：
1. stage timeout
2. stage exception
3. LLM 调用失败
"""

import asyncio
import json
import urllib.request
import time
from typing import Dict, Any

API_URL = "http://localhost:8000/agent/api/v1/chatagent/agent/chat"
API_KEY = "ak_bigdata_internal_2024"
PROMETHEUS_URL = "http://localhost:19090/api/v1/query"


def query_prometheus(metric: str) -> Dict[str, Any]:
    url = f"{PROMETHEUS_URL}?query={metric}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "error", "error": str(e)}


def print_error_rates():
    """打印当前各层异常率"""
    print("\n=== 当前异常率 ===")
    for stage in ["p0_rule", "p1_faiss", "p2_linear", "p3_llm", "embedding_prepare"]:
        total_q = f'sum(rate(shop_agent_tool_select_stage_total{{stage="{stage}"}}[5m]))'
        err_q = f'sum(rate(shop_agent_tool_select_stage_total{{stage="{stage}",outcome=~"error|timeout"}}[5m]))'
        
        total_data = query_prometheus(total_q)
        err_data = query_prometheus(err_q)
        
        total = 0
        err = 0
        if total_data.get("status") == "success" and total_data.get("data", {}).get("result"):
            total = float(total_data["data"]["result"][0]["value"][1])
        if err_data.get("status") == "success" and err_data.get("data", {}).get("result"):
            err = float(err_data["data"]["result"][0]["value"][1])
        
        rate = (err / total * 100) if total > 0 else 0
        status = "ALERT" if rate > 1 else "OK"
        print(f"  [{status}] {stage}: total={total:.2f}/s, error={err:.2f}/s, rate={rate:.1f}%")


async def send_request(message: str) -> Dict:
    """发送聊天请求"""
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({"message": message}).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": API_KEY}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"success": False, "error": str(e)}


async def test_normal_requests():
    """正常请求建立基线"""
    print("\n=== 发送正常请求建立基线 ===")
    for i in range(10):
        msg = f"查询订单 WB2024090100{i:02d} 详情"
        result = await send_request(msg)
        status = "OK" if result.get("success") else "FAIL"
        print(f"  {status} {msg}")
        await asyncio.sleep(0.2)
    await asyncio.sleep(2)  # 等待指标刷新
    print_error_rates()


async def test_timeout_injection():
    """通过超短 timeout 触发超时（需配合配置修改）"""
    print("\n=== 注入超时错误（需配置配合）===")
    print("  提示：将 docker-compose.yml 中 P2_LINEAR_TIMEOUT 设为 1ms 可触发")
    print("  当前跳过，需手动修改配置后重启")


async def test_llm_failure():
    """通过无效输入触发 LLM 错误"""
    print("\n=== 触发 LLM 错误 ===")
    # 发送可能导致 LLM 解析失败的消息
    for i in range(5):
        msg = "随机乱码输入 " + "x" * 1000  # 超长输入可能触发错误
        result = await send_request(msg)
        status = "OK" if result.get("success") else "FAIL"
        print(f"  {status} 超长输入测试 {i+1}")
        await asyncio.sleep(0.5)
    await asyncio.sleep(2)
    print_error_rates()


async def test_invalid_tool():
    """触发工具不存在错误"""
    print("\n=== 触发无效工具调用 ===")
    # 这里需要在代码层注入，HTTP 接口无法直接触发
    print("  需在代码中注入：在 tool_select_pipeline.py 添加 raise Exception()")


def main():
    print("工具选择管线异常率告警测试")
    print("=" * 50)
    
    # 1. 正常基线
    asyncio.run(test_normal_requests())
    
    # 2. 尝试触发 LLM 错误
    asyncio.run(test_llm_failure())
    
    # 3. 提示如何注入更多错误
    print("\n=== 更多注入方式 ===")
    print("1. 超时: 修改 docker-compose.yml P2_LINEAR_TIMEOUT=1 重启")
    print("2. 异常: 在 tool_select_pipeline.py _run_stage 中注入 raise")
    print("3. LLM 失败: 关闭 vLLM 容器 docker stop vllm-qwen3")
    print("4. Embedding 失败: 关闭 vllm-bge-m3 容器")


if __name__ == "__main__":
    main()