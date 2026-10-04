#!/usr/bin/env python3
"""
通过 shop-agent API 触发告警测试

三种方式：
1. 停止依赖服务（最真实）
2. 注入代码异常（需重启）
3. 配置极短超时（需重启）
"""

import asyncio
import json
import urllib.request
import time
import sys

API_URL = "http://localhost:8000/agent/api/v1/chatagent/agent/chat"
API_KEY = "ak_bigdata_internal_2024"
PROMETHEUS_URL = "http://localhost:19090/api/v1/query"


def query_prometheus(metric: str) -> dict:
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
        total_q = f'sum(rate(shop_agent_tool_select_stage_total{{stage="{stage}"}}[1m]))'
        err_q = f'sum(rate(shop_agent_tool_select_stage_total{{stage="{stage}",outcome=~"error|timeout"}}[1m]))'
        
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


async def send_request(message: str) -> dict:
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


async def test_normal_baseline():
    """建立正常基线"""
    print("\n=== 1. 发送正常请求建立基线 ===")
    for i in range(5):
        msg = f"查询订单 WB2024090100{i:02d} 详情"
        result = await send_request(msg)
        status = "OK" if result.get("success") else "FAIL"
        print(f"  {status} {msg}")
        await asyncio.sleep(0.3)
    await asyncio.sleep(2)
    print_error_rates()


async def test_long_input_llm_errors():
    """发送超长输入尝试触发 LLM 解析错误"""
    print("\n=== 2. 发送超长输入尝试触发 LLM 错误 ===")
    for i in range(10):
        msg = "测试超长输入 " + "x" * 2000
        result = await send_request(msg)
        status = "OK" if result.get("success") else "FAIL"
        print(f"  {status} 超长输入 {i+1}")
        await asyncio.sleep(0.2)
    await asyncio.sleep(2)
    print_error_rates()


async def test_rapid_fire():
    """快速并发请求尝试触发限流/超时"""
    print("\n=== 3. 快速并发请求（尝试触发限流）===")
    async def fire(msg):
        return await send_request(msg)
    
    tasks = [fire(f"并发查询 {i}") for i in range(20)]
    results = await asyncio.gather(*tasks)
    ok = sum(1 for r in results if r.get("success"))
    print(f"  完成: {ok}/20 成功")
    await asyncio.sleep(2)
    print_error_rates()


def test_stop_dependencies():
    """停止依赖服务（需手动执行）"""
    print("\n=== 4. 停止依赖服务（最有效，需手动执行）===")
    print("  在另一个终端运行以下命令之一：")
    print("    docker stop vllm-qwen3          # 触发 P3 LLM 错误")
    print("    docker stop vllm-bge-small-zh  # 触发 P1/P2 Embedding 错误")
    print("    docker stop vllm-bge-reranker   # 触发 Rerank 错误")
    print("  然后重新运行此脚本的选项 1 或 2 观察异常率飙升")
    print("  测试完记得重启：docker start vllm-qwen3 vllm-bge-small-zh vllm-bge-reranker")


async def main():
    print("Shop-Agent 告警触发测试")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("选择测试模式：")
        print("  1 - 正常基线")
        print("  2 - 超长输入触发 LLM 错误")
        print("  3 - 快速并发触发限流")
        print("  4 - 显示停止依赖服务命令")
        print("  all - 全部运行")
        mode = input("输入模式 (1/2/3/4/all): ").strip()
    
    if mode in ("1", "all"):
        await test_normal_baseline()
    
    if mode in ("2", "all"):
        await test_long_input_llm_errors()
    
    if mode in ("3", "all"):
        await test_rapid_fire()
    
    if mode in ("4", "all"):
        test_stop_dependencies()
    
    print("\n=== 最终异常率 ===")
    print_error_rates()
    
    print("\n=== Grafana 告警验证 ===")
    print("1. 打开 http://localhost:3000/d/shop-agent-dashboard-001")
    print("2. 观察面板 10 '工具选择异常率' - 红色阈值 >1%")
    print("3. 观察面板 7 '成本漏斗分布' - P3 占比升高")


if __name__ == "__main__":
    asyncio.run(main())