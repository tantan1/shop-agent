#!/usr/bin/env python3
"""
四层工具选择流水线测试脚本

运行测试用例并输出 Prometheus 指标摘要
"""

import asyncio
import json
import urllib.request
import time
from typing import Dict, List, Any

# 测试用例
TEST_CASES = [
    ("request-return", "我要退货，订单号 WB202409010077，原因是不想要了"),
    ("query-order", "查询订单 WB202409010077 的详情"),
    ("check-shipping", "查物流 WB202409010077 发货了吗"),
    ("check-balance", "查一下余额"),
    ("coupon-inquiry", "优惠券 TEST123 怎么用"),
]

PROMETHEUS_URL = "http://localhost:19090/api/v1/query"
API_URL = "http://localhost:8000/agent/api/v1/chatagent/agent/chat"
API_KEY = "ak_bigdata_internal_2024"

METRICS = [
    "shop_agent_tool_select_stage_total",
    "shop_agent_tool_select_stage_duration_ms_sum",
    "shop_agent_tool_select_stage_duration_ms_count",
    "shop_agent_tool_select_candidates_in_sum",
    "shop_agent_tool_select_candidates_out_sum",
    "shop_agent_tool_select_exit_total",
]


def query_prometheus(metric: str) -> Dict[str, Any]:
    """查询 Prometheus 单个指标"""
    url = f"{PROMETHEUS_URL}?query={metric}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "error", "error": str(e)}


def print_metric_summary(metric_name: str, data: Dict[str, Any]):
    """打印指标摘要"""
    if data.get("status") != "success":
        print(f"  {metric_name}: ERROR - {data.get('error')}")
        return
    
    results = data.get("data", {}).get("result", [])
    if not results:
        print(f"  {metric_name}: (no data)")
        return
    
    print(f"  {metric_name}:")
    for r in results:
        labels = r.get("metric", {})
        stage = labels.get("stage", "")
        source = labels.get("source", "")
        stop = labels.get("stop_condition", "")
        value = r.get("value", [None, ""])[1]
        label_str = f"stage={stage}"
        if source:
            label_str += f", source={source}"
        if stop:
            label_str += f", stop={stop}"
        print(f"    {label_str}: {value}")


async def run_test_cases():
    """通过 API 运行测试用例"""
    print("=== 运行测试用例 ===")
    for intent, query in TEST_CASES:
        req = urllib.request.Request(
            API_URL,
            data=json.dumps({"message": query}).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": API_KEY}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                status = "OK" if result.get("success") else "FAIL"
                print(f"  {status} [{intent}] {query[:40]}...")
        except Exception as e:
            print(f"  FAIL [{intent}] {query[:40]}... ERROR: {e}")


def print_all_metrics():
    """打印所有相关指标"""
    print("\n=== Prometheus 指标摘要 ===")
    for metric in METRICS:
        data = query_prometheus(metric)
        print_metric_summary(metric, data)


def main():
    print("四层工具选择流水线测试")
    print("=" * 50)
    
    # 运行测试用例
    asyncio.run(run_test_cases())
    
    # 等待指标刷新
    print("\n等待指标刷新...")
    time.sleep(2)
    
    # 打印指标
    print_all_metrics()
    
    # 计算漏斗汇总
    print("\n=== 漏斗汇总 ===")
    stages = ["p0_rule", "p1_faiss", "p2_linear", "p3_llm"]
    
    for metric_name in ["shop_agent_tool_select_candidates_in_sum", "shop_agent_tool_select_candidates_out_sum"]:
        data = query_prometheus(metric_name)
        if data.get("status") == "success":
            results = data.get("data", {}).get("result", [])
            stage_map = {r.get("metric", {}).get("stage", ""): float(r.get("value", [0, 0])[1]) for r in results}
            print(f"  {metric_name}:")
            for s in stages:
                val = stage_map.get(s, 0)
                print(f"    {s}: {int(val)}")


if __name__ == "__main__":
    main()