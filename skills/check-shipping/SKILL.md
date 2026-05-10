---
name: check-shipping
display_name: 物流查询
description: >
  查询物流配送进度。返回揽收→运输→派送每一步的时间线和状态。
  触发条件：用户问"到哪了""物流""快递""什么时候送到"。
  区分：与 query-order 不同——本工具返回明细物流轨迹，query-order 是订单宏观信息。
tags:
  - 物流
  - 快递
  - 配送
  - 跟踪
allowed-tools: check-shipping query-order
priority: 10
---

# 物流查询 Skill

## 触发条件
当用户询问"快递到哪了""物流进度""什么时候送到""配送信息"时激活。

## 参数说明
| 参数 | 必填 | 说明 |
|------|------|------|
| tracking_number | 否 | 快递单号 |
| order_id | 否 | 订单号 |

## 执行流程
1. 如果有快递单号 → 直接用 `check-shipping` 查询
2. 如果只有订单号 → 先调 `query-order` 获取运单号，再调 `check-shipping`
3. 将物流步骤整理成时间线展示给用户

## 注意事项
- 如果用户只是问"我的订单"（而非专门问物流），先用 `query-order`
- 本工具返回的是明细物流轨迹，query-order 返回的是订单宏观状态
