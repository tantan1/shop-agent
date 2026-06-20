---
name: query-order
display_name: 订单查询
description: >
  查询用户的订单列表或指定订单详情。触发条件：用户询问订单状态、订单号、我的订单。
  区分：与 check-shipping 不同——本工具查询订单信息全貌（状态/金额/列表），check-shipping 专门查询物流轨迹详情。
tags:
  - 订单
  - 查询
  - 订单号
allowed-tools: query-order check-shipping
priority: 10
---

# 订单查询 Skill

## 触发条件
当用户询问订单状态、订单号查询、订单列表、我的订单时激活。

## 执行流程
1. 如果用户提供了订单号 → 调用 `query-order` 查询指定订单
2. 如果用户接着问物流 → 调用 `check-shipping` 查物流轨迹
3. 将结果整合成自然、友好的回复

## 注意事项
- 与 `check-shipping` 的区别：本工具查订单宏观信息（状态/金额），check-shipping 查物流轨迹
- 用户只问"我的订单到哪了"时，先用 query-order 获取订单，再用 check-shipping 查物流
