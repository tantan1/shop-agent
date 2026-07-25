---
name: coupon-inquiry
display_name: 优惠券查询
description: >
  查询可用的优惠券列表，包括有效期、使用门槛。
  触发条件：用户问"优惠券""代金券""满减""有什么券""优惠码"。
  区分：与 check-balance 不同——本工具查优惠券而非账户余额。
tags:
  - 优惠券
  - 代金券
  - 满减
  - 折扣
allowed-tools: coupon-inquiry
priority: 10
---

# 优惠券查询 Skill

## 触发条件
当用户询问"优惠券""代金券""有什么券""满减券""优惠码可用吗"时激活。

## 执行流程
1. 调用 `coupon-inquiry` 获取可用优惠券列表
2. 展示每张券的名称、面额、使用门槛、有效期
3. 如果用户提供了券码，还可以验证券码是否可用

## 注意事项
- 与 `check-balance` 的区别：本工具查优惠券，check-balance 查余额积分
- 列出所有可用券，按过期时间从近到远排序
