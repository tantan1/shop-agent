---
name: check-balance
display_name: 余额积分查询
description: >
  查询账户余额和可用积分。
  触发条件：用户问"余额""钱包""有多少钱""积分""我的积分"。
  区分：与 coupon-inquiry 不同——本工具查账户资金/积分，coupon-inquiry 查优惠券。
tags:
  - 余额
  - 积分
  - 钱包
  - 账户
allowed-tools: check-balance
priority: 10
---

# 余额积分查询 Skill

## 触发条件
当用户询问"余额""钱包有多少钱""我的积分""积分剩余"时激活。

## 执行流程
1. 调用 `check-balance` 获取余额和积分
2. 以友好格式展示：账户余额 XX 元，可用积分 XX 分

## 注意事项
- 与 `coupon-inquiry` 的区别：本工具查账户资金/积分，coupon-inquiry 查优惠券
- 同时输出余额和积分，避免用户分别询问
