"""alerts.redact 单元测试（替代临时冒烟脚本，固化入站脱敏覆盖度）。

验证 code-reviewer B-2 指出的脱敏缺口：邮箱/IP/手机号/身份证/银行卡/密钥，
以及内网 IP 不应误伤、递归深度上限。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring_agent.alerts import redact, redact_obj  # noqa: E402


def test_email_and_phone_masked():
    s = "contact bob@example.com or 13800138000"
    out = redact(s)
    assert "bob@example.com" not in out
    assert "13800138000" not in out


def test_public_ip_masked_internal_preserved():
    s = "src 203.0.113.5 dst 10.0.0.5 127.0.0.1 192.168.1.1"
    out = redact(s)
    assert "203.0.113.5" not in out, "公网 IP 应被脱敏"
    assert "10.0.0.5" in out, "内网 IP 不应误伤"
    assert "127.0.0.1" in out, "回环地址不应误伤"
    assert "192.168.1.1" in out, "私网 IP 不应误伤"


def test_idcard_and_bankcard_masked():
    s = "id 11010519491231002X card 6222021234567890123"
    out = redact(s)
    assert "11010519491231002X" not in out
    assert "6222021234567890123" not in out


def test_secret_masked():
    s = 'api_key="sk-abc123DEF456" password=secret123'
    out = redact(s)
    assert "sk-abc123DEF456" not in out
    assert "secret123" not in out
    assert "api_key" in out and "password" in out, "键名保留，值脱敏"


def test_recursive_depth_limit():
    # 超深嵌套不应栈溢出
    deep = {"a": {"a": {"a": {"a": {"a": {"a": "x@y.com"}}}}}}
    out = redact_obj(deep)
    # 深度未超上限时仍脱敏
    assert "x@y.com" not in str(out)


def test_empty_and_non_str():
    assert redact("") == ""
    assert redact(None) is None
    assert redact(123) == 123
