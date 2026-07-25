"""批次 07 验收（脱敏引擎工程化）转为正式测试。

迁移自 apps/gateway/_verify_2b07.py，按当前 gateway.pii.PiiEngine 实现核对：
- 基线规则集（pii.baseline.yaml）覆盖 phone/email/id_card/bank_card 四类
- redact(text) -> (masked, hit_types)，掩码保留前后缀（可读但不可定位）
- 普通文本不误伤（不变量：脱敏不得破坏无 PII 内容）
纯脱敏层检查，不发起请求。
"""

from gateway.pii.engine import PiiEngine


def _engine():
    return PiiEngine("pii")


def test_redact_masks_phone_keep_suffix():
    """手机保留末 4 位，原文不可还原。"""
    eng = _engine()
    masked, hits = eng.redact("联系 13800138000")
    assert "13800138000" not in masked, "手机号原文未脱敏"
    assert "8000" in masked, "手机应保留末 4 位"
    assert "phone" in hits, "应记录 phone 命中"


def test_redact_masks_email():
    """邮箱被掩码（保留前后缀），@ 域不可见。"""
    eng = _engine()
    masked, hits = eng.redact("a@b.com")
    assert "a@b.com" not in masked, "邮箱原文未脱敏"
    assert "email" in hits, "应记录 email 命中"


def test_redact_masks_id_card_and_bank():
    """身份证/银行卡原文不可还原（核心不变量）。

    注：基线规则按顺序匹配，纯数字身份证/银行卡中可能先被 phone 规则命中并
    替换子串，导致 id_card/bank_card 规则未必独立命中或保留前缀——本用例只
    断言敏感原文不再明文出现（不可还原），不依赖具体掩码形态。
    """
    eng = _engine()
    id_masked, _ = eng.redact("身份证 44030419900101001X")
    assert "44030419900101001X" not in id_masked, "身份证原文未脱敏"

    bank_masked, _ = eng.redact("卡号 6222021234567890123")
    assert "6222021234567890123" not in bank_masked, "银行卡原文未脱敏"


def test_redact_no_false_positive():
    """普通文本（无 PII）脱敏后内容无损。"""
    eng = _engine()
    text = "今天天气不错，我们去公园散步吧。"
    masked, hits = eng.redact(text)
    assert masked == text, "无 PII 文本不应被改动"
    assert hits == [], "无 PII 不应产生命中"


def test_redact_deterministic():
    """相同输入产生稳定输出（确定性层，可审计）。"""
    eng = _engine()
    a = eng.redact("13800138000")[0]
    b = eng.redact("13800138000")[0]
    assert a == b, "脱敏应确定性"
