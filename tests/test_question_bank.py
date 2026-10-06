"""固定题库测试 —— V0.2 的实验地基。

如果 PRE/POST 不可比，所有分数差都是噪声。
"""

from __future__ import annotations

import pytest

from brain import question_bank as qb


def test_bank_is_intact():
    assert qb.assert_bank_intact() is None


def test_pre_and_post_have_equal_size():
    assert len(qb.PRE_SET) == len(qb.POST_SET) == 4


def test_pre_and_post_cover_same_kps():
    assert qb.kp_coverage(qb.PRE_SET) == qb.kp_coverage(qb.POST_SET)


def test_item_ids_are_unique():
    all_ids = [i.item_id for i in qb.ALL_ITEMS]
    assert len(all_ids) == len(set(all_ids))


def test_pre_and_post_questions_are_differently_worded():
    """POST 不能是 PRE 的复制——否则退化成同一份卷。"""
    pre_qs = {i.question for i in qb.PRE_SET}
    for item in qb.POST_SET:
        assert item.question not in pre_qs


def test_each_kp_appears_once_per_set():
    for name, items in (("pre", qb.PRE_SET), ("post", qb.POST_SET)):
        kps = [i.kp for i in items]
        assert len(kps) == len(set(kps)), f"{name} 卷有知识点重复"


def test_fingerprint_is_stable_across_calls():
    """指纹必须可重复计算，否则无法用它检测篡改。"""
    assert qb.BANK_FINGERPRINT == qb._fingerprint(qb.ALL_ITEMS)


def test_fingerprint_changes_when_question_changes():
    original = qb.ALL_ITEMS[0]
    tampered = original._replace(question=original.question + "（被改过）")
    assert qb._fingerprint((tampered,) + qb.ALL_ITEMS[1:]) != qb.BANK_FINGERPRINT


def test_pre_and_post_fingerprints_differ():
    assert qb.PRE_FINGERPRINT != qb.POST_FINGERPRINT


def test_get_set_rejects_unknown_name():
    with pytest.raises(ValueError, match="未知卷名"):
        qb.get_set("final")


def test_get_set_returns_correct_sets():
    # V0.4 起「post」这个卷名改成「memory」——它现在只服务 MEMORY TEST，
    # 而 TRANSFER TEST 是独立的一卷。旧名保留会让人误以为两者同卷。
    assert qb.get_set("pre") is qb.PRE_SET
    assert qb.get_set("memory") is qb.POST_SET
    assert qb.get_set("transfer") is qb.TRANSFER_SET
    with pytest.raises(ValueError, match="未知卷名"):
        qb.get_set("post")


def test_every_item_has_substantive_answer():
    for item in qb.ALL_ITEMS:
        assert len(item.answer) >= 30, f"{item.item_id} 参考答案过短，无法用于评分"
        assert item.question and item.kp


def test_kp_labels_cover_all_kps():
    for item in qb.ALL_ITEMS:
        assert item.kp in qb.KP_LABELS


def test_bank_meta_is_serializable():
    meta = qb.bank_meta()
    assert meta["version"] == qb.BANK_VERSION
    assert meta["fingerprint"] == qb.BANK_FINGERPRINT
    assert len(meta["kps"]) == 4


def test_mismatched_kp_triggers_bank_check(monkeypatch):
    """人为制造 kp 不配平时，必须报错拒绝跑实验。"""
    broken = (qb.PRE_ITEMS[0]._replace(kp="kp_ghost"),) + qb.PRE_ITEMS[1:]
    monkeypatch.setattr(qb, "PRE_SET", broken)
    with pytest.raises(ValueError, match="知识点不配平"):
        qb.assert_bank_intact()


def test_question_contains_no_answers():
    """题目文本里不能夹带答案，否则考生不看题就会了。"""
    for item in qb.ALL_ITEMS:
        for answer_line in item.answer.split("；"):
            core = answer_line.strip()[:12]
            if len(core) >= 8:
                assert core not in item.question
