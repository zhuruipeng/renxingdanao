"""tests/test_v06_causal.py —— V0.6 因果层：difference-in-gains / CI / 效应量 / 措辞门禁。

这一层测的是**指标算得对不对**、**结论许可由数据决定**。
它不碰数据库、不联网、不调模型——纯函数，全部可重放。

三组测试各自盯住一类曾经真的错过的错误：

1. 成对指标：把四类 gain 算成同一个东西
2. 统计量：CI 跨 0 时仍然说「有提升」
3. 措辞门禁：报告模板把自己拦下来
"""

from __future__ import annotations

import pytest

from brain import causal_v3 as c3

# ==========================================================================
# 1. 单次 trial 的成对指标
# ==========================================================================


def test_paired_trial_computes_three_differences():
    t = c3.paired_trial(
        learning_pre=20.0, learning_post=80.0,
        control_pre=20.0, control_post=30.0,
    )
    assert t[c3.KEY_LEARNING_GAIN] == 60.0
    assert t[c3.KEY_CONTROL_GAIN] == 10.0
    assert t[c3.KEY_DIFFERENCE] == 50.0


def test_paired_trial________():
    t = c3.paired_trial(20.0, 80.0, 25.0, 35.0)
    for k in (c3.KEY_LEARNING_PRE, c3.KEY_LEARNING_POST,
              c3.KEY_CONTROL_PRE, c3.KEY_CONTROL_POST):
        assert k in t


def test_paired_trial_difference____gain__():
    t = c3.paired_trial(20.0, 80.0, 25.0, 35.0)
    assert t[c3.KEY_DIFFERENCE] == (
        t[c3.KEY_LEARNING_GAIN] - t[c3.KEY_CONTROL_GAIN]
    )


def test_paired_trial________difference_0():
    """两组都涨 10 分 → 差值的差值 = 0。

    这正是 V0.6 存在的理由：V0.5 只看learning_gain=10，
    会把「做了一遍这件事的正常变化」当成学习效果。
    """
    t = c3.paired_trial(30.0, 40.0, 30.0, 40.0)
    assert t[c3.KEY_LEARNING_GAIN] == 10.0
    assert t[c3.KEY_CONTROL_GAIN] == 10.0
    assert t[c3.KEY_DIFFERENCE] == 0.0


def test_paired_trial________difference():
    """PRE 基线差不该被算成学习效果。"""
    a = c3.paired_trial(10.0, 50.0, 10.0, 30.0)   # 基线相同
    b = c3.paired_trial(30.0, 70.0, 30.0, 50.0)   # 基线高 20
    assert a[c3.KEY_DIFFERENCE] == b[c3.KEY_DIFFERENCE]


def test_paired_trial____difference__():
    t = c3.paired_trial(50.0, 40.0, 50.0, 60.0)
    assert t[c3.KEY_DIFFERENCE] == -20.0


def test_paired_trial_________():
    t = c3.paired_trial(1.0, 2.0, 3.0, 4.0)
    assert t["order"] == (
        c3.KEY_LEARNING_PRE, c3.KEY_LEARNING_POST, c3.KEY_LEARNING_GAIN,
        c3.KEY_CONTROL_PRE, c3.KEY_CONTROL_POST, c3.KEY_CONTROL_GAIN,
        c3.KEY_DIFFERENCE,
    )


def test_objective_trial__________():
    t = c3.objective_trial(
        learning_objective_pre=10.0, learning_objective_post=90.0,
        control_objective_pre=10.0, control_objective_post=15.0,
    )
    assert t[c3.KEY_OBJ_LEARNING_GAIN] == 80.0
    assert t[c3.KEY_OBJ_CONTROL_GAIN] == 5.0
    assert t[c3.KEY_OBJ_DIFFERENCE] == 75.0


def test_objective_trial________():
    """用户明确要求：不要用「实践机器分 - 考试分」当gain。

    两套口径各自内部做前后差，**从不跨口径相减**。
    这里验证两组分别算完后，差值的差值仍然在客观口径内闭合。
    """
    t = c3.objective_trial(0.0, 100.0, 0.0, 0.0)
    assert t[c3.KEY_OBJ_DIFFERENCE] == (
        t[c3.KEY_OBJ_LEARNING_GAIN] - t[c3.KEY_OBJ_CONTROL_GAIN]
    )


def test_objective_trial__________key():
    t = c3.objective_trial(0.0, 1.0, 0.0, 1.0)
    for k in (c3.KEY_LEARNING_PRE, c3.KEY_LEARNING_POST, c3.KEY_DIFFERENCE):
        assert k not in t


# ==========================================================================
# 2. Cohen's d
# ==========================================================================


def test_cohens_d_large_positive_for_separated_groups():
    es = c3.cohens_d([10.0, 12.0, 11.0, 13.0], [0.0, 1.0, 2.0, 1.0])
    assert es["d"] is not None
    assert es["d"] > 0.8


def test_cohens_d_______d_0():
    es = c3.cohens_d([5.0, 6.0, 7.0], [5.0, 6.0, 7.0])
    assert es["d"] == 0.0


def test_cohens_d_____1_____():
    """n<2 时标准差不可估，必须拒绝给数——不能拿 0 当分母硬除。"""
    es = c3.cohens_d([5.0], [1.0, 2.0])
    assert es["d"] is None
    assert es["method"] == "insufficient"


def test_cohens_d______0_____():
    es = c3.cohens_d([3.0, 3.0], [1.0, 1.0])
    assert es["d"] is None
    assert es["method"] == "zero_pooled_variance"


def test_cohens_d________():
    """合并 SD 不是两组 SD 的平均——这里给出可手算的算例。"""
    t = [10.0, 20.0]      # mean 15, v1 = 50
    c = [0.0, 2.0]        # mean 1,  v2 = 2
    es = c3.cohens_d(t, c)
    import math
    pooled = math.sqrt(((2 - 1) * 50 + (2 - 1) * 2) / (2 + 2 - 2))
    assert es["pooled_sd"] == round(pooled, 3)
    assert es["d"] == round((15 - 1) / pooled, 3)


def test_paired_effect_size_measures_gain_not_post():
    """两组 POST 差很多但 gain 相同 → d 不可计算/接近 0。

    「学习组考得高」和「学习带来的提升更大」是两件事。
    前者可能纯粹因为两组 PRE 基线不同。
    """
    identical = c3.paired_effect_size([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert identical["d"] in (0.0, None)

    # POST 差距大但 gain 无差距
    by_gain = c3.paired_effect_size([1.0, 1.0, 1.0], [1.1, 0.9, 1.0])
    assert by_gain["d"] is not None
    assert abs(by_gain["d"]) < c3.MIN_EFFECT_SIZE

    # 同样两组 gain，换成 POST 分数则差距巨大
    by_post = c3.cohens_d([90.0, 92.0, 91.0], [10.0, 12.0, 11.0])
    assert by_post["d"] > 5.0


def test_effect_size_label__Cohen____():
    assert c3.effect_size_label(None) == "不可计算"
    assert c3.effect_size_label(0.1) == "可忽略"
    assert c3.effect_size_label(0.3) == "小"
    assert c3.effect_size_label(0.6) == "中等"
    assert c3.effect_size_label(1.2) == "大"
    assert c3.effect_size_label(-0.6) == "中等"


# ==========================================================================
# 3. 聚合：CI / 效应量 / 三条件
# ==========================================================================


def _trials(diffs: list[tuple[float, float, float, float]]) -> list[dict]:
    return [
        c3.paired_trial(lp, lq, cp, cq) for lp, lq, cp, cq in diffs
    ]


def test_aggregate_output_has_five_lenses():
    s = c3.aggregate_difference(_trials([(0.0, 50.0, 0.0, 10.0)]))
    for k in c3.FIVE_LENSES:
        assert k in s, f"报告必须分清的五类口径缺了 {k}"
    assert set(c3.FIVE_LENSES) <= set(s)


def test_aggregate___trial______():
    s = c3.aggregate_difference(_trials([(0.0, 50.0, 0.0, 10.0)]))
    ci = s["confidence_interval"]
    assert ci["ci_low"] is None and ci["ci_high"] is None
    assert ci["method"] == "insufficient"
    assert s["conditions"]["has_at_least_two_trials"] is False


def test_aggregate___trial_observable__():
    s = c3.aggregate_difference(_trials([(0.0, 50.0, 0.0, 10.0)]))
    assert s["observable_effect"] is False


def test_aggregate__________observable__():
    s = c3.aggregate_difference(_trials([
        (0.0, 60.0, 0.0, 10.0),
        (0.0, 62.0, 0.0, 11.0),
        (0.0, 58.0, 0.0, 9.0),
    ]))
    assert s["conditions"]["has_at_least_two_trials"] is True
    assert s["conditions"]["ci_excludes_zero"] is True
    assert s["conditions"]["effect_size_reaches_threshold"] is True
    assert s["observable_effect"] is True


def test_aggregate______CI_0______():
    s = c3.aggregate_difference(_trials([
        (30.0, 40.0, 30.0, 40.0),
        (30.0, 41.0, 30.0, 41.0),
        (30.0, 39.0, 30.0, 39.0),
    ]))
    assert s["difference_in_gains"]["mean"] == 0.0
    assert s["conditions"]["ci_excludes_zero"] is False
    assert s["observable_effect"] is False


def test_aggregate______caveat():
    """确定性 provider 下σ=0 是常态，区间退化成点但不等于「真实无波动」。"""
    s = c3.aggregate_difference(_trials([
        (0.0, 60.0, 0.0, 10.0),
        (0.0, 60.0, 0.0, 10.0),
    ]))
    assert s["confidence_interval"]["method"] == "zero_variance"
    assert "caveat" in s["confidence_interval"]


def test_aggregate_effect_size_below_threshold():
    # 差异落在 0 附近，且波动远大于差异 → d 必须很小
    s = c3.aggregate_difference(_trials([
        (0.0, 51.0, 0.0, 50.2),
        (0.0, 52.0, 0.0, 49.0),
        (0.0, 49.0, 0.0, 51.0),
    ]))
    assert s["conditions"]["effect_size_reaches_threshold"] is False
    assert any("Cohen" in r for r in s["reasons_not_observable"])


def test_aggregate____________CI____():
    """「还不到能算的阶段」与「算了但与零不可区分」必须能区分。"""
    s = c3.aggregate_difference(_trials([(0.0, 50.0, 0.0, 10.0)]))
    assert any("样本量" in r for r in s["reasons_not_observable"])


def test_aggregate__trial____():
    s = c3.aggregate_difference([])
    assert s["n_trials"] == 0
    assert s["observable_effect"] is False


def test_objective_summary__________():
    trials = [
        c3.objective_trial(0.0, 100.0, 0.0, 0.0),
        c3.objective_trial(0.0, 100.0, 0.0, 0.0),
    ]
    s = c3.objective_difference_summary(trials)
    for k in c3.FIVE_LENSES:
        assert k in s


# ==========================================================================
# 4. 结论许可
# ==========================================================================


def _good_summary() -> dict:
    return c3.aggregate_difference(_trials([
        (0.0, 60.0, 0.0, 10.0),
        (0.0, 62.0, 0.0, 11.0),
        (0.0, 58.0, 0.0, 9.0),
    ]))


def _bad_summary() -> dict:
    return c3.aggregate_difference(_trials([(0.0, 50.0, 0.0, 10.0)]))


def test_claim_permission_________():
    p = c3.claim_permission(_good_summary(), True, 3)
    assert p["may_attribute_to_learning"] is True
    assert p["blocked_by"] == []


def test_claim_permission_blocks_on_contamination():
    p = c3.claim_permission(_good_summary(), False, 3)
    assert p["may_attribute_to_learning"] is False
    assert any("污染" in b for b in p["blocked_by"])


def test_claim_permission________():
    p = c3.claim_permission(_good_summary(), True, 1)
    assert p["may_attribute_to_learning"] is False
    assert p["conditions"]["enough_trials"] is False


def test_claim_permission_CI_0___():
    p = c3.claim_permission(_bad_summary(), True, 3)
    assert p["may_attribute_to_learning"] is False


def test_claim_permission_lists_all_failed_conditions():
    p = c3.claim_permission(_bad_summary(), False, 1)
    assert len(p["blocked_by"]) == 4      # 污染 + CI + 效应量 + 样本量


def test_claim_permission_exposes_all_four_conditions():
    p = c3.claim_permission(_good_summary(), True, 3)
    assert set(p["conditions"]) == {
        "no_contamination", "ci_excludes_zero",
        "effect_size_reaches_threshold", "enough_trials",
    }


def test_claim_permission___________():
    p = c3.claim_permission(_bad_summary(), True, 1)
    assert "不得表述为" in p["required_phrasing"]


# ==========================================================================
# 5. 越权措辞门禁
# ==========================================================================


@pytest.mark.parametrize("claim", [
    "训练成功",
    "模型已经学会了这套规则",
    "这次训练成功了",
    "它的能力已经改变",
    "证明了学习有效",
    "确实提升了 20 分",
])
def test_scan________(claim):
    assert c3.scan_forbidden_claims(claim) != []


def test_scan________():
    text = "观测到学习组 POST 64.25 分，对照组 22.4 分。"
    assert c3.scan_forbidden_claims(text) == []


def test_scan________():
    assert c3.scan_forbidden_claims("本次不能声称训练成功") == []


def test_scan_____________():
    hits = c3.scan_forbidden_claims("这个差异可归因于学习", may_attribute=False)
    assert "归因于学习" in hits


def test_scan_allows_attribution_when_permitted():
    """拿到许可后报告本来就要写「归因于学习」，
    不放行等于门禁把合规报告也拦下来。"""
    hits = c3.scan_forbidden_claims("这个差异可归因于学习", may_attribute=True)
    assert "归因于学习" not in hits


def test_scan____________():
    """归因许可只放行归因类措辞，「训练成功」这种更硬的断言仍要拦。"""
    hits = c3.scan_forbidden_claims("训练成功", may_attribute=True)
    assert "训练成功" in hits


def test_scan__________():
    assert c3.scan_forbidden_claims("这次训练失败") != []


def test_assert_claims_allowed______():
    with pytest.raises(RuntimeError, match="未经证据支持"):
        c3.assert_claims_allowed("训练成功", may_attribute=False)


def test_assert_claims_allowed______():
    c3.assert_claims_allowed("两组 POST 分数不同。", may_attribute=False)


def test__________():
    assert len(set(c3.FIVE_LENSES)) == 5
