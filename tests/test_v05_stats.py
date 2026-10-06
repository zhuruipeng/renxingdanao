"""tests/test_v05_stats.py —— V0.5 统计层与报告层的测试。

这一层测的不是「算得对不对」——那太浅了——而是
**统计层有没有把不该下的结论悄悄下了**。

V0.4.1 的两个真实翻车：
1. CI 跨 0 被写成「学习无效」——跨 0 只说明证据不足。
2. PRE 已经 90 分还在讨论「为什么增益小」——那是天花板，不是「没用」。
这两类错误都不会让程序崩，只会让报告说错话，所以必须测。
"""

from __future__ import annotations

import json

import pytest

from brain import report_v2 as r2
from brain import stats_v2 as st


# ==========================================================================
# 一、基础统计
# ==========================================================================


def test_mean():
    assert st.mean([1.0, 2.0, 3.0]) == 2.0
    assert st.mean([]) == 0.0


def test_stdev_uses_n_minus_1():
    #样本标准差：2.0；若误用总体标准差（n）会得1.63
    assert st.stdev([1.0, 2.0, 3.0]) == 1.0
    assert st.stdev([5.0]) == 0.0
    assert st.stdev([]) == 0.0


def test_stdev_single_sample_is_zero():
    """单样本没有可估的离散度，返回 0 而不是报错或 NaN。"""
    assert st.stdev([42.0]) == 0.0


def test_t_table_values():
    assert st.t_critical(1)[0] == 12.706
    assert st.t_critical(4)[0] == 2.776
    assert st.t_critical(9)[0] == 2.262
    assert st.t_critical(19)[0] == 2.093
    assert st.t_critical(30)[0] == 2.042


def test_t_table_switches_to_normal_above_30():
    t, method = st.t_critical(31)
    assert t == 1.96
    assert "normal" in method


def test_t_critical_zero_df_does_not_crash():
    assert st.t_critical(0)[0] > 0


# ==========================================================================
# 二、置信区间
# ==========================================================================


def test_ci_of_one_sample_is_refused():
    """单样本的区间退化为 [x, x]，看起来精确，实际什么都没说。"""
    ci = st.confidence_interval([70.0])
    assert ci["ci_low"] is None and ci["ci_high"] is None
    assert ci["method"] == "insufficient"
    assert ci["note"]


def test_ci_contains_mean():
    vals = [60.0, 65.0, 70.0, 75.0, 80.0]
    ci = st.confidence_interval(vals)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]


def test_ci_narrows_with_more_samples():
    few = st.confidence_interval([60.0, 70.0, 80.0])
    many = st.confidence_interval([60.0, 70.0, 80.0] * 5)
    width_few = few["ci_high"] - few["ci_low"]
    width_many = many["ci_high"] - many["ci_low"]
    assert width_many < width_few


def test_zero_variance_gets_caveat():
    """σ=0 的零宽度区间最容易被误读成「结果非常精确」。"""
    ci = st.confidence_interval([70.0, 70.0, 70.0])
    assert ci["method"] == "zero_variance"
    assert "方差" in ci["caveat"]


def test_zero_variance_caveat_propagates_through_describe():
    d = st.describe([70.0, 70.0])
    assert d["method"] == "zero_variance"
    assert "caveat" in d


def test_describe_of_empty_series():
    d = st.describe([])
    assert d["n"] == 0 and d["method"] == "empty"
    assert d["ci_low"] is None


# ==========================================================================
# 三、天花板 / 地板
# ==========================================================================


def test_ceiling_hit_at_90():
    d = st.detect_ceiling(90.0)
    assert d["hit"] is True
    assert d["headroom"] == 10.0
    assert "没有可测量的空间" in d["note"]


def test_ceiling_boundary_is_inclusive():
    assert st.detect_ceiling(85.0)["hit"] is True
    assert st.detect_ceiling(84.9)["hit"] is False


def test_no_ceiling_leaves_headroom():
    d = st.detect_ceiling(12.0)
    assert d["hit"] is False
    assert d["headroom"] == 88.0


def test_floor_hit_at_5():
    d = st.detect_floor(5.0)
    assert d["hit"] is True
    assert "本来就不会" in d["note"]


def test_floor_boundary_is_inclusive():
    assert st.detect_floor(15.0)["hit"] is True
    assert st.detect_floor(15.1)["hit"] is False


def test_ceiling_and_floor_can_both_miss():
    assert st.detect_ceiling(50.0)["hit"] is False
    assert st.detect_floor(50.0)["hit"] == False


# ==========================================================================
# 四、gain 判定
# ==========================================================================


def test_stable_positive_gain():
    v = st.verdict_from_gain([10.0, 12.0, 11.0, 13.0, 9.0])
    assert v["verdict"] == "positive"


def test_ci_crossing_zero_is_noisy_not_negative():
    """V0.4.1 犯过的错：CI 跨 0 被说成「学习无效」。

    跨 0 只能说明**证据不足**，不能说明**效应为零**，
    所以判定词是 noisy，绝不是 negative。
    """
    # 效应量要够大（> MIN_EFFECT）才能落进「CI 跨 0」这条分支，
    # 否则会被更早的「效应太小」分支截走，测的就不是同一件事。
    v = st.verdict_from_gain([20.0, -30.0, 25.0, -20.0, 22.0])
    assert v["verdict"] == "noisy"
    assert v["ci_low"] < 0 < v["ci_high"]
    assert "证据不足" in v["reason"]
    assert abs(v["mean"]) >= st.MIN_EFFECT


def test_no_verdict_value_is_called_negative():
    """防回归：整个统计层不允许出现 negative 判定值。"""
    for series in ([1.0, -3.0, 4.0], [0.0, 0.0, 0.0], [-5.0, -6.0, -4.0]):
        v = st.verdict_from_gain(series)
        assert v["verdict"] in ("positive", "non_positive", "noisy")
        assert v["verdict"] != "negative"


def test_stably_non_positive_gain():
    v = st.verdict_from_gain([-8.0, -9.0, -7.0, -10.0])
    assert v["verdict"] == "non_positive"


def test_tiny_effect_called_noisy():
    """方向明确但幅度只有 0.5 分：不能报成 positive。"""
    v = st.verdict_from_gain([0.5, 0.4, 0.6, 0.5])
    assert v["verdict"] == "noisy"


def test_single_measurement_is_noisy():
    v = st.verdict_from_gain([30.0])
    assert v["verdict"] == "noisy"
    assert "只有 1 次测量" in v["reason"]


def test_zero_variance_positive_is_marked_provisional():
    """确定性 provider 下全同分，positive 只能算暂定。"""
    v = st.verdict_from_gain([20.0, 20.0, 20.0])
    assert v["verdict"] == "positive"
    assert v["provisional"] is True


# ==========================================================================
# 五、聚合
# ==========================================================================


def _trial(eid: int, pre: float, ob_: float, mb: float, tr: float,
           op: float, nq: float = 60.0) -> dict:
    return {
        "experiment_id": eid,
        "trial_no": eid,
        "scores": {
            "pre_blind": pre,
            "open_book": ob_,
            "memory_blind": mb,
            "transfer_blind": tr,
            "objective_practice_avg": op,
            "objective_practice_pass_rate": 1.0,
            "evaluator_practice_avg": 40.0,
            "note_quality_avg": nq,
        },
        "per_kp_rows": [],
    }


def _good_batch(n: int = 6) -> list[dict]:
    return [
        _trial(i, 12.0, 66.0 + i * 0.5, 53.0 + i, 41.0 + i, 73.0 + i)
        for i in range(1, n + 1)
    ]


def test_aggregate_empty_batch():
    s = st.aggregate_trials([])
    assert s["trials"] == 0
    assert s["errors"]


def test_aggregate_basic_metrics():
    s = st.aggregate_trials(_good_batch(6))
    assert s["trials"] == 6
    m = s["metrics"]["pre_blind"]
    assert m["n"] == 6
    assert m["mean"] == 12.0
    assert m["std"] == 0.0
    assert m["ci_low"] is not None


def test_aggregate_computes_all_four_gains():
    s = st.aggregate_trials(_good_batch(6))
    for k in ("external_memory_gain", "memory_retention_gain",
              "transfer_gain", "objective_practice_gain"):
        assert k in s["gains"], k
        assert s["gains"][k]["description"]


def test_gain_definitions_use_correct_bases():
    """四类gain 的减数各不相同——把它们混为一谈是V0.4 的原罪。"""
    assert st._GAIN_BASE["external_memory_gain"] == ("open_book", "pre_blind")
    assert st._GAIN_BASE["memory_retention_gain"] == ("memory_blind", "pre_blind")
    assert st._GAIN_BASE["transfer_gain"] == ("transfer_blind", "pre_blind")
    assert st._GAIN_BASE["objective_practice_gain"] == (
        "objective_practice_avg", "pre_blind",
    )


def test_aggregate_detects_ceiling_and_floor():
    s = st.aggregate_trials([_trial(i, 92.0, 95.0, 94.0, 93.0, 96.0)
                             for i in range(1, 6)])
    assert s["ceiling"]["hit"] is True
    assert s["floor"]["hit"] is False


def test_aggregate_detects_floor():
    s = st.aggregate_trials(_good_batch(6))
    assert s["floor"]["hit"] is True


def test_sample_size_problem_when_too_few_trials():
    s = st.aggregate_trials(_good_batch(2))
    assert s["sample_size_diagnostics"]["sufficient_trials"] is False
    assert s["sample_size_diagnostics"]["problems"]


def test_per_kp_flags_insufficient_sample():
    trials = [{
        **_trial(1, 10.0, 60.0, 50.0, 40.0, 70.0),
        "per_kp_rows": [
            {"knowledge_point": "kp_a", "stage": "pre_blind", "score": 10.0},
        ],
    }]
    s = st.aggregate_trials(trials)
    entry = s["per_kp"]["kp_a"]["pre_blind"]
    assert entry["n"] == 1
    assert entry["insufficient_sample"] is True
    assert "不具备统计意义" in entry["caveat"]


def test_per_kp_clears_at_minimum_sample():
    rows = [{"knowledge_point": "kp_a", "stage": "pre_blind", "score": 10.0 + i}
            for i in range(4)]
    trials = [{**_trial(i + 1, 10.0, 60.0, 50.0, 40.0, 70.0), "per_kp_rows": rows}
              for i in range(4)]
    s = st.aggregate_trials(trials)
    assert s["per_kp"]["kp_a"]["pre_blind"]["insufficient_sample"] is False


def test_high_variance_gain_detected():
    noisy = [
        _trial(1, 12.0, 90.0, 50.0, 40.0, 70.0),
        _trial(2, 12.0, 20.0, 50.0, 40.0, 70.0),
        _trial(3, 12.0, 60.0, 50.0, 40.0, 70.0),
        _trial(4, 12.0, 35.0, 50.0, 40.0, 70.0),
    ]
    s = st.aggregate_trials(noisy)
    assert "external_memory_gain" in s["sample_size_diagnostics"]["high_variance_gains"]


def test_diagnose_batch_reports_unreliable_when_floor_hit():
    s = st.aggregate_trials(_good_batch(6))
    d = st.diagnose_batch(s)
    assert d["has_floor"] is True
    assert d["reliable"] is False
    assert d["caveats"]


def test_zero_variance_across_trials_is_diagnosed():
    """确定性 provider 下跑 5 次＝同一次抄 5 遍。

    这时若不报「无离散度」，`sufficient_trials=True` 与
    `[70, 70]` 这样的零宽度区间会一起把读者骗了。
    """
    same = [_trial(i, 70.0, 70.0, 70.0, 70.0, 100.0) for i in range(1, 6)]
    ssd = st.aggregate_trials(same)["sample_size_diagnostics"]
    assert ssd["n_trials"] == 5
    assert ssd["sufficient_trials"] is True
    assert ssd["all_zero_variance"] is True
    assert ssd["has_variation"] is False
    assert any("方差都是 0" in p for p in ssd["problems"])


def test_diagnose_batch_unreliable_on_zero_variance():
    same = [_trial(i, 70.0, 70.0, 70.0, 70.0, 100.0) for i in range(1, 6)]
    d = st.diagnose_batch(st.aggregate_trials(same))
    assert d["reliable"] is False, "方差为 0 时不允许报「可直接下结论」"


def test_partial_zero_variance_is_listed():
    """只有一个 gain 无方差时，列出来即可，不该说「全部无方差」。"""
    trials = [
        _trial(1, 70.0, 90.0, 70.0, 70.0, 100.0),
        _trial(2, 70.0, 50.0, 70.0, 70.0, 100.0),
        _trial(3, 70.0, 70.0, 70.0, 70.0, 100.0),
    ]
    ssd = st.aggregate_trials(trials)["sample_size_diagnostics"]
    assert "external_memory_gain" not in ssd["zero_variance_gains"]
    assert "transfer_gain" in ssd["zero_variance_gains"]
    assert ssd["all_zero_variance"] is False


def test_diagnose_batch_reliable_on_healthy_batch():
    """四项指标都要有真实离散度，否则会被方差诊断拦下——
    「没有诊断问题」不等于「样本多」，还要求样本之间确实不同。"""
    # 每段分数用**不同**的波动模式：若 memory = pre + 16 恒成立，
    # 那 memory_retention_gain 恒为 16、方差为 0 —— 这是测试数据的问题，
    # 不是统计层放过的问题。
    trials = [
        _trial(i,
               50.0 + i,
               70.0 + i * 2,
               60.0 + i * 3,
               55.0 + i * 4,
               75.0 + i * 5)
        for i in range(1, 7)
    ]
    # 补足逐 kp 样本，避免 thin_kps 拉低 reliable
    for t in trials:
        t["per_kp_rows"] = [
            {"knowledge_point": f"kp_{j}", "stage": "pre_blind",
             "score": 50.0 + j + t["trial_no"]}
            for j in range(4)
        ]
    d = st.diagnose_batch(st.aggregate_trials(trials))
    assert d["reliable"] is True, d["caveats"]


# ==========================================================================
# 六、报告层：禁止称「训练成功」
# ==========================================================================


def test_scan_detects_forbidden_phrases():
    assert r2.scan_forbidden("结论：模型训练成功")
    assert r2.scan_forbidden("已掌握该知识")
    assert r2.scan_forbidden("模型学会了")


def test_scan_passes_clean_text():
    assert r2.scan_forbidden("TRANSFER 增益为正，说明能处理新场景") == []


@pytest.mark.parametrize("phrase", r2.FORBIDDEN_PHRASES)
def test_assert_no_forbidden_raises(phrase: str):
    with pytest.raises(AssertionError) as ei:
        r2.assert_no_forbidden(f"本次实验{phrase}。")
    assert "训练" in str(ei.value) or "掌握" in str(ei.value) or "学会" in str(ei.value)


def test_forbidden_list_contains_training_success():
    assert "训练成功" in r2.FORBIDDEN_PHRASES


def test_transfer_claim_explicitly_denies_training():
    row = [r for r in r2._claim_table({"transfer_gain": {"mean": 30.0}})
           if r["metric"] == "transfer_gain"][0]
    assert "训练" in row["does_not_support"]


def test_negated_phrase_is_not_a_violation():
    """报告必须能自我限定——「本指标不代表训练成功」不算越权断言。"""
    assert r2.scan_forbidden("本指标不代表训练成功") == []
    assert r2.scan_forbidden("这不是训练成功") == []
    assert r2.scan_forbidden("不能称为「训练成功」") == []


def test_bare_assertion_is_still_a_violation():
    assert r2.scan_forbidden("实验表明训练成功") == ["训练成功"]
    assert r2.scan_forbidden("该模型已掌握该知识") == ["已掌握该知识"]


# ==========================================================================
# 七、报告层：结构
# ==========================================================================


def test_build_batch_report_shape():
    s = st.aggregate_trials(_good_batch(6))
    rep = r2.build_batch_report(s, {"batch_id": "b1", "model": "scripted"})
    assert rep["version"] == "V0.5"
    assert rep["trials"] == 6
    assert rep["model"] == "scripted"
    assert len(rep["claims"]) == 4


def test_build_batch_report_has_frozen_fingerprints():
    s = st.aggregate_trials(_good_batch(6))
    rep = r2.build_batch_report(s, {
        "batch_id": "b1",
        "bank_fingerprint": "a" * 64,
        "material_checksum": "b" * 64,
        "tasks_fingerprint": "c" * 64,
    })
    assert rep["frozen"]["bank_fingerprint"] == "a" * 64
    assert rep["frozen"]["material_checksum"] == "b" * 64


def test_claims_table_binds_supports_and_not_supports():
    s = st.aggregate_trials(_good_batch(6))
    rep = r2.build_batch_report(s, {})
    for row in rep["claims"]:
        assert row["supports"]
        assert row["does_not_support"]


def test_batch_text_contains_all_sections():
    s = st.aggregate_trials(_good_batch(6))
    txt = r2.format_batch_text(r2.build_batch_report(s, {"batch_id": "b1"}))
    for marker in ("口径声明", "四项观测指标", "四类 Gain", "天花板",
                   "样本量与方差诊断", "逐知识点", "结论可用性", "冻结指纹"):
        assert marker in txt, f"报告缺少章节：{marker}"


def test_batch_text_has_no_forbidden_phrase():
    s = st.aggregate_trials(_good_batch(6))
    txt = r2.format_batch_text(r2.build_batch_report(s, {}))
    assert r2.scan_forbidden(txt) == []


def test_batch_text_shows_ci_columns():
    s = st.aggregate_trials(_good_batch(6))
    txt = r2.format_batch_text(r2.build_batch_report(s, {}))
    assert "95% CI" in txt
    assert "标准差" in txt or "σ" in txt


def test_batch_text_flags_insufficient_trials():
    s = st.aggregate_trials(_good_batch(2))
    txt = r2.format_batch_text(r2.build_batch_report(s, {}))
    assert "样本量与方差诊断" in txt
    assert "建议 ≥5" in txt or "样本量不足" in txt


# ==========================================================================
# 八、机器分与 LLM 分必须分表
# ==========================================================================


def _practice_inputs():
    runs = [
        {"task_id": "obj_hash_01", "kp": "kp_lumen_hash", "kind": "algorithm_output",
         "title": "T1", "objective_score": 100.0, "objective_passed": True,
         "parse_error": "", "objective_result": {"ok": True}},
        {"task_id": "obj_hash_02", "kp": "kp_lumen_hash", "kind": "python_unit_test",
         "title": "T2", "objective_score": 20.0, "objective_passed": False,
         "parse_error": "", "objective_result": {"ok": False}},
    ]
    evals = [
        {"task_id": "obj_hash_01", "kp": "kp_lumen_hash", "score": 40.0,
         "usable": False, "verdict": "poor", "feedback": "写得对但没法用"},
        {"task_id": "obj_hash_02", "kp": "kp_lumen_hash", "score": 80.0,
         "usable": True, "verdict": "good", "feedback": "看起来不错"},
    ]
    return runs, evals


def test_practice_tables_keep_objective_and_evaluator_apart():
    runs, evals = _practice_inputs()
    t = r2._practice_tables(runs, evals)
    assert t["objective"]["avg"] == 60.0
    assert t["evaluator"]["avg"] == 60.0
    # 两表行数与内容各自独立
    assert len(t["objective"]["rows"]) == 2
    assert len(t["evaluator"]["rows"]) == 2
    assert "objective_score" in t["objective"]["rows"][0]
    assert "evaluator_score" in t["evaluator"]["rows"][0]


def test_practice_tables_never_merge_scores():
    """不提供任何加权平均字段——一旦合并就再也分不开是谁给的。"""
    runs, evals = _practice_inputs()
    t = r2._practice_tables(runs, evals)
    assert "combined" not in t
    assert "overall" not in t
    assert "weighted" not in t


def test_divergence_flags_disagreement():
    runs, evals = _practice_inputs()
    t = r2._practice_tables(runs, evals)
    assert t["divergence"]["count"] == 2
    directions = {r["direction"] for r in t["divergence"]["rows"]}
    assert directions == {"objective_higher", "llm_higher"}


def test_divergence_quiet_when_aligned():
    runs = [{"task_id": "t1", "kp": "k", "kind": "sql_exec", "title": "T",
             "objective_score": 80.0, "objective_passed": True,
             "parse_error": "", "objective_result": {}}]
    evals = [{"task_id": "t1", "kp": "k", "score": 82.0, "usable": True,
              "verdict": "ok", "feedback": ""}]
    assert r2._practice_tables(runs, evals)["divergence"]["count"] == 0


# ==========================================================================
# 九、报告层：隔离自检与答案泄漏
# ==========================================================================


def _stage(stage: str, **kw) -> dict:
    base = {
        "stage": stage,
        "item_id": f"it_{stage}",
        "context_used": "",
        "question": "题干",
        "reference_answer": "参考答案正文一二三四五",
    }
    base.update(kw)
    return base


_ALL_STAGES = (
    _stage("pre_blind"),
    _stage("open_book", context_used="笔记内容"),
    _stage("memory_blind"),
    _stage("transfer_blind"),
)


def test_consistency_passes_on_clean_stage_items():
    c = r2._consistency(_ALL_STAGES)
    assert c["blind_context_empty"] is True
    assert c["open_book_context_present"] is True
    assert c["answer_leak"] is False
    assert c["problems"] == []


def test_consistency_catches_dirty_blind_context():
    c = r2._consistency([_stage("pre_blind", context_used="偷看了笔记")])
    assert c["blind_context_empty"] is False
    assert any("盲考被污染" in p for p in c["problems"])


def test_consistency_catches_answer_in_context():
    """答案泄进 context 是最严重的污染——考生直接读到答案。"""
    c = r2._consistency([
        _stage("open_book", context_used="笔记里写着：参考答案正文一二三四五"),
    ])
    assert c["answer_leak"] is True
    assert any("泄进 prompt" in p for p in c["problems"])


def test_consistency_catches_answer_in_question():
    c = r2._consistency([
        _stage("transfer_blind", question="答案就是参考答案正文一二三四五，对吗"),
    ])
    assert c["answer_leak"] is True


def test_reference_answer_field_alone_is_not_a_leak():
    """reference_answer 必须落库（否则无法复核判分），它本身不是泄漏。"""
    c = r2._consistency([
        _stage("pre_blind", reference_answer="参考答案正文一二三四五"),
    ])
    assert c["answer_leak"] is False


def test_consistency_flags_missing_stage():
    c = r2._consistency([_stage("pre_blind")])
    assert any("缺少" in p for p in c["problems"])


def test_short_answer_does_not_trigger_false_leak():
    c = r2._consistency([_stage("open_book", context_used="是", reference_answer="是")])
    assert c["answer_leak"] is False, "单字答案不能当泄漏，否则全是误报"


# ==========================================================================
# 十、单次 trial 报告
# ==========================================================================


def _trial_payload() -> dict:
    return {
        "experiment_id": 1,
        "batch_id": "b1",
        "trial_no": 1,
        "scores": {
            "pre_blind": 10.0, "open_book": 60.0, "memory_blind": 50.0,
            "transfer_blind": 40.0, "objective_practice_avg": 75.0,
            "objective_practice_pass_rate": 0.75,
            "evaluator_practice_avg": 55.0, "note_quality_avg": 66.0,
        },
        "detail": {"stage_order": ["pre_blind", "study", "practice",
                                   "open_book", "memory_blind", "transfer_blind"]},
        "diagnostics": {},
    }


def test_build_trial_report_shape():
    runs, evals = _practice_inputs()
    rep = r2.build_trial_report(_trial_payload(), runs, evals, _ALL_STAGES)
    assert rep["version"] == "V0.5"
    assert rep["gains"]["external_memory_gain"] == 50.0
    assert rep["gains"]["memory_retention_gain"] == 40.0
    assert rep["gains"]["transfer_gain"] == 30.0
    assert rep["gains"]["objective_practice_gain"] == 65.0


def test_trial_report_transfer_vs_memory():
    """transfer< memory 说明「背住了但换场景不会用」。"""
    rep = r2.build_trial_report(_trial_payload(), [], [], _ALL_STAGES)
    assert rep["transfer_vs_memory"] == -10.0


def test_trial_report_lists_parse_errors_as_caveats():
    runs = [{"task_id": "t1", "kp": "k", "kind": "sql_exec", "title": "T",
             "objective_score": 0.0, "objective_passed": False,
             "parse_error": "不是合法 JSON", "objective_result": {}}]
    rep = r2.build_trial_report(_trial_payload(), runs, [], _ALL_STAGES)
    assert any("无法解析" in c for c in rep["caveats"])


def test_trial_report_warns_when_no_practice_rows():
    rep = r2.build_trial_report(_trial_payload(), [], [], _ALL_STAGES)
    assert any("没有" in c for c in rep["caveats"])


def test_trial_text_has_no_forbidden_phrase():
    runs, evals = _practice_inputs()
    txt = r2.format_trial_text(
        r2.build_trial_report(_trial_payload(), runs, evals, _ALL_STAGES)
    )
    assert r2.scan_forbidden(txt) == []
    assert "禁止称训练成功" in txt


def test_trial_text_shows_both_practice_sections():
    runs, evals = _practice_inputs()
    txt = r2.format_trial_text(
        r2.build_trial_report(_trial_payload(), runs, evals, _ALL_STAGES)
    )
    assert "机器判分（objective）" in txt
    assert "LLM 评审（evaluator）" in txt


def test_trial_report_json_serialisable():
    runs, evals = _practice_inputs()
    rep = r2.build_trial_report(_trial_payload(), runs, evals, _ALL_STAGES)
    json.dumps(rep, ensure_ascii=False)
