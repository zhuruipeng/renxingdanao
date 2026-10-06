"""tests/test_v06_isolation.py —— V0.6 隔离层：材料 / 组 / held-out / evaluator / checksum。

这一层测的是**实验合不合法**，不测分数好不好。
全部不联网、不调真实模型；污染检测那几条用真实 SQLite 库
构造污染场景，验证门禁真的能抓住——而不是只验证"干净时不报警"。

只测"干净时通过"是不够的：一套永远不会报警的检测器也能通过那些测试。
所以这里每一条检测都配一个**注入污染后必须报警**的反例。
"""

from __future__ import annotations

import pytest

from brain import benchmark_v2 as b2
from brain import causal_v3 as c3
from brain import control_material as cm
from brain import group_isolation as gi
from brain import heldout_bank as ho
from brain import novel_material as nm
from memory.db import Database, checksum
from practice.objective import OBJECTIVE_TASKS

# ==========================================================================
# 1. 对照材料
# ==========================================================================


def test_control_material_length_within_range():
    assert cm.LENGTH_RATIO_MIN <= cm.length_ratio() <= cm.LENGTH_RATIO_MAX


def test_control_material_has_no_leak():
    for label, hits in cm.leakage_hits().items():
        assert hits == [], f"对照材料在 {label} 上泄漏：{hits}"


def test_control_material_checksum_stable():
    assert checksum(cm.CONTROL_MATERIAL) == checksum(cm.CONTROL_MATERIAL)


def test_control_material_is_not_learning_material():
    assert cm.CONTROL_MATERIAL != nm.NOVEL_MATERIAL


def test_assert_control_intact_passes():
    cm.assert_control_intact()


def test_control_material_mentions_no_target_answer():
    """对照材料里不能出现目标系统的核心常数组合。"""
    for probe in ("65537", "LumenHash", "Z-Tag", "zephyra_buckets"):
        assert probe not in cm.CONTROL_MATERIAL, f"对照材料泄漏 {probe}"


# ==========================================================================
# 2. 组作用域
# ==========================================================================


def test_build_scopes_distinct_ids():
    scopes = gi.build_scopes("r1", 1, 2)
    assert scopes["learning"].experiment_id != scopes["control"].experiment_id


def test_build_scopes_rejects_shared_id():
    with pytest.raises(RuntimeError, match="共用"):
        gi.build_scopes("r1", 7, 7)


def test_group_scope_stage_of():
    scopes = gi.build_scopes("r1", 1, 2)
    assert scopes["learning"].stage_of("pre") == "pre_learning"
    assert scopes["control"].stage_of("pre") == "pre_control"


def test_group_scope_peer():
    scopes = gi.build_scopes("r1", 1, 2)
    assert scopes["learning"].peer() == "control"
    assert scopes["control"].peer() == "learning"


def test_group_scope_is_learning():
    scopes = gi.build_scopes("r1", 1, 2)
    assert scopes["learning"].is_learning is True
    assert scopes["control"].is_learning is False


def test_group_scope_rejects_unknown_group():
    with pytest.raises(ValueError):
        gi.GroupScope(
            run_id="r", exp_group="other", experiment_id=1,
            material_kind="k", material_text="t",
        )


def test_ten_stages_fixed_order():
    assert len(gi.STAGES_V3) == 10
    assert gi.STAGES_V3[:2] == ("pre_learning", "pre_control")
    assert gi.STAGES_V3[2:4] == ("study_learning", "control_material")
    assert gi.STAGES_V3[4:6] == ("practice_learning", "practice_control")
    assert gi.STAGES_V3[6:8] == ("post_learning", "post_control")
    assert gi.STAGES_V3[8:] == ("transfer_learning", "transfer_control")


def test_both_pre_before_any_material():
    pre_max = max(
        gi.STAGES_V3.index(s) for s in ("pre_learning", "pre_control")
    )
    for s in ("study_learning", "control_material"):
        assert gi.STAGES_V3.index(s) > pre_max


def test_blind_stages_cover_all_exams():
    for s in ("pre_learning", "pre_control", "post_learning",
              "post_control", "transfer_learning", "transfer_control"):
        assert s in gi.BLIND_STAGES_V3


# ==========================================================================
# 3. 隔离静态诊断
# ==========================================================================


def test_isolation_diagnostics_clean():
    assert gi.isolation_diagnostics()["problems"] == []


def test_assert_isolation_intact_passes():
    gi.assert_isolation_intact()


def test_pre_and_heldout_item_ids_disjoint():
    pre = {i.item_id for i in b2.PRE_ITEMS}
    held = {i.item_id for i in ho.ALL_HELDOUT_ITEMS}
    assert pre & held == set()


def test_heldout_not_in_learning_material():
    for it in ho.ALL_HELDOUT_ITEMS:
        core = it.question.strip()[:40]
        if len(core) >= 20:
            assert core not in nm.NOVEL_MATERIAL


def test_heldout_not_in_control_material():
    for it in ho.ALL_HELDOUT_ITEMS:
        core = it.question.strip()[:40]
        if len(core) >= 20:
            assert core not in cm.CONTROL_MATERIAL


# ==========================================================================
# 4. held-out 隔离
# ==========================================================================


def test_practice_param_overlap_empty():
    assert ho.practice_param_overlap() == {}


def test_assert_heldout_isolated_passes():
    ho.assert_heldout_isolated()


def test_assert_heldout_intact_passes():
    ho.assert_heldout_intact()


def test_assert_heldout_scorers_intact_passes():
    ho.assert_heldout_scorers_intact()


def test_heldout_eight_tasks():
    assert len(ho.HELD_OUT_TASKS) == 8


def test_heldout_two_tasks_per_kp():
    counts: dict[str, int] = {}
    for t in ho.HELD_OUT_TASKS:
        counts[t.kp] = counts.get(t.kp, 0) + 1
    assert set(counts.values()) == {2}
    assert len(counts) == 4


def test_heldout_fingerprint_differs_from_practice():
    from practice.objective import TASKS_FINGERPRINT
    assert ho.HO_TASKS_FINGERPRINT != TASKS_FINGERPRINT


def test_practice_and_heldout_task_ids_disjoint():
    assert {t.task_id for t in OBJECTIVE_TASKS} & {
        t.task_id for t in ho.HELD_OUT_TASKS
    } == set()


def test_invariant_exemptions_are_really_invariants():
    """被豁免的恒等式必须仍等于 7 —— 豁免不能变成漏检。"""
    from practice.objective import ref_lumen_hash
    assert ref_lumen_hash("") == 7


def test_residual_contamination_reported():
    """表达层无法隔离的两类必须显式报出，不能藏起来。"""
    r = ho.residual_contamination()
    assert any(k.startswith("python_unit_test") for k in r), r
    assert any(k.startswith("sql_exec") for k in r), r
    # 表达层无法隔离的两类必须都**非空**，否则等于没报
    assert all(v for v in r.values()), r


# ==========================================================================
# 5. checksum / 冻结指纹
# ==========================================================================


def test_frozen_fingerprints_all_seven_present():
    fp = gi.frozen_fingerprints()
    for k in (
        "pre_fingerprint", "post_fingerprint", "transfer_fingerprint",
        "learning_material_checksum", "control_material_checksum",
        "practice_tasks_fingerprint", "heldout_tasks_fingerprint",
    ):
        assert fp[k], f"缺少冻结指纹 {k}"


def test_frozen_fingerprints_stable():
    assert gi.frozen_fingerprints() == gi.frozen_fingerprints()


def test_two_fingerprints_are_distinct():
    fp = gi.frozen_fingerprints()
    assert fp["pre_fingerprint"] != fp["post_fingerprint"]
    assert fp["post_fingerprint"] != fp["transfer_fingerprint"]


def test_material_checksums_distinct():
    fp = gi.frozen_fingerprints()
    assert fp["learning_material_checksum"] != fp["control_material_checksum"]


# ==========================================================================
# 6. 六类污染检测（干净时通过）
# ==========================================================================


@pytest.fixture()
def clean_db(tmp_path):
    """每次一个独立的库文件。

    用 pytest 的 ``tmp_path`` 而不是自己拼临时目录 + ``os.remove``：
    后者会在Windows 上撞上长路径问题（``\\\\?\\`` 前缀 + 深目录），
    清理失败时会把整个测试进程卡住。
    """
    db = Database(str(tmp_path / "v06_iso.db"))
    db.init_schema()
    yield db
    db.close()


#: ``v3_stage_items`` 的列清单与一行默认值。
#:
#: 测试里插入这张表时**一律走这个助手**：它经过多次迁移
#: （V0.6 加过 context_used 等列），手写 VALUES 极易数错，
#: 而「16 values for 17 columns」这种报错会把人引到
#: 完全无关的方向上。
_STAGE_COLS: tuple[str, ...] = (
    "run_id", "experiment_id", "exp_group", "stage", "seq", "item_id",
    "knowledge_point", "question_type", "question", "answer", "score",
    "context_used", "model", "provider", "evaluator", "reference_answer",
    "mode", "created_at",
)


def _insert_stage_item(
    db: Database,
    *,
    run_id: str,
    exp_group: str,
    stage: str,
    seq: int,
    item_id: str = "x1",
    context_used: str = "",
    mode: str = "blind",
) -> None:
    ph = ",".join("?" for _ in _STAGE_COLS)
    db.execute(
        f"INSERT INTO v3_stage_items({','.join(_STAGE_COLS)}) VALUES ({ph})",
        (run_id, 1, exp_group, stage, seq, item_id, "kp", "application",
         "q", "a", 50.0, context_used, "cm", "cp", "ep", "", mode, "t"),
    )


def _insert_objective(
    db: Database,
    *,
    run_id: str,
    exp_group: str,
    stage: str,
    seq: int,
    task_id: str,
) -> None:
    obj_cols = [
        "run_id", "experiment_id", "exp_group", "stage", "seq", "task_id",
        "kp", "kind", "candidate_output", "checks_json", "objective_score",
        "objective_passed", "parse_error", "model", "provider",
        "context_used", "created_at",
    ]
    ph = ",".join("?" for _ in obj_cols)
    db.execute(
        f"INSERT INTO v3_objective_runs({','.join(obj_cols)}) VALUES ({ph})",
        (run_id, 1, exp_group, stage, seq, task_id, "kp", "k",
         "{}", "[]", 50.0, 1, "", "cm", "cp", "", "t"),
    )


def _seed_clean_run(db: Database, run_id: str = "r1") -> None:
    """落一份结构完整、无污染的 run。"""
    from brain.experiment_v3 import ProviderConfig, RunSpec, V3Store
    from brain.llm import LLMProvider, LLMResponse

    class Ok(LLMProvider):
        def complete(self, messages, *, temperature=0.7, max_tokens=None,
                     json_mode=False):
            return LLMResponse(text="{}", model="m")

    spec = RunSpec(
        run_id=run_id, trial_no=1, topic="t",
        candidate=ProviderConfig("cp", "cm", Ok()),
        evaluator=ProviderConfig("ep", "em", Ok()),
    )
    store = V3Store(db)
    store.create_run(spec)
    for g, kind, text in (
        ("learning", "novel_zephyra", nm.NOVEL_MATERIAL),
        ("control", "control_veyra", cm.CONTROL_MATERIAL),
    ):
        eid = store.create_group(run_id, g, kind, text)
        db.execute(
            "UPDATE v3_experiments SET exp_group=?, id=id WHERE id=?",
            (g, eid),
        )
    for stage in gi.STAGES_V3:
        seq = gi.STAGES_V3.index(stage) + 1
        group = "learning" if stage.endswith("learning") else "control"
        _insert_stage_item(
            db, run_id=run_id, exp_group=group, stage=stage, seq=seq,
        )
    # seq 必须写真实位置：顺序审计现在**两张表都查**，
    # 客观阶段 seq=0 会被判成「post_objective 早于 practice」。
    for group in ("learning", "control"):
        for stage, seq in (("pre_objective", 3), ("post_objective", 9)):
            for t in ho.HELD_OUT_TASKS:
                _insert_objective(
                    db, run_id=run_id, exp_group=group, stage=stage,
                    seq=seq, task_id=t.task_id,
                )


def test_contamination_clean_on_clean_run(clean_db):
    _seed_clean_run(clean_db)
    rep = gi.contamination_report(clean_db, "r1")
    assert rep["clean"] is True, rep["contaminations"]


def test_contamination_runs_all_six_checks(clean_db):
    _seed_clean_run(clean_db)
    rep = gi.contamination_report(clean_db, "r1")
    assert len(gi._CHECKS) == 7, "六类污染 + 调用失败必须都跑"
    for fn in gi._CHECKS:
        assert fn.__name__ in rep["checks_run"]
    # 干净时 checks_missing 必须为空——它报的是「没人负责的类别」
    assert rep["checks_missing"] == []
    # 六类是用户点名的，调用失败是第七类
    assert len(gi.ALL_CONTAMINATIONS) == 6
    assert len(gi.ALL_CONTAMINATION_KINDS) == 7
    assert set(gi.CHECK_FOR_KIND) == set(gi.ALL_CONTAMINATION_KINDS)


def test_assert_no_contamination_passes_on_clean(clean_db):
    _seed_clean_run(clean_db)
    gi.assert_no_contamination(clean_db, "r1")


def test_order_audit_clean_on_clean_run(clean_db):
    _seed_clean_run(clean_db)
    rep = gi.order_audit(clean_db, "r1")
    assert rep["clean"] is True, rep["problems"]


def test_assert_order_valid_passes_on_clean(clean_db):
    _seed_clean_run(clean_db)
    gi.assert_order_valid(clean_db, "r1")


# ==========================================================================
# 7. 污染检测：注入后必须报警（关键的反例）
# ==========================================================================


def test_detects_shared_learning_notes(clean_db):
    _seed_clean_run(clean_db)
    # learning_notes.experiment_id 是外键，必须指向 experiments 里的真实行。
    # 用参数化 INSERT + 显式列清单，避免手写 VALUES 顺序出错
    # （这列经历过多次V0.5/V0.6 迁移，手数字典序极易对不上）。
    cols = [
        "experiment_id", "topic", "kp", "concept", "principle", "example",
        "common_mistake", "application_rule", "checksum", "created_at",
        "verified", "quality_total", "quality_verdict", "v3_experiment_id",
    ]
    placeholders = ",".join("?" for _ in cols)
    for eid in (1, 2):
        clean_db.execute(
            "INSERT INTO experiments(id, topic, created_at) VALUES (?,?,?)",
            (9000 + eid, "t", "t"),
        )
        clean_db.execute(
            f"INSERT INTO learning_notes({','.join(cols)})"
            f" VALUES ({placeholders})",
            (
                9000 + eid, "t", "kp", "共享概念", "原理", "", "",
                "", "same-checksum", "t", 0, 0, "", eid,
            ),
        )
    hits = gi.check_shared_notes(clean_db, "r1")
    assert hits, "两组笔记 checksum 相同却没报警"


def test_detects_shared_knowledge_content(clean_db):
    _seed_clean_run(clean_db)
    for eid in (1, 2):
        clean_db.execute(
            """INSERT INTO knowledge(topic, title, content, keywords, source,
                 confidence, created_at, updated_at, experiment_id)
               VALUES ('t','x','完全一样的内容','k','s',0.5,'t','t',?)""",
            (eid,),
        )
    hits = gi.check_shared_knowledge(clean_db, "r1")
    assert hits, "两组 knowledge 内容相同却没报警"


def test_detects_heldout_in_study(clean_db):
    _seed_clean_run(clean_db)
    it = ho.HELD_OUT_POST[0]
    _insert_stage_item(
        clean_db, run_id="r1", exp_group="learning",
        stage="study_learning", seq=3, item_id=it.item_id, mode="open",
    )
    hits = gi.check_heldout_in_study(clean_db, "r1")
    assert hits, "held-out 题出现在 STUDY 阶段却没报警"


def test_detects_transfer_in_practice(clean_db):
    _seed_clean_run(clean_db)
    it = ho.HELD_OUT_TRANSFER[0]
    _insert_stage_item(
        clean_db, run_id="r1", exp_group="control",
        stage="practice_control", seq=6, item_id=it.item_id, mode="open",
    )
    hits = gi.check_transfer_in_practice(clean_db, "r1")
    assert hits, "TRANSFER 题出现在 PRACTICE 阶段却没报警"


def test_detects_heldout_task_in_practice(clean_db):
    """客观任务表必须被检查——早先完全没查，污染整个漏了过去。"""
    _seed_clean_run(clean_db)
    clean_db.execute("DELETE FROM v3_objective_runs")
    _insert_objective(
        clean_db, run_id="r1", exp_group="control",
        stage="practice_control", seq=6,
        task_id=ho.HELD_OUT_TASKS[0].task_id,
    )
    hits = gi.check_transfer_in_practice(clean_db, "r1")
    assert hits, "held-out 客观任务出现在 PRACTICE 阶段却没报警"


def test_detects_heldout_task_in_pre(clean_db):
    _seed_clean_run(clean_db)
    clean_db.execute("DELETE FROM v3_objective_runs")
    _insert_objective(
        clean_db, run_id="r1", exp_group="learning",
        stage="pre_learning", seq=1,
        task_id=ho.HELD_OUT_TASKS[0].task_id,
    )
    hits = gi.check_transfer_in_practice(clean_db, "r1")
    assert hits, "held-out 客观任务出现在 PRE 阶段却没报警"


def test_detects_post_objective_wrong_task_set(clean_db):
    """测试卷被换过 → 前后差的分母不同 → 不可比。"""
    _seed_clean_run(clean_db)
    clean_db.execute(
        "DELETE FROM v3_objective_runs WHERE stage='post_objective'"
        " AND exp_group='learning'"
    )
    for t in ho.HELD_OUT_TASKS[:-1]:
        _insert_objective(
            clean_db, run_id="r1", exp_group="learning",
            stage="post_objective", seq=9, task_id=t.task_id,
        )
    hits = gi.check_transfer_in_practice(clean_db, "r1")
    assert hits, "post_objective 少一题却没报警"


def test_detects_blind_stage_with_context(clean_db):
    _seed_clean_run(clean_db)
    clean_db.execute(
        "UPDATE v3_stage_items SET context_used='偷看到的材料'"
        " WHERE run_id='r1' AND stage='post_learning'"
    )
    hits = gi.check_evaluator_leak(clean_db, "r1")
    assert hits, "盲考阶段 context 非空却没报警"


def test_detects_same_candidate_and_evaluator(clean_db):
    _seed_clean_run(clean_db)
    clean_db.execute(
        "UPDATE v3_runs SET evaluator_provider=candidate_provider,"
        " evaluator_model=candidate_model WHERE run_id='r1'"
    )
    hits = gi.check_evaluator_leak(clean_db, "r1")
    assert hits, "考生与评估器同实例却没报警"


def test_detects_call_failures(clean_db):
    _seed_clean_run(clean_db)
    clean_db.execute(
        """INSERT INTO v3_call_failures(run_id, exp_group, stage, item_id,
             error, provider, model, created_at)
           VALUES ('r1','learning','post_learning','x1','Timeout','','','t')"""
    )
    hits = gi.check_call_failures(clean_db, "r1")
    assert hits, "有调用失败却没报警"


def test_detects_checksum_drift(clean_db):
    _seed_clean_run(clean_db)
    clean_db.execute(
        "UPDATE v3_runs SET post_fingerprint='deadbeef' WHERE run_id='r1'"
    )
    hits = gi.check_checksum(clean_db, "r1")
    assert hits, "指纹漂移却没报警"


def test_assert_no_contamination_raises(clean_db):
    _seed_clean_run(clean_db)
    clean_db.execute(
        "UPDATE v3_runs SET post_fingerprint='deadbeef' WHERE run_id='r1'"
    )
    with pytest.raises(RuntimeError, match="污染"):
        gi.assert_no_contamination(clean_db, "r1")


def test_detects_out_of_order_stage(clean_db):
    _seed_clean_run(clean_db)
    # 把 STUDY 的 seq 改到 PRE 之后 —— 顺序不再是「先跑完两个 PRE」
    clean_db.execute(
        "UPDATE v3_stage_items SET seq=9 WHERE run_id='r1'"
        " AND stage='study_learning'"
    )
    rep = gi.order_audit(clean_db, "r1")
    assert rep["clean"] is False
    assert rep["problems"]


def test_detects_missing_pre_stage(clean_db):
    _seed_clean_run(clean_db)
    clean_db.execute(
        "DELETE FROM v3_stage_items WHERE run_id='r1' AND stage='pre_control'"
    )
    rep = gi.order_audit(clean_db, "r1")
    assert rep["clean"] is False


# ==========================================================================
# 8. 隔离 → 归因许可的联动
# ==========================================================================


def test_contamination_blocks_attribution(clean_db):
    """污染存在时，即使统计量漂亮也不许归因。"""
    _seed_clean_run(clean_db)
    clean_db.execute(
        "UPDATE v3_runs SET post_fingerprint='deadbeef' WHERE run_id='r1'"
    )
    rep = gi.contamination_report(clean_db, "r1")
    trials = [
        c3.paired_trial(0.0, 60.0, 0.0, 10.0),
        c3.paired_trial(0.0, 62.0, 0.0, 11.0),
        c3.paired_trial(0.0, 58.0, 0.0, 9.0),
    ]
    s = c3.aggregate_difference(trials)
    perm = c3.claim_permission(s, rep["clean"], len(trials))
    assert s["observable_effect"] is True
    assert perm["may_attribute_to_learning"] is False


# ==========================================================================
# 9. evaluator 隔离：另一组笔记内容不得出现在本组 context
# ==========================================================================


def _add_note(clean_db, *, eid, concept, principle, legacy_id):
    cols = [
        "experiment_id", "topic", "kp", "concept", "principle", "example",
        "common_mistake", "application_rule", "checksum", "created_at",
        "verified", "quality_total", "quality_verdict", "v3_experiment_id",
    ]
    ph = ",".join("?" for _ in cols)
    clean_db.execute(
        f"INSERT INTO learning_notes({','.join(cols)}) VALUES ({ph})",
        (legacy_id, "t", "kp", concept, principle, "", "", "", "c",
         "t", 0, 0, "", eid),
    )


def test_detects_peer_note_text_in_own_context(clean_db):
    """另一组的笔记内容出现在本组 context 里 → 污染。

    早先用 ``LIKE '%<另一组 id>%'`` 判，把材料正文里的
    "bucket 5"、"SPAN=60000" 误当成 id 引用，跑了四条假污染；
    改成拿 concept 文本去找才是真实含义。
    """
    _seed_clean_run(clean_db)
    for eid, legacy in ((1, 9101), (2, 9102)):
        clean_db.execute(
            "INSERT INTO experiments(id, topic, created_at) VALUES (?,?,?)",
            (legacy, "t", "t"),
        )
    _add_note(clean_db, eid=2, concept="Veyra跨度常数", principle="SPAN=60000",
              legacy_id=9102)
    _insert_stage_item(
        clean_db, run_id="r1", exp_group="learning",
        stage="study_learning", seq=3, mode="open",
        context_used="学习材料里写了 Veyra跨度常数 这个概念",
    )
    hits = gi.check_evaluator_leak(clean_db, "r1")
    assert hits, "另一组笔记内容出现在本组 context 却没报警"
    assert any(h.kind == gi.CONTAM_EVALUATOR_LEAK for h in hits)


def test_material_numbers_are_not_treated_as_ids(clean_db):
    """材料正文里的数字不是 experiment_id 引用——不能误报。"""
    _seed_clean_run(clean_db)
    clean_db.execute(
        "INSERT INTO experiments(id, topic, created_at) VALUES (?,?,?)",
        (9103, "t", "t"),
    )
    _add_note(clean_db, eid=2, concept="唯一概念标记XYZ", principle="p",
              legacy_id=9103)
    _insert_stage_item(
        clean_db, run_id="r1", exp_group="learning",
        stage="study_learning", seq=3, mode="open",
        context_used="bucket 5 的 live=5, tombstones=0，SPAN=60000",
    )
    hits = gi.check_evaluator_leak(clean_db, "r1")
    assert not hits, f"材料里的普通数字被误判成 id 引用：{hits}"


# ==========================================================================
# 9. evaluator 隔离：另一组笔记内容不得出现在本组 context
# ==========================================================================


def _add_note(clean_db, *, eid, concept, principle, legacy_id):
    cols = [
        "experiment_id", "topic", "kp", "concept", "principle", "example",
        "common_mistake", "application_rule", "checksum", "created_at",
        "verified", "quality_total", "quality_verdict", "v3_experiment_id",
    ]
    ph = ",".join("?" for _ in cols)
    clean_db.execute(
        f"INSERT INTO learning_notes({','.join(cols)}) VALUES ({ph})",
        (legacy_id, "t", "kp", concept, principle, "", "", "", "c",
         "t", 0, 0, "", eid),
    )


def test_detects_peer_note_text_in_own_context(clean_db):
    """另一组的笔记内容出现在本组 context 里 → 污染。

    早先用 ``LIKE '%<另一组 id>%'`` 判，把材料正文里的
    "bucket 5"、"SPAN=60000" 误当成 id 引用，跑了四条假污染；
    改成拿 concept 文本去找才是真实含义。
    """
    _seed_clean_run(clean_db)
    for eid, legacy in ((1, 9101), (2, 9102)):
        clean_db.execute(
            "INSERT INTO experiments(id, topic, created_at) VALUES (?,?,?)",
            (legacy, "t", "t"),
        )
    _add_note(clean_db, eid=2, concept="Veyra跨度常数", principle="SPAN=60000",
              legacy_id=9102)
    _insert_stage_item(
        clean_db, run_id="r1", exp_group="learning",
        stage="study_learning", seq=3, mode="open",
        context_used="学习材料里写了 Veyra跨度常数 这个概念",
    )
    hits = gi.check_evaluator_leak(clean_db, "r1")
    assert hits, "另一组笔记内容出现在本组 context 却没报警"
    assert any(h.kind == gi.CONTAM_EVALUATOR_LEAK for h in hits)


def test_material_numbers_are_not_treated_as_ids(clean_db):
    """材料正文里的数字不是 experiment_id 引用——不能误报。"""
    _seed_clean_run(clean_db)
    clean_db.execute(
        "INSERT INTO experiments(id, topic, created_at) VALUES (?,?,?)",
        (9103, "t", "t"),
    )
    _add_note(clean_db, eid=2, concept="唯一概念标记XYZ", principle="p",
              legacy_id=9103)
    _insert_stage_item(
        clean_db, run_id="r1", exp_group="learning",
        stage="study_learning", seq=3, mode="open",
        context_used="bucket 5 的 live=5, tombstones=0，SPAN=60000",
    )
    hits = gi.check_evaluator_leak(clean_db, "r1")
    assert not hits, f"材料里的普通数字被误判成 id 引用：{hits}"


# ==========================================================================
# 10. held-out 客观判分器
# ==========================================================================


def test_heldout_reference_answers_all_full_marks():
    """判分器必须认得自己的标准答案——否则 held-out 分数全不可信。"""
    d = ho.selfcheck_answers()
    assert d["problems"] == [], d["problems"]
    assert d["fixture_restored"] is True


def test_heldout_empty_output_scores_zero():
    """空答案不能因为「没报错」而拿分。"""
    for task in ho.HELD_OUT_TASKS:
        res = ho.evaluate_heldout(task.task_id, "")
        assert res.objective_score < 50, f"{task.task_id} 空答案竟得高分"


def test_heldout_scorers_intact_cached():
    """自检结果有缓存，但 force=True 必须能强制重算。"""
    ho.assert_heldout_scorers_intact()
    ho.assert_heldout_scorers_intact()
    ho.assert_heldout_scorers_intact(force=True)


def test_heldout_checkers_are_independent_of_practice():
    """held-out 的判分器不能直接复用 practice 的。

    practice 的 checker 把practice 自己的参数写死进了检查项
    （0 号桶、``1 not in compress and 4 not in compress``），
    直接复用会把 held-out 的正确答案判错。
    """
    assert ho.HELD_OUT_CHECKERS is not None
    for name, fn in ho.HELD_OUT_CHECKERS.items():
        assert fn.__module__ != "practice.objective" or name in (
            "check_overload_sql", "check_unreadable_sql",
        ), f"{name} 直接复用了 practice 的判分器"


def test_heldout_split_uses_its_own_bucket():
    """held-out 的分裂题目标桶是 5，不是 practice 的 0。"""
    assert ho.HO_SPLIT_BUCKET == 5
    assert ho.HO_SPLIT_BUCKET != 0


def test_heldout_compact_derives_buckets_dynamically():
    """held-out 压缩的触发桶含 4号——practice checker 会把它判错。"""
    assert 4 in ho.HO_COMPACT_TRIGGERED
    assert 1 in ho.HO_COMPACT_TRIGGERED


def test_heldout_consistency_is_mixed_state():
    """held-out 的一致性题必须同时有可读与需重算的键。

    若全是失配，照搬 practice 的答案结构就能答对，
    「练习过」与「会做」在同一条结构上无法区分。
    """
    assert ho.HO_CONSIST_READABLE, "应至少有一个可读的键"
    assert ho.HO_CONSIST_NEED_RETAG, "应至少有一个需重算的键"
    assert set(ho.HO_CONSIST_READABLE) & set(ho.HO_CONSIST_NEED_RETAG) == set()


def test_heldout_new_tags_only_cover_needed_keys():
    """题目只要求给需重算的键算新标签，多给也算错。"""
    assert set(ho.HO_CONSIST_NEW_TAGS) == set(ho.HO_CONSIST_NEED_RETAG)


# ==========================================================================
# 10. held-out 客观判分器
# ==========================================================================


def test_heldout_reference_answers_all_full_marks():
    """判分器必须认得自己的标准答案——否则 held-out 分数全不可信。"""
    d = ho.selfcheck_answers()
    assert d["problems"] == [], d["problems"]
    assert d["fixture_restored"] is True


def test_heldout_empty_output_scores_zero():
    """空答案不能因为「没报错」而拿分。"""
    for task in ho.HELD_OUT_TASKS:
        res = ho.evaluate_heldout(task.task_id, "")
        assert res.objective_score < 50, f"{task.task_id} 空答案竟得高分"


def test_heldout_scorers_intact_cached():
    """自检结果有缓存，但 force=True 必须能强制重算。"""
    ho.assert_heldout_scorers_intact()
    ho.assert_heldout_scorers_intact()
    ho.assert_heldout_scorers_intact(force=True)


def test_heldout_checkers_are_independent_of_practice():
    """held-out 的判分器不能直接复用 practice 的。

    practice 的 checker 把practice 自己的参数写死进了检查项
    （0 号桶、``1 not in compress and 4 not in compress``），
    直接复用会把 held-out 的正确答案判错。
    """
    assert ho.HELD_OUT_CHECKERS is not None
    for name, fn in ho.HELD_OUT_CHECKERS.items():
        assert fn.__module__ != "practice.objective" or name in (
            "check_overload_sql", "check_unreadable_sql",
        ), f"{name} 直接复用了 practice 的判分器"


def test_heldout_split_uses_its_own_bucket():
    """held-out 的分裂题目标桶是 5，不是 practice 的 0。"""
    assert ho.HO_SPLIT_BUCKET == 5
    assert ho.HO_SPLIT_BUCKET != 0


def test_heldout_compact_derives_buckets_dynamically():
    """held-out 压缩的触发桶含 4号——practice checker 会把它判错。"""
    assert 4 in ho.HO_COMPACT_TRIGGERED
    assert 1 in ho.HO_COMPACT_TRIGGERED


def test_heldout_consistency_is_mixed_state():
    """held-out 的一致性题必须同时有可读与需重算的键。

    若全是失配，照搬 practice 的答案结构就能答对，
    「练习过」与「会做」在同一条结构上无法区分。
    """
    assert ho.HO_CONSIST_READABLE, "应至少有一个可读的键"
    assert ho.HO_CONSIST_NEED_RETAG, "应至少有一个需重算的键"
    assert set(ho.HO_CONSIST_READABLE) & set(ho.HO_CONSIST_NEED_RETAG) == set()


def test_heldout_new_tags_only_cover_needed_keys():
    """题目只要求给需重算的键算新标签，多给也算错。"""
    assert set(ho.HO_CONSIST_NEW_TAGS) == set(ho.HO_CONSIST_NEED_RETAG)
