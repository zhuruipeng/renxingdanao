"""tests/test_v06_experiment.py —— V0.6 端到端：十阶段编排、双组分离、报告渲染。

这一层跑完整链路（ScriptedProvider，不联网），验证的是**结构**：

- 两个组拿到**不同的** experiment_id，产物不互相可见
- 十阶段按固定顺序落库，每阶段字段齐全
- 考试与客观两套口径各自闭合
- 报告能渲染，且渲染结果不触发自己的越权措辞门禁
- 两组结果不同时（同一个 bug 的两种表现）

**不验证分数好不好**——离线脚本的分数是钦定的，
分数好不代表链路对；分数一样才说明链路没起作用。
"""

from __future__ import annotations

import json

import pytest

from brain import causal_v3 as c3
from brain import experiment_v3 as ev3
from brain import group_isolation as gi
from brain import heldout_bank as ho
from brain import report_v3 as r3
from brain.llm import LLMProvider, LLMResponse
from memory.db import GROUPS_V3, STAGES_V3, Database


# ==========================================================================
# 离线 provider
# ==========================================================================


_ANSWER_UNKNOWN = "我不知道这个虚构系统的规则，无法作答。"
_ANSWER_LEARNING = (
    "LumenHash 初值 7，逐字符 h=(h*131+ord(c)) mod 65537；bucket = hash mod 8；"
    "分裂迁移到 (b+8) mod 8 = b，是恒等变换、不减负载；"
    "压缩条件是墓碑>=2 **且** 活跃键<=1；"
    "Z-Tag=(LumenHash+epoch) mod 65537，epoch 一变全库存量键失配。"
)
_ANSWER_CONTROL = (
    "根据 Veyra 材料：SPAN=60000，seg = T // SPAN，"
    "活跃数超过 8 就按 2*seg / 2*seg+1 分裂，parent = seg // 2；"
    "check = (len(payload)*31 + seq) mod 8191。这些与题目问的规则无关。"
)

_PROBES = (
    ("压缩条件是「且」", ("且",)),
    ("哈希递推常数", ("131", "65537")),
    ("桶号=mod 8", ("mod 8",)),
    ("分裂恒等变换", ("(b+8) mod 8",)),
    ("Z-Tag 随 epoch", ("Z-Tag", "epoch")),
)


class Candidate(LLMProvider):
    """按阶段/组别分叉的离线考生。

    分叉依据是 prompt 末尾的 ``[[V3_STAGE:..]] [[V3_GROUP:..]]`` 标记。
    没有标记就抛错——**宁可炸也不要猜**：猜错时两组会拿到同一个分数，
    而「两组一样」在报告里看起来像「没差异」，实际上只是脚本没接上。
    """

    def complete(self, messages, *, temperature=0.7, max_tokens=None,
                 json_mode=False):
        joined = "\n".join(m.get("content", "") for m in messages)
        import re
        st = re.search(r"\[\[V3_STAGE:([a-z_]+)\]\]", joined)
        gp = re.search(r"\[\[V3_GROUP:([a-z]+)\]\]", joined)
        if not st or not gp:
            raise RuntimeError("prompt 里没有阶段/组别标记")
        stage, group = st.group(1), gp.group(1)

        # 客观任务：只有学习组的 post_objective 答得出
        if stage in ("pre_objective", "post_objective"):
            if group == "learning" and stage == "post_objective":
                return LLMResponse(
                    text=ho._reference_answer(ho.HELD_OUT_TASKS[0]),
                    model="cand",
                )
            return LLMResponse(text="{}", model="cand")

        if stage.endswith("_material_read") or stage in (
            "study_learning", "control_material"
        ):
            pts = [{"kp": "k1", "concept": "C1", "principle": "P1"}]
            if "Veyra" in joined:
                pts = [{"kp": "k2", "concept": "C2", "principle": "P2"}]
            return LLMResponse(
                text=json.dumps({"points": pts}, ensure_ascii=False),
                model="cand",
            )

        if stage.startswith("pre_"):
            return LLMResponse(text=_ANSWER_UNKNOWN, model="cand")
        if group == "control":
            return LLMResponse(text=_ANSWER_CONTROL, model="cand")
        return LLMResponse(text=_ANSWER_LEARNING, model="cand")


class Evaluator(LLMProvider):
    """离线评估器：只按答案有没有命中目标规则要点给分。

    **不看答案里出现哪个系统名**——对照组答的正是 Veyra 的规则，
    按「提到系统名就高分」判，两组 POST 会拿到同一个分数。
    """

    def complete(self, messages, *, temperature=0.7, max_tokens=None,
                 json_mode=False):
        joined = "\n".join(m.get("content", "") for m in messages)
        answer = joined.rsplit("考生答案：", 1)[-1]
        if "我不知道" in answer or "无法作答" in answer:
            score, hits = 10.0, []
        else:
            hits = [n for n, ps in _PROBES if any(p in answer for p in ps)]
            score = round(10.0 + 62.0 * len(hits) / len(_PROBES), 2)
        return LLMResponse(
            text=json.dumps({
                "score": score,
                "verdict": "pass" if score >= 60 else "fail",
                "feedback": f"hit={hits}",
                "missing_points": [n for n, _ in _PROBES if n not in hits],
            }, ensure_ascii=False),
            model="eval",
        )


class Boom(LLMProvider):
    """永远崩的 provider。"""

    def complete(self, messages, *, temperature=0.7, max_tokens=None,
                 json_mode=False):
        raise RuntimeError("模拟超时")


def _call(cfg, system, user):
    return cfg.provider.complete(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        json_mode=True,
    ).text


# ==========================================================================
# 夹具
# ==========================================================================


@pytest.fixture()
def db(tmp_path):
    d = Database(str(tmp_path / "v06_exp.db"))
    d.init_schema()
    yield d
    d.close()


def _spec(run_id: str, trial_no: int = 1) -> ev3.RunSpec:
    return ev3.RunSpec(
        run_id=run_id, trial_no=trial_no, topic="t",
        candidate=ev3.ProviderConfig("cand", "cand", Candidate()),
        evaluator=ev3.ProviderConfig("eval", "eval", Evaluator()),
    )


def _run(db, run_id="r1", trial_no=1):
    store = ev3.V3Store(db)
    return ev3.DualGroupRunner(store, _spec(run_id, trial_no), _call).run()


# ==========================================================================
# 1. schema
# ==========================================================================


def test_v3_tables_exist(db):
    for t in ("v3_runs", "v3_experiments", "v3_stage_items",
              "v3_objective_runs", "v3_call_failures"):
        rows = db.query_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)
        )
        assert rows, f"缺表 {t}"


def test_init_schema_is_idempotent(db):
    db.init_schema()
    db.init_schema()


def test_stages_v3_is_ten():
    assert len(STAGES_V3) == 10


# ==========================================================================
# 2. 双组结构
# ==========================================================================


def test_run_creates_two_distinct_experiment_ids(db):
    res = _run(db)
    ids = res["experiment_ids"]
    assert ids["learning"] != ids["control"]


def test_two_groups_are_isolated(db):
    """两组的笔记/knowledge 必须按 v3_experiment_id 天然分开。"""
    res = _run(db)
    ids = res["experiment_ids"]
    for eid in (ids["learning"], ids["control"]):
        notes = db.query_all(
            "SELECT concept FROM learning_notes WHERE v3_experiment_id=?",
            (eid,),
        )
        assert notes, f"experiment {eid} 没有笔记"


def test_two_groups_have_different_notes(db):
    """两组的知识点内容不能相同——相同就等于共享了学习产物。"""
    _run(db)
    rows = db.query_all(
        "SELECT v3_experiment_id, concept FROM learning_notes"
        " WHERE v3_experiment_id IS NOT NULL ORDER BY v3_experiment_id"
    )
    by_group: dict[int, set] = {}
    for r in rows:
        by_group.setdefault(r["v3_experiment_id"], set()).add(r["concept"])
    if len(by_group) == 2:
        values = list(by_group.values())
        assert values[0] != values[1], "两组笔记内容相同 = 共享学习产物"


def test_candidate_and_evaluator_recorded_separately(db):
    _run(db)
    row = db.query_one("SELECT * FROM v3_runs WHERE run_id='r1'")
    assert row["candidate_provider"] == "cand"
    assert row["evaluator_provider"] == "eval"
    assert row["candidate_provider"] != row["evaluator_provider"]


def test_run_freezes_all_fingerprints(db):
    _run(db)
    row = db.query_one("SELECT * FROM v3_runs WHERE run_id='r1'")
    for col in ("pre_fingerprint", "post_fingerprint", "transfer_fingerprint",
                "learning_material_checksum", "control_material_checksum",
                "practice_tasks_fingerprint", "heldout_tasks_fingerprint"):
        assert row[col], f"{col} 未冻结"


# ==========================================================================
# 3. 十阶段落库
# ==========================================================================


def test_all_ten_stages_present(db):
    """十阶段要**跨两张表**看：考试题在 stage_items，客观题在 objective_runs。"""
    _run(db)
    present = {
        r["stage"] for r in db.query_all(
            "SELECT DISTINCT stage FROM v3_stage_items WHERE run_id='r1'"
        )
    } | {
        r["stage"] for r in db.query_all(
            "SELECT DISTINCT stage FROM v3_objective_runs WHERE run_id='r1'"
        )
    }
    for s in STAGES_V3:
        assert s in present, f"阶段 {s} 没落库"


def test_order_audit_sees_practice_and_objective_stages(db):
    """顺序审计必须**两张表都查**。

    早先只查 v3_stage_items，于是 practice_* 与 *_objective
    四个阶段在审计里完全不存在——声称「十阶段顺序」的门禁
    实际只覆盖八个，而报告里还写着 clean=True。
    """
    res = _run(db)
    seq = res["order"]["seq_by_stage"]
    for s in ("practice_learning", "practice_control",
              "pre_objective", "post_objective"):
        assert s in seq, f"顺序审计看不见 {s}"


def test_pre_objective_is_before_material_stages(db):
    res = _run(db)
    seq = res["order"]["seq_by_stage"]
    material = max(seq[s] for s in (
        "study_learning", "control_material",
        "practice_learning", "practice_control"))
    assert seq["pre_objective"] < material


def test_post_objective_is_after_practice(db):
    res = _run(db)
    seq = res["order"]["seq_by_stage"]
    assert seq["post_objective"] > max(
        seq["practice_learning"], seq["practice_control"])


def test_order_audit_sees_practice_and_objective_stages(db):
    """顺序审计必须**两张表都查**。

    早先只查 v3_stage_items，于是 practice_* 与 *_objective
    四个阶段在审计里完全不存在——声称「十阶段顺序」的门禁
    实际只覆盖八个，而报告里还写着 clean=True。
    """
    res = _run(db)
    seq = res["order"]["seq_by_stage"]
    for s in ("practice_learning", "practice_control",
              "pre_objective", "post_objective"):
        assert s in seq, f"顺序审计看不见 {s}"


def test_pre_objective_is_before_material_stages(db):
    res = _run(db)
    seq = res["order"]["seq_by_stage"]
    material = max(seq[s] for s in (
        "study_learning", "control_material",
        "practice_learning", "practice_control"))
    assert seq["pre_objective"] < material


def test_post_objective_is_after_practice(db):
    res = _run(db)
    seq = res["order"]["seq_by_stage"]
    assert seq["post_objective"] > max(
        seq["practice_learning"], seq["practice_control"])


def test_stage_seq_follows_fixed_order(db):
    _run(db)
    audit = gi.order_audit(db, "r1")
    seq_of = audit["seq_by_stage"]
    by_fixed_order = [seq_of[s] for s in STAGES_V3]
    assert by_fixed_order == sorted(by_fixed_order)
    assert by_fixed_order == list(range(1, 11))


def test_both_pre_before_study(db):
    _run(db)
    seq = gi.order_audit(db, "r1")["seq_by_stage"]
    assert max(seq["pre_learning"], seq["pre_control"]) < seq["study_learning"]
    assert max(seq["pre_learning"], seq["pre_control"]) < seq["control_material"]


def test_post_after_practice(db):
    _run(db)
    seq = gi.order_audit(db, "r1")["seq_by_stage"]
    assert seq["post_learning"] > max(
        seq["practice_learning"], seq["practice_control"]
    )


def test_every_stage_item_has_required_columns(db):
    _run(db)
    row = db.query_one(
        "SELECT * FROM v3_stage_items WHERE run_id='r1'"
        " AND knowledge_point != '__material__' LIMIT 1"
    )
    for col in ("run_id", "experiment_id", "exp_group", "stage", "seq",
                "item_id", "answer", "score", "context_used",
                "model", "provider", "created_at"):
        assert col in row, f"缺列 {col}"
        assert row[col] is not None, f"{col} 为空"


def test_both_pre_use_same_question_bank(db):
    _run(db)
    l = {r["item_id"] for r in db.query_all(
        "SELECT item_id FROM v3_stage_items"
        " WHERE run_id='r1' AND stage='pre_learning'")}
    c = {r["item_id"] for r in db.query_all(
        "SELECT item_id FROM v3_stage_items"
        " WHERE run_id='r1' AND stage='pre_control'")}
    assert l == c, "两组的 PRE 必须是同一套题，否则基线不可比"


def test_both_post_use_same_question_bank(db):
    _run(db)
    l = {r["item_id"] for r in db.query_all(
        "SELECT item_id FROM v3_stage_items"
        " WHERE run_id='r1' AND stage='post_learning'")}
    c = {r["item_id"] for r in db.query_all(
        "SELECT item_id FROM v3_stage_items"
        " WHERE run_id='r1' AND stage='post_control'")}
    assert l == c, "两组的 POST 必须是同一套 held-out 题"


def test_blind_stages_have_empty_context(db):
    _run(db)
    for stage in ("pre_learning", "pre_control", "post_learning",
                  "post_control", "transfer_learning", "transfer_control"):
        rows = db.query_all(
            "SELECT context_used FROM v3_stage_items"
            " WHERE run_id='r1' AND stage=?", (stage,),
        )
        assert all(not r["context_used"].strip() for r in rows), \
            f"{stage} 是盲考阶段，context 必须为空"


def test_material_stages_have_nonempty_context(db):
    _run(db)
    for stage in ("study_learning", "control_material"):
        rows = db.query_all(
            "SELECT context_used FROM v3_stage_items"
            " WHERE run_id='r1' AND stage=?", (stage,),
        )
        assert all(r["context_used"].strip() for r in rows), \
            f"{stage} 必须带材料，否则「这一步做没做」在数据里看不见"


# ==========================================================================
# 4. 两组结果必须不同
# ==========================================================================


def test_groups_produce_different_exam_scores(db):
    """链路没起作用时，两组会拿到同一个分数。"""
    res = _run(db)
    t = res["trial"]
    assert t[c3.KEY_DIFFERENCE] != 0.0, "两组考试分完全相同 = 链路没起作用"


def test_groups_produce_different_objective_scores(db):
    res = _run(db)
    t = res["objective_trial"]
    assert t[c3.KEY_OBJ_DIFFERENCE] != 0.0, "两组客观分完全相同"


def test_learning_beats_control_on_objective(db):
    res = _run(db)
    t = res["objective_trial"]
    assert t[c3.KEY_OBJ_LEARNING_GAIN] > t[c3.KEY_OBJ_CONTROL_GAIN]


def test_both_pre_scores_are_identical(db):
    """PRE 是任何材料介入之前的基线，两组必须一样高（或一样低）。"""
    res = _run(db)
    t = res["trial"]
    assert t[c3.KEY_LEARNING_PRE] == t[c3.KEY_CONTROL_PRE]


def test_objective_pre_identical(db):
    res = _run(db)
    t = res["objective_trial"]
    assert t[c3.KEY_OBJ_LEARNING_PRE] == t[c3.KEY_OBJ_CONTROL_PRE]


# ==========================================================================
# 5. 客观任务集隔离
# ==========================================================================


def test_practice_uses_practice_tasks_not_heldout(db):
    """两组 practice 都必须用 practice 任务集。

    早先 Control 组 practice 用了 HELD_OUT_TASKS，
    而 post_objective 测的正是那批题——
    Control 组练过再测，objective_difference_in_gains 偏向假阴性。
    """
    _run(db)
    practice_ids = {
        r["task_id"] for r in db.query_all(
            "SELECT DISTINCT task_id FROM v3_objective_runs"
            " WHERE run_id='r1' AND stage LIKE 'practice_%'"
        )
    }
    ho_ids = {t.task_id for t in ho.HELD_OUT_TASKS}
    assert practice_ids and not (practice_ids & ho_ids), \
        "held-out 任务出现在了 practice 阶段"
    from practice.objective import OBJECTIVE_TASKS
    assert practice_ids == {t.task_id for t in OBJECTIVE_TASKS}


def test_both_pre_and_post_objective_use_heldout(db):
    _run(db)
    for stage in ("pre_objective", "post_objective"):
        used = {
            r["task_id"] for r in db.query_all(
                "SELECT DISTINCT task_id FROM v3_objective_runs"
                " WHERE run_id='r1' AND stage=?", (stage,),
            )
        }
        assert used == {t.task_id for t in ho.HELD_OUT_TASKS}, stage


def test_contamination_clean_after_run(db):
    res = _run(db)
    assert res["contamination"]["clean"] is True, res["contamination"]


def test_order_audit_clean_after_run(db):
    res = _run(db)
    assert res["order"]["clean"] is True, res["order"]


# ==========================================================================
# 6. 调用失败必须中止而不是变成分数
# ==========================================================================


def test_candidate_crash_aborts_run(db):
    spec = ev3.RunSpec(
        run_id="boom", trial_no=1, topic="t",
        candidate=ev3.ProviderConfig("cand", "cand", Boom()),
        evaluator=ev3.ProviderConfig("eval", "eval", Boom()),
    )
    runner = ev3.DualGroupRunner(ev3.V3Store(db), spec, _call)
    with pytest.raises(RuntimeError, match="调用考生失败"):
        runner.run()


def test_candidate_crash_is_recorded(db):
    spec = ev3.RunSpec(
        run_id="boom", trial_no=1, topic="t",
        candidate=ev3.ProviderConfig("cand", "cand", Boom()),
        evaluator=ev3.ProviderConfig("eval", "eval", Boom()),
    )
    runner = ev3.DualGroupRunner(ev3.V3Store(db), spec, _call)
    with pytest.raises(RuntimeError):
        runner.run()
    rows = db.query_all("SELECT * FROM v3_call_failures")
    assert rows, "调用失败必须结构化落库"
    assert rows[0]["run_id"] == "boom"
    assert rows[0]["provider"] == "cand"


def test_crash_is_flagged_as_contamination(db):
    """崩溃后污染检测要报警——失败样本不能被当成「模型答不出」。"""
    spec = ev3.RunSpec(
        run_id="boom", trial_no=1, topic="t",
        candidate=ev3.ProviderConfig("cand", "cand", Boom()),
        evaluator=ev3.ProviderConfig("eval", "eval", Boom()),
    )
    runner = ev3.DualGroupRunner(ev3.V3Store(db), spec, _call)
    with pytest.raises(RuntimeError):
        runner.run()
    hits = gi.check_call_failures(db, "boom")
    assert hits
    assert hits[0].kind == gi.CONTAM_CALL_FAILED


def test_no_call_failure_recorded_on_healthy_run(db):
    _run(db)
    assert db.query_all("SELECT * FROM v3_call_failures") == []


# ==========================================================================
# 7. 报告
# ==========================================================================


def test_report_renders_without_forbidden_claims(db):
    """报告模板自己含禁用子串时会被自己的门禁拦下——这条守住它。"""
    res = _run(db)
    rep = r3.build_report(
        batch_id="b", db=db, trials=[res["trial"]],
        objective_trials=[res["objective_trial"]], run_ids=["r1"],
    )
    text = r3.format_text(rep)          # 内部会过越权门禁
    assert "Agent-01 V0.6" in text


def test_report_lists_five_lenses_twice(db):
    res = _run(db)
    rep = r3.build_report(
        batch_id="b", db=db, trials=[res["trial"]],
        objective_trials=[res["objective_trial"]], run_ids=["r1"],
    )
    text = r3.format_text(rep)
    for label in r3.LENS_LABELS.values():
        assert text.count(label) == 2, f"{label} 应在考试与客观口径各出现一次"


def test_report_shows_contamination_and_order(db):
    res = _run(db)
    rep = r3.build_report(
        batch_id="b", db=db, trials=[res["trial"]],
        objective_trials=[res["objective_trial"]], run_ids=["r1"],
    )
    text = r3.format_text(rep)
    assert "污染检测" in text
    assert "实验顺序" in text
    assert "考生调用失败" in text


def test_report_shows_residual_contamination(db):
    """表达层无法隔离的两类必须印在报告上，不能藏起来。"""
    res = _run(db)
    rep = r3.build_report(
        batch_id="b", db=db, trials=[res["trial"]],
        objective_trials=[res["objective_trial"]], run_ids=["r1"],
    )
    assert "残留污染" in r3.format_text(rep)


def test_report_denies_attribution_on_single_trial(db):
    res = _run(db)
    rep = r3.build_report(
        batch_id="b", db=db, trials=[res["trial"]],
        objective_trials=[res["objective_trial"]], run_ids=["r1"],
    )
    assert rep["claim_permission"]["may_attribute_to_learning"] is False
    assert "学习效应归因许可 = 不允许" in r3.format_text(rep)


def test_report_providers_differ(db):
    res = _run(db)
    rep = r3.build_report(
        batch_id="b", db=db, trials=[res["trial"]],
        objective_trials=[res["objective_trial"]], run_ids=["r1"],
    )
    assert rep["providers"]["candidate"][0]["provider"] == "cand"
    assert rep["providers"]["evaluator"][0]["provider"] == "eval"


def test_report_shows_material_length_ratio(db):
    res = _run(db)
    rep = r3.build_report(
        batch_id="b", db=db, trials=[res["trial"]],
        objective_trials=[res["objective_trial"]], run_ids=["r1"],
    )
    assert 0.9 <= rep["materials"]["length_ratio"] <= 1.1
    assert "长度比" in r3.format_text(rep)


def test_report_has_no_call_failures_on_healthy_run(db):
    res = _run(db)
    rep = r3.build_report(
        batch_id="b", db=db, trials=[res["trial"]],
        objective_trials=[res["objective_trial"]], run_ids=["r1"],
    )
    assert rep["call_failures"] == []
