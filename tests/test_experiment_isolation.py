"""V0.2 隔离测试（V0.3 适配版）。

V0.3 把流程从两场考试扩到三场，被测行为变了：
- PRE 与 BLIND_POST 都是盲考，但题库卷不同
- 新增 OPEN_BOOK 场次，走完全不同的闸门
- 题目快照断言要按 mode 分开核对

这里保留 V0.2 时代建立的纪律（题目不可由模型产生、评估器独立、
V0.1 机制未被破坏），并按 V0.3 的三场结构重写。
"""

from __future__ import annotations

import json

import pytest

from brain import question_bank as qb
from brain.experiment import ExperimentRunner
from brain.materials import DEFAULT_MATERIAL
from memory.gate import FORBIDDEN_IN_EXAM, AccessPolicy, ExamAccessViolation

from tests.test_experiment import EXTRA_SCRIPT

TOPIC = "SQLite 索引"


@pytest.fixture()
def runner(db, provider):
    provider.script.update(EXTRA_SCRIPT)
    return ExperimentRunner(db, provider)


# ------------------------------------------------------------------
# 1. 禁读集合
# ------------------------------------------------------------------
def test_experiences_joined_forbidden_set():
    assert "experiences" in FORBIDDEN_IN_EXAM
    assert {"knowledge", "training_samples", "experiences"} <= FORBIDDEN_IN_EXAM
    # V0.3 新增
    assert "learning_notes" in FORBIDDEN_IN_EXAM
    assert "material" in FORBIDDEN_IN_EXAM
    assert "reference_answer" in FORBIDDEN_IN_EXAM


def test_isolation_selfcheck_passes(db):
    from brain.llm import ScriptedProvider

    r = ExperimentRunner(db, ScriptedProvider({}))
    assert r._assert_exam_isolation() is None
    assert r._assert_blind_isolation() is None


def test_isolation_selfcheck_fails_if_gate_broken(db, monkeypatch):
    """闸门一旦失效，实验必须拒绝跑，而不是静默产出无效分数。"""
    from brain.llm import ScriptedProvider
    from memory import gate

    monkeypatch.setattr(gate, "BLIND_MODEL_FORBIDDEN", frozenset())
    r = ExperimentRunner(db, ScriptedProvider({}))
    with pytest.raises(RuntimeError, match="访问闸门失效"):
        r._assert_blind_isolation()


def test_exam_stores_are_blind(db):
    from experience.store import ExperienceStore
    from knowledge.store import KnowledgeStore
    from learning_notes.store import LearningNoteStore
    from training_data.generator import TrainingSampleStore

    blind = AccessPolicy.blind_model(stage="exam", experiment_id=1)
    with pytest.raises(ExamAccessViolation):
        KnowledgeStore(db, blind).for_experiment(1)
    with pytest.raises(ExamAccessViolation):
        LearningNoteStore(db, blind).for_experiment(1)
    with pytest.raises(ExamAccessViolation):
        TrainingSampleStore(db, blind).list_by_topic("t")
    with pytest.raises(ExamAccessViolation):
        ExperienceStore(db, blind).recent()


# ------------------------------------------------------------------
# 2. 盲考不注入上下文
# ------------------------------------------------------------------
def test_exam_answer_never_receives_context(runner, provider):
    """PRE 与 BLIND_POST 共 8 次答题，prompt 里不能出现任何学习资料。"""
    provider.calls.clear()
    runner.run(TOPIC, DEFAULT_MATERIAL)
    answer_prompts = [
        "\n".join(m["content"] for m in call)
        for call in provider.calls
        if "[[TASK:ANSWER]]" in "\n".join(m["content"] for m in call)
    ]
    # PRE 4 + MEMORY_BLIND 4 + TRANSFER 4 = 12 次无上下文；MEMORY_OPEN 4 题带笔记
    assert len(answer_prompts) == 16
    blind_calls = [p for p in answer_prompts if "学习笔记" not in p]
    assert len(blind_calls) == 12, "PRE + MEMORY_BLIND + TRANSFER 应为 12 次无上下文答题"
    for p in blind_calls:
        assert "参考资料" not in p, "盲考不得注入参考资料"
        assert "学习材料" not in p, "盲考不得注入学习材料"
        assert "学习笔记" not in p, "盲考不得注入学习笔记"
        assert "B-Tree 平衡多路树" not in p, "盲考不得泄漏知识点标题"


def test_pre_and_post_answers_are_not_cheated_by_study(runner):
    """学习后、笔记入库后，盲考闸门依然有效。"""
    material = runner.materials.save(TOPIC, TOPIC, DEFAULT_MATERIAL, version="v1")
    bank_id = runner.bank_store.register()
    exp_id = runner.store.create(TOPIC, material["id"], bank_id)
    runner.study(TOPIC, DEFAULT_MATERIAL, experiment_id=exp_id)
    runner.make_notes(exp_id, TOPIC, DEFAULT_MATERIAL)
    assert runner.db.count("learning_notes") > 0
    runner._assert_blind_isolation()


# ------------------------------------------------------------------
# 3. 题目不可由模型产生
# ------------------------------------------------------------------
def test_question_making_is_never_called(runner, monkeypatch):
    """实验流程中，LLM 出题接口一次都不能被调用。"""
    calls = {"n": 0}
    original = runner.builder.make_questions

    def spy(*a, **k):  # pragma: no cover - 不应执行
        calls["n"] += 1
        return original(*a, **k)

    monkeypatch.setattr(runner.builder, "make_questions", spy)
    runner.run(TOPIC, DEFAULT_MATERIAL)
    assert calls["n"] == 0, "实验题目必须来自固定题库，不许 LLM 出题"


def test_exam_questions_match_bank_exactly(runner):
    """PRE 用 PRE 卷；两场 POST 都用 POST 卷，且题目一字不差。"""
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    rows = runner.store.items(res.experiment_id)
    pre = [r for r in rows if r["phase"] == "pre"]
    post = [r for r in rows if r["phase"] == "post"]
    open_rows = [r for r in post if r["mode"] == "open_book_memory"]
    blind_rows = [r for r in post if r["mode"] == "blind_model"]

    assert [r["item_id"] for r in pre] == [i.item_id for i in qb.PRE_SET]
    assert [r["item_id"] for r in open_rows] == [i.item_id for i in qb.POST_SET]
    assert [r["item_id"] for r in blind_rows] == [i.item_id for i in qb.POST_SET]

    for row, item in zip(pre, qb.PRE_SET):
        assert row["question"] == item.question
        assert row["reference_answer"] == item.answer
    for row, item in zip(open_rows + blind_rows, list(qb.POST_SET) * 2):
        assert row["question"] == item.question
        assert row["reference_answer"] == item.answer


def test_tasks_from_experiment_are_flagged_blind(runner):
    runner.run(TOPIC, DEFAULT_MATERIAL)
    rows = runner.db.query_all(
        "SELECT * FROM tasks WHERE stage IN ('pre_test','post_test')"
    )
    # PRE 4 + POST 两场各 4 = 12
    assert len(rows) == 12
    assert all(r["blind"] == 1 for r in rows)


# ------------------------------------------------------------------
# 4. 评估器独立
# ------------------------------------------------------------------
def test_evaluator_prompt_excludes_study_material(runner, provider):
    """评估 prompt 只能有题目+参考答案+考生答案，不能有学习材料或笔记。"""
    provider.calls.clear()
    runner.run(TOPIC, DEFAULT_MATERIAL)
    eval_prompts = [
        "\n".join(m["content"] for m in call)
        for call in provider.calls
        if "[[TASK:EVALUATE]]" in "\n".join(m["content"] for m in call)
    ]
    assert eval_prompts
    for p in eval_prompts:
        assert "学习材料" not in p
        assert "参考资料" not in p, "评估器不得看到考生答题时的上下文"
        assert "学习笔记" not in p, "评估器不得看到考生用的笔记"


def test_evaluator_uses_separate_system_prompt(runner):
    """答题与判分必须是两套人格。"""
    from evaluation.evaluator import EVAL_SYSTEM
    from tasks.builder import ANSWER_SYSTEM

    assert EVAL_SYSTEM != ANSWER_SYSTEM
    assert "评估" in EVAL_SYSTEM
    assert "考生" in ANSWER_SYSTEM


def test_separate_evaluator_provider_is_used(db, provider):
    """传入独立 evaluator provider 时，必须真的用它判分。"""
    from brain.llm import ScriptedProvider

    judge = ScriptedProvider({"EVALUATE": json.dumps({"score": 100, "verdict": "pass"})})
    r = ExperimentRunner(db, provider, evaluator_provider=judge)
    res = r.run(TOPIC, DEFAULT_MATERIAL)
    assert res.pre_score == res.memory_open_score == 100.0
    assert res.memory_blind_score == res.transfer_blind_score == 100.0
    # 16 道题判分（PRE 4 + MEMORY_OPEN 4 + MEMORY_BLIND 4 + TRANSFER 4）
    # + 4 次笔记质量 + 4 次实践评审 = 24
    expected = (
        len(qb.PRE_SET) + 2 * len(qb.POST_SET) + len(qb.TRANSFER_ITEMS)
        + 2 * len(qb.PRE_SET)
    )
    assert len(judge.calls) == expected, (
        f"独立评估器应被调用 {expected} 次（16 判分 + 4 笔记质量 + 4 实践评审），"
        f"实际 {len(judge.calls)}"
    )
    assert not any(
        "[[TASK:EVALUATE]]" in "\n".join(m["content"] for m in c) for c in provider.calls
    ), "考生 provider 不该承担判分"


# ------------------------------------------------------------------
# 5. V0.1 隔离机制未被破坏
# ------------------------------------------------------------------
def test_v0_1_exam_still_works(agent):
    """V0.1 的 exam 流程在 V0.3 之后必须照常工作。"""
    agent.learn("SQLite 索引原理", max_items=2)
    rr = agent.exam("SQLite 索引原理", count=1, stage="pre_test")
    assert rr.blind is True
    assert rr.evaluations


def test_v0_1_gate_still_blocks(db):
    from knowledge.store import KnowledgeStore

    blind = KnowledgeStore(db, AccessPolicy(stage="pre_test", blind=True))
    with pytest.raises(ExamAccessViolation):
        blind.list_by_topic("x")


def test_full_cycle_still_works(agent):
    out = agent.full_cycle("SQLite 索引原理", count=1)
    assert "delta" in out
    assert out["post_test"]["blind"] is True
