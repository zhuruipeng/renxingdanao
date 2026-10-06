"""V0.3 专项测试：两种考试模式 + 结构化学习笔记。

V0.3 要解决的结构性问题：
    V0.2 的盲考禁读知识库 → 学到的知识在考试时根本用不上，
    于是「学习有没有用」这个问题在结构上就无法回答。

V0.3 的解法是把「使用外部记忆」和「模型自身能力」拆成两场考试：
    OPEN_BOOK_MEMORY  放行本实验的 knowledge + learning_notes
    BLIND_MODEL       只给题目本身

但拆开之后立刻冒出新风险：
    开卷是不是把原始材料全文抄给模型了？
    两场的数据会不会互相串？
    笔记能不能伪装成参考答案？
    换个 experiment_id 能不能读到别人的笔记？

本文件逐条把这些风险钉死。原则同 V0.1/V0.2：
**凡是靠 prompt 君子协定保证的事，一律不算保证，必须由代码层强制 + 测试锁住。**
"""

from __future__ import annotations

import json

import pytest

from brain import question_bank as qb
from brain.experiment import ExperimentRunner
from brain.llm import ScriptedProvider
from brain.materials import DEFAULT_MATERIAL
from brain.report import interpret
from knowledge.store import KnowledgeStore
from learning_notes.store import (
    LearningNoteStore,
    note_checksum,
    render_notes_for_exam,
)
from memory import gate
from memory.gate import (
    BLIND_MODEL_FORBIDDEN,
    OPEN_BOOK_FORBIDDEN,
    AccessPolicy,
    ExamAccessViolation,
)

from tests.test_experiment import EXTRA_SCRIPT

TOPIC = "SQLite 索引"

#: 放进材料里的唯一哨兵串。只要它出现在考试 prompt 中，
#: 就说明代码把原始材料塞进了上下文 —— 那等于把答案抄给模型。
MATERIAL_SENTINEL = "ZQXJ-SENTINEL-7734-不要进入考试上下文"


@pytest.fixture()
def runner(db, provider):
    provider.script.update(EXTRA_SCRIPT)
    return ExperimentRunner(db, provider)


@pytest.fixture()
def tainted_runner(db):
    """材料里埋了哨兵串的 runner，用来检测「原书内容进入考试 prompt」。"""
    from main import demo_script

    material = DEFAULT_MATERIAL + "\n\n" + MATERIAL_SENTINEL + "\n"
    script = demo_script()
    script.update(EXTRA_SCRIPT)
    return ExperimentRunner(db, ScriptedProvider(script)), material


def _answer_prompts(provider: ScriptedProvider) -> list[str]:
    return [
        "\n".join(m["content"] for m in call)
        for call in provider.calls
        if "[[TASK:ANSWER]]" in "\n".join(m["content"] for m in call)
    ]


def _seeded_experiment(runner: ExperimentRunner) -> int:
    """跑完 STUDY + NOTES，返回 experiment_id（不跑考试，省 token）。"""
    from brain import question_bank as bank

    bank.assert_bank_intact()
    bank_id = runner.bank_store.register()
    material = runner.materials.save(TOPIC, TOPIC, DEFAULT_MATERIAL, version="v1")
    eid = runner.store.create(TOPIC, material["id"], bank_id)
    runner.study(TOPIC, DEFAULT_MATERIAL, experiment_id=eid)
    runner.make_notes(eid, TOPIC, DEFAULT_MATERIAL)
    return eid


# ======================================================================
# 1. OPEN_BOOK 能读取本实验的 knowledge 与 learning_notes
# ======================================================================
def test_open_book_can_read_own_experiment_knowledge(runner):
    """开卷的正向能力：本实验产出的知识必须读得到，否则开卷毫无意义。"""
    eid = _seeded_experiment(runner)
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=eid)
    rows = KnowledgeStore(runner.db, policy).for_experiment(eid)
    assert len(rows) >= 1, "开卷必须能读到本实验的 knowledge"
    assert all(r["experiment_id"] == eid for r in rows)


def test_open_book_can_read_own_experiment_notes(runner):
    """开卷的正向能力：本实验产出的结构化笔记必须读得到。"""
    eid = _seeded_experiment(runner)
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=eid)
    notes = LearningNoteStore(runner.db, policy).for_experiment(eid)
    assert len(notes) >= 1, "开卷必须能读到本实验的 learning_notes"
    for n in notes:
        assert n["experiment_id"] == eid
        assert n["concept"] and n["principle"]


def test_open_book_context_is_actually_populated(runner):
    """开卷上下文不能是空串——空串等于开卷没开。"""
    eid = _seeded_experiment(runner)
    context, meta = runner._build_exam_context(eid, "open_book_memory")
    assert context.strip(), "开卷上下文必须有内容"
    assert meta["context_len"] == len(context)
    assert meta["notes_used"] >= 1
    assert meta["knowledge_used"] >= 1


def test_open_book_context_only_renders_note_elements(runner):
    """上下文只能由笔记五要素 + 知识点拼成，不允许夹带别的。"""
    eid = _seeded_experiment(runner)
    context, _ = runner._build_exam_context(eid, "open_book_memory")
    for label in ("- 概念：", "- 原理：", "- 例子：", "- 易错点：", "- 应用规则："):
        assert label in context, f"笔记渲染缺少字段 {label}"
    assert "学习笔记" in context, "应标明信息来源是 Agent 自己的笔记"


# ======================================================================
# 2. OPEN_BOOK 不能读取参考答案 / 原始材料 / 经验
# ======================================================================
@pytest.mark.parametrize(
    "resource", ["reference_answer", "material", "training_samples",
                 "evaluation", "exam_history", "experiences"]
)
def test_open_book_cannot_read_answer_sources(resource):
    """开卷放行笔记，但答案类资源一个都不能放。"""
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=1)
    with pytest.raises(ExamAccessViolation):
        policy.guard(resource)


def test_open_book_cannot_read_training_store(db):
    eid = _seeded_experiment(ExperimentRunner(db, ScriptedProvider({})))
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=eid)
    from training_data.generator import TrainingSampleStore

    with pytest.raises(ExamAccessViolation):
        TrainingSampleStore(db, policy).list_by_topic(TOPIC)


def test_open_book_cannot_read_experience_store(db):
    eid = _seeded_experiment(ExperimentRunner(db, ScriptedProvider({})))
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=eid)
    from experience.store import ExperienceStore

    with pytest.raises(ExamAccessViolation):
        ExperienceStore(db, policy).recent()


def test_open_book_still_blocks_writing_notes(db):
    """考试期不允许伪造/篡改笔记——否则等于自己给自己发小抄。"""
    eid = _seeded_experiment(ExperimentRunner(db, ScriptedProvider({})))
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=eid)
    with pytest.raises(ExamAccessViolation):
        LearningNoteStore(db, policy).add(eid, TOPIC, "kp_btree", "c", "p")


# ======================================================================
# 3. BLIND 不能读取任何学习资料
# ======================================================================
@pytest.mark.parametrize(
    "resource",
    sorted(BLIND_MODEL_FORBIDDEN),
)
def test_blind_blocks_every_learning_artifact(resource):
    policy = AccessPolicy.blind_model(stage="exam")
    with pytest.raises(ExamAccessViolation):
        policy.guard(resource)


def test_blind_context_is_always_empty(runner):
    """盲考上下文恒为空串，一个字符都不能有。"""
    eid = _seeded_experiment(runner)
    for _ in range(3):
        context, meta = runner._build_exam_context(eid, "blind_model")
        assert context == ""
        assert meta == {"notes_used": 0, "knowledge_used": 0, "context_len": 0}


def test_blind_cannot_read_knowledge_store(runner):
    eid = _seeded_experiment(runner)
    blind = AccessPolicy.blind_model(stage="exam", experiment_id=eid)
    with pytest.raises(ExamAccessViolation):
        KnowledgeStore(runner.db, blind).for_experiment(eid)


def test_blind_cannot_read_note_store(runner):
    eid = _seeded_experiment(runner)
    blind = AccessPolicy.blind_model(stage="exam", experiment_id=eid)
    with pytest.raises(ExamAccessViolation):
        LearningNoteStore(runner.db, blind).for_experiment(eid)


def test_blind_violation_message_explains_open_book_only(runner):
    """拦截提示要说清「开卷才放行笔记」，否则排查时看不懂。"""
    eid = _seeded_experiment(runner)
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=eid)
    with pytest.raises(ExamAccessViolation, match="开卷模式"):
        policy.guard("material")


# ======================================================================
# 4. 原书内容不能直接进入考试 prompt
# ======================================================================
def test_material_sentinel_never_enters_exam_prompt(tainted_runner):
    """最关键的一条：材料原文里的独特串绝不能出现在任何一次答题 prompt 中。

    如果它出现了，说明代码在放行原始材料全文 —— 那种实验测的是
    「抄写能力」而不是「学习效果」，分数没有意义。
    """
    runner, material = tainted_runner
    from main import demo_script

    provider = ScriptedProvider({**demo_script(), **EXTRA_SCRIPT})
    runner.provider = provider
    runner.note_generator.provider = provider
    runner.builder.provider = provider
    runner.evaluator.provider = provider
    runner.reflector.provider = provider
    runner.evaluator_provider = provider

    res = runner.run(TOPIC, material)
    assert MATERIAL_SENTINEL in material, "哨兵串必须真的在材料里，否则测试无效"

    prompts = _answer_prompts(provider)
    # V0.4 四场：PRE(1卷) + MEMORY_OPEN(1卷) + MEMORY_BLIND(1卷，同卷复用) + TRANSFER(1卷)
    expected = len(qb.PRE_SET) + 2 * len(qb.POST_SET) + len(qb.TRANSFER_ITEMS)
    assert len(prompts) == expected, (
        f"答题 prompt 数应为 PRE+2×MEMORY+TRANSFER = {expected}，"
        f"实际 {len(prompts)} —— 某场没跑或跑了两遍"
    )
    for p in prompts:
        assert MATERIAL_SENTINEL not in p, "原始材料原文泄漏进了考试上下文"

    # 落库的 context_used 同样不能含哨兵
    for row in runner.store.items(res.experiment_id):
        assert MATERIAL_SENTINEL not in (row["context_used"] or "")


def test_open_book_context_has_no_material_sentinel(tainted_runner):
    runner, material = tainted_runner
    bank_id = runner.bank_store.register()
    m = runner.materials.save(TOPIC, TOPIC, material, version="v1")
    eid = runner.store.create(TOPIC, m["id"], bank_id)
    runner.study(TOPIC, material, experiment_id=eid)
    runner.make_notes(eid, TOPIC, material)
    context, _ = runner._build_exam_context(eid, "open_book_memory")
    assert context.strip()
    assert MATERIAL_SENTINEL not in context


def test_reference_answers_do_not_leak_into_open_book_context(runner):
    """题库参考答案不得出现在开卷上下文里（防止用答案反过来写笔记）。"""
    eid = _seeded_experiment(runner)
    context, _ = runner._build_exam_context(eid, "open_book_memory")
    for item in qb.ALL_ITEMS:
        assert item.answer[:20] not in context, f"参考答案泄漏：{item.item_id}"


def test_exam_questions_do_not_leak_into_open_book_context(runner):
    """考题本身也不该出现在上下文里——否则模型是被告知「要答什么」。"""
    eid = _seeded_experiment(runner)
    context, _ = runner._build_exam_context(eid, "open_book_memory")
    for item in qb.ALL_ITEMS:
        assert item.question[:15] not in context, f"题目泄漏：{item.item_id}"


# ======================================================================
# 5. PRE / POST 题目不能被模型修改
# ======================================================================
def test_bank_fingerprint_unchanged_after_experiment(runner):
    """跑完实验，题库指纹必须与代码常量一致（=期间没人改过题）。"""
    runner.run(TOPIC, DEFAULT_MATERIAL)
    row = runner.db.query_one("SELECT * FROM question_bank ORDER BY id DESC LIMIT 1")
    assert row["fingerprint"] == qb.BANK_FINGERPRINT


def test_experiment_stores_bank_constants_not_model_output(runner):
    """即使 LLM 返回了 MAKE_QUESTION，实验记录的题目也必须来自题库常量。"""
    from main import demo_script

    poisoned = {**demo_script(), **EXTRA_SCRIPT}
    poisoned["MAKE_QUESTION"] = json.dumps(
        {"items": [{"question": "这是模型自己编的题", "reference_answer": "编的答案"}]},
        ensure_ascii=False,
    )
    provider = ScriptedProvider(poisoned)
    r = ExperimentRunner(runner.db, provider)
    res = r.run(TOPIC, DEFAULT_MATERIAL)

    rows = r.store.items(res.experiment_id)
    bank_questions = {i.question for i in qb.ALL_ITEMS}
    for row in rows:
        assert row["question"] in bank_questions, "题目被模型替换了"
    assert not any(
        "[[TASK:MAKE_QUESTION]]" in "\n".join(m["content"] for m in c)
        for c in provider.calls
    ), "实验流程根本不该调用出题接口"


def test_bank_verify_aborts_experiment_on_tampering(runner, monkeypatch):
    """题目指纹对不上必须抛错，不能带着可疑数据继续跑。"""
    runner.run(TOPIC, DEFAULT_MATERIAL)  # 先注册题库
    bid = runner.db.scalar("SELECT MAX(id) FROM question_bank")
    runner.db.execute(
        "UPDATE question_bank SET fingerprint = ? WHERE id = ?", ("0" * 64, bid)
    )
    with pytest.raises(RuntimeError, match="指纹不一致"):
        runner.bank_store.verify(bid)


# ======================================================================
# 6. learning_notes 不能伪装成 reference_answer
# ======================================================================
def test_note_table_has_no_reference_answer_column(db):
    """结构上就不许有：笔记表里没有参考答案字段。

    这是「笔记不能伪装成答案」的物理保证——不是靠约定，
    是表结构里根本没有那个位置。
    """
    cols = {r["name"] for r in db.query_all("PRAGMA table_info(learning_notes)")}
    assert "reference_answer" not in cols
    assert "answer" not in cols
    assert {"concept", "principle", "example", "common_mistake",
            "application_rule", "checksum", "experiment_id"} <= cols


def test_render_notes_contains_only_five_elements(runner):
    eid = _seeded_experiment(runner)
    notes = runner.notes.for_experiment(eid)
    text = render_notes_for_exam(notes)
    for n in notes:
        assert n["concept"] in text
        assert n["principle"] in text
    # 渲染文本里不得出现任何题库参考答案片段
    for item in qb.ALL_ITEMS:
        assert item.answer[:20] not in text


def test_render_empty_notes_returns_empty_string():
    assert render_notes_for_exam([]) == ""


def test_note_checksum_detects_tampering(db):
    """改了笔记内容，checksum 必须对不上。"""
    from main import demo_script

    eid = _seeded_experiment(ExperimentRunner(db, ScriptedProvider(demo_script())))
    store = LearningNoteStore(db)
    nid = store.add(eid, TOPIC, "kp_btree", "概念A", "原理A", "例A", "错A", "规则A")
    assert store.verify_checksum(nid) is True
    db.execute("UPDATE learning_notes SET principle = '被偷偷改过的原理' WHERE id = ?", (nid,))
    assert store.verify_checksum(nid) is False


def test_note_checksum_is_order_sensitive():
    a = note_checksum("c", "p", "e", "m", "r")
    b = note_checksum("c", "p", "e", "r", "m")
    assert a != b, "五要素顺序变了 checksum 必须变，否则字段可能被串位"


def test_exam_cannot_write_notes(runner):
    """考试模式下伪造笔记必须被拦。"""
    eid = _seeded_experiment(runner)
    blind = AccessPolicy.blind_model(stage="exam", experiment_id=eid)
    with pytest.raises(ExamAccessViolation):
        LearningNoteStore(runner.db, blind).add(eid, TOPIC, "kp_btree", "c", "p")


# ======================================================================
# 7. experiment_id 作用域隔离
# ======================================================================
def test_open_book_cannot_read_another_experiment_notes(db):
    """换个 experiment_id 想读别人的笔记 —— 必须被拒。"""
    from main import demo_script

    r = ExperimentRunner(db, ScriptedProvider({**demo_script(), **EXTRA_SCRIPT}))
    first = _seeded_experiment(r)
    second = _seeded_experiment(r)
    assert first != second
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=second)
    with pytest.raises(ExamAccessViolation, match="跨实验"):
        LearningNoteStore(db, policy).for_experiment(first)


def test_open_book_cannot_read_another_experiment_knowledge(db):
    from main import demo_script

    r = ExperimentRunner(db, ScriptedProvider({**demo_script(), **EXTRA_SCRIPT}))
    first = _seeded_experiment(r)
    second = _seeded_experiment(r)
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=second)
    with pytest.raises(ExamAccessViolation, match="跨实验"):
        KnowledgeStore(db, policy).for_experiment(first)


def test_open_book_without_experiment_id_is_rejected():
    """开卷没有 experiment_id 就没法做作用域隔离 —— 直接拒绝，不猜。"""
    policy = AccessPolicy(stage="exam", blind=True, mode="open_book_memory")
    with pytest.raises(ExamAccessViolation, match="experiment_id"):
        policy.guard_scope("learning_notes", 1)


def test_scope_check_message_names_both_experiments(db):
    from main import demo_script

    r = ExperimentRunner(db, ScriptedProvider({**demo_script(), **EXTRA_SCRIPT}))
    first = _seeded_experiment(r)
    second = _seeded_experiment(r)
    policy = AccessPolicy.open_book_memory(stage="exam", experiment_id=second)
    with pytest.raises(ExamAccessViolation) as exc:
        policy.guard_scope("learning_notes", first)
    msg = str(exc.value)
    assert f"#{first}" in msg and f"#{second}" in msg, "报错要说清是哪两个实验"


def test_non_exam_policy_has_no_scope_restriction(db):
    """非考试场景（学习/实践/报告）不该被作用域限制挡住。"""
    store = LearningNoteStore(db, AccessPolicy.open_mode())
    assert isinstance(store.for_experiment(999), list)


# ======================================================================
# 8. 两种考试模式不串数据
# ======================================================================
def test_open_book_then_blind_does_not_leak_context(runner):
    """先开卷再盲考：盲考必须拿不到刚才开卷时的上下文。"""
    eid = _seeded_experiment(runner)
    open_ctx, _ = runner._build_exam_context(eid, "open_book_memory")
    blind_ctx, _ = runner._build_exam_context(eid, "blind_model")
    assert open_ctx and blind_ctx == ""
    for line in open_ctx.splitlines():
        if line.startswith("- 原理："):
            assert line not in blind_ctx, "开卷上下文残留到了盲考"


def test_stored_context_only_on_open_book_rows(runner):
    """落库层面：只有 open_book_memory 的行能有 context_used。"""
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    for row in runner.store.items(res.experiment_id):
        if row["mode"] == "blind_model":
            assert not (row["context_used"] or "").strip(), (
                f"{row['item_id']} 标为盲考却带了上下文"
            )
        else:
            assert (row["context_used"] or "").strip(), (
                f"{row['item_id']} 标为开卷却没带上下文"
            )


def test_modes_are_recorded_separately(runner):
    """三场考试的 mode 必须都落库，事后才能复核当时考的是哪种。"""
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    rows = runner.store.items(res.experiment_id)
    pre = [r for r in rows if r["phase"] == "pre"]
    post = [r for r in rows if r["phase"] == "post"]
    assert len(pre) == len(qb.PRE_SET)
    assert all(r["mode"] == "blind_model" for r in pre)
    assert {r["mode"] for r in post} == {"blind_model", "open_book_memory"}
    for r in post:
        bank_post = [i.item_id for i in qb.POST_SET]
        assert r["item_id"] in bank_post


def test_three_exams_produce_distinct_task_rows(runner):
    """同一道 POST 题被考两次 → 必须是两条独立 task 记录，不能复用。"""
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    rows = runner.store.items(res.experiment_id)
    post_task_ids = [r["task_id"] for r in rows if r["phase"] == "post"]
    assert len(post_task_ids) == len(set(post_task_ids)), "两次考试复用了同一条 task"
    eval_ids = [r["evaluation_id"] for r in rows]
    assert len(set(eval_ids)) == len(eval_ids), "判分记录被复用了"


def test_two_experiments_do_not_share_notes(db):
    """连跑两次实验，笔记必须各归各的，互不覆盖。"""
    from main import demo_script

    r = ExperimentRunner(db, ScriptedProvider({**demo_script(), **EXTRA_SCRIPT}))
    first = _seeded_experiment(r)
    second = _seeded_experiment(r)
    all_notes = db.query_all("SELECT experiment_id, COUNT(*) c FROM learning_notes GROUP BY experiment_id")
    assert {row["experiment_id"] for row in all_notes} == {first, second}


# ======================================================================
# 9. 闸门自检：失效必须中止，不得静默产出无效分数
# ======================================================================
def test_run_aborts_when_blind_gate_is_broken(runner, monkeypatch):
    monkeypatch.setattr(gate, "BLIND_MODEL_FORBIDDEN", frozenset())
    with pytest.raises(RuntimeError, match="访问闸门失效"):
        runner.run(TOPIC, DEFAULT_MATERIAL)


def test_run_aborts_when_open_book_gate_is_broken(runner, monkeypatch):
    monkeypatch.setattr(gate, "OPEN_BOOK_FORBIDDEN", frozenset())
    with pytest.raises(RuntimeError, match="访问闸门失效"):
        runner.run(TOPIC, DEFAULT_MATERIAL)


def test_unknown_exam_mode_is_rejected(runner):
    eid = _seeded_experiment(runner)
    with pytest.raises(ValueError, match="未知考试模式"):
        runner._build_exam_context(eid, "open_book_everything")


def test_old_style_blind_policy_defaults_to_blind_model():
    """V0.1 的AccessPolicy(stage, blind=True) 不能静默退化成开卷。"""
    old = AccessPolicy(stage="pre_test", blind=True)
    assert old.mode == "blind_model"
    with pytest.raises(ExamAccessViolation):
        old.guard("knowledge")
    with pytest.raises(ExamAccessViolation):
        old.guard("learning_notes")


# ======================================================================
# 10. 三个 gain 的解释纪律（V0.4：加上了 TRANSFER 与两个新指标）
# ======================================================================
def test_interpret_memory_only_case():
    """50/90/50/50 → 只学会用笔记，模型本身没变。"""
    r = interpret(50, 90, 50, 50, practice_usable_rate=1.0, note_quality_avg=80)
    assert r["used_external_memory"] is True
    assert r["remembered"] is False
    assert r["formed_concept"] is False
    assert "外部记忆" in r["headline"]
    assert r["alternative_explanations"] == []


def test_interpret_migration_case_requires_caveat():
    """50/90/50/70 → 只能叫「迁移迹象」，不能叫训练成功。"""
    r = interpret(50, 90, 50, 70, practice_usable_rate=1.0, note_quality_avg=80)
    assert r["used_external_memory"] is True
    assert r["formed_concept"] is True
    assert "不是训练成功" in r["headline"]
    assert r["alternative_explanations"], "盲考侧有提升必须列替代解释"


def test_interpret_recall_but_not_transfer():
    """V0.4 新增的关键情形：背住了同题，但换场景答不出 -> 背诵不是学习。

    这正是 V0.3 的两场 POST 同题设计会误判成「学会了」的情况。
    """
    r = interpret(50, 90, 80, 52, practice_usable_rate=1.0, note_quality_avg=80)
    assert r["used_external_memory"] is True
    assert r["remembered"] is True
    assert r["formed_concept"] is False
    assert "背住了" in r["headline"]
    assert r["transfer_vs_memory"] > 0, "迁移不如记忆强，折损应为正"


def test_interpret_no_gain_case():
    r = interpret(70, 72, 70, 70, practice_usable_rate=0.5, note_quality_avg=50)
    assert r["used_external_memory"] is False
    assert r["formed_concept"] is False
    assert r["proves"] and "未能观测到" in r["proves"][0]


def test_interpret_anomalous_case():
    """开卷没提升、盲考却提升 —— 数据组合可疑，不能硬给结论。"""
    r = interpret(80, 82, 95, 95, practice_usable_rate=1.0, note_quality_avg=80)
    assert "异常" in r["headline"]
    assert r["proves"] == []
    assert r["disproves"]


def test_interpret_threshold_ignores_noise():
    """3 分以内的波动不算提升 —— 否则报告会把噪声写成结论。"""
    r = interpret(80, 83, 80, 80, practice_usable_rate=1.0, note_quality_avg=80)
    assert r["used_external_memory"] is False
    assert r["formed_concept"] is False


def test_interpret_requires_chain_intact_for_migration_claim():
    """笔记不靠谱时，即便 TRANSFER 高也不能下迁移结论。

    笔记本身是错的，开卷高分只说明「抄对了自己写的东西」。
    """
    r = interpret(50, 90, 50, 75, practice_usable_rate=0.2, note_quality_avg=30)
    assert r["formed_concept"] is True
    assert r["chain_intact"] is False
    assert any("笔记质量不达标" in a for a in r["alternative_explanations"])


# ======================================================================
# 11. ScriptedProvider 的上下文分叉
# ======================================================================
def test_scripted_provider_forks_on_context_marker():
    provider = ScriptedProvider(
        {"ANSWER": {"带了笔记": "开卷答案", "default": "盲考答案"}}
    )
    assert provider.ask_text("s", "[[TASK:ANSWER]] 带了笔记的题目") == "开卷答案"
    assert provider.ask_text("s", "[[TASK:ANSWER]] 光秃秃的题目") == "盲考答案"


def test_scripted_provider_prefers_first_matching_marker():
    provider = ScriptedProvider({"ANSWER": {"甲": "A", "乙": "B", "default": "D"}})
    assert provider.ask_text("s", "[[TASK:ANSWER]] 乙") == "B", "字典序=优先级"


def test_demo_script_answers_differ_per_stage():
    """离线演示脚本必须让 PRE 与 OPEN_BOOK 返回不同答案。

    三场一模一样的话，离线实验永远测不出东西 —— 那是自欺欺人。
    """
    from main import demo_script

    script = demo_script()
    assert set(script) >= {
        "STUDY", "LEARNING_NOTES", "PRACTICE", "ANSWER", "EVALUATE"
    }
    assert isinstance(script["ANSWER"], dict), "ANSWER 必须按上下文分叉"
    assert script["ANSWER"]["以下是你自己此前整理的学习笔记"] != script["ANSWER"]["default"]
    assert isinstance(script["EVALUATE"], dict), "EVALUATE 必须按上下文分叉"
    assert "notes" in json.loads(script["LEARNING_NOTES"])


def test_evaluate_fork_marker_never_appears_in_question_bank():
    """回归测试：EVALUATE 的分叉标记绝不能出现在题库题目/参考答案里。

    踩过的坑：用「内部节点分叉数固定」当标记，而它恰好在 post_btree_01 的
    参考答案里 —— 于是盲答也被误判成开卷，白白多出 47 分假提升。
    评估 prompt 里含参考答案，标记只要碰它就废了。
    """
    from main import demo_script

    script = demo_script()
    fork = script["EVALUATE"]
    assert isinstance(fork, dict)
    for marker in fork:
        if marker == "default":
            continue
        assert marker, "分叉标记不能为空"
        for item in qb.ALL_ITEMS:
            assert marker not in item.question, f"分叉标记泄漏到题目：{item.item_id}"
            assert marker not in item.answer, f"分叉标记泄漏到参考答案：{item.item_id}"


def test_answer_fork_marker_only_in_open_book_context():
    """回归测试：ANSWER 的分叉标记只能被开卷上下文命中，不能出现在题目里。"""
    from main import demo_script

    script = demo_script()
    marker = "以下是你自己此前整理的学习笔记"
    for item in qb.ALL_ITEMS:
        assert marker not in item.question
        assert marker not in item.answer


def test_offline_experiment_produces_nonzero_open_book_gain():
    """用户要求：离线实验必须能测出 OPEN_BOOK_GAIN > 0。

    测不出的话说明分叉机制坏了 —— 那整套离线管线就是自欺欺人。
    """
    from main import demo_script

    from memory.db import Database

    import tempfile, os

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "gain.db")
    db = Database(path)
    db.init_schema()
    try:
        r = ExperimentRunner(db, ScriptedProvider({**demo_script(), **EXTRA_SCRIPT}))
        res = r.run(TOPIC, DEFAULT_MATERIAL)
        assert res.open_book_gain > 0, "离线脚本应让开卷明显高于起点"
        assert res.memory_open_score > res.pre_score
    finally:
        db.close()


def test_offline_blind_post_matches_pre_when_no_context():
    """盲考两场用的是同一份盲答，分数应完全一致。

    分数不一致就说明盲考里混进了上下文 —— 那 MEMORY_GAIN 是假的。
    V0.4 的 TRANSFER 是新题，不参与这条同分断言。
    """
    from main import demo_script

    from memory.db import Database

    import tempfile, os

    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "blind.db"))
    db.init_schema()
    try:
        r = ExperimentRunner(db, ScriptedProvider({**demo_script(), **EXTRA_SCRIPT}))
        res = r.run(TOPIC, DEFAULT_MATERIAL)
        assert res.memory_gain == 0.0, (
            f"盲答两场应同分，实际 MEMORY_GAIN={res.memory_gain}，疑似上下文残留"
        )
    finally:
        db.close()


def test_demo_script_notes_cover_all_five_elements():
    from main import demo_script

    notes = json.loads(demo_script()["LEARNING_NOTES"])["notes"]
    assert len(notes) >= 4, "四个知识点都该有笔记，否则开卷有缺口"
    kps = {n["kp"] for n in notes}
    assert kps == set(qb.kp_coverage(qb.PRE_SET)), "笔记必须覆盖 PRE 卷全部知识点"
    for n in notes:
        for f in ("concept", "principle", "example", "common_mistake",
                  "application_rule"):
            assert n[f].strip(), f"笔记缺字段 {f}"
