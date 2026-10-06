"""tests/test_v05_experiment.py —— V0.5 端到端：流程、落库、隔离、污染。

这一层跑完整六阶段链路（ScriptedProvider，不联网），
验证的是「结构对不对」而不是「分数好不好」：

- 六阶段顺序固定
- 每阶段落齐 8 列
- 盲考 context恒空、开卷 context 非空
- 实践的机器判分与 LLM 评审分表
- 两个实验互不污染
- V0.4 的数据不被 V0.5 读走，反之亦然
"""

from __future__ import annotations

import json

import pytest

from brain import experiment_v2 as ev2
from brain import novel_material as nm
from brain import report_v2 as r2
from brain import stats_v2 as st
from brain.llm import LLMResponse, ScriptedProvider
from memory.db import Database, create_v2_sentinel, v2_experiment_key
from practice.objective import OBJECTIVE_TASKS, TASKS_FINGERPRINT

# ==========================================================================
# 离线脚本：正确产出 + 正确笔记
# ==========================================================================

_CORRECT: dict[str, str] = {
    "obj_hash_01": json.dumps(
        {"key": "abc", "steps": [7, 1014, 1858, 46886],
         "lumen_hash": 46886, "bucket": 6}),
    "obj_hash_02": json.dumps(
        {"code": "def lumen_hash(key):\n    h = 7\n"
                 "    for c in key:\n        h = (h * 131 + ord(c)) % 65537\n"
                 "    return h"}),
    "obj_bucket_01": json.dumps(
        {"assignments": [{"key": n, "from_bucket": 0, "to_bucket": 0}
                         for n in ("k0", "k8", "k16", "k24", "k32")],
         "migrated_count": 0, "still_over_limit": True}),
    "obj_bucket_02": json.dumps(
        {"sql": "SELECT bucket FROM zephyra_buckets"
                " WHERE live_count > 4 ORDER BY bucket"}),
    "obj_compact_01": json.dumps({"compress": [0, 3], "epoch_after": 3}),
    "obj_compact_02": json.dumps(
        {"code": "def should_compact(tombstones, live):\n"
                 "    return tombstones >= 2 and live <= 1"}),
    "obj_consist_01": json.dumps(
        {"readable": [], "need_retag": ["x1", "x2", "x3", "x4"],
         "new_tags": {"x1": 2502, "x2": 4002, "x3": 6002, "x4": 8202}}),
    "obj_consist_02": json.dumps(
        {"sql": "SELECT key FROM zephyra_keys WHERE ztag != (lumen_hash +"
                " (SELECT value FROM zephyra_meta WHERE name='epoch')) % 65537"
                " ORDER BY key"}),
}

#: 用题面片段而非 task_id 分派——prompt 里没有 task_id
_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("obj_hash_01", "LumenHash 值，并给出它所属的桶号"),
    ("obj_hash_02", "def lumen_hash(key)"),
    ("obj_bucket_01", "0、8、16、24、32"),
    ("obj_bucket_02", "live_count > 4"),
    ("obj_compact_01", "tombstones=9"),
    ("obj_compact_02", "def should_compact(tombstones, live)"),
    ("obj_consist_01", "4000、6000、8200"),
    ("obj_consist_02", "zephyra_meta"),
)

_SCRIPT: dict[str, object] = {
    "ANSWER": {
        "default": "我不知道这个虚构系统的规则。",
        "参考资料": (
            "根据我自己的笔记：LumenHash 初值 7，(h*131+ord(c)) mod 65537；"
            "桶号 = hash mod 8；分裂是恒等变换 (b+8) mod 8 = b；"
            "压缩需墓碑>=2 且活跃<=1；Z-Tag=(hash+epoch) mod 65537，"
            "epoch 变了全库失配。"
        ),
    },
    "EVALUATE": '{"score":70,"verdict":"pass","feedback":"覆盖了部分要点",'
                '"missing_points":[]}',
    "STUDY": json.dumps({"items": [
        {"title": "LumenHash", "content": "h=7 起逐字符 (h*131+ord(c)) mod 65537",
         "keywords": ["hash"]},
        {"title": "Z-8 分桶", "content": "bucket=hash mod 8，分裂是恒等变换",
         "keywords": ["bucket"]},
        {"title": "压缩", "content": "墓碑>=2 且活跃<=1 时触发", "keywords": ["compact"]},
        {"title": "Epoch", "content": "ZTag=(hash+epoch) mod 65537", "keywords": ["epoch"]},
    ]}, ensure_ascii=False),
    "LEARNING_NOTES": json.dumps({"notes": [
        {"kp": "kp_lumen_hash", "concept": "LumenHash",
         "principle": "h=7 起逐字符 (h*131+ord(c)) mod 65537",
         "example": 'LumenHash("ab")=1858', "common_mistake": "初值写成0",
         "application_rule": "算哈希再对8取模"},
        {"kp": "kp_lumen_bucket", "concept": "Z-8 分桶", "principle": "分裂是恒等变换",
         "example": "(b+8) mod 8 = b", "common_mistake": "以为分裂有效",
         "application_rule": "容量按8桶规划"},
        {"kp": "kp_lumen_compact", "concept": "压缩", "principle": "墓碑>=2 且活跃<=1",
         "example": "3墓碑2活跃不触发", "common_mistake": "把且看成或",
         "application_rule": "两条件同时满足才压"},
        {"kp": "kp_lumen_consist", "concept": "Epoch 标签",
         "principle": "ZTag=(hash+epoch) mod 65537", "example": "epoch+1 全库失配",
         "common_mistake": "以为标签自动更新", "application_rule": "压缩后必须批量重算"},
    ]}, ensure_ascii=False),
    "QUALITY": '{"accuracy":85,"completeness":80,"transferable":75,'
               '"no_hallucination":90,"feedback":"笔记基本准确","problems":[]}',
    "PRACTICE_JUDGE": '{"usable":true,"score":75,"verdict":"pass",'
                     '"feedback":"产出可执行"}',
}


class V2Scripted(ScriptedProvider):
    """实践任务按题面片段分派，其余走基类的标记分叉。"""

    def complete(self, messages, *, temperature=0.7, max_tokens=None,
                 json_mode=False):
        joined = "\n".join(m.get("content", "") for m in messages)
        for task_id, frag in _FRAGMENTS:
            if frag in joined:
                return LLMResponse(text=_CORRECT[task_id], model="scripted")
        return super().complete(messages, temperature=temperature,
                                max_tokens=max_tokens, json_mode=json_mode)


@pytest.fixture()
def v2db(tmp_path) -> Database:
    database = Database(str(tmp_path / "v05.db"))
    database.init_schema()
    yield database
    database.close()


@pytest.fixture()
def v2runner(v2db: Database) -> ev2.V2ExperimentRunner:
    prov = V2Scripted(dict(_SCRIPT))
    return ev2.V2ExperimentRunner(v2db, prov, prov)


@pytest.fixture(scope="module")
def _shared_run():
    """跑一次完整链路，供多个只读断言复用（省 8 次全流程）。"""
    import tempfile
    from pathlib import Path
    d = tempfile.mkdtemp()
    db = Database(str(Path(d) / "shared.db"))
    db.init_schema()
    prov = V2Scripted(dict(_SCRIPT))
    runner = ev2.V2ExperimentRunner(db, prov, prov)
    res = runner.run(batch_id="shared", trial_no=1)
    yield db, res, ev2.V2Store(db)
    db.close()


# ==========================================================================
# 一、流程与阶段
# ==========================================================================


def test_stage_order_is_fixed():
    assert ev2.STAGES == (
        "pre_blind", "study", "practice",
        "open_book", "memory_blind", "transfer_blind",
    )


def test_blind_stages_exclude_open_book():
    assert "open_book" not in ev2.BLIND_STAGES
    assert {"pre_blind", "memory_blind", "transfer_blind"} <= ev2.BLIND_STAGES


def test_full_run_produces_result(v2runner):
    res = v2runner.run(batch_id="t1", trial_no=1)
    assert res.experiment_id > 0
    assert set(res.scores) >= {
        "pre_blind", "open_book", "memory_blind", "transfer_blind",
        "objective_practice_avg", "objective_practice_pass_rate",
        "evaluator_practice_avg", "note_quality_avg",
    }


def test_run_records_four_gains(v2runner):
    res = v2runner.run(batch_id="t2", trial_no=1)
    assert set(res.gains) == {
        "external_memory_gain", "memory_retention_gain",
        "transfer_gain", "objective_practice_gain",
    }


def test_four_exams_each_have_eight_items(_shared_run):
    db, res, store = _shared_run
    for stage in ("pre_blind", "open_book", "memory_blind", "transfer_blind"):
        n = db.scalar(
            "SELECT COUNT(*) FROM v2_stage_items WHERE experiment_id=? AND stage=?",
            (res.experiment_id, stage),
        )
        assert n == 8, f"{stage} 不是 8 题"


def test_stage_items_have_all_eight_required_columns(_shared_run):
    db, res, store = _shared_run
    rows = db.query_all(
        "SELECT * FROM v2_stage_items WHERE experiment_id=? AND stage='pre_blind'",
        (res.experiment_id,),
    )
    required = ("question", "answer", "score", "evaluator", "context_used",
                "experiment_id", "knowledge_point", "question_type")
    for row in rows:
        for col in required:
            assert col in row, f"缺列 {col}"
        assert row["question"]
        assert row["answer"] != "" or row["score"] == 0.0
        assert row["knowledge_point"]
        assert row["question_type"] == "basic"


# ==========================================================================
# 二、隔离：盲考 / 开卷
# ==========================================================================


def test_blind_context_always_empty(_shared_run):
    db, res, store = _shared_run
    rows = db.query_all(
        "SELECT stage, context_used FROM v2_stage_items"
        " WHERE experiment_id=? AND stage IN ('pre_blind','memory_blind','transfer_blind')",
        (res.experiment_id,),
    )
    assert rows
    for r in rows:
        assert (r["context_used"] or "").strip() == "", f"{r['stage']} 上下文非空"


def test_open_book_has_context(_shared_run):
    db, res, store = _shared_run
    n = db.scalar(
        "SELECT COUNT(*) FROM v2_stage_items WHERE experiment_id=? AND stage='open_book'"
        " AND LENGTH(TRIM(context_used)) > 0",
        (res.experiment_id,),
    )
    assert n == 8


def test_open_book_reads_only_own_experiment(v2runner, v2db):
    """先跑一个实验产生笔记，再跑第二个——第二个开卷不能读到第一个的。"""
    v2runner.run(batch_id="a", trial_no=1)
    v2runner.run(batch_id="b", trial_no=1)
    rows = v2db.query_all(
        "SELECT experiment_id, context_used FROM v2_stage_items"
        " WHERE stage='open_book'"
    )
    # 两个实验的上下文都非空，但两者的 experiment_id 不同，
    # 即各自只拿自己的笔记（若串了，AccessPolicy 会在读取时报错）
    ids = {r["experiment_id"] for r in rows}
    assert len(ids) == 2
    for r in rows:
        assert r["context_used"].strip()


def test_blind_isolation_probe_passes(v2runner):
    v2runner.assert_blind_isolation()


def test_open_book_isolation_probe_passes(v2runner, v2db):
    eid = v2db.scalar("SELECT MAX(id) FROM v2_experiments") or 1
    v2runner.assert_open_book_isolation(int(eid))


def test_transfer_question_states_premise(_shared_run):
    """transfer 卷的题面必须自带前提，否则题目不可解。"""
    db, res, store = _shared_run
    rows = db.query_all(
        "SELECT item_id, question FROM v2_stage_items"
        " WHERE experiment_id=? AND stage='transfer_blind'",
        (res.experiment_id,),
    )
    assert len(rows) == 8
    for r in rows:
        assert len(r["question"]) > 40, f"{r['item_id']} 题面过短，前提可能没写清"


# ==========================================================================
# 三、实践：机器判分与 LLM 评审分表
# ==========================================================================


def test_practice_runs_has_eight_rows(_shared_run):
    db, res, store = _shared_run
    assert db.count("practice_runs") == len(OBJECTIVE_TASKS)


def test_practice_run_columns_present(_shared_run):
    db, _, store = _shared_run
    rows = db.query_all("SELECT * FROM practice_runs")
    for col in ("task_id", "candidate_output", "objective_result",
                "objective_score", "objective_passed", "parse_error"):
        assert col in rows[0], f"practice_runs 缺列 {col}"


def test_practice_runs_have_no_evaluator_column(_shared_run):
    """机器判分表里不该出现 evaluator_score——两套分数必须物理分开。"""
    db, _, store = _shared_run
    cols = {r["name"] for r in db.query_all("PRAGMA table_info(practice_runs)")}
    assert "evaluator_score" not in cols
    assert "objective_score" in cols


def test_evaluations_stored_in_legacy_table(_shared_run):
    db, res, store = _shared_run
    rows = db.query_all(
        "SELECT * FROM practice_evaluations WHERE experiment_id=?",
        (v2_experiment_key(res.experiment_id),),
    )
    assert len(rows) == len(OBJECTIVE_TASKS)
    assert "score" in rows[0] and "usable" in rows[0]


def test_evaluations_carry_task_id(_shared_run):
    """评审行必须带 task_id，否则与机器判分无法对齐。

    没有它时看到「一个 55 分一个 70 分」不知道是谁，
    分歧就成了无法定位的噪声。
    """
    db, res, store = _shared_run
    evals = store.evaluations_for(res.experiment_id)
    ids = {e["task_id"] for e in evals}
    assert "" not in ids, "存在缺 task_id 的评审行"
    assert ids == {t.task_id for t in OBJECTIVE_TASKS}


def test_objective_and_evaluator_can_be_joined_by_task_id(_shared_run):
    db, res, store = _shared_run
    runs = {r["task_id"]: r["objective_score"] for r in store.practice_runs(res.experiment_id)}
    evals = {e["task_id"]: e["score"] for e in store.evaluations_for(res.experiment_id)}
    assert set(runs) == set(evals), "两侧 task_id 集合不一致，分歧无法逐任务分析"
    assert len(runs) == len(OBJECTIVE_TASKS)


def test_objective_and_evaluator_can_disagree(_shared_run):
    """两套分数独立存在，允许不一致——不一致本身就是数据。"""
    db, res, store = _shared_run
    run = db.scalar("SELECT AVG(objective_score) FROM practice_runs")
    ev = db.scalar(
        "SELECT AVG(score) FROM practice_evaluations WHERE experiment_id=?",
        (v2_experiment_key(res.experiment_id),),
    )
    assert run is not None and ev is not None
    assert run != ev, "脚本里机器分 100 / LLM 75，两表不应被合并成同一个数"


def test_practice_candidate_output_is_stored(_shared_run):
    db, _, store = _shared_run
    rows = db.query_all("SELECT task_id, candidate_output FROM practice_runs")
    assert all(r["candidate_output"] for r in rows)


# ==========================================================================
# 四、experiment 隔离与 V0.4/V0.5 互不污染
# ==========================================================================


def test_v2_experiment_key_is_negative():
    assert v2_experiment_key(1) == -1
    assert v2_experiment_key(2) == -2


def test_v2_rows_use_negative_id(_shared_run):
    db, res, store = _shared_run
    for table in ("learning_notes", "note_quality", "practice_evaluations"):
        n = db.scalar(f"SELECT COUNT(*) FROM {table} WHERE experiment_id > 0")
        assert n == 0, f"{table} 里出现了正 id，V2 数据会混进 V0.4 查询"


def test_v2_rows_are_readable_with_negative_id(_shared_run):
    db, res, store = _shared_run
    key = v2_experiment_key(res.experiment_id)
    for table in ("learning_notes", "note_quality", "practice_evaluations"):
        n = db.scalar(f"SELECT COUNT(*) FROM {table} WHERE experiment_id = ?", (key,))
        assert n > 0, f"{table} 用负 id 读不到 V2 数据"


def test_sentinel_row_exists_for_each_v2_experiment(_shared_run):
    db, res, store = _shared_run
    row = db.query_one("SELECT * FROM experiments WHERE id = ?",
                       (v2_experiment_key(res.experiment_id),))
    assert row is not None
    assert row["topic"].startswith("[V2]")


def test_two_experiments_do_not_pollute_each_other(v2db):
    prov_a = V2Scripted(dict(_SCRIPT))
    prov_b = V2Scripted(dict(_SCRIPT))
    ra = ev2.V2ExperimentRunner(v2db, prov_a, prov_a)
    rb = ev2.V2ExperimentRunner(v2db, prov_b, prov_b)
    a = ra.run(batch_id="pa", trial_no=1)
    b = rb.run(batch_id="pb", trial_no=1)
    assert a.experiment_id != b.experiment_id
    items_a = v2db.query_all(
        "SELECT * FROM v2_stage_items WHERE experiment_id=?", (a.experiment_id,))
    items_b = v2db.query_all(
        "SELECT * FROM v2_stage_items WHERE experiment_id=?", (b.experiment_id,))
    assert len(items_a) == len(items_b)
    assert not ({i["id"] for i in items_a} & {i["id"] for i in items_b})
    # 各 runner 只能看到自己的笔记
    notes_a = ra.notes.for_v2_experiment(a.experiment_id)
    notes_b = rb.notes.for_v2_experiment(b.experiment_id)
    assert notes_a and notes_b
    assert {n["id"] for n in notes_a} != {n["id"] for n in notes_b}


def test_v04_data_not_visible_to_v05(v2db):
    """用真实的 V0.4 写入路径造一条正 id 笔记，V2 查询必须读不到它。

    刻意走 store 而不是手写 INSERT：手写会漏掉 checksum/topic 这类
    NOT NULL 列，写出来的「V0.4 数据」其实不是 V0.4 真实会写出的形态。
    """
    v2db.execute(
        "INSERT INTO experiments(id, topic, created_at) VALUES (?,?,?)",
        (999, "V0.4 老实验", "2026-01-01 00:00:00"),
    )
    from knowledge.store import KnowledgeStore
    from learning_notes.store import LearningNoteStore
    from memory.gate import AccessPolicy

    policy = AccessPolicy.open_mode()
    old_notes = LearningNoteStore(v2db, policy)
    old_notes.add(experiment_id=999, topic="V0.4 老实验",
                  kp="kp_old", concept="旧概念", principle="旧原理",
                  example="旧例子", common_mistake="旧错误",
                  application_rule="旧规则")

    prov = V2Scripted(dict(_SCRIPT))
    runner = ev2.V2ExperimentRunner(v2db, prov, prov)
    res = runner.run(batch_id="iso", trial_no=1)
    notes = runner.notes.for_v2_experiment(res.experiment_id)
    assert notes
    assert all(n["experiment_id"] < 0 for n in notes)
    assert all(n["concept"] != "旧概念" for n in notes), "V0.4 的笔记漏进了 V2"


def test_v05_does_not_pollute_v04_queries(v2db):
    """反向：V2 跑完后，按正 id 查老表必须仍然是空的。"""
    prov = V2Scripted(dict(_SCRIPT))
    runner = ev2.V2ExperimentRunner(v2db, prov, prov)
    runner.run(batch_id="rev", trial_no=1)
    for table in ("learning_notes", "note_quality", "practice_evaluations"):
        n = v2db.scalar(
            f"SELECT COUNT(*) FROM {table} WHERE experiment_id > 0")
        assert n == 0, f"{table} 出现正 id，V0.4 报表会把 V2 数据算进去"


# ==========================================================================
# 五、指纹冻结
# ==========================================================================


def test_batch_freezes_three_fingerprints(v2db, v2runner):
    v2runner.run(batch_id="fz", trial_no=1)
    row = v2db.query_one("SELECT * FROM v2_batches WHERE batch_id='fz'")
    assert len(row["bank_fingerprint"]) == 64
    assert len(row["material_checksum"]) == 64
    assert row["tasks_fingerprint"] == TASKS_FINGERPRINT


def test_verify_frozen_passes_when_untouched(v2db, v2runner):
    v2runner.run(batch_id="fz2", trial_no=1)
    v2runner.store.verify_frozen("fz2")


def test_changed_bank_invalidates_batch(v2db, v2runner, monkeypatch):
    """改了题库指纹 → 整批作废。这条不守，跨批次比较全是假的。"""
    v2runner.run(batch_id="fz3", trial_no=1)
    from brain import benchmark_v2 as b2
    monkeypatch.setattr(b2, "BANK_FINGERPRINT", "0" * 64)
    with pytest.raises(RuntimeError) as ei:
        v2runner.store.verify_frozen("fz3")
    assert "题库" in str(ei.value)
    assert "作废" in str(ei.value)


def test_changed_material_invalidates_batch(v2db, v2runner, monkeypatch):
    v2runner.run(batch_id="fz4", trial_no=1)
    from brain import novel_material as material
    monkeypatch.setattr(material, "NOVEL_MATERIAL", material.NOVEL_MATERIAL + "\n改动")
    with pytest.raises(RuntimeError) as ei:
        v2runner.store.verify_frozen("fz4")
    assert "学习材料" in str(ei.value)


def test_changed_tasks_invalidates_batch(v2db, v2runner, monkeypatch):
    v2runner.run(batch_id="fz5", trial_no=1)
    from practice import objective as ob
    monkeypatch.setattr(ob, "TASKS_FINGERPRINT", "0" * 64)
    with pytest.raises(RuntimeError) as ei:
        v2runner.store.verify_frozen("fz5")
    assert "实践任务集" in str(ei.value)


def test_register_batch_raises_planned_trials(v2db, v2runner):
    v2runner.run(batch_id="pt", trial_no=1)
    v2runner.run(batch_id="pt", trial_no=2)
    row = v2db.query_one("SELECT planned_trials FROM v2_batches WHERE batch_id='pt'")
    assert row["planned_trials"] == 2, "planned_trials 不能永远停在 1"


# ==========================================================================
# 六、幂等性与 schema
# ==========================================================================


def test_init_schema_is_idempotent(v2db):
    v2db.init_schema()
    v2db.init_schema()
    assert set(("v2_experiments", "v2_stage_items", "practice_runs",
                "v2_batches")) <= set(v2db.table_names())


def test_sentinel_is_idempotent(v2db):
    create_v2_sentinel(v2db.conn, 1, "[V2] 测试")
    create_v2_sentinel(v2db.conn, 1, "[V2] 测试")
    assert v2db.scalar("SELECT COUNT(*) FROM experiments WHERE id = -1") == 1


# ==========================================================================
# 七、报告能从库里重建
# ==========================================================================


def test_batch_report_from_store(_shared_run):
    db, _, store = _shared_run
    rep = r2.build_batch_report_from_store(store, "shared")
    assert rep["trials"] == 1
    assert len(rep["claims"]) == 4
    assert r2.scan_forbidden(str(rep)) == []


def test_trial_report_from_store(_shared_run):
    db, res, store = _shared_run
    rep = r2.build_trial_report_from_store(store, res.experiment_id)
    assert rep["experiment_id"] == res.experiment_id
    assert rep["consistency_check"]["blind_context_empty"] is True
    assert rep["consistency_check"]["answer_leak"] is False


def test_trial_report_from_store_unknown_id(_shared_run):
    db, _, store = _shared_run
    with pytest.raises(KeyError):
        r2.build_trial_report_from_store(store, 987654)


def test_batch_text_from_store_has_no_forbidden(_shared_run):
    db, _, store = _shared_run
    rep = r2.build_batch_report_from_store(store, "shared")
    txt = r2.format_batch_text(rep)
    assert r2.scan_forbidden(txt) == []


def test_aggregated_stats_from_store(_shared_run):
    db, _, store = _shared_run
    trials = st.load_trials(store, "shared")
    summary = st.aggregate_trials(trials)
    assert summary["trials"] == 1
    # 单次 trial → 所有 gain 判定都必须是 noisy
    for g in summary["gains"].values():
        assert g["verdict"] == "noisy"
    assert summary["sample_size_diagnostics"]["sufficient_trials"] is False
