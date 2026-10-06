"""实验流程测试 —— V0.2 核心。"""

from __future__ import annotations

import json

import pytest

from brain import question_bank as qb
from brain.experiment import ExperimentRunner
from brain.materials import DEFAULT_MATERIAL
from brain.report import build_report, format_text

TOPIC = "SQLite 索引"

# 离线脚本：STUDY / PRACTICE 由 conftest 的 SCRIPT 补齐，这里只覆盖实验专用标记
EXTRA_SCRIPT = {
    "STUDY": json.dumps(
        {
            "items": [
                {
                    "title": "B-Tree 平衡结构",
                    "content": "所有叶子同层，查找次数不随数据量增长。",
                    "keywords": ["B-Tree"],
                },
                {
                    "title": "覆盖索引",
                    "content": "查询列全在索引中时无需回表。",
                    "keywords": ["覆盖索引"],
                },
            ]
        },
        ensure_ascii=False,
    ),
    "PRACTICE": json.dumps(
        {"outcome": "success", "result": "已按材料完成实践任务。", "kp": "kp_btree"},
        ensure_ascii=False,
    ),
}


@pytest.fixture()
def runner(db, provider):
    provider.script.update(EXTRA_SCRIPT)
    return ExperimentRunner(db, provider)


def test_bank_registered_once(runner):
    a = runner.bank_store.register()
    b = runner.bank_store.register()
    assert a == b


def test_bank_verify_detects_tampering(runner):
    bid = runner.bank_store.register()
    runner.db.execute(
        "UPDATE question_bank SET fingerprint = ? WHERE id = ?", ("deadbeef", bid)
    )
    with pytest.raises(RuntimeError, match="指纹不一致"):
        runner.bank_store.verify(bid)


def test_material_checksum_is_recorded(runner):
    m = runner.materials.save(TOPIC, "t", "内容内容内容")
    assert len(m["checksum"]) == 64
    assert runner.materials.get(m["id"])["checksum"] == m["checksum"]


def test_identical_material_is_not_duplicated(runner):
    """同内容同版本的材料重复实验时应复用，不堆重复行。"""
    a = runner.materials.save(TOPIC, "t", "同样的内容")
    b = runner.materials.save(TOPIC, "t", "同样的内容")
    assert a["id"] == b["id"]
    assert b["reused"] is True
    assert runner.db.count("materials") == 1


def test_changed_creates_new_material_row(runner):
    a = runner.materials.save(TOPIC, "t", "内容甲")
    b = runner.materials.save(TOPIC, "t", "内容乙")
    assert a["id"] != b["id"]
    assert a["checksum"] != b["checksum"]
    assert runner.db.count("materials") == 2


def test_experiment_runs_leave_one_material_row(runner):
    """一次实验只应占用一行材料记录。"""
    runner.run(TOPIC, DEFAULT_MATERIAL)
    assert runner.db.count("materials") == 1


def test_full_experiment_runs(runner):
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    assert res.experiment_id > 0
    for score in (
        res.pre_score, res.memory_open_score,
        res.memory_blind_score, res.transfer_blind_score,
    ):
        assert 0 <= score <= 100
    assert res.open_book_gain == round(res.memory_open_score - res.pre_score, 2)
    assert res.memory_gain == round(res.memory_blind_score - res.pre_score, 2)
    assert res.transfer_gain == round(res.transfer_blind_score - res.pre_score, 2)


def test_experiment_records_all_evidence(runner):
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    items = runner.store.items(res.experiment_id)
    pre = [i for i in items if i["exam_kind"] == "pre"]
    memory = [i for i in items if i["exam_kind"] == "memory"]
    transfer = [i for i in items if i["exam_kind"] == "transfer"]
    # V0.4：PRE 1 场 + MEMORY 2 场（同卷）+ TRANSFER 1 场
    assert len(pre) == len(qb.PRE_SET)
    assert len(memory) == len(qb.POST_SET) * 2
    assert len(transfer) == len(qb.TRANSFER_ITEMS)
    for row in pre + memory + transfer:
        assert row["question"], "题目快照必须留存"
        assert row["reference_answer"], "参考答案必须留存"
        assert row["candidate_answer"], "考生答案必须留存"
        assert row["score"] is not None
        assert row["task_id"] and row["evaluation_id"]
        assert row["mode"] in {"blind_model", "open_book_memory"}


def test_transfer_never_runs_open_book(runner):
    """TRANSFER 只跑盲考 —— 开卷能查笔记就测不出迁移能力了。"""
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    transfer = [
        i for i in runner.store.items(res.experiment_id) if i["exam_kind"] == "transfer"
    ]
    assert transfer, "TRANSFER 卷必须真的考"
    for row in transfer:
        assert row["mode"] == "blind_model"
        assert not (row["context_used"] or "").strip()


def test_experiment_persists_practice_records(runner):
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    recs = runner.store.practice(res.experiment_id)
    assert len(recs) == len(qb.PRE_SET)
    for r in recs:
        assert r["outcome"] in {"success", "failure"}
        assert r["lesson"], "每次实践都要沉淀经验"
        assert r["kp"]


def test_experiment_marks_status_done(runner):
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    exp = runner.store.get(res.experiment_id)
    assert exp["status"] == "done"
    assert exp["finished_at"]
    assert exp["material_id"] and exp["bank_id"]


def test_detail_json_records_bank_and_material(runner):
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    detail = json.loads(runner.store.get(res.experiment_id)["detail_json"])
    assert detail["bank_fingerprint"] == qb.BANK_FINGERPRINT
    assert len(detail["material_checksum"]) == 64
    assert detail["pre_results"] and detail["memory_open_results"]
    assert detail["memory_blind_results"] and detail["transfer_results"]


def test_kp_comparison_covers_all_kps(runner):
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    cmp_result = runner.compare(res.experiment_id)
    all_rows = (
        cmp_result["improved"] + cmp_result["flat"] + cmp_result["regressed"]
    )
    assert len(all_rows) == 4
    for row in all_rows:
        assert row["label"]
        assert row["memory_delta"] == round(row["memory_blind"] - row["pre"], 2)
        assert row["transfer_delta"] == round(row["transfer_blind"] - row["pre"], 2)
        assert row["transfer_vs_memory"] == round(
            row["memory_delta"] - row["transfer_delta"], 2
        )


def test_study_extracts_knowledge(runner):
    out = runner.study(TOPIC, DEFAULT_MATERIAL)
    assert len(out["knowledge_ids"]) == 2
    assert runner.db.count("knowledge") == 2


def test_study_degrades_when_llm_fails(db, provider):
    provider.script.pop("STUDY", None)
    provider.script["STUDY"] = "不是 JSON"
    r = ExperimentRunner(db, provider)
    out = r.study(TOPIC, DEFAULT_MATERIAL)
    assert len(out["knowledge_ids"]) == 4, "LLM 失效应按 max_items 兜底"


def test_practice_scores_success_and_failure():
    long_text = "已按学习材料完成实践：先写出 B-Tree 结构的验证步骤，再实测回表次数。"
    assert ExperimentRunner._practice_score("success", long_text) == 85.0
    assert ExperimentRunner._practice_score("success", "太短") == 60.0
    assert ExperimentRunner._practice_score("failure", long_text) == 40.0


def test_report_contains_all_required_sections(runner):
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    rep = build_report(runner, res.experiment_id)
    assert rep["scores"]["pre_blind"] == res.pre_score
    assert rep["scores"]["open_book"] == res.memory_open_score
    assert rep["scores"]["memory_blind"] == res.memory_blind_score
    assert rep["scores"]["transfer_blind"] == res.transfer_blind_score
    assert rep["scores"]["open_book_gain"] == res.open_book_gain
    assert rep["scores"]["memory_gain"] == res.memory_gain
    assert rep["scores"]["transfer_gain"] == res.transfer_gain
    # 老板点名的六项指标一个都不能少
    for key in (
        "pre", "open_book", "memory_blind", "transfer_blind",
        "practice_usable_rate", "note_quality_avg",
    ):
        assert key in rep["metrics"], f"报告缺少指标 {key}"
    assert len(rep["pre_items"]) == len(qb.PRE_SET)
    assert len(rep["memory_open_items"]) == len(qb.POST_SET)
    assert len(rep["memory_blind_items"]) == len(qb.POST_SET)
    assert len(rep["transfer_items"]) == len(qb.TRANSFER_ITEMS)
    assert rep["practice"]["total"] == 4
    assert rep["practice_judgments"], "实践产出必须经独立评审"
    assert rep["note_quality"], "笔记质量必须有独立评估明细"
    assert rep["interpretation"]["headline"]
    assert rep["caveats"]
    assert rep["learning_notes"], "报告应列出学习笔记"


def test_report_never_calls_it_training(runner):
    """报告必须明确声明这是能力迁移实验，而非训练实验。"""
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    rep = build_report(runner, res.experiment_id)
    # 必须明确声明能力迁移实验
    assert any("能力迁移实验" in c for c in rep["caveats"])
    # 任何提到「训练」的地方都必须是否定语境（不能宣称训练成功）
    import re

    for text in [rep["interpretation"]["headline"], *rep["caveats"],
                 rep["interpretation"]["reading"]]:
        for m in re.finditer(r"训练", text):
            window = text[max(0, m.start() - 30): m.start() + 30]
            assert any(
                neg in window
                for neg in ("不是", "没有", "不能", "不等于", "未", "无")
            ), f"发现无否定语境的训练表述：{window!r}"


def test_report_does_not_fabricate_gains(runner):
    """分数没提升时，报告不能出现「有提升」这类措辞。"""
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    rep = build_report(runner, res.experiment_id)
    it = rep["interpretation"]
    # 结论必须与实际 gain 一致
    if res.open_book_gain <= 5:
        assert it["used_external_memory"] is False
    if res.transfer_gain <= 5:
        assert it["formed_concept"] is False
    text = format_text(rep)
    assert "【三、这个实验证明了什么】" in text
    assert "【八、局限】" in text


def test_report_text_renders(runner):
    res = runner.run(TOPIC, DEFAULT_MATERIAL)
    text = format_text(build_report(runner, res.experiment_id))
    assert "Agent-01 V0.4 学习效果实验报告" in text
    # 六项指标必须都印出来
    for label in (
        "PRE", "OPEN_BOOK", "MEMORY_BLIND", "TRANSFER_BLIND",
        "PRACTICE", "NOTE_QUALITY",
    ):
        assert label in text, f"报告缺少指标 {label}"
    for gain in ("OPEN_BOOK_GAIN", "MEMORY_GAIN", "TRANSFER_GAIN"):
        assert gain in text, f"报告缺少 {gain}"


def test_report_rejects_unknown_experiment(runner):
    with pytest.raises(ValueError, match="不存在"):
        build_report(runner, 9999)
