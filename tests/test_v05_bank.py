"""tests/test_v05_bank.py —— V0.5 固定题库与客观实践任务的硬约束测试。

这些测试的作用是**在跑实验之前**就把题库/任务集的问题拦住。
V0.4.1 的教训：一道歧义题让报告上出现「该知识点迁移 -32.9」的假象，
真相是题目问错了；而当时没有任何机器检查能发现它。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pytest

from brain import benchmark_v2 as b2
from brain import novel_material as nm
from memory.db import checksum
from practice import objective as ob

ALL_KPS = (
    "kp_lumen_hash",
    "kp_lumen_bucket",
    "kp_lumen_compact",
    "kp_lumen_consist",
)


# ==========================================================================
# 一、题库规模与固定性
# ==========================================================================


def test_bank_total_is_24():
    assert len(b2.ALL_ITEMS) == 24


def test_four_knowledge_points():
    assert sorted(b2.kp_coverage(b2.ALL_ITEMS)) == sorted(ALL_KPS)
    assert len(b2.kp_coverage(b2.ALL_ITEMS)) == 4


def test_each_kp_has_six_questions():
    counts: dict[str, int] = {}
    for it in b2.ALL_ITEMS:
        counts[it.kp] = counts.get(it.kp, 0) + 1
    assert counts == {kp: 6 for kp in ALL_KPS}


def test_each_exam_has_eight_questions():
    for name, items in b2.EXAM_SETS.items():
        assert len(items) == 8, f"{name} 卷不是 8 题"


def test_each_kp_has_two_questions_per_exam():
    for name, items in b2.EXAM_SETS.items():
        counts: dict[str, int] = {}
        for it in items:
            counts[it.kp] = counts.get(it.kp, 0) + 1
        assert counts == {kp: 2 for kp in ALL_KPS}, f"{name} 卷逐 kp 配平失败"


def test_question_types_are_exactly_three_per_kp():
    types = ("basic", "application", "transfer")
    assert set(b2.QUESTION_TYPES) == set(types)
    for kp in ALL_KPS:
        got = {it.question_type for it in b2.ALL_ITEMS if it.kp == kp}
        assert got == set(types), f"{kp} 的题型不全"


def test_item_ids_are_unique():
    ids = [it.item_id for it in b2.ALL_ITEMS]
    assert len(set(ids)) == 24


def test_item_ids_are_stable_across_imports():
    """题目 id 是锚点，改 id 等于换题库。"""
    assert b2.PRE_ITEMS[0].item_id == "v2_pre_hash_01"
    assert b2.MEMORY_ITEMS[0].item_id.startswith("v2_mem_")
    assert b2.TRANSFER_ITEMS[0].item_id.startswith("v2_tr_")


# ==========================================================================
# 二、premise 显式声明与 Transfer 门禁
# ==========================================================================


def test_no_item_declares_missing_premise():
    bad = [it.item_id for it in b2.ALL_ITEMS if it.premise_in_question == "no"]
    assert not bad, f"这些题声明题面无判断依据，等于不可解：{bad}"


def test_premise_flag_values_are_legal():
    for it in b2.ALL_ITEMS:
        assert it.premise_in_question in ("yes", "asserted"), it.item_id


def test_all_transfer_items_assert_premise():
    bad = [
        it.item_id for it in b2.TRANSFER_ITEMS
        if it.premise_in_question != "asserted"
    ]
    assert not bad, f"TRANSFER 题必须 asserted：{bad}"


def test_all_transfer_items_have_author_note():
    missing = [
        it.item_id for it in b2.TRANSFER_ITEMS
        if it.item_id not in b2.TRANSFER_NOTES
    ]
    assert not missing, f"TRANSFER 题缺少出题说明：{missing}"


def test_transfer_answers_are_long_enough_to_judge():
    short = [it.item_id for it in b2.TRANSFER_ITEMS if len(it.answer) < 120]
    assert not short, f"TRANSFER 参考答案过短，判分会失真：{short}"


def test_bank_diagnostics_clean():
    d = b2.bank_diagnostics()
    assert d["problems"] == []
    assert d["total_items"] == 24


def test_assert_bank_intact_passes():
    b2.assert_bank_intact()  # 不抛异常即通过


# ==========================================================================
# 三、checksum 与指纹
# ==========================================================================


def test_bank_fingerprint_is_sha256_hex():
    assert re.fullmatch(r"[0-9a-f]{64}", b2.BANK_FINGERPRINT)


def test_bank_fingerprint_covers_kp_and_premise():
    """指纹必须覆盖 premise 字段——只覆盖题干时改前提不会触发作废。"""
    payload = json.dumps(
        [
            {
                "item_id": it.item_id,
                "kp": it.kp,
                "question": it.question,
                "answer": it.answer,
                "question_type": it.question_type,
                "premise_in_question": it.premise_in_question,
            }
            for it in b2.ALL_ITEMS
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == b2.BANK_FINGERPRINT


def test_per_exam_fingerprints_differ():
    fps = {b2.PRE_FINGERPRINT, b2.MEMORY_FINGERPRINT, b2.TRANSFER_FINGERPRINT}
    assert len(fps) == 3, "三卷指纹不能相同，否则说明卷内容没区分开"


def test_material_checksum_is_sha256_hex():
    cks = checksum(nm.NOVEL_MATERIAL)
    assert re.fullmatch(r"[0-9a-f]{64}", cks)


def test_material_checksum_changes_with_content():
    assert checksum(nm.NOVEL_MATERIAL) != checksum(nm.NOVEL_MATERIAL + "\n改动")


def test_material_contains_all_five_sections():
    for kw in ("LumenHash", "桶", "分裂", "压缩", "Z-Tag"):
        assert kw in nm.NOVEL_MATERIAL, f"材料缺少「{kw}」相关规则"


def test_material_mentions_no_real_system():
    """novel 材料的全部规则必须在现实世界里不存在。

    只要出现真实存储引擎的名字，「切断先验知识」这个前提就作废了——
    模型可能本来就知道它的真实行为。
    """
    for real in ("Redis", "Memcached", "LevelDB", "RocksDB", "Cassandra",
                 "Riak", "Dynamo", "RiakKV", "Ketama", "Jump Consistent"):
        assert real not in nm.NOVEL_MATERIAL, f"材料出现真实系统名 {real}"


def test_tasks_fingerprint_is_sha256_hex():
    assert re.fullmatch(r"[0-9a-f]{64}", ob.tasks_meta()["fingerprint"])


# ==========================================================================
# 四、客观实践任务
# ==========================================================================


def test_at_least_six_objective_tasks():
    assert len(ob.OBJECTIVE_TASKS) >= 6


def test_exactly_eight_objective_tasks():
    assert len(ob.OBJECTIVE_TASKS) == 8


def test_each_kp_has_two_tasks():
    for kp in ALL_KPS:
        assert len(ob.tasks_for_kp(kp)) == 2, kp


def test_all_four_checker_kinds_covered():
    kinds = {t.kind for t in ob.OBJECTIVE_TASKS}
    assert kinds == set(ob.KINDS), f"缺少判分方式：{set(ob.KINDS) - kinds}"


def test_task_ids_unique():
    ids = [t.task_id for t in ob.OBJECTIVE_TASKS]
    assert len(set(ids)) == len(ids)


def test_task_diagnostics_clean():
    d = ob.task_diagnostics()
    assert d["problems"] == []
    assert d["warnings"] == []
    assert d["total"] == 8


def test_assert_tasks_intact_passes():
    ob.assert_tasks_intact()


def test_submit_hint_is_valid_json():
    """提交格式示例本身必须是合法 JSON，否则在教模型输出非法格式。"""
    for t in ob.OBJECTIVE_TASKS:
        json.loads(t.submit_hint), t.task_id


def test_submit_hint_keys_match_submit_fields():
    for t in ob.OBJECTIVE_TASKS:
        hinted = set(json.loads(t.submit_hint).keys())
        assert hinted == set(t.submit_fields), (
            f"{t.task_id} 的示例字段 {hinted} 与 submit_fields {set(t.submit_fields)} 不符"
        )


def test_prompts_do_not_leak_expected_values():
    """题面/示例里出现期望**值**就是泄题——门禁必须能拦住。

    只探答案值不探输入值：`expected["key"] == "abc"` 是**题目给的输入**，
    必须出现在题面里；`steps` / `bucket` 这类字段名同理。
    门禁要抓的是「模型本该算出来的东西被写在了题面上」。
    """
    for t in ob.OBJECTIVE_TASKS:
        haystack = t.prompt + t.submit_hint
        for value in _leaf_values(_answer_part(t.expected)):
            if value in ("", 0, 1, 2, -1, True, False):
                continue  # 无信息量的值，出现不构成泄题
            if len(value) < 4:
                continue  # 单字符/短片段在任何长文本里都会偶合，探了是噪声
            assert value not in haystack, f"{t.task_id} 题面泄漏期望值 {value!r}"


#: `expected` 里属于「题目输入」而非「待求答案」的键
_INPUT_KEYS = ("key",)


def _answer_part(expected: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in expected.items() if k not in _INPUT_KEYS}


def _leaf_values(expected: object) -> list[str]:
    """展开 expected 的所有叶子值，渲染成题面里可能出现的字符串。"""
    out: list[str] = []
    if isinstance(expected, dict):
        for v in expected.values():
            out.extend(_leaf_values(v))
    elif isinstance(expected, (list, tuple)):
        for v in expected:
            out.extend(_leaf_values(v))
    elif isinstance(expected, bool):
        out.append(str(expected).lower())
    elif isinstance(expected, (int, float)):
        out.append(str(expected))
    elif isinstance(expected, str):
        out.append(expected)
    return out


def test_prompts_do_not_leak_expected_step_sequence():
    """逐步序列与SQL 期望行必须整块不出现在题面。"""
    t = ob.get_task("obj_hash_01")
    assert str(t.expected["steps"]) not in t.prompt
    assert "1014" not in t.prompt and "1858" not in t.prompt
    t2 = ob.get_task("obj_bucket_02")
    for b in ob.EXPECTED_OVERLOADED_BUCKETS:
        # 桶号 3/4/7 若作为「答案」出现在题面即为泄题；
        # 允许出现在 0~4 的容量说明里，所以只查多位数与组合形态
        assert f"[{b}]" not in t2.prompt
    t3 = ob.get_task("obj_consist_02")
    assert "ztag != (lumen_hash" not in t3.prompt, "题面直接给出了答案 SQL"


def test_prompts_contain_no_expected_blob():
    """期望值对象不得整块出现在题面中。"""
    for t in ob.OBJECTIVE_TASKS:
        assert json.dumps(t.expected, ensure_ascii=False) not in t.prompt


# ==========================================================================
# 五、参考实现与期望值一致（防止手算错）
# ==========================================================================


def test_reference_lumen_hash_values():
    assert ob.ref_lumen_hash("") == 7
    assert ob.ref_lumen_hash("a") == 1014
    assert ob.ref_lumen_hash("ab") == 1858
    assert ob.ref_lumen_hash("abc") == 46886
    assert ob.ref_lumen_hash("user") == 14991


def test_reference_bucket_is_hash_mod_8():
    for k in ("", "a", "ab", "abc", "user", "zzz"):
        assert ob.ref_bucket(k) == ob.ref_lumen_hash(k) % 8, k


def test_reference_split_is_identity():
    """一级分裂 (b+8) mod 8 = b，是恒等变换——期望值必须反映这一点。"""
    for h in (0, 8, 16, 24, 32, 999):
        assert ob.ref_split_target(h, 0) == 0


def test_reference_compact_requires_and():
    """压缩条件是「且」：只满足一个不触发。"""
    assert ob.ref_should_compact(2, 1) is True
    assert ob.ref_should_compact(2, 5) is False
    assert ob.ref_should_compact(1, 1) is False
    assert ob.ref_should_compact(9, 3) is False


def test_expected_hash_task_matches_reference():
    t = ob.get_task("obj_hash_01")
    assert t.expected["lumen_hash"] == ob.ref_lumen_hash("abc")
    assert t.expected["bucket"] == ob.ref_bucket("abc")


def test_expected_compact_task_matches_reference():
    t = ob.get_task("obj_compact_01")
    # 桶 0(1,2) 与桶 3(0,4) 满足 tombstones>=2 且 live<=1
    assert t.expected["compress"] == [0, 3]
    assert t.expected["epoch_after"] == 1 + 2


def test_fixture_bucket_equals_hash_mod_8():
    """fixture 自己必须遵守材料规则，否则 SQL 任务在考一个自相矛盾的库。"""
    for key, lh, bucket, _ztag, _epoch in ob.FIXTURE_KEYS:
        assert bucket == lh % 8, f"{key}: bucket {bucket} != {lh} mod 8"


def test_fixture_rows_consistent_with_expected_sql():
    conn = ob._build_fixture_db()
    try:
        over = [
            r[0] for r in conn.execute(
                "SELECT bucket FROM zephyra_buckets WHERE live_count > 4"
                " ORDER BY bucket"
            )
        ]
        assert over == ob.EXPECTED_OVERLOADED_BUCKETS
        epoch = conn.execute(
            "SELECT value FROM zephyra_meta WHERE name='epoch'"
        ).fetchone()[0]
        assert epoch == 2
    finally:
        conn.close()


# ==========================================================================
# 六、判分器：正例
# ==========================================================================


def _submit(task_id: str, payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_correct_hash_answer_scores_full():
    t = ob.get_task("obj_hash_01")
    r = ob.evaluate_objective(t, _submit(t.task_id, t.expected))
    assert r.passed is True
    assert r.objective_score == 100.0
    assert r.parse_error == ""


def test_correct_split_answer_scores_full():
    t = ob.get_task("obj_bucket_01")
    r = ob.evaluate_objective(t, _submit(t.task_id, t.expected))
    assert r.passed is True and r.objective_score == 100.0


def test_correct_compact_answer_scores_full():
    t = ob.get_task("obj_compact_01")
    r = ob.evaluate_objective(t, _submit(t.task_id, t.expected))
    assert r.passed is True and r.objective_score == 100.0


def test_correct_consistency_answer_scores_full():
    t = ob.get_task("obj_consist_01")
    r = ob.evaluate_objective(t, _submit(t.task_id, t.expected))
    assert r.passed is True and r.objective_score == 100.0


def test_correct_overload_sql_scores_full():
    t = ob.get_task("obj_bucket_02")
    sql = ("SELECT bucket FROM zephyra_buckets WHERE live_count > 4"
           " ORDER BY bucket")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"sql": sql}))
    assert r.passed is True and r.objective_score == 100.0


def test_correct_hash_unit_test_scores_full():
    t = ob.get_task("obj_hash_02")
    code = (
        "def lumen_hash(key):\n"
        "    h = 7\n"
        "    for c in key:\n"
        "        h = (h * 131 + ord(c)) % 65537\n"
        "    return h\n"
    )
    r = ob.evaluate_objective(t, _submit(t.task_id, {"code": code}))
    assert r.passed is True, r.detail
    assert r.objective_score == 100.0


def test_correct_compact_unit_test_scores_full():
    t = ob.get_task("obj_compact_02")
    code = "def should_compact(tombstones, live):\n    return tombstones >= 2 and live <= 1\n"
    r = ob.evaluate_objective(t, _submit(t.task_id, {"code": code}))
    assert r.passed is True, r.detail


def test_correct_unreadable_sql_scores_full():
    t = ob.get_task("obj_consist_02")
    sql = (
        "SELECT key FROM zephyra_keys "
        "WHERE ztag != (lumen_hash + (SELECT value FROM zephyra_meta"
        " WHERE name='epoch')) % 65537 ORDER BY key"
    )
    r = ob.evaluate_objective(t, _submit(t.task_id, {"sql": sql}))
    assert r.passed is True, r.detail


# ==========================================================================
# 七、判分器：反例（错误答案必须被扣分）
# ==========================================================================


def test_wrong_hash_value_not_full_score():
    t = ob.get_task("obj_hash_01")
    bad = dict(t.expected)
    bad["lumen_hash"] = 46887
    r = ob.evaluate_objective(t, _submit(t.task_id, bad))
    assert r.objective_score < 100.0
    assert r.passed is False


def test_hash_with_wrong_initial_value_penalised():
    """把初值当 0 是最典型的错误——steps 会暴露。"""
    t = ob.get_task("obj_hash_01")
    bad = dict(t.expected)
    bad["steps"] = [0, 97, 13298, 46886]
    r = ob.evaluate_objective(t, _submit(t.task_id, bad))
    assert r.objective_score < 100.0


def test_wrong_bucket_penalised():
    t = ob.get_task("obj_hash_01")
    bad = dict(t.expected)
    bad["bucket"] = 2          # 46886 mod 8 实际是 6
    r = ob.evaluate_objective(t, _submit(t.task_id, bad))
    assert r.objective_score < 100.0


def test_thinking_split_moves_keys_penalised():
    """误以为分裂有效（+8 不是恒等）是本题的核心陷阱。"""
    t = ob.get_task("obj_bucket_01")
    bad = {
        "assignments": [
            {"key": f"k{lh}", "from_bucket": 0, "to_bucket": (0 + 8) % 8}
            for lh in (0, 8, 16, 24, 32)
        ],
        "migrated_count": 5,
        "still_over_limit": False,
    }
    r = ob.evaluate_objective(t, _submit(t.task_id, bad))
    assert r.objective_score < 100.0
    assert r.passed is False


def test_using_or_instead_of_and_penalised():
    """把「且」写成「或」会把桶 1、2、4 也算成触发压缩。"""
    code = "def should_compact(tombstones, live):\n    return tombstones >= 2 or live <= 1\n"
    t = ob.get_task("obj_compact_02")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"code": code}))
    assert r.objective_score < 100.0


def test_forgetting_epoch_offset_penalised():
    t = ob.get_task("obj_consist_01")
    bad = {
        "readable": ["x1", "x2", "x3", "x4"],
        "need_retag": [],
        "new_tags": {},
    }
    r = ob.evaluate_objective(t, _submit(t.task_id, bad))
    assert r.objective_score < 100.0


def test_wrong_compact_bucket_set_penalised():
    t = ob.get_task("obj_compact_01")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"compress": [0, 1, 2, 3, 4],
                                                    "epoch_after": 6}))
    assert r.objective_score < 100.0


def test_empty_output_scores_zero():
    t = ob.get_task("obj_hash_01")
    r = ob.evaluate_objective(t, "")
    assert r.objective_score == 0.0
    assert r.parse_error != ""


def test_non_json_output_scores_zero():
    t = ob.get_task("obj_hash_01")
    r = ob.evaluate_objective(t, "我认为答案是 46886")
    assert r.objective_score == 0.0
    assert r.parse_error != ""


def test_missing_field_loses_points():
    """缺字段按字段给分（部分分），但绝不能满分。"""
    t = ob.get_task("obj_hash_01")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"lumen_hash": 46886}))
    assert r.objective_score < 100.0
    assert r.passed is False


def test_fenced_json_is_accepted():
    """模型常把 JSON 包在 ```json 里，判分器必须能接住。"""
    t = ob.get_task("obj_hash_01")
    raw = "```json\n" + _submit(t.task_id, t.expected) + "\n```"
    r = ob.evaluate_objective(t, raw)
    assert r.objective_score == 100.0


def test_sql_wrong_result_not_full_score():
    t = ob.get_task("obj_bucket_02")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"sql": "SELECT bucket FROM zephyra_buckets"}))
    assert r.objective_score < 100.0


def test_sql_wrong_order_not_full_score():
    """结果集对但顺序错：必须扣分，否则「排序」这一要求形同虚设。"""
    t = ob.get_task("obj_bucket_02")
    r = ob.evaluate_objective(
        t, _submit(t.task_id, {"sql": "SELECT bucket FROM zephyra_buckets"
                                  " WHERE live_count > 4 ORDER BY bucket DESC"})
    )
    assert r.objective_score < 100.0


# ==========================================================================
# 八、代码执行沙箱
# ==========================================================================


@pytest.mark.parametrize(
    "code",
    [
        "import os\ndef lumen_hash(k):\n    return os.urandom(1)[0]",
        "import subprocess\ndef lumen_hash(k):\n    return 0",
        "from os import path\ndef lumen_hash(k):\n    return 0",
        "def lumen_hash(k):\n    return __import__('os').getcwd()",
        "def lumen_hash(k):\n    return eval('1+1')",
        "def lumen_hash(k):\n    exec('x=1')\n    return 0",
    ],
)
def test_dangerous_code_rejected(code: str):
    """`python -I` 挡不住 `import os; os.system(...)`，必须静态拦。"""
    assert ob.scan_code_safety(code), f"未拦住危险代码：{code!r}"
    t = ob.get_task("obj_hash_02")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"code": code}))
    assert r.objective_score == 0.0


def test_safe_code_passes_safety_scan():
    code = "def lumen_hash(key):\n    h = 7\n    return h\n"
    assert ob.scan_code_safety(code) == []


def test_infinite_loop_times_out():
    t = ob.get_task("obj_hash_02")
    code = "def lumen_hash(key):\n    while True:\n        pass\n"
    r = ob.evaluate_objective(t, _submit(t.task_id, {"code": code}))
    assert r.objective_score == 0.0


# ==========================================================================
# 九、SQL 只读护栏
# ==========================================================================


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE zephyra_keys",
        "DELETE FROM zephyra_keys",
        "UPDATE zephyra_keys SET ztag = 0",
        "INSERT INTO zephyra_keys VALUES ('x', 1, 1, 1, 1)",
        "ALTER TABLE zephyra_keys ADD COLUMN c INT",
        "CREATE TABLE t (a INT)",
        "ATTACH DATABASE 'x.db' AS x",
        "SELECT bucket FROM zephyra_buckets; DROP TABLE zephyra_keys",
        "PRAGMA writable_schema = 1",
    ],
)
def test_write_sql_rejected(sql: str):
    t = ob.get_task("obj_bucket_02")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"sql": sql}))
    assert r.objective_score == 0.0, f"写操作未被拒绝：{sql}"


def test_sql_on_nonexistent_table_fails_cleanly():
    t = ob.get_task("obj_bucket_02")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"sql": "SELECT * FROM not_a_table"}))
    assert r.objective_score == 0.0


# ==========================================================================
# 十、写死答案不能通过
# ==========================================================================


def test_hardcoded_hash_fails_unseen_case():
    """写死 `if key == 'abc': return 46886` 在空串用例上必须露馅。"""
    code = (
        "def lumen_hash(key):\n"
        "    if key == 'abc':\n"
        "        return 46886\n"
        "    return 0\n"
    )
    t = ob.get_task("obj_hash_02")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"code": code}))
    assert r.objective_score < 100.0


def test_hardcoded_compact_fails_unseen_case():
    code = (
        "def should_compact(tombstones, live):\n"
        "    if (tombstones, live) in ((2, 1), (3, 0)):\n"
        "        return True\n"
        "    return False\n"
    )
    t = ob.get_task("obj_compact_02")
    r = ob.evaluate_objective(t, _submit(t.task_id, {"code": code}))
    assert r.objective_score < 100.0
