"""V0.4 专项测试。

三条主线：
    1. TRANSFER 卷的题目质量（它是不是真的在测「概念」而不是「记忆」）
    2. Note quality / Practice Judge 两个独立评估器
    3. 重复实验与跨次统计（判定门槛、隔离、批次归属）

**离线脚本的定位要说清楚**：本文件的分数全部来自 demo_script()，
是写死的常量。本文件验证的是**管线能不能正确区分四个阶段**，
不是「学习效果如何」。真实效果必须 --provider openai 实跑。
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain import question_bank as qb  # noqa: E402
from brain import replicate as rep  # noqa: E402
from brain import report  # noqa: E402
from brain.experiment import ExperimentRunner  # noqa: E402
from brain.llm import ScriptedProvider  # noqa: E402
from brain.materials import DEFAULT_MATERIAL  # noqa: E402
from brain.report import batch_verdict, build_batch_report  # noqa: E402
from learning_notes.quality import (  # noqa: E402
    DIMENSIONS,
    PASS_LINE,
    PLACEHOLDER_MARKERS,
    WEIGHTS,
    NoteQualityEvaluator,
    NoteQualityStore,
    weighted_total,
)
from main import demo_script  # noqa: E402
from memory.db import Database  # noqa: E402
from memory.gate import AccessPolicy  # noqa: E402
from practice.judge import USABLE_LINE, PracticeJudge, PracticeJudgeStore  # noqa: E402

TOPIC = "SQLite 索引原理"


def _fresh_db(tmp_path, name="v4.db") -> Database:
    database = Database(str(tmp_path / name))
    database.init_schema()
    return database


def _runner(database: Database) -> ExperimentRunner:
    return ExperimentRunner(database, ScriptedProvider(demo_script()))


# ======================================================================
# 1. TRANSFER 卷的题目质量
# ======================================================================
class TestTransferBank:
    """TRANSFER 卷是 V0.4 的核心新增，它必须真的是「新题测旧原理」。"""

    def test_transfer_items_exist_and_are_four(self):
        assert len(qb.TRANSFER_ITEMS) == len(qb.PRE_SET) == len(qb.POST_SET)

    def test_transfer_questions_differ_from_pre_and_post(self):
        """TRANSFER 若是 PRE/POST 的复制，它就退化成 MEMORY 题。"""
        known = {i.question for i in qb.PRE_ITEMS} | {i.question for i in qb.POST_ITEMS}
        for item in qb.TRANSFER_ITEMS:
            assert item.question not in known, f"{item.item_id} 与 PRE/POST 重复"

    def test_transfer_covers_exactly_the_same_kps(self):
        """知识点必须配平，否则跨卷比较会把「换了考点」误读成「迁移失败」。"""
        assert qb.kp_coverage(qb.TRANSFER_SET) == qb.kp_coverage(qb.PRE_SET)

    def test_every_transfer_item_documents_its_design(self):
        """每道 TRANSFER 题都要能回答：迁移自哪条原理？先验直觉错在哪？"""
        for item in qb.TRANSFER_ITEMS:
            note = qb.TRANSFER_NOTES.get(item.item_id)
            assert note, f"{item.item_id} 缺少 TRANSFER_NOTES"
            assert note.get("transfers_from"), f"{item.item_id} 未记录迁移来源"
            assert note.get("prior_intuition"), f"{item.item_id} 未记录先验直觉错在哪"

    def test_transfer_answers_are_substantive(self):
        """答案太短判分会失真 —— 判分器没有可对照的要点。"""
        for item in qb.TRANSFER_ITEMS:
            assert len(item.answer) >= 60, f"{item.item_id} 参考答案过短"

    def test_transfer_answer_does_not_verbatim_copy_material(self):
        """逐字抄材料就退化成「抄写测试」了。"""
        material_segments = [
            seg for seg in DEFAULT_MATERIAL.split("\n") if len(seg.strip()) >= 20
        ]
        for item in qb.TRANSFER_ITEMS:
            for seg in material_segments:
                assert seg.strip() not in item.answer, (
                    f"{item.item_id} 的答案整段抄了材料"
                )

    def test_transfer_has_its_own_fingerprint(self):
        """TRANSFER 必须有独立指纹，否则改题不会被发现。"""
        assert qb.TRANSFER_FINGERPRINT != qb.BANK_FINGERPRINT
        assert len(qb.TRANSFER_FINGERPRINT) == 64

    def test_bank_meta_reports_all_three_sets(self):
        meta = qb.bank_meta()
        assert meta["transfer_count"] == len(qb.TRANSFER_ITEMS)
        assert meta.get("transfer_fingerprint")

    def test_bank_integrity_check_covers_transfer(self):
        qb.assert_bank_intact()  # 不抛异常即通过


# ======================================================================
# 2. Note quality 独立评估
# ======================================================================
class TestNoteQuality:
    def _seeded(self, tmp_path):
        database = _fresh_db(tmp_path)
        runner = _runner(database)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        return database, runner, res

    def test_dimensions_and_weights_cover_four_axes(self):
        assert set(DIMENSIONS) == {
            "accuracy", "completeness", "transferable", "no_hallucination",
        }
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
        # accuracy 权重最高：笔记如果是错的，讲得再透也不能用。
        # transferable 紧随其后 —— 它是「能不能迁移」的关键维度，
        # 但必须排在准确性之后：可靠性优先于有用性。
        assert WEIGHTS["accuracy"] == max(WEIGHTS.values())
        assert WEIGHTS["transferable"] == sorted(WEIGHTS.values())[-2]
        assert WEIGHTS["transferable"] >= 0.25, "可迁移性不能被稀释成配角"

    def test_weighted_total_matches_manual_sum(self):
        scores = {
            "accuracy": 80.0, "completeness": 60.0,
            "transferable": 100.0, "no_hallucination": 40.0,
        }
        manual = sum(scores[d] * WEIGHTS[d] for d in DIMENSIONS)
        assert weighted_total(scores) == pytest.approx(manual, abs=0.01)

    def test_quality_rows_persisted_for_every_note(self, tmp_path):
        database, runner, res = self._seeded(tmp_path)
        rows = NoteQualityStore(database).for_experiment(res.experiment_id)
        notes = runner.notes.for_experiment(res.experiment_id)
        assert rows, "每条笔记都要有质量评估"
        assert len(rows) == len(notes)
        try:
            for r in rows:
                assert 0 <= r["accuracy"] <= 100
                assert 0 <= r["transferable"] <= 100
                assert r["verdict"] in {"pass", "fail"}
        finally:
            database.close()

    def test_placeholder_text_forces_low_score(self, tmp_path):
        """模型挂了写出的占位符会污染开卷上下文，必须硬压分。"""
        database = _fresh_db(tmp_path, "ph.db")
        runner = _runner(database)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        notes = runner.notes.for_experiment(res.experiment_id)
        store = NoteQualityStore(database)
        ev = NoteQualityEvaluator(database, store, ScriptedProvider(demo_script()))
        note = dict(notes[0])
        note["example"] = f"（{PLACEHOLDER_MARKERS[0]}）"
        out = ev.evaluate(res.experiment_id, note, DEFAULT_MATERIAL, "标签")
        assert out["completeness"] <= 20.0
        assert out["transferable"] <= 20.0
        assert any("占位" in p for p in out["problems"])
        database.close()

    def test_rule_fallback_never_guesses_accuracy(self, tmp_path):
        """LLM 失效时的兜底不得虚构准确性分数，且必须留痕。"""
        database = _fresh_db(tmp_path, "fb.db")
        runner = _runner(database)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        notes = runner.notes.for_experiment(res.experiment_id)
        broken = ScriptedProvider({"NOTE_QUALITY": "这不是 JSON"})
        store = NoteQualityStore(database)
        ev = NoteQualityEvaluator(database, store, broken)
        out = ev.evaluate(res.experiment_id, dict(notes[0]), DEFAULT_MATERIAL, "标签")
        assert out["evaluator"] == "rule_fallback", "必须标注走了兜底"
        assert out["accuracy"] == 50.0, "不猜准确性时必须给中性值并留痕"
        assert "兜底" in out["feedback"]
        # 落库的 evaluator 也必须是 rule_fallback，不能一律写 llm_judge
        rows = store.for_experiment(res.experiment_id)
        assert rows[-1]["evaluator"] == "rule_fallback"
        database.close()

    def test_quality_is_evaluated_before_any_post_exam(self, tmp_path):
        """笔记质量必须在 PRE 之后、其余三场考试之前评完。

        考试时不能再改笔记（store 层禁止 blind 写），所以质量评估
        必须落在 PRE 结束与 MEMORY/TRANSFER 之间 —— 否则开卷考的
        是一份「还没被审过」的笔记。
        """
        database = _fresh_db(tmp_path, "ord.db")
        provider = ScriptedProvider(demo_script())
        runner = ExperimentRunner(database, provider)
        runner.run(TOPIC, DEFAULT_MATERIAL)
        order = []
        for call in provider.calls:
            joined = "\n".join(m["content"] for m in call)
            if "[[TASK:NOTE_QUALITY]]" in joined:
                order.append("quality")
            elif "[[TASK:ANSWER]]" in joined:
                order.append("answer")
        assert "quality" in order
        quality_at = order.index("quality")
        # 第一次 quality 之前只允许有 PRE 的 4 次答题
        assert order[:quality_at] == ["answer"] * len(qb.PRE_SET), (
            f"质量评估前应恰好是 PRE 的 {len(qb.PRE_SET)} 次答题，"
            f"实际 {order[:quality_at]}"
        )
        database.close()


# ======================================================================
# 3. Practice Judge
# ======================================================================
class TestPracticeJudge:
    def _seeded(self, tmp_path, name="pj.db"):
        database = _fresh_db(tmp_path, name)
        runner = _runner(database)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        return database, runner, res

    def test_offline_practice_is_judged_usable(self, tmp_path):
        """演示脚本里的实践产出含具体命令，应被判为可用。"""
        database, runner, res = self._seeded(tmp_path)
        assert res.practice_usable_rate == 1.0
        assert res.practice_avg_score >= USABLE_LINE
        database.close()

    def test_vague_answer_is_not_usable(self, tmp_path):
        """只停留在「我要做什么」的产出，即使 LLM 判了可用，规则也要能拦住。

        这里刻意让 LLM 判 usable=True —— 兜底规则的价值就在于
        「模型说得好听，但产出里没有任何可操作的东西」这种情况。
        """
        database = _fresh_db(tmp_path, "vague.db")
        store = PracticeJudgeStore(database)
        judge = PracticeJudge(database, store, ScriptedProvider(demo_script()))
        eid = _seed_exp(database)
        out = judge.judge(
            eid, "kp_btree", "标签", "任务：验证索引是否生效",
            "我将会先检查表结构，然后建立索引，接着观察执行计划。",
        )
        # demo_script 的 PRACTICE_JUDGE 是写死的 usable=True，
        # 所以这里只能验证「规则兜底路径」能识别空话
        fallback = PracticeJudge._rule_fallback(
            "我将会先检查表结构，然后建立索引，接着观察执行计划。"
        )
        score, usable, verdict, _ = fallback
        assert usable is False
        assert score < USABLE_LINE
        assert verdict == "fail"
        assert out["evaluator"] == "llm_judge"
        database.close()

    def test_concrete_answer_is_usable(self, tmp_path):
        database = _fresh_db(tmp_path, "conc.db")
        store = PracticeJudgeStore(database)
        judge = PracticeJudge(database, store, ScriptedProvider(demo_script()))
        eid = _seed_exp(database)
        out = judge.judge(
            eid, "kp_btree", "标签", "任务：验证索引是否生效",
            "执行 sqlite3 test.db \"EXPLAIN QUERY PLAN SELECT * FROM t WHERE k='x';\" "
            "判据：输出出现 SEARCH t USING INDEX 算命中，出现 SCAN t 算未命中。",
        )
        assert out["usable"] is True
        database.close()

    def test_judge_never_sees_material_or_notes(self, tmp_path):
        """评审只看产出本身 —— 「好不好」和「有没有用上材料」是两件事。"""
        database = _fresh_db(tmp_path, "blind.db")
        store = PracticeJudgeStore(database)
        spy = ScriptedProvider(demo_script())
        judge = PracticeJudge(database, store, spy)
        eid = _seed_exp(database)
        judge.judge(eid, "kp_btree", "标签", "任务提示", "候选产出：EXPLAIN QUERY PLAN ...")
        joined = "\n".join(
            m["content"] for call in spy.calls for m in call
        )
        assert "PRACTICE_JUDGE" in joined
        assert DEFAULT_MATERIAL[:60] not in joined, "评审不应看到原始材料"
        assert "学习笔记" not in joined, "评审不应看到学习笔记"

    def test_outcome_follows_judge_not_self_report(self, tmp_path):
        """模型自报 success 不作数 —— 以独立评审为准。"""
        script = demo_script()
        script["PRACTICE"] = json.dumps(
            {"outcome": "success", "result": "我将会先做 A，再做 B。", "kp": "kp_btree"},
            ensure_ascii=False,
        )
        script.pop("PRACTICE_JUDGE", None)  # 让评审走规则兜底
        database = _fresh_db(tmp_path, "sr.db")
        runner = ExperimentRunner(database, ScriptedProvider(script))
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        recs = runner.store.practice(res.experiment_id)
        assert recs[0]["outcome"] == "failure", "自报 success 不应直接采信"
        assert res.practice_usable_rate == 0.0
        database.close()

    def test_json_failure_retries_as_plain_text(self, tmp_path):
        """JSON 解析失败时必须重取原文，不能拿占位符当产出。

        实测踩过的坑：解析失败时代码把「（模型未返回结构化结果）」
        当成模型的实践产出丢给评审器，评审器判它「未返回任何产出」→ 0 分。
        模型明明答了，整段内容却被占位符顶掉，
        「实践可用率」从此测的是「听不听 JSON 格式指令」。

        这里让 PRACTICE 返非 JSON（解析必失败），ask_text 走 fallback 分支。
        """
        script = demo_script()
        script["PRACTICE"] = "这不是 JSON，但我答了："
        script.pop("PRACTICE_JUDGE", None)
        database = _fresh_db(tmp_path, "retry.db")

        class TextFallback(ScriptedProvider):
            def ask_text(self, system, user, **kw):
                return (
                    "sqlite3 test.db \"EXPLAIN QUERY PLAN SELECT 1;\" "
                    "判据：出现 SEARCH 算命中，出现 SCAN 算未命中。"
                )

        runner = ExperimentRunner(database, TextFallback(script))
        runner.run(TOPIC, DEFAULT_MATERIAL)
        rows = PracticeJudgeStore(database).for_experiment(
            runner.store.latest()["id"]
        )
        assert rows
        for r in rows:
            assert "模型未返回结构化结果" not in r["candidate_answer"], (
                "占位符不得冒充模型产出"
            )
            assert r["candidate_answer"].strip(), "重试后不应为空"
        database.close()

    def test_practice_record_keeps_actual_output(self, tmp_path):
        """practice_records.result 必须存实际产出，便于事后复核判分。"""
        database = _fresh_db(tmp_path, "keep.db")
        runner = _runner(database)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        for rec in runner.store.practice(res.experiment_id):
            assert rec["result"].strip(), "result 不得为空"
            assert "judged_usable=" not in rec["result"], (
                "result 应存实际产出，不是评审摘要"
            )
        database.close()

    def test_rule_fallback_caps_at_90(self, tmp_path):
        """规则兜底永远不给满分 —— 它没检查正确性。"""
        database = _fresh_db(tmp_path, "cap.db")
        store = PracticeJudgeStore(database)
        broken = ScriptedProvider({"PRACTICE_JUDGE": "非 JSON"})
        judge = PracticeJudge(database, store, broken)
        eid = _seed_exp(database)
        out = judge.judge(
            eid, "kp_btree", "标签", "任务",
            "sqlite3 test.db \"EXPLAIN QUERY PLAN SELECT 1;\" 判据：出现 SEARCH 算命中。",
        )
        assert out["score"] <= 90.0
        assert out["evaluator"] == "rule_fallback", "兜底必须留痕"
        database.close()


def _seed_exp(database: Database) -> int:
    """造一个最小可用 experiment 供评审类单测挂载。

    走 store 的公开 API 而不是裸 SQL —— 裸 INSERT 会在 schema 加了
    NOT NULL 列时静默炸掉，而这里只是想拿到一个合法 experiment_id。
    """
    from brain.experiment import BankStore, ExperimentStore, MaterialStore

    material_id = MaterialStore(database).save(
        TOPIC, TOPIC, DEFAULT_MATERIAL, version="v1"
    )["id"]
    bank_id = BankStore(database).register()
    return ExperimentStore(database).create(TOPIC, material_id, bank_id)


# ======================================================================
# 4. 重复实验与跨次统计
# ======================================================================
class TestStatistics:
    def test_describe_reports_n_mean_std_min_max(self):
        d = rep.describe([10, 20, 30])
        assert d["n"] == 3
        assert d["mean"] == 20.0
        assert d["min"] == 10.0 and d["max"] == 30.0
        assert d["std"] > 0

    def test_stdev_uses_sample_formula(self):
        assert rep.stdev([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.13809, abs=1e-4)

    def test_stdev_of_single_value_is_zero(self):
        assert rep.stdev([7]) == 0.0

    def test_mean_of_empty_is_zero(self):
        assert rep.mean([]) == 0.0
        assert rep.describe([])["n"] == 0

    def test_verdict_requires_enough_trials(self):
        """3 次以内不给结论 —— std 估计太不稳。"""
        v = rep.gain_verdict([10, 12, 11])
        assert v["state"] == "insufficient"
        assert "低于" in v["reason"]

    def test_verdict_positive_only_when_mean_exceeds_two_sigma(self):
        v = rep.gain_verdict([10, 10, 10, 10, 10])
        assert v["lower_bound"] == 10.0
        assert v["state"] == "positive"

    def test_verdict_noisy_when_mean_positive_but_within_noise(self):
        """均值 +2 但 σ=5 —— 小于 2σ，只能说「与噪声无法区分」。"""
        v = rep.gain_verdict([7, 2, -3, 0, 4])
        assert v["mean"] > 0
        assert v["lower_bound"] <= 0
        assert v["state"] == "noisy"

    def test_verdict_non_positive(self):
        assert rep.gain_verdict([0, -1, 1, 0, 0])["state"] == "non_positive"

    def test_min_trials_constant_is_five(self):
        assert rep.MIN_TRIALS_FOR_VERDICT == 5
        assert rep.DEFAULT_REPEATS == 5

    def test_aggregate_computes_all_seven_metrics(self):
        trials = [
            {
                "pre_score": 40.0, "memory_open_score": 85.0,
                "memory_blind_score": 50.0, "transfer_blind_score": 60.0,
                "practice_usable_rate": 1.0, "practice_avg_score": 70.0,
                "note_quality_avg": 75.0, "note_quality_pass_rate": 1.0,
            }
            for _ in range(5)
        ]
        out = rep.aggregate_trials(trials)
        assert out["n_trials"] == 5
        for key in rep.METRIC_KEYS:
            assert out["metrics"][key]["n"] == 5
        assert out["verdicts"]["open_book_gain"]["state"] == "positive"
        assert out["verdicts"]["transfer_gain"]["state"] == "positive"
        # 背住了 vs 学会了：MEMORY_GAIN(10) - TRANSFER_GAIN(20) = -10
        assert out["memory_vs_transfer_gap"]["mean"] == -10.0

    def test_aggregate_handles_partial_trials(self):
        """老数据没有 V0.4 的列时不能崩。"""
        trials = [{"pre_score": 40.0, "memory_open_score": 80.0}]
        out = rep.aggregate_trials(trials)
        assert out["n_trials"] == 1
        assert out["metrics"]["transfer_blind_score"]["n"] == 0


class TestReplicateRunner:
    def test_runs_five_trials_and_persists_batch(self, tmp_path):
        database = _fresh_db(tmp_path, "rep.db")
        runner = rep.ReplicateRunner(
            database, lambda: ScriptedProvider(demo_script())
        )
        summary = runner.run_repeated(TOPIC, DEFAULT_MATERIAL, repeats=5)
        assert summary["n_trials"] == 5
        assert len(summary["trials"]) == 5
        stored = runner.batch.get(summary["batch_id"])
        assert stored is not None and stored["status"] == "done"
        assert len(runner.batch.trials(summary["batch_id"])) == 5
        database.close()

    def test_each_trial_gets_fresh_provider(self, tmp_path):
        """每次新建 provider，否则上一轮上下文会渗进下一轮。"""
        made = []

        def factory():
            p = ScriptedProvider(demo_script())
            made.append(p)
            return p

        database = _fresh_db(tmp_path, "fresh.db")
        runner = rep.ReplicateRunner(database, factory)
        runner.run_repeated(TOPIC, DEFAULT_MATERIAL, repeats=5)
        assert len(made) == 5
        assert len({id(p) for p in made}) == 5, "每次 trial 必须是新实例"
        database.close()

    def test_repeat_one_still_records_trial(self, tmp_path):
        database = _fresh_db(tmp_path, "one.db")
        runner = rep.ReplicateRunner(database, lambda: ScriptedProvider(demo_script()))
        summary = runner.run_repeated(TOPIC, DEFAULT_MATERIAL, repeats=1)
        assert summary["n_trials"] == 1
        assert summary["verdicts"]["transfer_gain"]["state"] == "insufficient"
        database.close()

    def test_rejects_zero_repeats(self, tmp_path):
        database = _fresh_db(tmp_path, "zero.db")
        runner = rep.ReplicateRunner(database, lambda: ScriptedProvider(demo_script()))
        with pytest.raises(ValueError, match="至少为 1"):
            runner.run_repeated(TOPIC, DEFAULT_MATERIAL, repeats=0)
        database.close()

    def test_batch_id_is_unique_per_run(self):
        a = rep.make_batch_id(TOPIC, 0)
        b = rep.make_batch_id(TOPIC, 1)
        assert a != b

    def test_progress_callback_fires_per_trial(self, tmp_path):
        seen = []
        database = _fresh_db(tmp_path, "prog.db")
        runner = rep.ReplicateRunner(database, lambda: ScriptedProvider(demo_script()))
        runner.run_repeated(
            TOPIC, DEFAULT_MATERIAL, repeats=3,
            progress=lambda n, res: seen.append(n),
        )
        assert seen == [1, 2, 3]
        database.close()

    def test_batch_report_reads_back_from_db(self, tmp_path):
        database = _fresh_db(tmp_path, "back.db")
        runner = rep.ReplicateRunner(database, lambda: ScriptedProvider(demo_script()))
        summary = runner.run_repeated(TOPIC, DEFAULT_MATERIAL, repeats=5)
        got = build_batch_report(database, summary["batch_id"])
        assert got["n_trials"] == 5
        assert got["verdict"]["headline"]
        database.close()

    def test_batch_report_without_batch_raises(self, tmp_path):
        database = _fresh_db(tmp_path, "none.db")
        with pytest.raises(ValueError, match="重复实验"):
            build_batch_report(database)
        database.close()


class TestBatchVerdict:
    def _summary(self, open_state, mem_state, tr_state):
        def v(state):
            return {"state": state, "mean": 10.0, "std": 1.0,
                    "lower_bound": 8.0, "n": 5, "reason": ""}
        return {
            "n_trials": 5,
            "verdicts": {
                "open_book_gain": v(open_state),
                "memory_gain": v(mem_state),
                "transfer_gain": v(tr_state),
            },
        }

    def test_migration_verdict(self):
        out = batch_verdict(self._summary("positive", "positive", "positive"))
        assert "迁移迹象" in out["headline"]
        assert "不是训练成功" in out["reading"]

    def test_recall_without_transfer_verdict(self):
        out = batch_verdict(self._summary("positive", "positive", "non_positive"))
        assert "背诵" in out["headline"] or "没有可重复" in out["headline"]

    def test_no_memory_use_verdict(self):
        out = batch_verdict(self._summary("non_positive", "non_positive", "non_positive"))
        assert "未观察到稳定" in out["headline"]

    def test_insufficient_is_not_decided(self):
        out = batch_verdict(self._summary("insufficient", "insufficient", "insufficient"))
        assert out["decided"] is False


# ======================================================================
# 5. 四场考试的语义隔离
# ======================================================================
class TestFourStageIsolation:
    def test_offline_pipeline_separates_all_four_stages(self, tmp_path):
        """管线验证：四场必须给出四个不同的数。

        如果 TRANSFER 也拿高分，说明这套实验分不清「记住了」
        和「学会了」—— 那 V0.4 的核心改造就白做了。
        """
        database = _fresh_db(tmp_path, "stages.db")
        res = _runner(database).run(TOPIC, DEFAULT_MATERIAL)
        assert res.memory_open_score > res.pre_score, "开卷应明显高于起点"
        assert res.memory_blind_score == res.pre_score, "盲答两场应同分"
        assert res.transfer_blind_score < res.memory_open_score, (
            "演示脚本刻意让 TRANSFER 答非所问；"
            "若它也接近开卷分，说明 TRANSFER 没起作用"
        )
        database.close()

    def test_transfer_never_receives_context(self, tmp_path):
        database = _fresh_db(tmp_path, "ctx.db")
        provider = ScriptedProvider(demo_script())
        runner = ExperimentRunner(database, provider)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        for row in runner.store.items(res.experiment_id):
            if row["exam_kind"] == "transfer":
                assert not (row["context_used"] or "").strip()
        database.close()

    def test_memory_open_and_blind_share_the_same_items(self, tmp_path):
        """MEMORY 两场必须同题，否则「记忆保持」这个比较不成立。"""
        database = _fresh_db(tmp_path, "same.db")
        runner = _runner(database)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        items = runner.store.items(res.experiment_id)
        opened = {i["item_id"] for i in items
                  if i["exam_kind"] == "memory" and i["mode"] == "open_book_memory"}
        blinded = {i["item_id"] for i in items
                   if i["exam_kind"] == "memory" and i["mode"] == "blind_model"}
        assert opened == blinded and opened
        database.close()

    def test_experiment_records_batch_and_trial(self, tmp_path):
        database = _fresh_db(tmp_path, "meta.db")
        runner = rep.ReplicateRunner(database, lambda: ScriptedProvider(demo_script()))
        summary = runner.run_repeated(TOPIC, DEFAULT_MATERIAL, repeats=2)
        rows = runner.batch.trials(summary["batch_id"])
        assert [r["trial_no"] for r in rows] == [1, 2]
        assert all(r["batch_id"] == summary["batch_id"] for r in rows)
        database.close()


# ======================================================================
# 6. 报告纪律
# ======================================================================
class TestReportDiscipline:
    def test_report_never_claims_training_success(self, tmp_path):
        """任何「训练」表述都必须在否定语境里。"""
        from brain.report import build_report, format_batch_text, format_text

        database = _fresh_db(tmp_path, "disc.db")
        runner = rep.ReplicateRunner(database, lambda: ScriptedProvider(demo_script()))
        summary = runner.run_repeated(TOPIC, DEFAULT_MATERIAL, repeats=5)
        eid = summary["trials"][0]["experiment_id"]
        single_runner = _runner(database)  # build_report 要的是 ExperimentRunner
        texts = [
            format_text(build_report(single_runner, eid)),
            format_batch_text({**summary, "verdict": batch_verdict(summary)}),
        ]
        import re
        for text in texts:
            for m in re.finditer("训练", text):
                window = text[max(0, m.start() - 30): m.start() + 30]
                assert any(
                    neg in window
                    for neg in ("不是", "没有", "不能", "不等于", "未", "无", "绝")
                ), f"发现无否定语境的训练表述：{window!r}"
        database.close()

    def test_report_lists_six_metrics(self, tmp_path):
        from brain.report import build_report

        database = _fresh_db(tmp_path, "six.db")
        runner = _runner(database)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        rep_data = build_report(runner, res.experiment_id)
        m = rep_data["metrics"]
        assert m["pre"] == res.pre_score
        assert m["open_book"] == res.memory_open_score
        assert m["memory_blind"] == res.memory_blind_score
        assert m["transfer_blind"] == res.transfer_blind_score
        assert 0 <= m["practice_usable_rate"] <= 100
        assert m["note_quality_avg"] > 0
        database.close()

    def test_report_shows_transfer_versus_memory_gap(self, tmp_path):
        """「背住了 vs 学会了」的差值必须出现在报告里。"""
        from brain.report import build_report

        database = _fresh_db(tmp_path, "gap.db")
        runner = _runner(database)
        res = runner.run(TOPIC, DEFAULT_MATERIAL)
        rep_data = build_report(runner, res.experiment_id)
        gap = rep_data["interpretation"]["transfer_vs_memory"]
        assert gap == round(res.memory_gain - res.transfer_gain, 2)
        database.close()

    def test_blind_store_forbids_writing_notes_during_exam(self, tmp_path):
        """任何 blind 模式都禁止写笔记 —— 边考边改小抄等于作弊。"""
        from learning_notes.store import LearningNoteStore
        from memory.gate import ExamAccessViolation

        database = _fresh_db(tmp_path, "w.db")
        blind = AccessPolicy(mode="blind_model", stage="exam", blind=True)
        store = LearningNoteStore(database, blind)
        with pytest.raises(ExamAccessViolation):
            store.add(1, "kp_btree", "c", "p", "e", "x", "r")
        database.close()


# ---------------------------------------------------------------------------
# v0.4.1：题库诊断（题目歧义会伪装成模型能力不足）
# ---------------------------------------------------------------------------


class TestBankDiagnostics:
    """起因：真实模型跑 10 次后才发现 tr_covering_01 题面自身有歧义。

    旧题面问「要让它是覆盖索引还缺什么」，但查询只 SELECT month，
    索引 (month) 里就有 month —— 按字面推它已经是覆盖索引。
    考生答「什么都不缺」逻辑上是对的，却被判45 分，报告上看起来像
    「这个知识点没迁移」，真相是「这道题问错了」。

    所以歧义必须机器可检，不能靠跑完实验靠人眼发现。
    """

    def test_current_bank_has_no_ambiguity_problem(self):
        diag = qb.bank_diagnostics()
        assert diag["problems"] == [], f"当前题库不该有歧义问题：{diag['problems']}"

    def test_all_transfer_items_declare_premise(self):
        """每道 TRANSFER 题必须显式声明前提在不在题面里。"""
        for item in qb.TRANSFER_ITEMS:
            assert item.premise_in_question in ("yes", "no", "asserted"), (
                f"{item.item_id} 未声明 premise_in_question"
            )

    def test_premise_no_is_rejected_as_hard_error(self):
        """显式声明「前提不在题面」的题必须被门禁拦住。"""
        broken = qb.ExamItem(
            item_id="tr_broken_01",
            kp="kp_covering",
            question="这个索引还缺什么？",
            answer="x" * 80,
            premise_in_question="no",
        )
        saved = qb.TRANSFER_ITEMS
        try:
            patched = saved + (broken,)
            qb.TRANSFER_ITEMS = patched
            qb.ALL_ITEMS = qb.PRE_ITEMS + qb.POST_ITEMS + patched
            # 补出题说明，好让门禁走到 premise 检查那一关
            qb.TRANSFER_NOTES["tr_broken_01"] = {
                "transfers_from": "x", "prior_intuition": "y",
            }
            with pytest.raises(ValueError, match="premise_in_question='no'"):
                qb.assert_bank_intact()
        finally:
            qb.TRANSFER_NOTES.pop("tr_broken_01", None)
            qb.TRANSFER_ITEMS = saved
            qb.ALL_ITEMS = qb.PRE_ITEMS + qb.POST_ITEMS + saved

    def test_illegal_premise_value_is_rejected(self):
        bad = qb.ExamItem(
            item_id="tr_bad_01",
            kp="kp_covering",
            question="正常题目？",
            answer="x" * 80,
            premise_in_question="maybe",
        )
        saved = qb.TRANSFER_ITEMS
        try:
            patched = saved + (bad,)
            qb.TRANSFER_ITEMS = patched
            qb.ALL_ITEMS = qb.PRE_ITEMS + qb.POST_ITEMS + patched
            qb.TRANSFER_NOTES["tr_bad_01"] = {
                "transfers_from": "x", "prior_intuition": "y",
            }
            with pytest.raises(ValueError, match="取值非法"):
                qb.assert_bank_intact()
        finally:
            qb.TRANSFER_NOTES.pop("tr_bad_01", None)
            qb.TRANSFER_ITEMS = saved
            qb.ALL_ITEMS = qb.PRE_ITEMS + qb.POST_ITEMS + saved

    def test_bank_diagnostics_flags_thin_kp_coverage(self):
        """每 kp 每卷只1 题时必须报警 —— 这是逐 kp σ 被放大的根因。"""
        diag = qb.bank_diagnostics()
        assert any("只 1 道题" in w for w in diag["warnings"]), (
            "题量诊断应在当前题库（每 kp 每卷 1 题）上报警"
        )

    def test_items_per_kp_is_reported(self):
        diag = qb.bank_diagnostics()
        for name in ("pre", "memory", "transfer"):
            assert name in diag["items_per_kp"]
            assert set(diag["items_per_kp"][name]) == qb.kp_coverage(qb.PRE_SET)

    def test_regression_old_covering_question_was_ill_posed(self):
        """把 v0.4.0 的原题面拿回来，确认它确实是不可解的。

        这条是回归锚点：证明我们修的不是「模型答错」而是「题目问错」。
        """
        old = (
            "一条报表查询 SELECT month, COUNT(*) FROM orders GROUP BY month，"
            "数据量很大。有人提议建索引 (month) 来加速。"
            "请指出这个索引为什么救不了这条查询，并说明要让它是覆盖索引还缺什么。"
        )
        # 题面只给了 SELECT month，索引 (month) 已含 month → 无从判断「还缺什么」
        assert "还缺什么" in old
        assert "COUNT(*) 需要" not in old, "题面没有说明聚合要读哪些列"
        # 新题面把错误说法写进去了，考点唯一
        new = next(i.question for i in qb.TRANSFER_ITEMS if i.item_id == "tr_covering_01")
        assert "已经是覆盖索引了" in new, "新题面应把错误说法直接写出来让考生去驳"


# ---------------------------------------------------------------------------
# v0.4.1：题库诊断（题目歧义会伪装成模型能力不足）
# ---------------------------------------------------------------------------


class TestBankDiagnostics:
    """起因：真实模型跑 10 次后才发现 tr_covering_01 题面自身有歧义。

    旧题面问「要让它是覆盖索引还缺什么」，但查询只 SELECT month，
    索引 (month) 里就有 month —— 按字面推它已经是覆盖索引。
    考生答「什么都不缺」逻辑上是对的，却被判45 分，报告上看起来像
    「这个知识点没迁移」，真相是「这道题问错了」。

    所以歧义必须机器可检，不能靠跑完实验靠人眼发现。
    """

    def test_current_bank_has_no_ambiguity_problem(self):
        diag = qb.bank_diagnostics()
        assert diag["problems"] == [], f"当前题库不该有歧义问题：{diag['problems']}"

    def test_all_transfer_items_declare_premise(self):
        """每道 TRANSFER 题必须显式声明前提在不在题面里。"""
        for item in qb.TRANSFER_ITEMS:
            assert item.premise_in_question in ("yes", "no", "asserted"), (
                f"{item.item_id} 未声明 premise_in_question"
            )

    def test_premise_no_is_rejected_as_hard_error(self):
        """显式声明「前提不在题面」的题必须被门禁拦住。"""
        broken = qb.ExamItem(
            item_id="tr_broken_01",
            kp="kp_covering",
            question="这个索引还缺什么？",
            answer="x" * 80,
            premise_in_question="no",
        )
        saved = qb.TRANSFER_ITEMS
        try:
            patched = saved + (broken,)
            qb.TRANSFER_ITEMS = patched
            qb.ALL_ITEMS = qb.PRE_ITEMS + qb.POST_ITEMS + patched
            # 补出题说明，好让门禁走到 premise 检查那一关
            qb.TRANSFER_NOTES["tr_broken_01"] = {
                "transfers_from": "x", "prior_intuition": "y",
            }
            with pytest.raises(ValueError, match="premise_in_question='no'"):
                qb.assert_bank_intact()
        finally:
            qb.TRANSFER_NOTES.pop("tr_broken_01", None)
            qb.TRANSFER_ITEMS = saved
            qb.ALL_ITEMS = qb.PRE_ITEMS + qb.POST_ITEMS + saved

    def test_illegal_premise_value_is_rejected(self):
        bad = qb.ExamItem(
            item_id="tr_bad_01",
            kp="kp_covering",
            question="正常题目？",
            answer="x" * 80,
            premise_in_question="maybe",
        )
        saved = qb.TRANSFER_ITEMS
        try:
            patched = saved + (bad,)
            qb.TRANSFER_ITEMS = patched
            qb.ALL_ITEMS = qb.PRE_ITEMS + qb.POST_ITEMS + patched
            qb.TRANSFER_NOTES["tr_bad_01"] = {
                "transfers_from": "x", "prior_intuition": "y",
            }
            with pytest.raises(ValueError, match="取值非法"):
                qb.assert_bank_intact()
        finally:
            qb.TRANSFER_NOTES.pop("tr_bad_01", None)
            qb.TRANSFER_ITEMS = saved
            qb.ALL_ITEMS = qb.PRE_ITEMS + qb.POST_ITEMS + saved

    def test_bank_diagnostics_flags_thin_kp_coverage(self):
        """每 kp 每卷只1 题时必须报警 —— 这是逐 kp σ 被放大的根因。"""
        diag = qb.bank_diagnostics()
        assert any("只 1 道题" in w for w in diag["warnings"]), (
            "题量诊断应在当前题库（每 kp 每卷 1 题）上报警"
        )

    def test_items_per_kp_is_reported(self):
        diag = qb.bank_diagnostics()
        for name in ("pre", "memory", "transfer"):
            assert name in diag["items_per_kp"]
            assert set(diag["items_per_kp"][name]) == qb.kp_coverage(qb.PRE_SET)

    def test_regression_old_covering_question_was_ill_posed(self):
        """把 v0.4.0 的原题面拿回来，确认它确实是不可解的。

        这条是回归锚点：证明我们修的不是「模型答错」而是「题目问错」。
        """
        old = (
            "一条报表查询 SELECT month, COUNT(*) FROM orders GROUP BY month，"
            "数据量很大。有人提议建索引 (month) 来加速。"
            "请指出这个索引为什么救不了这条查询，并说明要让它是覆盖索引还缺什么。"
        )
        # 题面只给了 SELECT month，索引 (month) 已含 month → 无从判断「还缺什么」
        assert "还缺什么" in old
        assert "COUNT(*) 需要" not in old, "题面没有说明聚合要读哪些列"
        # 新题面把错误说法写进去了，考点唯一
        new = next(i.question for i in qb.TRANSFER_ITEMS if i.item_id == "tr_covering_01")
        assert "已经是覆盖索引了" in new, "新题面应把错误说法直接写出来让考生去驳"


# ---------------------------------------------------------------------------
# v0.4.1：天花板效应必须在报告里显式说出来
# ---------------------------------------------------------------------------


class TestCeilingDetection:
    """真实实验逼出来的问题：PRE 已90 分，「增益测不出来」有两种解释。

    DeepSeek 10 次实测 PRE=90.0 → 开卷最多只剩 10 分空间，
    而 σ=3~5。OPEN_BOOK_GAIN 均值 +6.8 看着是正的，但 mean-2σ=-1.9<0。

    关键：**「无法与噪声区分」不能被读成「学习没起作用」**。
    也可能是根本没有可测量的空间。这两种在数据上分不开，
    所以报告必须把这层不确定性写出来，而不是让读者自己脑补。
    """

    def test_ceiling_hit_at_high_pre(self):
        r = report.detect_ceiling(90.0)
        assert r["hit"] is True
        assert r["headroom"] == 10.0
        assert "天花板" in r["note"]

    def test_ceiling_not_hit_at_low_pre(self):
        r = report.detect_ceiling(45.0)
        assert r["hit"] is False
        assert r["headroom"] == 55.0
        assert "空间充足" in r["note"]

    def test_ceiling_boundary(self):
        assert report.detect_ceiling(report.CEILING_LINE)["hit"] is True
        assert report.detect_ceiling(report.CEILING_LINE - 0.1)["hit"] is False

    def test_ceiling_note_explains_both_explanations(self):
        """必须同时说出两种可能，否则读者会得到一个错误的否定结论。"""
        note = report.detect_ceiling(90.0)["note"]
        assert "学习没用" in note
        assert "没有可测量的空间" in note

    def test_weak_conclusion_gets_ceiling_caveat(self):
        """OPEN_BOOK 未通过判定 + 天花板命中 → 必须打上「结论不可靠」的警告。"""
        summary = {
            "n_trials": 10,
            "metrics": {"pre_score": {"mean": 90.0, "std": 1.3, "n": 10}},
            "verdicts": {
                "open_book_gain": {
                    "state": "noisy", "mean": 6.8, "std": 4.3,
                    "lower_bound": -1.9, "n": 10, "reason": "",
                },
                "memory_gain": {
                    "state": "noisy", "mean": 3.1, "std": 2.1,
                    "lower_bound": -1.1, "n": 10, "reason": "",
                },
                "transfer_gain": {
                    "state": "non_positive", "mean": -9.3, "std": 4.5,
                    "lower_bound": -18.2, "n": 10, "reason": "",
                },
            },
        }
        v = report.batch_verdict(summary)
        assert v["ceiling"]["hit"] is True
        assert any("不可靠" in c for c in v["caveats"]), (
            f"天花板下「未观察到提升」必须标注为不可靠：{v['caveats']}"
        )

    def test_strong_conclusion_still_gets_ceiling_caveat(self):
        """即使结论为正，天花板限制也要写 —— 它影响的是「提升幅度」的解读。"""
        summary = {
            "n_trials": 10,
            "metrics": {"pre_score": {"mean": 90.0, "std": 1.3, "n": 10}},
            "verdicts": {
                "open_book_gain": {
                    "state": "positive", "mean": 6.8, "std": 1.0,
                    "lower_bound": 4.8, "n": 10, "reason": "",
                },
                "memory_gain": {
                    "state": "positive", "mean": 3.1, "std": 0.5,
                    "lower_bound": 2.1, "n": 10, "reason": "",
                },
                "transfer_gain": {
                    "state": "noisy", "mean": -9.3, "std": 1.0,
                    "lower_bound": -11.3, "n": 10, "reason": "",
                },
            },
        }
        v = report.batch_verdict(summary)
        assert v["ceiling"]["hit"] is True
        assert len(v["caveats"]) >= 1, "即使结论为正也要写天花板"

    def test_caveats_always_include_thin_kp_warning(self):
        """题量诊断必须进 caveats —— 每 kp 只 1 题是逐 kp 结论的硬限制。"""
        summary = {
            "n_trials": 5,
            "metrics": {"pre_score": {"mean": 50.0, "std": 1.0, "n": 5}},
            "verdicts": {
                "open_book_gain": {
                    "state": "positive", "mean": 20.0, "std": 1.0,
                    "lower_bound": 18.0, "n": 5, "reason": "",
                },
            },
        }
        v = report.batch_verdict(summary)
        assert v["ceiling"]["hit"] is False
        assert any("只 1 道题" in c for c in v["caveats"]), (
            f"题量诊断必须进 caveats：{v['caveats']}"
        )

    def test_report_text_contains_caveat_section(self):
        """文本报告必须有「结论的限制」这一段，不能只印结论。"""
        summary = {
            "n_trials": 10, "topic": "T", "model": "m", "batch_id": "b",
            "bank_version": "v0.4.1", "bank_fingerprint": "f" * 64,
            "material_checksum": "c" * 64,
            "metrics": {
                "pre_score": {
                    "mean": 90.0, "std": 1.3, "n": 10,
                    "min": 85.8, "max": 91.8,
                },
            },
            "verdicts": {
                "open_book_gain": {
                    "state": "noisy", "mean": 6.8, "std": 4.3,
                    "lower_bound": -1.9, "n": 10, "reason": "波动大于均值",
                },
            },
            "note_quality_pass_rate": 1.0,
        }
        summary["verdict"] = report.batch_verdict(summary)
        text = report.format_batch_text(summary)
        assert "结论的限制" in text
        assert "天花板" in text


class TestCeilingDataPath:
    """天花板检测必须读真实字段名。

    实跑时踩过：`batch_verdict` 一开始从 `summary["gains"]["pre"]` 取 PRE 均值，
    但`aggregate_trials` 产出的真实路径是 `summary["metrics"]["pre_score"]["mean"]`。
    取错路径 → 静默得到 0.0 → 天花板永不命中 → 检测形同虚设，
    而且不报错。这类 bug 只有真实数据才暴露得出来，所以钉死它。
    """

    def test_aggregate_trials_uses_metrics_pre_score(self):
        """真实聚合产物的字段名就是 metrics.pre_score.mean。"""
        s = rep.aggregate_trials([
            {
                "trial_no": i, "experiment_id": i,
                "pre_score": 90.0, "memory_open_score": 97.0,
                "memory_blind_score": 93.0, "transfer_blind_score": 86.0,
                "practice_usable_rate": 1.0, "practice_avg_score": 76.0,
                "note_quality_avg": 85.0, "note_quality_pass_rate": True,
            }
            for i in (1, 2)
        ])
        assert "pre_score" in s["metrics"]
        assert s["metrics"]["pre_score"]["mean"] == 90.0
        assert "pre" not in s.get("gains", {}), (
            "gains 里不该有 pre 条目 —— 读错这个key 会拿到 0"
        )

    def test_verdict_detects_ceiling_from_real_summary(self):
        """用 aggregate_trials 的真实产物直接喂 batch_verdict。"""
        s = rep.aggregate_trials([
            {
                "trial_no": i, "experiment_id": i,
                "pre_score": 90.0 + i, "memory_open_score": 97.0,
                "memory_blind_score": 93.0, "transfer_blind_score": 86.0,
                "practice_usable_rate": 1.0, "practice_avg_score": 76.0,
                "note_quality_avg": 85.0, "note_quality_pass_rate": True,
            }
            for i in range(5)
        ])
        v = report.batch_verdict(s)
        assert v["ceiling"]["hit"] is True, (
            f"真实 summary 必须命中天花板检测：{v['ceiling']}"
        )
        assert v["ceiling"]["headroom"] < 15


class TestCeilingDataPath:
    """天花板检测必须读真实字段名。

    实跑时踩过：`batch_verdict` 一开始从 `summary["gains"]["pre"]` 取 PRE 均值，
    但`aggregate_trials` 产出的真实路径是 `summary["metrics"]["pre_score"]["mean"]`。
    取错路径 → 静默得到 0.0 → 天花板永不命中 → 检测形同虚设，
    而且不报错。这类 bug 只有真实数据才暴露得出来，所以钉死它。
    """

    def test_aggregate_trials_uses_metrics_pre_score(self):
        """真实聚合产物的字段名就是 metrics.pre_score.mean。"""
        s = rep.aggregate_trials([
            {
                "trial_no": i, "experiment_id": i,
                "pre_score": 90.0, "memory_open_score": 97.0,
                "memory_blind_score": 93.0, "transfer_blind_score": 86.0,
                "practice_usable_rate": 1.0, "practice_avg_score": 76.0,
                "note_quality_avg": 85.0, "note_quality_pass_rate": True,
            }
            for i in (1, 2)
        ])
        assert "pre_score" in s["metrics"]
        assert s["metrics"]["pre_score"]["mean"] == 90.0
        assert "pre" not in s.get("gains", {}), (
            "gains 里不该有 pre 条目 —— 读错这个key 会拿到 0"
        )

    def test_verdict_detects_ceiling_from_real_summary(self):
        """用 aggregate_trials 的真实产物直接喂 batch_verdict。"""
        s = rep.aggregate_trials([
            {
                "trial_no": i, "experiment_id": i,
                "pre_score": 90.0 + i, "memory_open_score": 97.0,
                "memory_blind_score": 93.0, "transfer_blind_score": 86.0,
                "practice_usable_rate": 1.0, "practice_avg_score": 76.0,
                "note_quality_avg": 85.0, "note_quality_pass_rate": True,
            }
            for i in range(5)
        ])
        v = report.batch_verdict(s)
        assert v["ceiling"]["hit"] is True, (
            f"真实 summary 必须命中天花板检测：{v['ceiling']}"
        )
        assert v["ceiling"]["headroom"] < 15
