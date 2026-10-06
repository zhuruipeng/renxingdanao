"""brain/experiment_v2.py —— V0.5 实验流程：Benchmark V2 六阶段。

六阶段（顺序不可换）
--------------------
    1. PRE_BLIND       裸能力基线，禁读一切
    2. STUDY           读 Zephyra 材料 → 抽知识点 → 写结构化笔记
    3. PRACTICE        8 个客观实践任务，**机器判分**
    4. OPEN_BOOK       放行本实验笔记与知识，考 MEMORY 卷
    5. MEMORY_BLIND    收回一切，考同一批题
    6. TRANSFER_BLIND  全新场景，只盲考

与 V0.4 的三处关键差异
----------------------
1. **实践判分不靠 LLM**：`PRACTICE` 阶段走
   `practice/objective.py` 的机器判分，产出落`practice_runs` 表。
   LLM 评审仍会跑（存`evaluator_practice_score`），
   但两者**分表存**，报告里也分开呈现——
   机器说 100 分而LLM 说 40 分时，这个矛盾本身就是数据。

2. **每阶段落齐 8 列**：`v2_stage_items` 强制记录
   question / answer / score / evaluator / context_used /
   experiment_id / knowledge_point / question_type。
   尤其 `context_used`：它证明「盲考时考生上下文真的是空的」，
   而不是「我们以为它是空的」。

3. **三份指纹一起冻**：题库 + 材料 + 实践任务集。
   V0.4 只冻前两份，于是「改了实践任务却拿旧结论比」是可能的。
   任何一份变化 → `verify_frozen()` 抛异常 → 整批作废。

隔离为什么必须在代码层面
------------------------
V0.5 的四类gain 全靠「上下文差异」来定义：

    external-memory gain  = OPEN_BOOK     - PRE_BLIND
    memory-retention gain  = MEMORY_BLIND  - PRE_BLIND
    transfer gain          = TRANSFER_BLIND- PRE_BLIND

这三个减法的**唯一前提**是 PRE_BLIND 阶段真的什么都没给考生。
如果它偷偷读到了笔记，PRE 会虚高，三个gain 全部被压低，
而且**压低多少算正常范围无法估计**——
这正是 V0.4.1 那句「增益测不出来有两种不可区分解释」的来源之一。

所以 PRE_BLIND 的 context 恒为空串，且每次都跑一遍闸门自检，
不是「配好了就不用看」。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from brain import benchmark_v2 as b2
from brain import novel_material as nm
from brain.llm import LLMProvider
from evaluation.evaluator import EvaluationStore, Evaluator
from knowledge.store import KnowledgeStore
from learning_notes.quality import NoteQualityEvaluator, NoteQualityStore
from learning_notes.store import LearningNoteStore, NoteGenerator
from memory.db import (
    Database,
    checksum,
    create_v2_sentinel,
    v2_experiment_key,
)
from memory.gate import (
    RESOURCE_EVALUATION,
    RESOURCE_EXPERIENCES,
    RESOURCE_HISTORY,
    RESOURCE_KNOWLEDGE,
    RESOURCE_MATERIAL,
    RESOURCE_NOTES,
    RESOURCE_REFERENCE,
    RESOURCE_SAMPLES,
    AccessPolicy,
    ExamAccessViolation,
)
from practice.judge import PracticeJudge, PracticeJudgeStore
from practice.objective import (
    OBJECTIVE_SYSTEM,
    OBJECTIVE_TASKS,
    ObjectiveResult,
    assert_tasks_intact,
    evaluate_objective,
    task_diagnostics,
    tasks_meta,
)
from tasks.builder import TaskStore

OBJECTIVE_TASK_TAG = "[[TASK:OBJECTIVE_PRACTICE]]"

#: 六阶段的规范顺序
STAGES: tuple[str, ...] = (
    "pre_blind",
    "study",
    "practice",
    "open_book",
    "memory_blind",
    "transfer_blind",
)

#: 三个盲考阶段：上下文必须恒为空
BLIND_STAGES = frozenset({"pre_blind", "memory_blind", "transfer_blind"})

#: 每个阶段考哪一卷
STAGE_SET: dict[str, str] = {
    "pre_blind": "pre",
    "open_book": "memory",
    "memory_blind": "memory",
    "transfer_blind": "transfer",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ==========================================================================
# 存储
# ==========================================================================


@dataclass
class V2Store:
    """V0.5 的四张表读写。"""

    db: Database

    # ---------- 批次 ----------
    def register_batch(
        self, batch_id: str, topic: str, planned_trials: int, model: str
    ) -> int:
        meta = b2.bank_meta()
        tm = tasks_meta()
        cks = checksum(nm.NOVEL_MATERIAL)
        existing = self.db.query_one(
            "SELECT id FROM v2_batches WHERE batch_id = ?", (batch_id,)
        )
        if existing:
            # 批次已存在：指纹保持冻结（不改），但把 planned_trials 往上抬。
            # 固定写 1 会让「计划 10 次」与库里的 1 不一致，
            # 事后按 planned_trials 核对完成度时必然误报。
            self.db.execute(
                "UPDATE v2_batches SET planned_trials = MAX(planned_trials, ?),"
                " model = ? WHERE batch_id = ?",
                (int(planned_trials), model, batch_id),
            )
            return int(existing["id"])
        cur = self.db.execute(
            """INSERT INTO v2_batches(batch_id, topic, planned_trials, bank_fingerprint,
                                       material_checksum, tasks_fingerprint, model,
                                       status, summary_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (batch_id, topic, planned_trials, meta["fingerprint"], cks,
             tm["fingerprint"], model, "running", "{}", _now()),
        )
        return int(cur.lastrowid)

    def finish_batch(self, batch_id: str, summary: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE v2_batches SET status = 'done', summary_json = ?, finished_at = ?"
            " WHERE batch_id = ?",
            (json.dumps(summary, ensure_ascii=False), _now(), batch_id),
        )

    def verify_frozen(self, batch_id: str) -> None:
        """核对三份指纹。任一变化 → 抛异常，本次批次结果全部作废。"""
        row = self.db.query_one("SELECT * FROM v2_batches WHERE batch_id = ?", (batch_id,))
        if row is None:
            raise RuntimeError(f"批次 {batch_id!r} 不存在")
        checks = (
            ("题库", row["bank_fingerprint"], b2.BANK_FINGERPRINT),
            ("学习材料", row["material_checksum"], checksum(nm.NOVEL_MATERIAL)),
            ("实践任务集", row["tasks_fingerprint"], tasks_meta()["fingerprint"]),
        )
        for label, frozen, current in checks:
            if frozen != current:
                raise RuntimeError(
                    f"{label}指纹不一致：实验期间被修改过，本次实验结果作废。\n"
                    f"  冻结时: {frozen[:16]}...\n"
                    f"  当前值: {current[:16]}..."
                )

    # ---------- 实验 ----------
    def create_experiment(
        self, topic: str, batch_id: str, trial_no: int
    ) -> int:
        meta = b2.bank_meta()
        tm = tasks_meta()
        cur = self.db.execute(
            """INSERT INTO v2_experiments(topic, batch_id, trial_no, bank_fingerprint,
                                          material_checksum, tasks_fingerprint,
                                          status, detail_json, diagnostics_json,
                                          created_at, trial_key)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (topic, batch_id, trial_no, meta["fingerprint"],
             checksum(nm.NOVEL_MATERIAL), tm["fingerprint"],
             "running", "{}", "{}", _now(), f"{batch_id}#{trial_no}"),
        )
        eid = int(cur.lastrowid)
        # 同步在老experiments 表建哨兵行，让三张老表的负 id 外键成立。
        # 没有它，learning_notes 等表的写入会被 FOREIGN KEY constraint 拒绝。
        create_v2_sentinel(self.db.conn, eid, topic)
        return eid

    def add_stage_item(
        self,
        experiment_id: int,
        stage: str,
        item_id: str,
        knowledge_point: str,
        question_type: str,
        question: str,
        answer: str,
        reference_answer: str,
        score: float | None,
        evaluator: str,
        context_used: str,
        mode: str,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO v2_stage_items(experiment_id, stage, item_id, knowledge_point,
                                          question_type, question, answer, reference_answer,
                                          score, evaluator, context_used, mode, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (experiment_id, stage, item_id, knowledge_point, question_type,
             question, answer, reference_answer, score, evaluator,
             context_used, mode, _now()),
        )
        return int(cur.lastrowid)

    def add_practice_run(
        self, experiment_id: int, result: ObjectiveResult,
        candidate_output: str, title: str,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO practice_runs(experiment_id, task_id, kp, kind, title,
                                         candidate_output, objective_result,
                                         objective_score, objective_passed,
                                         parse_error, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (experiment_id, result.task_id, result.kp, result.kind, title,
             candidate_output, json.dumps(result.checks, ensure_ascii=False),
             result.objective_score, 1 if result.passed else 0,
             result.parse_error, _now()),
        )
        return int(cur.lastrowid)

    def finish_experiment(
        self,
        experiment_id: int,
        scores: dict[str, Any],
        detail: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> None:
        self.db.execute(
            """UPDATE v2_experiments SET status = 'done',
                   pre_score = ?, open_book_score = ?, memory_blind_score = ?,
                   transfer_blind_score = ?, objective_practice_score = ?,
                   objective_practice_pass_rate = ?, evaluator_practice_score = ?,
                   note_quality_avg = ?, detail_json = ?, diagnostics_json = ?,
                   finished_at = ?
               WHERE id = ?""",
            (scores.get("pre_blind"), scores.get("open_book"),
             scores.get("memory_blind"), scores.get("transfer_blind"),
             scores.get("objective_practice_avg"),
             scores.get("objective_practice_pass_rate"),
             scores.get("evaluator_practice_avg"),
             scores.get("note_quality_avg"),
             json.dumps(detail, ensure_ascii=False),
             json.dumps(diagnostics, ensure_ascii=False),
             _now(), experiment_id),
        )

    # ---------- 读 ----------
    def experiment(self, eid: int) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM v2_experiments WHERE id = ?", (eid,))

    def items(self, eid: int, stage: str | None = None) -> list[dict[str, Any]]:
        if stage:
            return self.db.query_all(
                "SELECT * FROM v2_stage_items WHERE experiment_id = ? AND stage = ?"
                " ORDER BY id",
                (eid, stage),
            )
        return self.db.query_all(
            "SELECT * FROM v2_stage_items WHERE experiment_id = ? ORDER BY stage, id",
            (eid,),
        )

    def practice_runs(self, eid: int) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM practice_runs WHERE experiment_id = ? ORDER BY id", (eid,)
        )

    def batch_trials(self, batch_id: str) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM v2_experiments WHERE batch_id = ? ORDER BY trial_no, id",
            (batch_id,),
        )

    def batch_meta(self, batch_id: str) -> dict[str, Any] | None:
        """批次头信息（含三份冻结指纹）。"""
        return self.db.query_one(
            "SELECT * FROM v2_batches WHERE batch_id = ?", (batch_id,)
        )

    def evaluations_for(self, eid: int) -> list[dict[str, Any]]:
        """本实验的 LLM 评审行。

        V2 的实验 id 映射成负数写进老表（外键约束所迫），
        所以这里必须用同一个映射读回来，不能用 eid 直接查。
        """
        key = v2_experiment_key(eid)
        return self.db.query_all(
            "SELECT * FROM practice_evaluations WHERE experiment_id = ?"
            " ORDER BY id", (key,),
        )


# ==========================================================================
# 结果对象
# ==========================================================================


@dataclass
class V2TrialResult:
    """一次 V2 trial 的完整结果。"""

    experiment_id: int
    batch_id: str
    trial_no: int
    scores: dict[str, float] = field(default_factory=dict)
    objective_detail: list[dict[str, Any]] = field(default_factory=list)
    evaluator_practice: list[dict[str, Any]] = field(default_factory=list)
    gains: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "batch_id": self.batch_id,
            "trial_no": self.trial_no,
            "scores": self.scores,
            "objective_detail": self.objective_detail,
            "evaluator_practice": self.evaluator_practice,
            "gains": self.gains,
            "diagnostics": self.diagnostics,
            "detail": self.detail,
        }


# ==========================================================================
# Runner
# ==========================================================================


class V2ExperimentRunner:
    """Benchmark V2 的完整实验流程。"""

    def __init__(
        self,
        db: Database,
        provider: LLMProvider,
        evaluator_provider: LLMProvider | None = None,
    ) -> None:
        self.db = db
        self.provider = provider
        self.evaluator_provider = evaluator_provider or provider
        self.store = V2Store(db)

        open_policy = AccessPolicy.open_mode()
        self.knowledge = KnowledgeStore(db, open_policy)
        self.notes = LearningNoteStore(db, open_policy)
        self.note_quality_store = NoteQualityStore(db, open_policy)
        self.practice_judge_store = PracticeJudgeStore(db, open_policy)
        self.tasks = TaskStore(db, open_policy)
        self.evaluations = EvaluationStore(db)
        self.evaluator = Evaluator(db, self.evaluations, self.evaluator_provider)
        self.note_quality = NoteQualityEvaluator(
            db, self.note_quality_store, self.evaluator_provider
        )
        self.practice_judge = PracticeJudge(
            db, self.practice_judge_store, self.evaluator_provider
        )
        self.note_generator = NoteGenerator(
            db, provider, sorted(b2.kp_coverage(b2.PRE_SET))
        )

    # ------------------------------------------------------------------
    # 闸门自检
    # ------------------------------------------------------------------
    def assert_blind_isolation(self) -> None:
        """BLIND_MODEL 自检：确认任何学习产物都读不到。"""
        blind = AccessPolicy.blind_model(stage="v2_blind_selfcheck")
        probes = [
            (lambda: blind.guard(RESOURCE_KNOWLEDGE), RESOURCE_KNOWLEDGE),
            (lambda: blind.guard(RESOURCE_NOTES), RESOURCE_NOTES),
            (lambda: blind.guard(RESOURCE_MATERIAL), RESOURCE_MATERIAL),
            (lambda: blind.guard(RESOURCE_SAMPLES), RESOURCE_SAMPLES),
            (lambda: blind.guard(RESOURCE_EXPERIENCES), RESOURCE_EXPERIENCES),
            (lambda: blind.guard(RESOURCE_REFERENCE), RESOURCE_REFERENCE),
            (lambda: blind.guard(RESOURCE_EVALUATION), RESOURCE_EVALUATION),
            (lambda: blind.guard(RESOURCE_HISTORY), RESOURCE_HISTORY),
        ]
        self._run_probes(probes, "BLIND_MODEL")

    def assert_open_book_isolation(self, experiment_id: int) -> None:
        """OPEN_BOOK 自检：只放行本实验产物，禁材料/答案/跨实验。"""
        policy = AccessPolicy.open_book_memory(
            stage="v2_open_book_selfcheck", experiment_id=experiment_id
        )
        probes = [
            (lambda: policy.guard(RESOURCE_MATERIAL), RESOURCE_MATERIAL),
            (lambda: policy.guard(RESOURCE_REFERENCE), RESOURCE_REFERENCE),
            (lambda: policy.guard(RESOURCE_SAMPLES), RESOURCE_SAMPLES),
            (lambda: policy.guard(RESOURCE_EVALUATION), RESOURCE_EVALUATION),
            (lambda: policy.guard(RESOURCE_HISTORY), RESOURCE_HISTORY),
            (lambda: policy.guard(RESOURCE_EXPERIENCES), RESOURCE_EXPERIENCES),
            (lambda: policy.guard_scope(RESOURCE_NOTES, experiment_id + 999), RESOURCE_NOTES),
            (lambda: policy.guard_scope(RESOURCE_KNOWLEDGE, experiment_id + 999),
             RESOURCE_KNOWLEDGE),
        ]
        self._run_probes(probes, "OPEN_BOOK_MEMORY")

    @staticmethod
    def _run_probes(probes: list[tuple[Any, str]], label: str) -> None:
        for probe, name in probes:
            try:
                probe()
            except ExamAccessViolation:
                continue
            raise RuntimeError(  # pragma: no cover
                f"访问闸门失效：{label} 竟能访问 {name}，本次实验结果不可信"
            )

    # ------------------------------------------------------------------
    # 上下文构造
    # ------------------------------------------------------------------
    def build_exam_context(self, experiment_id: int, stage: str) -> tuple[str, dict[str, Any]]:
        """按阶段构造考生可见上下文。

        **BLIND_STAGES 里的三个阶段恒返回空串**，
        且每次都重跑闸门自检——不是「配好了就不用看」。
        理由见模块 docstring：三个 gain 的减法全靠这个前提。
        """
        if stage in BLIND_STAGES:
            self.assert_blind_isolation()
            return "", {"notes_used": 0, "knowledge_used": 0, "context_len": 0}

        if stage != "open_book":
            raise ValueError(f"未知考试阶段 {stage!r}")

        self.assert_open_book_isolation(experiment_id)
        policy = AccessPolicy.open_book_memory(stage="open_book", experiment_id=experiment_id)
        notes = LearningNoteStore(self.db, policy).for_v2_experiment(experiment_id)
        knowledge = KnowledgeStore(self.db, policy).for_experiment(experiment_id)
        from learning_notes.store import render_notes_for_exam

        parts = []
        note_text = render_notes_for_exam(notes)
        if note_text:
            parts.append(note_text)
        if knowledge:
            parts.append(
                "以下是本实验中抽取的知识点：\n"
                + "\n".join(f"- {k['title']}：{k['content']}" for k in knowledge)
            )
        context = "\n\n".join(parts)
        return context, {
            "notes_used": len(notes),
            "knowledge_used": len(knowledge),
            "context_len": len(context),
        }

    # ------------------------------------------------------------------
    # 考试阶段
    # ------------------------------------------------------------------
    def run_exam(self, experiment_id: int, stage: str) -> tuple[float, list[dict[str, Any]]]:
        """跑一场考试并落齐 8 列。"""
        set_name = STAGE_SET[stage]
        items = b2.get_set(set_name)
        context, ctx_meta = self.build_exam_context(experiment_id, stage)
        mode = "blind_model" if stage in BLIND_STAGES else "open_book_memory"

        results: list[dict[str, Any]] = []
        for it in items:
            tid = self.tasks.create(
                task_type="exam",
                stage=f"{stage}_test",
                topic=nm.NOVEL_TOPIC,
                question=it.question,
                reference_answer=it.answer,
                difficulty=it.difficulty,
                blind=True,
            )
            # reference_answer 永远不进prompt
            answer = self.provider.ask_text(
                "你是考生。只依据题目自身与你已有的知识作答，简洁、分点，"
                "不要编造引用来源。",
                f"题目：{it.question}" + (f"\n参考资料：\n{context}" if context else ""),
                temperature=0.4,
            ).strip()
            self.tasks.set_answer(tid, answer)

            ev = self.evaluator.evaluate(tid, it.question, answer, it.answer)
            self.store.add_stage_item(
                experiment_id=experiment_id,
                stage=stage,
                item_id=it.item_id,
                knowledge_point=it.kp,
                question_type=it.question_type,
                question=it.question,
                answer=answer,
                reference_answer=it.answer,
                score=ev["score"],
                evaluator="llm_judge",
                context_used=context,
                mode=mode,
            )
            results.append({
                "item_id": it.item_id, "kp": it.kp, "question_type": it.question_type,
                "score": ev["score"], "evaluator": "llm_judge", **ctx_meta,
            })
        avg = round(sum(r["score"] for r in results) / len(results), 2) if results else 0.0
        return avg, results

    # ------------------------------------------------------------------
    # 学习阶段
    # ------------------------------------------------------------------
    def run_study(self, experiment_id: int) -> dict[str, Any]:
        """读 Zephyra 材料 → 抽知识点 → 写结构化笔记。"""
        content = nm.NOVEL_MATERIAL
        topic = nm.NOVEL_TOPIC

        # 抽知识点
        user = (
            "[[TASK:STUDY]]\n学习材料：\n" + content +
            "\n请抽取 6 个核心知识点。只输出 JSON："
            '{"items":[{"title":"标题","content":"准确复述要点（100-300字）",'
            '"keywords":["关键词"]}]}'
        )
        try:
            data = self.provider.ask_json(
                "你是学习助手。阅读材料，抽取核心知识点。只输出 JSON。",
                user, temperature=0.3,
            )
        except ValueError:
            data = None
        items = []
        if isinstance(data, dict):
            items = [i for i in (data.get("items") or [])
                     if isinstance(i, dict) and i.get("title") and i.get("content")]
        if not items:
            items = [{"title": f"{topic} 要点{i+1}", "content": content[:300],
                      "keywords": [topic]} for i in range(4)]
        for it in items[:6]:
            self.knowledge.add(
                topic=topic, title=str(it["title"]), content=str(it["content"]),
                keywords=[str(k) for k in (it.get("keywords") or [])],
                source="study", experiment_id=experiment_id,
            )

        # 写笔记（V0.3 的五要素，沿用）
        notes = self.note_generator.generate(
            self.notes, experiment_id, topic, content, v2=True
        )
        for n in notes:
            if not self.notes.verify_checksum(n["id"]):  # pragma: no cover
                raise RuntimeError(f"学习笔记 #{n['id']} checksum 校验失败")
            # 笔记也落进 stage_items，让「study 阶段到底产出了什么」可查
            self.store.add_stage_item(
                experiment_id=experiment_id, stage="study",
                item_id=f"note_{n['id']}", knowledge_point=n["kp"],
                question_type="note",
                question=f"为知识点 {b2.KP_LABELS.get(n['kp'], n['kp'])} 写结构化笔记",
                answer=f"概念：{n['concept']}\n原理：{n['principle']}\n"
                       f"例子：{n['example']}\n易错点：{n['common_mistake']}\n"
                       f"应用规则：{n['application_rule']}",
                reference_answer="",
                score=n.get("quality_total"),
                evaluator="note_checksum",
                context_used="", mode="open",
            )
        return {
            "knowledge_count": len(items[:6]),
            "note_ids": [n["id"] for n in notes],
            "note_checksums": {n["kp"]: n["checksum"] for n in notes},
        }

    def run_note_quality(self, experiment_id: int, note_ids: list[int]) -> float:
        """笔记质量独立评估（必须在考试之前做完）。"""
        notes = [self.notes.get(nid) for nid in note_ids]
        notes = [n for n in notes if n]
        if not notes:
            return 0.0
        quality = self.note_quality.evaluate_all(
            experiment_id, notes, nm.NOVEL_MATERIAL, b2.KP_LABELS, v2=True
        )
        for row in quality["rows"]:
            self.notes.set_quality(row["note_id"], row["total"], row["verdict"])
        return quality["average_total"]

    # ------------------------------------------------------------------
    # 实践阶段（客观判分 + LLM 评审，两者分开）
    # ------------------------------------------------------------------
    def run_practice(self, experiment_id: int) -> dict[str, Any]:
        """跑 8 个客观实践任务。

        每个任务**跑两遍判分**：
          1. `evaluate_objective` 机器判分 → practice_runs 表
          2. `PracticeJudge` LLM 评审 → practice_evaluations 表

        两者的分数刻意不合并。机器说 100 而 LLM 说 40 时，
        这个差值本身就是结论的一部分：它说明产出形式正确但工程上不可用，
        或者反过来。合并成一个数字就把这层信息抹掉了。
        """
        rows: list[dict[str, Any]] = []
        llm_rows: list[dict[str, Any]] = []
        for task in OBJECTIVE_TASKS:
            # 实践阶段可读材料（非考试）：测「学了能不能用上」
            user = (
                f"{OBJECTIVE_TASK_TAG}\n学习材料：\n{nm.NOVEL_MATERIAL}\n"
                f"任务：{task.title}\n{task.prompt}\n"
                f"提交格式示例（注意示例里没有答案，值要你自己算）：{task.submit_hint}"
            )
            try:
                candidate = self.provider.ask_text(
                    OBJECTIVE_SYSTEM, user, temperature=0.3
                ).strip()
            except Exception:  # noqa: BLE001 — 拿不到产出也是数据
                candidate = ""

            result = evaluate_objective(task, candidate)
            self.store.add_practice_run(
                experiment_id, result, candidate, task.title
            )
            rows.append(result.as_dict())

            # LLM 评审：只判「作为工程实践好不好用」，看不到材料。
            # task_id 必须传——否则评审行无法与 practice_runs 的机器判分对齐，
            # 出现分歧时定位不到是哪个任务。
            judged = self.practice_judge.judge(
                experiment_id, task.kp, task.title, task.prompt, candidate,
                v2=True, task_id=task.task_id,
            )
            llm_rows.append(judged)
            # 落进 stage_items，让 practice 阶段也有完整 8 列
            self.store.add_stage_item(
                experiment_id=experiment_id, stage="practice",
                item_id=task.task_id, knowledge_point=task.kp,
                question_type="practice", question=task.prompt,
                answer=candidate, reference_answer="",
                score=result.objective_score, evaluator=result.evaluator,
                context_used="", mode="open",
            )

        n = len(rows) or 1
        return {
            "runs": rows,
            "llm": llm_rows,
            "objective_avg": round(sum(r["objective_score"] for r in rows) / n, 2),
            "pass_rate": round(sum(1 for r in rows if r["passed"]) / n, 4),
            "evaluator_avg": round(
                sum(r["score"] for r in llm_rows) / (len(llm_rows) or 1), 2
            ),
        }

    # ------------------------------------------------------------------
    # 一键完整流程
    # ------------------------------------------------------------------
    def run(
        self, batch_id: str = "", trial_no: int = 1, register: bool = True
    ) -> V2TrialResult:
        # --- 前置门禁：题库/任务集任一有问题就拒绝跑 ---
        b2.assert_bank_intact()
        assert_tasks_intact()

        topic = nm.NOVEL_TOPIC
        model = getattr(self.provider, "name", "unknown")
        if register:
            # planned_trials 传 trial_no：库不存在就按本次建，存在就往上抬。
            # 写死 1 会让「计划 N 次」永远停在 1。
            self.store.register_batch(batch_id or "default", topic, trial_no, model)
        if batch_id:
            self.store.verify_frozen(batch_id)

        experiment_id = self.store.create_experiment(topic, batch_id, trial_no)

        # 1) PRE_BLIND —— 裸能力，context 恒空
        pre, pre_rows = self.run_exam(experiment_id, "pre_blind")

        # 2) STUDY —— 读材料抽知识点写笔记
        study = self.run_study(experiment_id)
        nq = self.run_note_quality(experiment_id, study["note_ids"])

        # 3) PRACTICE —— 机器判分
        practice = self.run_practice(experiment_id)

        # 4) OPEN_BOOK —— 可读本实验笔记与知识
        open_book, ob_rows = self.run_exam(experiment_id, "open_book")

        # 5) MEMORY_BLIND —— 收回一切
        memory_blind, mb_rows = self.run_exam(experiment_id, "memory_blind")

        # 6) TRANSFER_BLIND —— 全新场景
        transfer_blind, tr_rows = self.run_exam(experiment_id, "transfer_blind")

        # 结束后再核对一次三份指纹
        if batch_id:
            self.store.verify_frozen(batch_id)

        scores = {
            "pre_blind": pre,
            "open_book": open_book,
            "memory_blind": memory_blind,
            "transfer_blind": transfer_blind,
            "objective_practice_avg": practice["objective_avg"],
            "objective_practice_pass_rate": practice["pass_rate"],
            "evaluator_practice_avg": practice["evaluator_avg"],
            "note_quality_avg": nq,
        }
        # 三个 gain 语义严格分离，不允许混称
        gains = {
            "external_memory_gain": round(open_book - pre, 2),
            "memory_retention_gain": round(memory_blind - pre, 2),
            "transfer_gain": round(transfer_blind - pre, 2),
            "objective_practice_gain": round(
                practice["objective_avg"] - pre, 2
            ),
        }
        diagnostics = {
            "bank": b2.bank_diagnostics(),
            "tasks": task_diagnostics(),
        }
        detail = {
            "bank_version": b2.BANK_VERSION,
            "bank_fingerprint": b2.BANK_FINGERPRINT,
            "material_checksum": checksum(nm.NOVEL_MATERIAL),
            "tasks_fingerprint": tasks_meta()["fingerprint"],
            "novel_topic": nm.NOVEL_TOPIC,
            "pre_rows": pre_rows,
            "open_book_rows": ob_rows,
            "memory_blind_rows": mb_rows,
            "transfer_blind_rows": tr_rows,
            "study": study,
            "gains": gains,
            "stage_order": list(STAGES),
        }
        self.store.finish_experiment(experiment_id, scores, detail, diagnostics)

        return V2TrialResult(
            experiment_id=experiment_id,
            batch_id=batch_id,
            trial_no=trial_no,
            scores=scores,
            objective_detail=practice["runs"],
            evaluator_practice=practice["llm"],
            gains=gains,
            diagnostics=diagnostics,
            detail=detail,
        )


__all__ = [
    "BLIND_STAGES",
    "STAGES",
    "STAGE_SET",
    "V2ExperimentRunner",
    "V2Store",
    "V2TrialResult",
]
