"""brain/experiment.py —— V0.2 核心：验证「学习是否真的带来能力提升」。

实验流程（顺序不可换）：
    1. 冻结题库 + 冻结学习材料（记checksum）
    2. PRE TEST  —— 用固定 PRE 卷，盲考，独立评分
    3. 学习       —— 读指定材料，抽取知识点
    4. 实践       —— 按材料完成实践任务，记录成功/失败/经验
    5. POST TEST —— 用固定 POST 卷（同 kp、不同表述），盲考，独立评分
    6. 对比       —— 总分 + 逐知识点
    7. 存全部原始记录

三条不可让步的实验纪律：
- 考生上下文为空：答pre/post 时 context 参数强制空串，闸门开启
- 题目不可由模型产生：题目来自 question_bank 模块常量，LLM 出题接口在
  本流程中完全不被调用
- 评估器独立：evaluate 用独立的 evaluation provider，只看题目+参考答案+
  考生答案，看不到考生答题时的任何上下文
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from brain import question_bank as qb
from brain.llm import LLMProvider
from evaluation.evaluator import EvaluationStore, Evaluator
from experience.store import ExperienceReflector, ExperienceStore
from knowledge.store import KnowledgeStore
from learning_notes.store import (
    LearningNoteStore,
    NoteGenerator,
    render_notes_for_exam,
)
from learning_notes.quality import NoteQualityEvaluator, NoteQualityStore
from memory.db import Database, checksum
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
from skills.store import SkillStore
from training_data.generator import TrainingSampleStore

STUDY_SYSTEM = (
    "你是学习助手。阅读给定的学习材料，抽取核心知识点。"
    "只输出 JSON："
    '{"items":[{"title":"知识点标题","content":"准确复述材料要点（100-300字）",'
    '"keywords":["关键词"]}]}'
)
STUDY_TASK_TAG = "[[TASK:STUDY]]"

PRACTICE_SYSTEM = (
    "你是练习者。基于给定的学习材料回答任务要求，"
    "答案要能体现你确实掌握了材料中的具体内容。只输出 JSON："
    '{"outcome":"success|failure","result":"实际结果","kp":"涉及的知识点id"}'
)
PRACTICE_TASK_TAG = "[[TASK:PRACTICE]]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MaterialStore:
    """学习材料版本存储。"""

    db: Database

    def save(self, topic: str, title: str, content: str, version: str = "v1") -> dict[str, Any]:
        """保存材料。同 topic+version+checksum 已存在则复用，避免实验跑多次堆重复行。"""
        cks = checksum(content)
        existing = self.db.query_one(
            "SELECT * FROM materials WHERE topic = ? AND version = ? AND checksum = ?",
            (topic, version, cks),
        )
        if existing:
            return {
                "id": int(existing["id"]),
                "topic": topic,
                "version": version,
                "title": title,
                "checksum": cks,
                "reused": True,
            }
        cur = self.db.execute(
            "INSERT INTO materials(topic, version, title, content, checksum, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (topic, version, title, content, cks, _now()),
        )
        return {
            "id": int(cur.lastrowid),
            "topic": topic,
            "version": version,
            "title": title,
            "checksum": cks,
            "reused": False,
        }

    def get(self, mid: int) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM materials WHERE id = ?", (mid,))

    def latest(self, topic: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM materials WHERE topic = ? ORDER BY id DESC LIMIT 1", (topic,)
        )


@dataclass
class BankStore:
    """题库指纹存储。实验开始时登记一次，结束时核对。"""

    db: Database

    def register(self) -> int:
        meta = qb.bank_meta()
        existing = self.db.query_one(
            "SELECT id FROM question_bank WHERE fingerprint = ?", (meta["fingerprint"],)
        )
        if existing:
            return int(existing["id"])
        payload = json.dumps(
            {
                "pre": [
                    {"item_id": i.item_id, "kp": i.kp, "question": i.question,
                     "answer": i.answer, "difficulty": i.difficulty}
                    for i in qb.PRE_SET
                ],
                "post": [
                    {"item_id": i.item_id, "kp": i.kp, "question": i.question,
                     "answer": i.answer, "difficulty": i.difficulty}
                    for i in qb.POST_SET
                ],
            },
            ensure_ascii=False,
        )
        meta_full = qb.bank_meta()
        cur = self.db.execute(
            """INSERT INTO question_bank(version, fingerprint, pre_fingerprint,
                                         post_fingerprint, pre_count, post_count,
                                         payload_json, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                meta_full["version"],
                meta_full["fingerprint"],
                meta_full["pre_fingerprint"],
                meta_full["post_fingerprint"],
                meta_full["pre_count"],
                meta_full["post_count"],
                payload,
                _now(),
            ),
        )
        return int(cur.lastrowid)

    def verify(self, bank_id: int) -> None:
        """核对库中指纹与当前代码常量一致。"""
        row = self.db.query_one("SELECT * FROM question_bank WHERE id = ?", (bank_id,))
        if row is None:
            raise RuntimeError(f"题库记录 #{bank_id} 不存在")
        if row["fingerprint"] != qb.BANK_FINGERPRINT:
            raise RuntimeError(
                "题库指纹不一致：实验期间题库被修改过，本次实验结果作废。\n"
                f"  库中记录: {row['fingerprint'][:16]}...\n"
                f"  当前代码: {qb.BANK_FINGERPRINT[:16]}..."
            )


@dataclass
class ExperimentStore:
    """实验主记录。"""

    db: Database

    def create(
        self,
        topic: str,
        material_id: int,
        bank_id: int,
        batch_id: str = "",
        trial_no: int = 1,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO experiments(topic, material_id, bank_id, status,
                                       detail_json, batch_id, trial_no, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (topic, material_id, bank_id, "running", "{}", batch_id, trial_no, _now()),
        )
        return int(cur.lastrowid)

    def add_item(
        self,
        experiment_id: int,
        phase: str,
        item_id: str,
        kp: str,
        question: str,
        reference_answer: str,
        candidate_answer: str,
        score: float | None,
        verdict: str,
        feedback: str,
        missing_points: list[str],
        task_id: int | None,
        evaluation_id: int | None,
        mode: str = "blind_model",
        context_used: str = "",
        exam_kind: str = "memory",
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO experiment_items(experiment_id, phase, item_id, kp, question,
                                           reference_answer, candidate_answer, score,
                                           verdict, feedback, missing_points, task_id,
                                           evaluation_id, created_at, mode, context_used,
                                           exam_kind)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                experiment_id, phase, item_id, kp, question, reference_answer,
                candidate_answer, score, verdict, feedback,
                json.dumps(missing_points, ensure_ascii=False), task_id,
                evaluation_id, _now(), mode, context_used, exam_kind,
            ),
        )
        return int(cur.lastrowid)

    def finish(
        self,
        experiment_id: int,
        pre_score: float,
        detail: dict[str, Any],
        memory_open_score: float | None = None,
        memory_blind_score: float | None = None,
        transfer_blind_score: float | None = None,
        practice_usable_rate: float | None = None,
        practice_avg_score: float | None = None,
        note_quality_avg: float | None = None,
        delta: float | None = None,
        delta_pct: float | None = None,
        open_book_score: float | None = None,
        post_score: float | None = None,
    ) -> None:
        """写回实验结果。

        老的 pre/post/delta/open_book_score 列继续写（V0.3 报告读它们），
        V0.4 的四个新分数写各自的列 —— 两代并存，report 层再选读哪代。
        """
        self.db.execute(
            """UPDATE experiments SET pre_score = ?, status = 'done',
                   detail_json = ?, finished_at = ?,
                   memory_open_score = ?, memory_blind_score = ?,
                   transfer_blind_score = ?, practice_usable_rate = ?,
                   practice_avg_score = ?, note_quality_avg = ?,
                   delta = ?, delta_pct = ?,
                   open_book_score = ?, post_score = ?
               WHERE id = ?""",
            (
                pre_score, json.dumps(detail, ensure_ascii=False), _now(),
                memory_open_score, memory_blind_score, transfer_blind_score,
                practice_usable_rate, practice_avg_score, note_quality_avg,
                delta, delta_pct, open_book_score, post_score,
                experiment_id,
            ),
        )

    def add_practice(
        self,
        experiment_id: int,
        outcome: str,
        kp: str,
        scenario: str,
        action: str,
        result: str,
        lesson: str,
        score: float | None,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO practice_records(experiment_id, stage, outcome, kp, scenario,
                                            action, result, lesson, score, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (experiment_id, "practice", outcome, kp, scenario, action, result,
             lesson, score, _now()),
        )
        return int(cur.lastrowid)

    def get(self, eid: int) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM experiments WHERE id = ?", (eid,))

    def items(self, eid: int, phase: str | None = None) -> list[dict[str, Any]]:
        if phase:
            return self.db.query_all(
                "SELECT * FROM experiment_items WHERE experiment_id = ? AND phase = ?"
                " ORDER BY id",
                (eid, phase),
            )
        return self.db.query_all(
            "SELECT * FROM experiment_items WHERE experiment_id = ? ORDER BY phase, id",
            (eid,),
        )

    def practice(self, eid: int) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM practice_records WHERE experiment_id = ? ORDER BY id", (eid,)
        )

    def latest(self) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM experiments ORDER BY id DESC LIMIT 1")


@dataclass
class ExperimentResult:
    """V0.4：四个考试分数 + 两个非考试指标。

    四场考试的分工（这是 V0.4 的核心设计）：

      PRE              学习前裸能力，不给任何资料。基线。
      MEMORY_OPEN      记忆测试开卷：同一批新题，可读本实验笔记。
                       测「能不能用外部记忆」。
      MEMORY_BLIND     记忆测试盲考：同一批题，收回全部外部记忆。
                       测「拿掉资料以后是不是还能记住」——记忆保持，不是概念形成。
      TRANSFER_BLIND   迁移测试盲考：TRANSFER 卷，题目完全不同但考同一知识点。
                       测「有没有形成概念」——这是 V0.4 真正想看的东西。

    **为什么必须有 TRANSFER**
    MEMORY_BLIND 高只能说明模型还能想起和这道题相关的措辞；
    TRANSFER_BLIND 高才说明它能在新场景里用同一个原理。
    前者可以是背了答案，后者才是学会了。

    两个非考试指标：
      practice_usable_rate  学习能否转化为实际任务能力（独立评审判产出可执行性）
      note_quality_avg      学习笔记本身是否可靠（四维度加权）

    语义纪律：
      **这是能力迁移实验，不是训练实验。** 本项目没有任何参数更新或微调。
      TRANSFER_BLIND 再高也不能解释为「模型被改造过」。
    """

    experiment_id: int
    topic: str
    pre_score: float
    memory_open_score: float
    memory_blind_score: float
    transfer_blind_score: float
    practice_usable_rate: float = 0.0
    practice_avg_score: float = 0.0
    note_quality_avg: float = 0.0
    note_quality_detail: dict[str, Any] = field(default_factory=dict)
    transfer_gain: float = 0.0
    memory_gain: float = 0.0
    open_book_gain: float = 0.0
    trial_no: int = 1
    batch_id: str = ""
    improved_kps: list[dict[str, Any]] = field(default_factory=list)
    flat_kps: list[dict[str, Any]] = field(default_factory=list)
    regressed_kps: list[dict[str, Any]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "topic": self.topic,
            "pre_score": self.pre_score,
            "memory_open_score": self.memory_open_score,
            "memory_blind_score": self.memory_blind_score,
            "transfer_blind_score": self.transfer_blind_score,
            "practice_usable_rate": self.practice_usable_rate,
            "practice_avg_score": self.practice_avg_score,
            "note_quality_avg": self.note_quality_avg,
            "note_quality_detail": self.note_quality_detail,
            "memory_gain": self.memory_gain,
            "open_book_gain": self.open_book_gain,
            "transfer_gain": self.transfer_gain,
            "trial_no": self.trial_no,
            "batch_id": self.batch_id,
            "improved_kps": self.improved_kps,
            "flat_kps": self.flat_kps,
            "regressed_kps": self.regressed_kps,
            "detail": self.detail,
        }


class ExperimentRunner:
    """一键跑完整实验。"""

    def __init__(
        self,
        db: Database,
        provider: LLMProvider,
        evaluator_provider: LLMProvider | None = None,
    ) -> None:
        self.db = db
        self.provider = provider
        # 评估器可以有独立 provider；默认同 provider 但仍是两次独立调用
        self.evaluator_provider = evaluator_provider or provider

        open_policy = AccessPolicy.open_mode()
        self.knowledge = KnowledgeStore(db, open_policy)
        self.samples = TrainingSampleStore(db, open_policy)
        self.experiences = ExperienceStore(db, open_policy)
        self.notes = LearningNoteStore(db, open_policy)
        self.evaluations = EvaluationStore(db)
        self.skills = SkillStore(db)
        self.materials = MaterialStore(db)
        self.bank_store = BankStore(db)
        self.store = ExperimentStore(db)
        self.note_generator = NoteGenerator(db, provider, sorted(qb.kp_coverage(qb.PRE_SET)))

        # 考生侧：只能答题，不能出题
        from tasks.builder import QuestionBuilder, TaskStore

        self.tasks = TaskStore(db, open_policy)
        self.builder = QuestionBuilder(db, self.tasks, provider)
        # 评估侧：独立 provider 实例
        self.evaluator = Evaluator(db, self.evaluations, self.evaluator_provider)
        self.reflector = ExperienceReflector(db, self.experiences, self.provider)

        # V0.4：笔记质量评审与实践评审。
        # 两者都是独立 provider 实例 + 独立人格，且都**看不到**考生答题上下文：
        #   笔记评审能看到原始材料（判断幻觉必须对照原文）
        #   实践评审看不到材料（判的是「产出能不能用」，
        #                如果给材料就变成了「有没有用上材料」）
        self.note_quality_store = NoteQualityStore(db, open_policy)
        self.note_quality = NoteQualityEvaluator(
            db, self.note_quality_store, self.evaluator_provider
        )
        self.practice_judge_store = PracticeJudgeStore(db, open_policy)
        self.practice_judge = PracticeJudge(
            db, self.practice_judge_store, self.evaluator_provider
        )

    # ------------------------------------------------------------------
    # 闸门自检
    # ------------------------------------------------------------------
    def _assert_blind_isolation(self) -> None:
        """BLIND_MODEL 自检：确认任何学习产物都读不到。

        注意用 lambda 惰性调用 —— 探针本身必须先被构造再执行，
        否则会在这里就直接抛异常，把「自检」变成「崩溃」。
        """
        blind = AccessPolicy.blind_model(stage="blind_selfcheck")
        probes = [
            (lambda: KnowledgeStore(self.db, blind).for_experiment(1), RESOURCE_KNOWLEDGE),
            (lambda: LearningNoteStore(self.db, blind).for_experiment(1), RESOURCE_NOTES),
            (lambda: TrainingSampleStore(self.db, blind).list_by_topic("x", limit=1),
             RESOURCE_SAMPLES),
            (lambda: ExperienceStore(self.db, blind).recent(limit=1), RESOURCE_EXPERIENCES),
            (lambda: blind.guard(RESOURCE_REFERENCE), RESOURCE_REFERENCE),
            (lambda: blind.guard(RESOURCE_EVALUATION), RESOURCE_EVALUATION),
            (lambda: blind.guard(RESOURCE_MATERIAL), RESOURCE_MATERIAL),
        ]
        self._run_probes(probes, "BLIND_MODEL")

    def _assert_open_book_isolation(self, experiment_id: int) -> None:
        """OPEN_BOOK_MEMORY 自检：只放行本实验笔记/知识，其余全禁。"""
        policy = AccessPolicy.open_book_memory(stage="open_book_selfcheck",
                                               experiment_id=experiment_id)
        probes = [
            # 禁读项
            (lambda: policy.guard(RESOURCE_MATERIAL), RESOURCE_MATERIAL),
            (lambda: policy.guard(RESOURCE_REFERENCE), RESOURCE_REFERENCE),
            (lambda: policy.guard(RESOURCE_SAMPLES), RESOURCE_SAMPLES),
            (lambda: policy.guard(RESOURCE_EVALUATION), RESOURCE_EVALUATION),
            (lambda: policy.guard(RESOURCE_HISTORY), RESOURCE_HISTORY),
            (lambda: policy.guard(RESOURCE_EXPERIENCES), RESOURCE_EXPERIENCES),
            # 跨实验访问
            (lambda: policy.guard_scope(RESOURCE_NOTES, experiment_id + 999), RESOURCE_NOTES),
            (lambda: policy.guard_scope(RESOURCE_KNOWLEDGE, experiment_id + 999),
             RESOURCE_KNOWLEDGE),
        ]
        self._run_probes(probes, "OPEN_BOOK_MEMORY")

    @staticmethod
    def _run_probes(probes: list[tuple[Any, str]], mode_label: str) -> None:
        for probe, name in probes:
            try:
                probe()
            except ExamAccessViolation:
                continue
            raise RuntimeError(  # pragma: no cover
                f"访问闸门失效：{mode_label} 竟能访问 {name}，本次实验结果不可信"
            )

    def _assert_exam_isolation(self) -> None:
        """向后兼容：等价于 BLIND_MODEL 自检。"""
        self._assert_blind_isolation()

    # ------------------------------------------------------------------
    # 2. 考试（V0.3：三种 mode）
    # ------------------------------------------------------------------
    def _build_exam_context(
        self, experiment_id: int, mode: str
    ) -> tuple[str, dict[str, Any]]:
        """按考试模式构造考生可见的上下文。

        返回 (context_text, meta)。这是 V0.3 的关键分叉点：

        - BLIND_MODEL     → 恒返回空串。只允许看到题目本身。
        - OPEN_BOOK_MEMORY→ 只放行**本实验产出**的 knowledge + learning_notes。
                            原始材料、参考答案、训练样本、经验、历史答案一律不放行。

        meta 里记录实际读了多少条笔记/知识，落库供事后复核。
        """
        if mode == "blind_model":
            self._assert_blind_isolation()
            return "", {"notes_used": 0, "knowledge_used": 0, "context_len": 0}

        if mode != "open_book_memory":
            raise ValueError(f"未知考试模式 {mode!r}")

        # 开卷：先过作用域检查，再取数据
        policy = AccessPolicy.open_book_memory(stage="open_book", experiment_id=experiment_id)
        note_store = LearningNoteStore(self.db, policy)
        know_store = KnowledgeStore(self.db, policy)

        notes = note_store.for_experiment(experiment_id)  # guard_scope 生效
        knowledge = know_store.for_experiment(experiment_id)

        note_text = render_notes_for_exam(notes)
        know_text = "\n".join(f"- {k['title']}：{k['content']}" for k in knowledge)
        parts = []
        if note_text:
            parts.append(note_text)
        if know_text:
            parts.append("以下是本实验中抽取的知识点：\n" + know_text)
        context = "\n\n".join(parts)
        return context, {
            "notes_used": len(notes),
            "knowledge_used": len(knowledge),
            "context_len": len(context),
        }

    def _run_exam(
        self,
        experiment_id: int,
        phase: str,
        mode: str,
        set_name: str = "post",
        exam_kind: str = "memory",
    ) -> tuple[float, list[dict[str, Any]]]:
        """跑一场考试。

        phase:    'pre' | 'post'      —— 落库用的阶段名
        mode:     'blind_model' | 'open_book_memory'
        set_name: 'pre' | 'memory' | 'transfer' —— 取哪一卷题
        exam_kind:'memory' | 'transfer' —— 这场属于哪类测试，落库供事后复核

        题目来自固定卷，LLM 出题接口在本流程中零调用。
        """
        context, ctx_meta = self._build_exam_context(experiment_id, mode)
        items = qb.get_set(set_name)
        results: list[dict[str, Any]] = []
        for it in items:
            # 题目快照落库：确保事后可复核「当时考的是这道题、什么模式、哪一卷」
            tid = self.tasks.create(
                task_type="exam",
                stage=f"{phase}_test",
                topic=qb.PRE_TOPIC,
                question=it.question,
                reference_answer=it.answer,
                difficulty=it.difficulty,
                blind=True,
            )
            # 考生答题：context 按模式决定，reference_answer 永远不进来
            answer = self.builder.answer(it.question, context=context)
            self.tasks.set_answer(tid, answer)

            # 独立评估：只看题目 + 参考答案 + 考生答案
            ev = self.evaluator.evaluate(tid, it.question, answer, it.answer)

            self.store.add_item(
                experiment_id=experiment_id,
                phase=phase,
                mode=mode,
                exam_kind=exam_kind,
                item_id=it.item_id,
                kp=it.kp,
                question=it.question,
                reference_answer=it.answer,
                candidate_answer=answer,
                score=ev["score"],
                verdict=ev["verdict"],
                feedback=ev["feedback"],
                missing_points=ev["missing_points"],
                task_id=tid,
                evaluation_id=ev["evaluation_id"],
                context_used=context,
            )
            results.append(
                {"item_id": it.item_id, "kp": it.kp, "mode": mode,
                 "exam_kind": exam_kind, "score": ev["score"],
                 "answer": answer, "feedback": ev["feedback"], **ctx_meta}
            )
        avg = round(sum(r["score"] for r in results) / len(results), 2) if results else 0.0
        return avg, results

    # ------------------------------------------------------------------
    # 3. 学习
    # ------------------------------------------------------------------
    def study(
        self, topic: str, content: str, max_items: int = 4,
        experiment_id: int | None = None,
    ) -> dict[str, Any]:
        """读材料学知识点。返回抽取结果 + 材料信息。

        experiment_id 传入时会把产出的知识绑定到该实验，
        供开卷考试做作用域隔离读取。
        """
        material = self.materials.save(topic, topic, content, version="v1")
        user = f"{STUDY_TASK_TAG}\n学习材料：\n{content}\n请抽取 {max_items} 个核心知识点。"
        try:
            data = self.provider.ask_json(STUDY_SYSTEM, user, temperature=0.3)
        except ValueError:
            data = None
        items = []
        if isinstance(data, dict):
            items = [i for i in (data.get("items") or []) if i.get("title") and i.get("content")]
        if not items:  # 降级
            items = [
                {"title": f"{topic} 要点{i+1}", "content": content[:300],
                 "keywords": [topic]}
                for i in range(max_items)
            ]
        ids = []
        for it in items[:max_items]:
            kid = self.knowledge.add(
                topic=topic,
                title=str(it["title"]),
                content=str(it["content"]),
                keywords=[str(k) for k in (it.get("keywords") or [])],
                source="study",
                experiment_id=experiment_id,
            )
            ids.append(kid)
        return {"material": material, "knowledge_ids": ids}

    def make_notes(
        self, experiment_id: int, topic: str, content: str
    ) -> list[dict[str, Any]]:
        """生成结构化学习笔记（V0.3 核心产物）。

        笔记是考试时唯一允许读取的「外部记忆」载体，
        必须在 study 阶段就固化，否则开卷考试无上下文可用。
        """
        notes = self.note_generator.generate(self.notes, experiment_id, topic, content)
        for n in notes:
            if not self.notes.verify_checksum(n["id"]):  # pragma: no cover
                raise RuntimeError(f"学习笔记 #{n['id']} checksum 校验失败")
        return notes

    # ------------------------------------------------------------------
    # 4. 实践（成功 / 失败 / 经验）
    # ------------------------------------------------------------------
    def practice(self, experiment_id: int, topic: str, content: str) -> dict[str, Any]:
        """按材料做实践任务，记录成功/失败/经验，并过独立评审。

        实践阶段可以读材料（非考试），但**评审环节看不到材料**——
        评审判的是「产出是否可执行」，不是「有没有用上材料」。
        """
        records: list[dict[str, Any]] = []
        for it in qb.get_set("pre"):  # 用 pre 卷的 kp 列表作为实践任务清单
            task_prompt = (
                f"针对知识点「{qb.KP_LABELS.get(it.kp, it.kp)}」，"
                f"请给出一个能验证你是否掌握它的实践任务，并说明你会如何完成。"
                "要求给出**可直接执行**的操作（命令 / SQL / 步骤），"
                "以及如何判断结果是否符合预期。"
            )
            user = f"{PRACTICE_TASK_TAG}\n学习材料：\n{content}\n任务：{task_prompt}"
            self_reported = "success"
            result_text = ""
            try:
                data = self.provider.ask_json(PRACTICE_SYSTEM, user, temperature=0.3)
            except ValueError:
                data = None
            if isinstance(data, dict) and data.get("outcome") in {"success", "failure"}:
                self_reported = str(data["outcome"])
                result_text = str(data.get("result", ""))
            # JSON 解析失败时**必须重取原文**。
            # 曾踩的坑：解析失败就把占位串当「实践产出」丢给评审器，
            # 评审器看着一句「模型未返回结构化结果」判 0 分不可用——
            # 模型明明答了，整段内容却被占位符顶掉了。
            # 指标失真比指标缺失更危险：它看起来仍然是个数。
            if not result_text.strip():
                result_text = self._recover_practice_text(user)

            # 独立评审：只看产出本身，不给材料、不给笔记。
            # outcome 以评审为准，不采信模型自报。
            judged = self.practice_judge.judge(
                experiment_id, it.kp, qb.KP_LABELS.get(it.kp, it.kp),
                task_prompt, result_text,
            )

            lesson = self.reflector.reflect(
                "practice", task_prompt, result_text, judged["score"], None
            )["lesson"]
            self.store.add_practice(
                experiment_id=experiment_id,
                outcome="success" if judged["usable"] else "failure",
                kp=it.kp,
                scenario=task_prompt,
                action=result_text,
                # result 存**实际产出**而不是评审摘要：
                # 事后要能翻出「当时到底产出了什么」才能复核判分对不对。
                # 评审结论另存practice_evaluations 表。
                result=result_text,
                lesson=lesson,
                score=judged["score"],
            )
            records.append({
                "kp": it.kp,
                "self_reported": self_reported,
                "usable": judged["usable"],
                "score": judged["score"],
                "evaluator": judged.get("evaluator", ""),
                "output_len": len(result_text),
                "lesson": lesson,
            })
        return {"records": records}

    def _recover_practice_text(self, user: str) -> str:
        """实践产出 JSON 解析失败时，重取一次纯文本。

        为什么要重试而不是记失败：模型答没答对它自己不一定清楚，
        但「答没答」是可观测的。把它降级成空串或占位符，
        实践可用率这个指标就变成在测「模型听不听 JSON 格式指令」，
        而不是「学习能否转化为实际任务能力」。

        重试仍失败就返回空串 —— 那时候判不可用才是对的。
        """
        try:
            text = self.provider.ask_text(
                "你是练习者。基于给定的学习材料完成任务，"
                "直接给出可执行的操作（命令 / SQL / 步骤）与判据，不要输出 JSON。",
                user,
                temperature=0.3,
            )
        except Exception:  # noqa: BLE001 — 兜底路径不能再抛
            return ""
        return (text or "").strip()

    @staticmethod
    def _practice_score(outcome: str, result_text: str) -> float:
        """V0.3 的粗粒度自评分，**V0.4 报告不再使用**。

        只看模型自报的 outcome 加结果长度，测不出真实能力——
        模型写一大段看起来很像样的话就能拿高分。
        V0.4 改用独立评审的 judge 分数。保留此函数仅为旧数据可读。
        """
        if outcome == "failure":
            return 40.0
        return 85.0 if len(result_text.strip()) >= 15 else 60.0

    # ------------------------------------------------------------------
    # 6. 对比
    # ------------------------------------------------------------------
    def compare(self, experiment_id: int) -> dict[str, Any]:
        """逐知识点对比：PRE / MEMORY_BLIND / TRANSFER_BLIND 三列盲考并列。

        只在三者都存在的 kp 上比较——kp 缺失说明题库或笔记出问题，
        宁可不比也不给假结论。

        为什么把 TRANSFER 放进「逐知识点」而不是单独看总分：
        总分只能说明「整体有没有进步」，看不出是哪个知识点形成了概念、
        哪个只是背住了话。逐 kp 看才能定位。
        """
        rows = self.store.items(experiment_id)
        pre = {r["kp"]: r for r in rows if r["phase"] == "pre"}
        memory = {
            r["kp"]: r for r in rows
            if r["phase"] == "post" and r.get("mode") == "blind_model"
        }
        transfer = {
            r["kp"]: r for r in rows
            if r.get("exam_kind") == "transfer"
        }
        improved, flat, regressed = [], [], []
        for kp in sorted(set(pre) & set(memory) & set(transfer)):
            p_score = float(pre[kp]["score"])
            m_score = float(memory[kp]["score"])
            t_score = float(transfer[kp]["score"])
            entry = {
                "kp": kp,
                "label": qb.KP_LABELS.get(kp, kp),
                "pre": p_score,
                "memory_blind": m_score,
                "transfer_blind": t_score,
                # 记忆保持 vs 概念迁移，差值本身就是信息：
                # 记忆涨得多但迁移没涨 = 背住了，没学会
                "memory_delta": round(m_score - p_score, 2),
                "transfer_delta": round(t_score - p_score, 2),
                # 迁移相对记忆的折损：正数=迁移不如记忆强
                "transfer_vs_memory": round(t_score - m_score, 2),
            }
            if entry["transfer_delta"] > 0:
                improved.append(entry)
            elif entry["transfer_delta"] < 0:
                regressed.append(entry)
            else:
                flat.append(entry)
        return {"improved": improved, "flat": flat, "regressed": regressed}

    # ------------------------------------------------------------------
    # 8. 一键完整实验（V0.4 流程）
    #
    #   PRE (盲) → 阅读材料 → 整理笔记 → 实践
    #            → MEMORY:OPEN_BOOK → MEMORY:BLIND
    #            → TRANSFER:BLIND
    #
    # 四个考试回答四个不同问题：
    #   PRE                 学习前裸能力，基线
    #   MEMORY_OPEN_BOOK    能不能用自己写的外部记忆
    #   MEMORY_BLIND        拿掉资料后还记不记得住（记忆保持）
    #   TRANSFER_BLIND      新场景里能不能用同一个原理（概念形成）
    #
    # 两个非考试指标：
    #   practice_usable_rate  学习能否转化为可执行的任务产出
    #   note_quality_avg      笔记本身可靠吗（四维度加权，独立评估）
    # ------------------------------------------------------------------
    def run(
        self,
        topic: str,
        material_content: str,
        batch_id: str = "",
        trial_no: int = 1,
    ) -> ExperimentResult:
        # 冻结题库 + 材料
        qb.assert_bank_intact()
        bank_id = self.bank_store.register()
        self.bank_store.verify(bank_id)
        material = self.materials.save(topic, topic, material_content, version="v1")
        experiment_id = self.store.create(
            topic, material["id"], bank_id, batch_id=batch_id, trial_no=trial_no
        )

        # 1) PRE —— 学习前裸能力，只给题目
        pre_score, pre_results = self._run_exam(
            experiment_id, "pre", "blind_model", set_name="pre", exam_kind="pre"
        )

        # 2) 阅读材料 —— 抽知识点
        study = self.study(topic, material_content, experiment_id=experiment_id)

        # 3) 自己整理学习笔记 —— 结构化五要素
        notes = self.make_notes(experiment_id, topic, material_content)

        # 3b) 笔记质量独立评估（必须在考试之前做完：
        #     如果笔记有幻觉，我们要能看出开卷高分是虚的）
        quality = self.note_quality.evaluate_all(
            experiment_id, notes, material_content, qb.KP_LABELS
        )
        for row in quality["rows"]:
            self.notes.set_quality(row["note_id"], row["total"], row["verdict"])

        # 4) 实践 —— 产出 + 独立评审可执行性
        practice = self.practice(experiment_id, topic, material_content)
        usable_rate = self.practice_judge_store.usable_rate(experiment_id)
        practice_score = self.practice_judge_store.average_score(experiment_id)

        # 5) MEMORY TEST 开卷 —— 放行本实验笔记与知识
        self._assert_open_book_isolation(experiment_id)
        mem_open, mem_open_results = self._run_exam(
            experiment_id, "post", "open_book_memory",
            set_name="memory", exam_kind="memory",
        )

        # 6) MEMORY TEST 盲考 —— 同一批题，收回一切
        mem_blind, mem_blind_results = self._run_exam(
            experiment_id, "post", "blind_model",
            set_name="memory", exam_kind="memory",
        )

        # 7) TRANSFER TEST —— 题目全新，只盲考。
        #    不给它开卷是有意的：开卷能查笔记就说明不了迁移能力，
        #    TRANSFER 要测的正是「不依赖笔记能不能用原理」。
        transfer_blind, transfer_results = self._run_exam(
            experiment_id, "transfer", "blind_model",
            set_name="transfer", exam_kind="transfer",
        )

        # 题库二次核对：确保整个实验期间没人改题
        self.bank_store.verify(bank_id)

        # 三个 gain，语义各不相同，不可混用
        open_book_gain = round(mem_open - pre_score, 2)
        memory_gain = round(mem_blind - pre_score, 2)
        transfer_gain = round(transfer_blind - pre_score, 2)
        delta_pct = (
            round((transfer_gain / pre_score) * 100, 2) if pre_score else 0.0
        )
        cmp_result = self.compare(experiment_id)

        detail = {
            "bank_version": qb.BANK_VERSION,
            "bank_fingerprint": qb.BANK_FINGERPRINT,
            "material_checksum": material["checksum"],
            "material_id": material["id"],
            "pre_results": pre_results,
            "memory_open_results": mem_open_results,
            "memory_blind_results": mem_blind_results,
            "transfer_results": transfer_results,
            "study_knowledge_ids": study["knowledge_ids"],
            "note_ids": [n["id"] for n in notes],
            "note_checksums": {n["kp"]: n["checksum"] for n in notes},
            "note_quality": quality,
            "practice_records": practice["records"],
            "practice_usable_rate": usable_rate,
            "practice_avg_score": practice_score,
            "open_book_gain": open_book_gain,
            "memory_gain": memory_gain,
            "transfer_gain": transfer_gain,
        }
        # 同时写回 V0.3 的列，老报告不至于读空
        self.store.finish(
            experiment_id, pre_score, detail,
            memory_open_score=mem_open,
            memory_blind_score=mem_blind,
            transfer_blind_score=transfer_blind,
            practice_usable_rate=usable_rate,
            practice_avg_score=practice_score,
            note_quality_avg=quality["average_total"],
            delta=transfer_gain,
            delta_pct=delta_pct,
            open_book_score=mem_open,
            post_score=mem_blind,
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            topic=topic,
            pre_score=pre_score,
            memory_open_score=mem_open,
            memory_blind_score=mem_blind,
            transfer_blind_score=transfer_blind,
            practice_usable_rate=usable_rate,
            practice_avg_score=practice_score,
            note_quality_avg=quality["average_total"],
            note_quality_detail=quality,
            memory_gain=memory_gain,
            open_book_gain=open_book_gain,
            transfer_gain=transfer_gain,
            trial_no=trial_no,
            batch_id=batch_id,
            improved_kps=cmp_result["improved"],
            flat_kps=cmp_result["flat"],
            regressed_kps=cmp_result["regressed"],
            detail=detail,
        )
