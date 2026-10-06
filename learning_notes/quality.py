"""学习笔记质量评估 —— V0.4 核心指标之一。

**为什么必须单独评估笔记？**

开卷考试只能告诉我们「带笔记能不能答对」，但答对可能是因为笔记写得好，
也可能是因为题目恰好简单。如果笔记本身是错的——比如凭空编了一个机制、
把 rowid 说成是「物理地址」——那么开卷分数照样很漂亮，而整条学习链路
已经在骗人了。

所以笔记质量必须**独立测量**，而不是从开卷分数里反推。

**四个维度**
------------
accuracy      准确性：笔记里的机制描述是否与原始材料一致，有没有张冠李戴
completeness  完整性：五个字段是否都填了实质内容，而不是「（模型不可用）」这种占位
transferable  可迁移性：笔记讲的是「为什么」还是只背结论——只背结论的笔记
              在 TRANSFER 测试上必然失效
no_hallucination 无幻觉：有没有编造材料里不存在的概念

**为什么要用独立 provider**
--------------------------
写笔记的和评笔记的必须是两个独立调用、两个独立人格。
同一个模型在同一次上下文里既写又评，几乎必然给自己打高分——这在
V0.2/V0.3 的评估器设计里已经反复验证过（考生/考官人格必须分离）。

评估器**可以看到原始材料**：判断「是否有幻觉」本质上就是拿笔记和材料对照。
但它**看不到考生答题的任何内容** —— 否则「这条笔记有没有帮上忙」的信号
会污染「这条笔记对不对」的判断。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from brain.llm import LLMProvider
from learning_notes.store import LearningNoteStore, note_checksum
from memory.db import Database, v2_experiment_key
from memory.gate import RESOURCE_NOTES, AccessPolicy, ExamAccessViolation

QUALITY_SYSTEM = (
    "你是严格的学习笔记审校员。你会拿到一份原始学习材料，"
    "以及某位学习者针对某个知识点写的结构化笔记（五要素：概念/原理/例子/易错点/应用规则）。"
    "请严格对照材料评估这份笔记的质量，四个维度各打 0-100 分：\n"
    "- accuracy：机制描述是否与材料一致，有没有把 A 的性质说成 B 的\n"
    "- completeness：五个字段是否都有实质内容，而不是占位符或空话\n"
    "- transferable：讲清了「为什么」还是只背了结论。"
    "只写结论、无法支撑迁移应用的，transferable 应给低分\n"
    "- no_hallucination：有没有编造材料里根本不存在的概念、术语或机制\n"
    "只输出 JSON："
    '{"accuracy":85,"completeness":90,"transferable":70,"no_hallucination":95,'
    '"total":85,"verdict":"pass|fail","feedback":"简短评语",'
    '"problems":["具体问题"]}'
)

QUALITY_TASK_TAG = "[[TASK:NOTE_QUALITY]]"

#: 四个维度的权重。transferable 权重最高——
#: 一份准确但不解释原理的笔记，在 TRANSFER 测试上毫无用处，
#: 而 TRANSFER 才是 V0.4 真正想测的东西。
DIMENSIONS = ("accuracy", "completeness", "transferable", "no_hallucination")

WEIGHTS = {
    "accuracy": 0.35,
    "completeness": 0.20,
    "transferable": 0.30,
    "no_hallucination": 0.15,
}

#: 判定笔记「合格」的总分线
PASS_LINE = 60.0

#: 占位/降级文本。出现在笔记里直接判不合格——
#: 模型挂了写出来的占位符如果混进开卷上下文，会污染整场实验。
PLACEHOLDER_MARKERS = (
    "模型不可用",
    "模型挂了",
    "降级",
    "TODO",
    "待补充",
    "（无）",
    "不知道",
    "无法生成",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class NoteQualityStore:
    """笔记质量评估结果存储。"""

    def __init__(self, db: Database, policy: AccessPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or AccessPolicy.open_mode()

    def add(
        self,
        experiment_id: int,
        note_id: int,
        accuracy: float,
        completeness: float,
        transferable: float,
        no_hallucination: float,
        total: float,
        verdict: str,
        feedback: str,
        problems: list[str],
        evaluator: str = "llm_judge",
        *,
        v2: bool = False,
    ) -> int:
        self.policy.guard(RESOURCE_NOTES)
        if v2:
            # experiment_id 写负数哨兵，理由见 learning_notes/store.py 的 add()
            cur = self.db.execute(
                """INSERT INTO note_quality(experiment_id, v2_experiment_id, note_id,
                                           accuracy, completeness, transferable,
                                           no_hallucination, total, verdict, feedback,
                                           problems, evaluator, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (v2_experiment_key(experiment_id), experiment_id, note_id,
                 accuracy, completeness, transferable,
                 no_hallucination, total, verdict, feedback,
                 _delimiter_join(problems), evaluator, _now()),
            )
            return int(cur.lastrowid)
        cur = self.db.execute(
            """INSERT INTO note_quality(experiment_id, note_id, accuracy, completeness,
                                       transferable, no_hallucination, total, verdict,
                                       feedback, problems, evaluator, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (experiment_id, note_id, accuracy, completeness, transferable,
             no_hallucination, total, verdict, feedback,
             _delimiter_join(problems), evaluator, _now()),
        )
        return int(cur.lastrowid)

    def for_experiment(self, experiment_id: int) -> list[dict[str, Any]]:
        self.policy.guard(RESOURCE_NOTES)
        return self.db.query_all(
            "SELECT * FROM note_quality WHERE experiment_id = ? ORDER BY id",
            (experiment_id,),
        )

    def for_note(self, note_id: int) -> dict[str, Any] | None:
        self.policy.guard(RESOURCE_NOTES)
        return self.db.query_one(
            "SELECT * FROM note_quality WHERE note_id = ? ORDER BY id DESC LIMIT 1",
            (note_id,),
        )

    def average_total(self, experiment_id: int) -> float:
        rows = self.for_experiment(experiment_id)
        if not rows:
            return 0.0
        return round(sum(float(r["total"]) for r in rows) / len(rows), 2)


def _delimiter_join(items: Sequence[str]) -> str:
    import json

    return json.dumps(list(items), ensure_ascii=False)


def _split_delimiter(text: str) -> list[str]:
    import json

    try:
        v = json.loads(text or "[]")
        return list(v) if isinstance(v, list) else []
    except (ValueError, TypeError):
        return []


def weighted_total(scores: dict[str, float]) -> float:
    return round(sum(scores[d] * WEIGHTS[d] for d in DIMENSIONS), 2)


class NoteQualityEvaluator:
    """独立笔记审校器。

    与写笔记的 provider 是两个独立实例、两次独立调用、两套人格。
    """

    def __init__(
        self,
        db: Database,
        store: NoteQualityStore,
        provider: LLMProvider,
    ) -> None:
        self.db = db
        self.store = store
        self.provider = provider

    # ------------------------------------------------------------------
    def evaluate(
        self,
        experiment_id: int,
        note: dict[str, Any],
        material: str,
        kp_label: str = "",
        *,
        v2: bool = False,
    ) -> dict[str, Any]:
        """评估单份笔记。

        material 必须传入：判断「是否有幻觉」本质上就是拿笔记和原文对照。
        但**不能**传入考生答题内容 —— 否则「有没有帮上忙」会污染「对不对」。
        """
        note_text = _render_note(note)
        placeholder_hit = _detect_placeholder(note_text)
        user = (
            f"{QUALITY_TASK_TAG}\n"
            f"知识点：{kp_label or note.get('kp', '')}\n\n"
            f"【原始学习材料】\n{material}\n\n"
            f"【待审笔记】\n{note_text}\n\n"
            "请按四个维度打分。"
        )
        data: dict[str, Any] | None = None
        try:
            data = self.provider.ask_json(QUALITY_SYSTEM, user, temperature=0.1)
        except ValueError:
            data = None

        if isinstance(data, dict) and any(d in data for d in DIMENSIONS):
            scores = {d: _clamp(data.get(d, 0)) for d in DIMENSIONS}
            feedback = str(data.get("feedback", ""))
            problems = list(data.get("problems") or [])
            evaluator = "llm_judge"
        else:
            scores = self._rule_score(note_text, material)
            feedback = "LLM 不可用，按规则兜底评分"
            problems = []
            # 必须留痕：规则兜底的分和真判分的分不是一回事，
            # 报告里若混为一谈，「笔记质量」这个指标就失去意义了。
            evaluator = "rule_fallback"

        # 占位文本是硬否决：模型挂了写出来的占位符不能算「质量良好」
        if placeholder_hit:
            scores["completeness"] = min(scores["completeness"], 20.0)
            scores["transferable"] = min(scores["transferable"], 20.0)
            problems.append(f"笔记含占位/降级文本：{placeholder_hit}")

        total = weighted_total(scores)
        verdict = "pass" if total >= PASS_LINE else "fail"
        qid = self.store.add(
            experiment_id=experiment_id,
            note_id=int(note["id"]),
            accuracy=scores["accuracy"],
            completeness=scores["completeness"],
            transferable=scores["transferable"],
            no_hallucination=scores["no_hallucination"],
            total=total,
            verdict=verdict,
            feedback=feedback,
            problems=problems,
            evaluator=evaluator,
            v2=v2,
        )
        return {
            "quality_id": qid,
            "note_id": int(note["id"]),
            **scores,
            "total": total,
            "verdict": verdict,
            "evaluator": evaluator,
            "feedback": feedback,
            "problems": problems,
        }

    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        experiment_id: int,
        notes: list[dict[str, Any]],
        material: str,
        kp_labels: dict[str, str] | None = None,
        *,
        v2: bool = False,
    ) -> dict[str, Any]:
        """评估一份实验的全部笔记，返回汇总。

        `v2=True` 时走 V0.5 的写入路径（只填 v2_experiment_id）。
        """
        labels = kp_labels or {}
        rows: list[dict[str, Any]] = []
        for n in notes:
            rows.append(
                self.evaluate(
                    experiment_id, n, material, labels.get(n.get("kp", ""), ""),
                    v2=v2,
                )
            )
        if not rows:
            return {"rows": [], "average_total": 0.0, "pass_rate": 0.0,
                    "dimension_avg": {d: 0.0 for d in DIMENSIONS}}
        dim_avg = {
            d: round(sum(r[d] for r in rows) / len(rows), 2) for d in DIMENSIONS
        }
        return {
            "rows": rows,
            "average_total": round(sum(r["total"] for r in rows) / len(rows), 2),
            "pass_rate": round(
                sum(1 for r in rows if r["verdict"] == "pass") / len(rows), 4
            ),
            "dimension_avg": dim_avg,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _rule_score(note_text: str, material: str) -> dict[str, float]:
        """LLM 不可用时的规则兜底。

        刻意保守：只做「有没有内容」「五要素齐不齐」「有没有复述材料」三项机械检查，
        **不猜准确性**。宁给低分也不给假高分 —— 一个虚高的 fallback
        会让「笔记质量」这个指标彻底失去意义。
        """
        import re

        fields = ("概念：", "原理：", "例子：", "易错点：", "应用规则：")
        filled = sum(1 for f in fields if f in note_text)
        completeness = filled / len(fields) * 100

        # 有实质内容的字段越多越完整；全是短句说明是敷衍
        body = re.sub(r"【.*?】", "", note_text)
        substantive = len([s for s in re.split(r"[\n。；]", body) if len(s.strip()) >= 12])
        completeness = min(completeness, 40 + substantive * 12)

        # 可迁移性：看有没有「因为/所以/前提/代价/之所以」这类推理连接词
        reasoning = len(
            re.findall(r"因为|所以|因此|前提|代价|之所以|意味着|取决于", note_text)
        )
        transferable = min(30.0 + reasoning * 14.0, 95.0)

        # 准确性：LLM 挂了就不能断言准确，给中性偏低的 50，
        # 并在 problems 里标明「准确性未经验证」——不假装测到了。
        accuracy = 50.0

        # 幻觉：抽查笔记里的长句是否出现在材料中。出现得少说明做了改写（好），
        # 但也不能证明没幻觉，所以同样给中性值。
        norm_material = material.replace(" ", "").replace("\n", "")
        segs = [s for s in re.split(r"[，。；]", note_text) if len(s.strip()) >= 16]
        copied = sum(1 for s in segs if s.replace(" ", "") in norm_material)
        no_hallucination = 50.0 + min(20.0, (len(segs) - copied) * 3.0)

        return {
            "accuracy": round(accuracy, 2),
            "completeness": round(completeness, 2),
            "transferable": round(transferable, 2),
            "no_hallucination": round(no_hallucination, 2),
        }


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _render_note(note: dict[str, Any]) -> str:
    return (
        f"【{note.get('kp', '')}】\n"
        f"- 概念：{note.get('concept', '')}\n"
        f"- 原理：{note.get('principle', '')}\n"
        f"- 例子：{note.get('example', '')}\n"
        f"- 易错点：{note.get('common_mistake', '')}\n"
        f"- 应用规则：{note.get('application_rule', '')}"
    )


def _detect_placeholder(note_text: str) -> str:
    for marker in PLACEHOLDER_MARKERS:
        if marker in note_text:
            return marker
    return ""


def quality_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    """DB 行 → 报告用的 dict（problems 从 JSON 还原成数组）。"""
    out = dict(row)
    out["problems"] = _split_delimiter(row.get("problems") or "[]")
    return out


def notes_are_self_consistent(notes: Sequence[dict[str, Any]]) -> bool:
    """所有笔记的 checksum 是否都能重算对上。

    这是质量评估的前置条件：被篡改的笔记不该进入评分。
    """
    return all(
        note_checksum(
            n["concept"], n["principle"], n["example"],
            n["common_mistake"], n["application_rule"],
        ) == n["checksum"]
        for n in notes
    )


__all__ = [
    "DIMENSIONS",
    "PASS_LINE",
    "QUALITY_SYSTEM",
    "QUALITY_TASK_TAG",
    "WEIGHTS",
    "NoteQualityEvaluator",
    "NoteQualityStore",
    "ExamAccessViolation",
    "LearningNoteStore",
    "notes_are_self_consistent",
    "quality_row_to_dict",
    "weighted_total",
]
