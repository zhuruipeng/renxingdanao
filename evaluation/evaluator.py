"""evaluation 子系统：独立评估器。

关键点：**评估器与答题器分离**。
- 答题：ANSWER_SYSTEM（考生人格）
- 评估：EVAL_SYSTEM（考官人格）
两者用不同的 system prompt、不同的 LLM 调用，评估器只看题目+参考答案+考生答案，
不看考生答题时的思维过程，避免自评虚高。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from brain.llm import LLMProvider
from memory.db import Database

EVAL_SYSTEM = (
    "你是严格独立的评估员。对照参考答案给考生答案打分（0-100）。"
    "只看要点覆盖与正确性，不看文采。只输出 JSON："
    '{"score":85,"verdict":"pass|fail","feedback":"简短评语",'
    '"missing_points":["缺失要点"]}'
)
EVAL_TASK_TAG = "[[TASK:EVALUATE]]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EvaluationStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(
        self,
        task_id: int | None,
        score: float,
        verdict: str = "",
        feedback: str = "",
        evaluator: str = "llm_judge",
        max_score: float = 100.0,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO evaluations(task_id, evaluator, score, max_score, verdict,
                                       feedback, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (task_id, evaluator, score, max_score, verdict, feedback, _now()),
        )
        return int(cur.lastrowid)

    def for_task(self, task_id: int) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM evaluations WHERE task_id = ? ORDER BY id", (task_id,)
        )

    def average(self, task_ids: list[int]) -> float:
        if not task_ids:
            return 0.0
        marks = ",".join("?" * len(task_ids))
        return float(
            self.db.scalar(
                f"SELECT AVG(score) FROM evaluations WHERE task_id IN ({marks})", task_ids
            )
            or 0.0
        )

    def count(self) -> int:
        return self.db.count("evaluations")


class Evaluator:
    """独立评估：LLM 判分 + 规则兜底。"""

    def __init__(self, db: Database, store: EvaluationStore, provider: LLMProvider) -> None:
        self.db = db
        self.store = store
        self.provider = provider

    def evaluate(
        self,
        task_id: int,
        question: str,
        candidate_answer: str,
        reference_answer: str | None = None,
    ) -> dict[str, Any]:
        user = (
            f"{EVAL_TASK_TAG}\n题目：{question}\n"
            f"参考答案：{reference_answer or '（无）'}\n"
            f"考生答案：{candidate_answer}"
        )
        data: dict[str, Any] | None = None
        try:
            data = self.provider.ask_json(EVAL_SYSTEM, user, temperature=0.2)
        except ValueError:
            data = None
        if not isinstance(data, dict) or "score" not in data:
            score, verdict, feedback = self._rule_score(
                candidate_answer, reference_answer
            )
            missing: list[str] = []
        else:
            score = max(0.0, min(100.0, float(data.get("score", 0))))
            verdict = str(data.get("verdict") or ("pass" if score >= 60 else "fail"))
            feedback = str(data.get("feedback", ""))
            missing = list(data.get("missing_points") or [])
        eid = self.store.add(
            task_id=task_id,
            score=score,
            verdict=verdict,
            feedback=feedback,
        )
        self.db.execute(
            "UPDATE tasks SET status = 'evaluated', updated_at = ? WHERE id = ?",
            (_now(), task_id),
        )
        return {
            "evaluation_id": eid,
            "task_id": task_id,
            "score": score,
            "verdict": verdict,
            "feedback": feedback,
            "missing_points": missing,
        }

    @staticmethod
    def _rule_score(candidate: str, reference: str | None) -> tuple[float, str, str]:
        """LLM 不可用时的兜底：按参考答案的要点覆盖率打分。"""
        if not reference:
            return (50.0, "fail", "无参考答案，兜底计分")
        keys = [k for k in _split_points(reference) if k]
        if not keys:
            return (60.0, "pass", "参考答案无法切分，兜底计分")
        hit = sum(1 for k in keys if k in candidate)
        score = round(100.0 * hit / len(keys), 1)
        verdict = "pass" if score >= 60 else "fail"
        return (score, verdict, f"规则兜底：命中 {hit}/{len(keys)} 个要点")


def _split_points(text: str) -> list[str]:
    import re

    parts = re.split(r"[。；;，,\n\r、\d\.\)）]+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 2]
