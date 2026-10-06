"""tasks 子系统：统一任务模型 + 出题器。

task_type: learn(学习) | practice(实践) | exam(考试)
stage:     learn | practice | pre_test | post_test
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from brain.llm import LLMProvider
from memory.db import Database
from memory.gate import AccessPolicy

QUESTION_SYSTEM = (
    "你是考官。根据给定主题出题，只输出 JSON："
    '{"items":[{"question":"题目","reference_answer":"参考答案",'
    '"difficulty":"easy|medium|hard"}]}'
)
QUESTION_TASK_TAG = "[[TASK:MAKE_QUESTION]]"

ANSWER_SYSTEM = (
    "你是考生。只依据题目自身的先验知识作答，简洁、分点，不要编造引用来源。"
)
ANSWER_TASK_TAG = "[[TASK:ANSWER]]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TaskStore:
    def __init__(self, db: Database, policy: AccessPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or AccessPolicy(stage="open", blind=False)

    def create(
        self,
        task_type: str,
        topic: str,
        question: str,
        reference_answer: str | None = None,
        difficulty: str = "medium",
        blind: bool = False,
        stage: str = "",
    ) -> int:
        ts = _now()
        cur = self.db.execute(
            """INSERT INTO tasks(task_type, stage, topic, question, reference_answer,
                                 difficulty, blind, status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (task_type, stage, topic, question, reference_answer, difficulty,
             1 if blind else 0, "pending", ts, ts),
        )
        return int(cur.lastrowid)

    def set_answer(self, task_id: int, candidate_answer: str) -> None:
        self.db.execute(
            "UPDATE tasks SET candidate_answer = ?, status = 'answered', updated_at = ?"
            " WHERE id = ?",
            (candidate_answer, _now(), task_id),
        )

    def get(self, task_id: int) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM tasks WHERE id = ?", (task_id,))

    def by_stage(self, stage: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM tasks WHERE stage = ? ORDER BY id LIMIT ?", (stage, limit)
        )

    def by_topic(self, topic: str, task_type: str | None = None, limit: int = 100):
        if task_type:
            return self.db.query_all(
                "SELECT * FROM tasks WHERE topic = ? AND task_type = ? ORDER BY id LIMIT ?",
                (topic, task_type, limit),
            )
        return self.db.query_all(
            "SELECT * FROM tasks WHERE topic = ? ORDER BY id LIMIT ?", (topic, limit)
        )

    def count(self) -> int:
        return self.db.count("tasks")


class QuestionBuilder:
    """出题 + 作答。blind=True 时不注入任何知识内容。"""

    def __init__(self, db: Database, tasks: TaskStore, provider: LLMProvider) -> None:
        self.db = db
        self.tasks = tasks
        self.provider = provider

    def make_questions(
        self, topic: str, count: int = 1, blind: bool = False, stage: str = ""
    ) -> list[dict[str, Any]]:
        user = f"{QUESTION_TASK_TAG}\n主题：{topic}\n请出 {count} 道题。"
        items: list[dict[str, Any]] = []
        try:
            data = self.provider.ask_json(QUESTION_SYSTEM, user, temperature=0.6)
            raw = data.get("items", data if isinstance(data, list) else [])
            items = [i for i in raw if i.get("question")]
        except ValueError:
            items = []
        if not items:
            items = [
                {
                    "question": f"请阐述「{topic}」的核心概念与实践要点。",
                    "reference_answer": f"围绕 {topic} 的定义、原理、实践三方面作答。",
                    "difficulty": "medium",
                }
            ][:count]
        out = []
        for i in items[:count]:
            tid = self.tasks.create(
                task_type="exam",
                stage=stage,
                topic=topic,
                question=i["question"],
                reference_answer=i.get("reference_answer"),
                difficulty=i.get("difficulty", "medium"),
                blind=blind,
            )
            out.append(
                {
                    "id": tid,
                    "question": i["question"],
                    "reference_answer": i.get("reference_answer"),
                    "difficulty": i.get("difficulty", "medium"),
                }
            )
        return out

    def answer(self, question: str, context: str = "") -> str:
        """context 仅在非盲模式传入知识片段。"""
        user = f"{ANSWER_TASK_TAG}\n题目：{question}"
        if context:
            user += f"\n参考资料：\n{context}"
        return self.provider.ask_text(ANSWER_SYSTEM, user, temperature=0.4).strip()
