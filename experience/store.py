"""experience 子系统：经验沉淀。

每次评估后把「场景 / 动作 / 结果 / 教训」落库，下次可以检索复用。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from brain.llm import LLMProvider
from memory.db import Database
from memory.gate import RESOURCE_EXPERIENCES, AccessPolicy

REFLECT_SYSTEM = (
    "你是经验沉淀助手。根据一次任务作答与评估结果，输出一句话教训。"
    '必须只输出 JSON：{"lesson":"教训","action":"下次怎么做"}'
)
REFLECT_TASK_TAG = "[[TASK:REFLECT]]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ExperienceStore:
    """经验读写。

    V0.2：读路径也过闸门。经验里含有具体答法与得分提示，
    严格考试模式下必须禁读，否则等于变相泄题。
    写入不设限（考试期间沉淀自己的失败经验是允许的）。
    """

    def __init__(self, db: Database, policy: AccessPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or AccessPolicy(stage="open", blind=False)

    def add(
        self,
        scenario: str,
        action: str = "",
        result: str = "",
        lesson: str = "",
        score: float | None = None,
        stage: str = "",
        task_id: int | None = None,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO experiences(task_id, stage, scenario, action, result,
                                      score, lesson, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (task_id, stage, scenario, action, result, score, lesson, _now()),
        )
        return int(cur.lastrowid)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        self.policy.guard(RESOURCE_EXPERIENCES)
        return self.db.query_all(
            "SELECT * FROM experiences ORDER BY id DESC LIMIT ?", (limit,)
        )

    def by_stage(self, stage: str) -> list[dict[str, Any]]:
        self.policy.guard(RESOURCE_EXPERIENCES)
        return self.db.query_all(
            "SELECT * FROM experiences WHERE stage = ? ORDER BY id DESC", (stage,)
        )

    def count(self) -> int:
        self.policy.guard(RESOURCE_EXPERIENCES)
        return self.db.count("experiences")


class ExperienceReflector:
    """把评估结果转成结构化经验；LLM 不可用时用规则兜底。"""

    def __init__(self, db: Database, store: ExperienceStore, provider: LLMProvider) -> None:
        self.db = db
        self.store = store
        self.provider = provider

    def reflect(
        self, stage: str, question: str, answer: str, score: float, task_id: int | None = None
    ) -> dict[str, Any]:
        lesson, action = self._llm_reflect(question, answer, score)
        eid = self.store.add(
            scenario=question,
            action=answer,
            result=f"score={score:.1f}",
            lesson=lesson,
            score=score,
            stage=stage,
            task_id=task_id,
        )
        return {"id": eid, "lesson": lesson, "action": action, "score": score}

    def _llm_reflect(self, question: str, answer: str, score: float) -> tuple[str, str]:
        user = (
            f"{REFLECT_TASK_TAG}\n题目：{question}\n"
            f"我的作答：{answer[:500]}\n得分：{score:.1f}"
        )
        try:
            data = self.provider.ask_json(REFLECT_SYSTEM, user, temperature=0.3)
            return str(data.get("lesson", "")).strip(), str(data.get("action", "")).strip()
        except ValueError:
            if score < 60:
                return "作答不完整，知识点覆盖不足", "下次先列提纲再作答"
            return "作答结构清晰，可继续压缩表达", "下次控制在要点式回答"
