"""training_data 子系统：由知识生成问答样本。

样本是「带答案的练习题」。严格考试模式下禁止读取——否则等于开卷考试。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from brain.llm import LLMProvider
from knowledge.store import KnowledgeStore
from memory.db import Database
from memory.gate import RESOURCE_SAMPLES, AccessPolicy

GEN_SYSTEM = (
    "你是训练数据生成器。根据给定知识点生成练习题。"
    "必须只输出 JSON，格式："
    '{"samples":[{"question":"题目","answer":"参考答案","difficulty":"easy|medium|hard"}]}'
)
GEN_TASK_TAG = "[[TASK:GEN_SAMPLES]]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrainingSampleStore:
    def __init__(self, db: Database, policy: AccessPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or AccessPolicy(stage="open", blind=False)

    def add(
        self,
        topic: str,
        question: str,
        answer: str,
        source_knowledge_id: int | None = None,
        difficulty: str = "medium",
    ) -> int:
        self.policy.guard(RESOURCE_SAMPLES)
        cur = self.db.execute(
            """INSERT INTO training_samples(topic, question, answer, source_knowledge_id,
                                           difficulty, created_at)
               VALUES (?,?,?,?,?,?)""",
            (topic, question, answer, source_knowledge_id, difficulty, _now()),
        )
        return int(cur.lastrowid)

    def list_by_topic(self, topic: str, limit: int = 50) -> list[dict[str, Any]]:
        self.policy.guard(RESOURCE_SAMPLES)
        return self.db.query_all(
            "SELECT * FROM training_samples WHERE topic = ? ORDER BY id DESC LIMIT ?",
            (topic, limit),
        )

    def count(self) -> int:
        self.policy.guard(RESOURCE_SAMPLES)
        return self.db.count("training_samples")


class SampleGenerator:
    """用 LLM 把知识条目转成训练样本。LLM 失败时降级为模板生成。"""

    def __init__(
        self,
        db: Database,
        knowledge: KnowledgeStore,
        store: TrainingSampleStore,
        provider: LLMProvider,
    ) -> None:
        self.db = db
        self.knowledge = knowledge
        self.store = store
        self.provider = provider

    def generate(self, topic: str, limit: int = 3) -> list[dict[str, Any]]:
        entries = self.knowledge.list_by_topic(topic, limit=limit * 2)
        if not entries:
            return []
        material = "\n".join(
            f"- [{e['title']}] {e['content']}" for e in entries
        )
        user = f"{GEN_TASK_TAG}\n知识点：\n{material}\n请生成 {limit} 道练习题。"
        samples: list[dict[str, Any]] = []
        try:
            data = self.provider.ask_json(GEN_SYSTEM, user, temperature=0.4)
            raw = data.get("samples", data if isinstance(data, list) else [])
            for item in raw:
                q, a = item.get("question"), item.get("answer")
                if q and a:
                    samples.append(
                        {
                            "question": q,
                            "answer": a,
                            "difficulty": item.get("difficulty", "medium"),
                        }
                    )
        except ValueError:
            samples = []
        if not samples:  # 降级：模板化
            samples = [
                {
                    "question": f"请解释「{e['title']}」的核心内容。",
                    "answer": e["content"],
                    "difficulty": "medium",
                }
                for e in entries[:limit]
            ]
        out = []
        for s in samples[:limit]:
            sid = self.store.add(
                topic=topic,
                question=s["question"],
                answer=s["answer"],
                difficulty=s.get("difficulty", "medium"),
            )
            out.append({"id": sid, **s, "topic": topic})
        return out
