"""knowledge 子系统：知识条目存储与检索。

检索用纯 SQL LIKE + 关键词打分，不引入向量数据库（V0.1 明确排除）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from memory.db import Database
from memory.gate import RESOURCE_KNOWLEDGE, AccessPolicy


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeStore:
    """知识库读写。所有方法都先过 policy.guard('knowledge')。"""

    def __init__(self, db: Database, policy: AccessPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or AccessPolicy.open_mode()

    def add(
        self,
        topic: str,
        title: str,
        content: str,
        keywords: list[str] | None = None,
        source: str = "llm",
        confidence: float = 1.0,
        experiment_id: int | None = None,
    ) -> int:
        self.policy.guard(RESOURCE_KNOWLEDGE)
        ts = _now()
        cur = self.db.execute(
            """INSERT INTO knowledge(topic, title, content, keywords, source, confidence,
                                    created_at, updated_at, experiment_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (topic, title, content, json.dumps(keywords or [], ensure_ascii=False),
             source, confidence, ts, ts, experiment_id),
        )
        return int(cur.lastrowid)

    def get(self, kid: int) -> dict[str, Any] | None:
        self.policy.guard(RESOURCE_KNOWLEDGE)
        return self.db.query_one("SELECT * FROM knowledge WHERE id = ?", (kid,))

    def list_by_topic(self, topic: str, limit: int = 50) -> list[dict[str, Any]]:
        self.policy.guard(RESOURCE_KNOWLEDGE)
        return self.db.query_all(
            "SELECT * FROM knowledge WHERE topic = ? ORDER BY id DESC LIMIT ?",
            (topic, limit),
        )

    def for_experiment(self, experiment_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """按实验读取知识 —— OPEN_BOOK_MEMORY 用它。

        作用域强制：policy 有 experiment_id 时必须匹配，
        防止开卷考试读到别的实验产出的知识。
        """
        if self.policy.blind:
            self.policy.guard_scope(RESOURCE_KNOWLEDGE, experiment_id)
        else:
            self.policy.guard(RESOURCE_KNOWLEDGE)
        return self.db.query_all(
            "SELECT * FROM knowledge WHERE experiment_id = ? ORDER BY id LIMIT ?",
            (experiment_id, limit),
        )

    def all(self, limit: int = 200) -> list[dict[str, Any]]:
        self.policy.guard(RESOURCE_KNOWLEDGE)
        return self.db.query_all("SELECT * FROM knowledge ORDER BY id DESC LIMIT ?", (limit,))

    def search(self, topic: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """关键词检索：标题/正文命中 + 关键词命中，按相关度排序。"""
        self.policy.guard(RESOURCE_KNOWLEDGE)
        rows = self.db.query_all(
            "SELECT * FROM knowledge WHERE topic = ? ORDER BY id DESC", (topic,)
        )
        terms = [t for t in _tokenize(query) if t]
        if not terms:
            return rows[:limit]
        scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            hay = f"{r['title']} {r['content']} {r['keywords']}".lower()
            score = 0.0
            for t in terms:
                if t in hay:
                    score += 2.0 if t in r["title"].lower() else 1.0
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: (-x[0], -x[1]["id"]))
        return [r for _, r in scored[:limit]]

    def count(self) -> int:
        self.policy.guard(RESOURCE_KNOWLEDGE)
        return self.db.count("knowledge")


def _tokenize(text: str) -> list[str]:
    """极简分词：英文按词、中文按 2-gram。够用且无依赖。"""
    import re

    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    cjk = re.findall(r"[一-鿿]+", text)
    for run in cjk:
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens
