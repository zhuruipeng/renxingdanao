"""skills 子系统：能力项与升级。

技能是「我会不会某类题」的度量，由评估分数驱动升级。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memory.db import Database

# 升一级的分数线（满分 100）
LEVEL_THRESHOLDS = (85.0, 70.0, 55.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SkillStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def ensure(self, name: str, description: str = "") -> int:
        row = self.db.query_one("SELECT id FROM skills WHERE name = ?", (name,))
        if row:
            return int(row["id"])
        ts = _now()
        cur = self.db.execute(
            "INSERT INTO skills(name, description, level, xp, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (name, description, 1, 0, ts, ts),
        )
        return int(cur.lastrowid)

    def record_score(self, name: str, score: float, xp_gain: int = 10) -> dict[str, Any]:
        """记录一次表现，按分数升级 / 扣经验。"""
        sid = self.ensure(name)
        row = self.db.query_one("SELECT * FROM skills WHERE id = ?", (sid,))
        assert row is not None
        level, xp = int(row["level"]), int(row["xp"])
        xp += xp_gain
        if score >= 90:
            level, xp = level + 1, xp  # 满分表现直接升级
        elif score < 40:
            level = max(1, level - 1)
        self.db.execute(
            "UPDATE skills SET level = ?, xp = ?, updated_at = ? WHERE id = ?",
            (level, xp, _now(), sid),
        )
        return {"id": sid, "name": name, "level": level, "xp": xp, "score": score}

    def list_all(self) -> list[dict[str, Any]]:
        return self.db.query_all("SELECT * FROM skills ORDER BY level DESC, xp DESC")

    def get(self, name: str) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM skills WHERE name = ?", (name,))

    def count(self) -> int:
        return self.db.count("skills")
