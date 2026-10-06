"""数据库层测试：表结构、增删改查、统计。"""

from __future__ import annotations

import pytest

EXPECTED_TABLES = {
    "knowledge",
    "experiences",
    "skills",
    "tasks",
    "evaluations",
    "training_samples",
}


def test_schema_creates_all_tables(db):
    assert EXPECTED_TABLES.issubset(set(db.table_names()))


def test_init_schema_is_idempotent(db):
    db.init_schema()
    db.init_schema()
    assert EXPECTED_TABLES.issubset(set(db.table_names()))


def test_counts_start_at_zero(db):
    for table in EXPECTED_TABLES:
        assert db.count(table) == 0


def test_foreign_keys_enabled(db):
    rows = db.query_all("PRAGMA foreign_keys")
    assert rows[0]["foreign_keys"] == 1


def test_transaction_rollback(db):
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO knowledge(topic,title,content,created_at,updated_at)"
                " VALUES ('t','t','c','now','now')"
            )
            raise RuntimeError("boom")
    assert db.count("knowledge") == 0


def test_transaction_commit(db):
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO knowledge(topic,title,content,created_at,updated_at)"
            " VALUES ('t','t','c','now','now')"
        )
    assert db.count("knowledge") == 1


def test_scalar_and_query_one(db):
    from knowledge.store import KnowledgeStore

    ks = KnowledgeStore(db)
    kid = ks.add("主题", "标题", "正文", ["kw"])
    assert ks.get(kid)["title"] == "标题"
    assert ks.get(9999) is None
    assert db.scalar("SELECT COUNT(*) FROM knowledge WHERE topic = ?", ("主题",)) == 1
