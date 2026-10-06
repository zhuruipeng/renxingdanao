"""pytest 公共夹具：内存数据库 + 离线ScriptedProvider。"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.agent import Agent  # noqa: E402
from brain.llm import ScriptedProvider  # noqa: E402
from memory.db import Database  # noqa: E402

TOPIC = "SQLite 索引原理"

# 离线脚本：按 [[TASK:KEY]] 标记返回预设 JSON，无需联网、不烧 token
SCRIPT = {
    "LEARN": json.dumps(
        {
            "items": [
                {
                    "title": "B-Tree 索引结构",
                    "content": "SQLite 采用 B-Tree 存储索引，所有叶子节点在同一深度，"
                    "因此查询路径长度稳定。聚簇索引的数据即表本身。",
                    "keywords": ["B-Tree", "索引"],
                },
                {
                    "title": "覆盖索引",
                    "content": "当查询所需的全部列都包含在索引中时，无需回表读取数据行，"
                    "称为覆盖索引，可显著减少随机 IO。",
                    "keywords": ["覆盖索引", "回表"],
                },
            ]
        },
        ensure_ascii=False,
    ),
    "GEN_SAMPLES": json.dumps(
        {
            "samples": [
                {
                    "question": "什么是覆盖索引？",
                    "answer": "查询所需列全部在索引中，无需回表。",
                    "difficulty": "easy",
                },
                {
                    "question": "B-Tree 为什么查询稳定？",
                    "answer": "所有叶子节点在同一深度。",
                    "difficulty": "medium",
                },
            ]
        },
        ensure_ascii=False,
    ),
    "MAKE_QUESTION": json.dumps(
        {
            "items": [
                {
                    "question": "简述 SQLite 索引的工作原理。",
                    "reference_answer": "B-Tree 结构；聚簇索引；覆盖索引减少回表。",
                    "difficulty": "medium",
                }
            ]
        },
        ensure_ascii=False,
    ),
    "ANSWER": "SQLite 用 B-Tree 组织索引，叶子节点同深度，查询路径稳定。",
    "EVALUATE": json.dumps(
        {
            "score": 82,
            "verdict": "pass",
            "feedback": "要点基本覆盖，表达略简。",
            "missing_points": ["回表代价"],
        },
        ensure_ascii=False,
    ),
    "REFLECT": json.dumps(
        {"lesson": "结构化作答比堆砌术语有效", "action": "先列提纲再展开"},
        ensure_ascii=False,
    ),
}


@pytest.fixture()
def db(tmp_path) -> Database:
    database = Database(str(tmp_path / "test_agent.db"))
    database.init_schema()
    yield database
    database.close()


@pytest.fixture()
def provider() -> ScriptedProvider:
    return ScriptedProvider(dict(SCRIPT))


@pytest.fixture()
def agent(db: Database, provider: ScriptedProvider) -> Agent:
    return Agent(db, provider)
