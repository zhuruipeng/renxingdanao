"""learning_notes 子系统：结构化学习笔记。

V0.3 的核心。笔记是「学习阶段 Agent 自己写出来的二次加工产物」，
结构固定为五要素：

    concept          这是什么
    principle        为什么是这样（底层机制）
    example          具体例子
    common_mistake   容易错在哪
    application_rule 什么时候用、怎么用

**为什么必须是结构化笔记，而不是原始材料切片？**

如果考试时放行原始材料全文，等于把答案抄给模型——它只需要复制粘贴，
测出来的是「抄写能力」而不是「学习效果」。V0.2 的实验（0% 提升）
已经说明这条路走不通。

结构化笔记强制 Agent 做二次加工：抽概念、说原理、举例子、指误区、给规则。
考场上放行笔记、放行材料，等于测「Agent 能否利用自己加工过的外部记忆」。

**checksum 的作用**：五个要素拼起来算SHA256，落在库里。
除了防篡改，还能在测试里断言「笔记内容确实变了」——
防止有人用「写点什么糊弄过去」的方式骗过流程。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from brain.llm import LLMProvider
from memory.db import Database, v2_experiment_key
from memory.gate import (
    RESOURCE_KNOWLEDGE,
    RESOURCE_NOTES,
    AccessPolicy,
    ExamAccessViolation,
)

NOTE_SYSTEM = (
    "你是学习助手。阅读材料后，为每个知识点写一份结构化学习笔记。"
    "笔记必须用自己的话重新组织，不要整段复制材料原文。"
    "只输出 JSON："
    '{"notes":[{"kp":"知识点id","concept":"这是什么（一两句）",'
    '"principle":"为什么是这样，讲底层机制（100-300字）",'
    '"example":"一个具体例子或场景",'
    '"common_mistake":"最容易犯的错是什么",'
    '"application_rule":"什么时候用、怎么用（可操作）"}]}'
)
NOTE_TASK_TAG = "[[TASK:LEARNING_NOTES]]"

#: 笔记五要素，顺序即checksum 拼接顺序
NOTE_FIELDS = ("concept", "principle", "example", "common_mistake", "application_rule")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def note_checksum(
    concept: str, principle: str, example: str, common_mistake: str, application_rule: str
) -> str:
    """五要素拼接后的 SHA256。"""
    payload = "\x1f".join(
        [concept, principle, example, common_mistake, application_rule]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LearningNoteStore:
    """学习笔记存储。

    读路径全部过 policy +作用域检查：
    - BLIND_MODEL 完全禁读
    - OPEN_BOOK_MEMORY 可读，但只能读本experiment_id 的笔记
    """

    def __init__(self, db: Database, policy: AccessPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or AccessPolicy.open_mode()

    def add(
        self,
        experiment_id: int,
        topic: str,
        kp: str,
        concept: str,
        principle: str,
        example: str = "",
        common_mistake: str = "",
        application_rule: str = "",
        *,
        v2: bool = False,
    ) -> int:
        # 写入只允许发生在非考试场景。
        # 开卷考试虽然「放行」笔记的读，但放行的是**学习阶段固化**的那份；
        # 若考试期间还能写，就等于考生边考边改小抄，OPEN_BOOK_GAIN 立刻失去意义。
        self.policy.guard(RESOURCE_NOTES)
        if self.policy.blind:
            raise ExamAccessViolation(
                f"考试模式 {self.policy.name} 禁止写入 {RESOURCE_NOTES}；"
                "笔记必须在学习阶段固化，考试期间只能读不能改。"
            )
        cks = note_checksum(concept, principle, example, common_mistake, application_rule)
        if v2:
            # experiment_id 写**负数哨兵**（见 memory/db.py 的说明）：
            # V2 的 id 属于另一套空间，直接写正数会挂到同号的 V0.4 实验名下，
            # 数字撞上即串数据，而且不会报任何错。
            cur = self.db.execute(
                """INSERT INTO learning_notes(experiment_id, v2_experiment_id, topic, kp,
                                              concept, principle, example,
                                              common_mistake, application_rule,
                                              checksum, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (v2_experiment_key(experiment_id), experiment_id, topic, kp,
                 concept, principle, example, common_mistake, application_rule,
                 cks, _now()),
            )
            return int(cur.lastrowid)
        cur = self.db.execute(
            """INSERT INTO learning_notes(experiment_id, topic, kp, concept, principle,
                                          example, common_mistake, application_rule,
                                          checksum, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (experiment_id, topic, kp, concept, principle, example,
             common_mistake, application_rule, cks, _now()),
        )
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    # 读路径：必须同时过资源闸门 + experiment_id 作用域
    # ------------------------------------------------------------------
    def get(self, note_id: int) -> dict[str, Any] | None:
        self.policy.guard(RESOURCE_NOTES)
        return self.db.query_one("SELECT * FROM learning_notes WHERE id = ?", (note_id,))

    def for_experiment(self, experiment_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """按 V0.4 实验读取。考试模式下强制校验 experiment_id 匹配。"""
        self.policy.guard_scope(RESOURCE_NOTES, experiment_id)
        return self.db.query_all(
            "SELECT * FROM learning_notes WHERE experiment_id = ? ORDER BY id LIMIT ?",
            (experiment_id, limit),
        )

    def for_v2_experiment(self, v2_experiment_id: int, limit: int = 50) -> list[dict[str, Any]]:
        """按 V0.5 实验读取。作用域检查用同一个 policy。"""
        self.policy.guard_scope(RESOURCE_NOTES, v2_experiment_id)
        return self.db.query_all(
            "SELECT * FROM learning_notes WHERE v2_experiment_id = ? ORDER BY id LIMIT ?",
            (v2_experiment_id, limit),
        )

    def for_kp(self, experiment_id: int, kp: str) -> dict[str, Any] | None:
        self.policy.guard_scope(RESOURCE_NOTES, experiment_id)
        return self.db.query_one(
            "SELECT * FROM learning_notes WHERE experiment_id = ? AND kp = ?"
            " ORDER BY id LIMIT 1",
            (experiment_id, kp),
        )

    def count(self) -> int:
        self.policy.guard(RESOURCE_NOTES)
        return self.db.count("learning_notes")

    def verify_checksum(self, note_id: int) -> bool:
        """重算 checksum 与库中记录比对，检测是否被篡改。"""
        row = self.get(note_id)
        if row is None:
            return False
        expect = note_checksum(
            row["concept"], row["principle"], row["example"],
            row["common_mistake"], row["application_rule"],
        )
        return expect == row["checksum"]

    # ------------------------------------------------------------------
    # 质量字段回写（V0.4）
    # ------------------------------------------------------------------
    def set_quality(self, note_id: int, total: float, verdict: str) -> None:
        """把独立评估器给出的质量分回写到笔记行。

        之所以**回写一份**而不是每次都去 join note_quality 表：
        开卷渲染时经常需要「这份笔记质量如何」的快速判断，
        而 note_quality 可能对同一份笔记有多次评估（不同实验重跑）。
        回写的语义是「最近一次评估结果」，权威明细仍在 note_quality 表。
        """
        self.policy.guard(RESOURCE_NOTES)
        if self.policy.blind:
            raise ExamAccessViolation(
                f"考试模式 {self.policy.name} 禁止修改 {RESOURCE_NOTES} 的质量字段"
            )
        self.db.execute(
            "UPDATE learning_notes SET quality_total = ?, quality_verdict = ? WHERE id = ?",
            (float(total), verdict, note_id),
        )

    def quality_average(self, experiment_id: int) -> float:
        """本实验笔记的平均质量分。无评估记录时返回 0.0。"""
        self.policy.guard(RESOURCE_NOTES)
        row = self.db.query_one(
            "SELECT AVG(quality_total) AS a FROM learning_notes"
            " WHERE experiment_id = ? AND quality_total IS NOT NULL",
            (experiment_id,),
        )
        if row is None or row["a"] is None:
            return 0.0
        return round(float(row["a"]), 2)


def render_notes_for_exam(notes: list[dict[str, Any]]) -> str:
    """把笔记渲染成可进prompt 的文本。

    只渲染五要素，**不渲染任何参考答案、原材料、题目历史**。
    格式刻意标注这是「你自己的笔记」，让考生知道信息来源是外部记忆。
    """
    if not notes:
        return ""
    blocks: list[str] = []
    for n in notes:
        blocks.append(
            f"【{n['kp']}】\n"
            f"- 概念：{n['concept']}\n"
            f"- 原理：{n['principle']}\n"
            f"- 例子：{n['example']}\n"
            f"- 易错点：{n['common_mistake']}\n"
            f"- 应用规则：{n['application_rule']}"
        )
    return "以下是你自己此前整理的学习笔记：\n\n" + "\n\n".join(blocks)


class NoteGenerator:
    """从学习材料生成结构化笔记。LLM 失效时按知识点降级。"""

    def __init__(
        self, db: Database, provider: LLMProvider, kps: list[str]
    ) -> None:
        self.db = db
        self.provider = provider
        self.kps = kps

    def generate(
        self, store: LearningNoteStore, experiment_id: int, topic: str, material: str,
        *, v2: bool = False,
    ) -> list[dict[str, Any]]:
        user = (
            f"{NOTE_TASK_TAG}\n学习材料：\n{material}\n"
            f"请为这些知识点写笔记：{', '.join(self.kps)}"
        )
        try:
            data = self.provider.ask_json(NOTE_SYSTEM, user, temperature=0.3)
        except ValueError:
            data = None
        items: list[dict[str, Any]] = []
        if isinstance(data, dict):
            for n in (data.get("notes") or []):
                if n.get("concept") and n.get("principle"):
                    items.append(n)
        if not items:  # 降级：每个 kp 一条模板笔记
            items = [
                {
                    "kp": kp,
                    "concept": f"{kp} 的基本概念",
                    "principle": material[:200],
                    "example": "（模型不可用，示例降级）",
                    "common_mistake": "（模型不可用，易错点降级）",
                    "application_rule": "（模型不可用，应用规则降级）",
                }
                for kp in self.kps
            ]
        out: list[dict[str, Any]] = []
        for it in items:
            kp = str(it.get("kp") or self.kps[0])
            concept = str(it.get("concept", ""))
            principle = str(it.get("principle", ""))
            example = str(it.get("example", ""))
            mistake = str(it.get("common_mistake", ""))
            rule = str(it.get("application_rule", ""))
            nid = store.add(
                experiment_id=experiment_id,
                topic=topic,
                kp=kp,
                concept=concept,
                principle=principle,
                example=example,
                common_mistake=mistake,
                application_rule=rule,
                v2=v2,
            )
            row = store.get(nid)
            assert row is not None
            out.append(row)
        return out
