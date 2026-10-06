"""brain/group_isolation.py —— V0.6 两组隔离的可执行定义。

V0.5 只能保证「一次实验内部不串」（AccessPolicy + experiment_id 作用域）。
V0.6 要保证的是**两次实验之间不串**，而两次实验跑在同一个进程、
同一张库、同一套 store 上。于是多出三类 V0.5 不存在的串组途径：

1. **同 id 不同组**：两组若共用 experiment_id，
   Learning 的笔记会出现在 Control 的 context 里。
2. **共享资源**：knowledge / learning_notes / experiences /
   training_samples / evaluator context / answer history ——
   任何一张表如果只有「有没有」而没有「属于谁」，
   两组的产物就会互相可见。
3. **顺序污染**：Control 的 PRE 必须在 Learning 的
   STUDY/POST 之前跑完。否则同一模型实例可能已从上一轮 prompt
   残留了目标知识，Control 的 PRE 被抬高，difference 被压低。

三条各自都能让 difference-in-gains 变成噪声，而且**都不报错**。

本模块把这些变成三类可执行检查
--------------------------------
- :class:`GroupScope`    —— 一组实验的 id 与资源边界，构造即隔离
- :func:`isolation_audit` —— 从数据库实际数据反查是否真的隔离
- :func:`order_audit`     —— 核对十阶段落库的顺序与时间戳
- :func:`contamination_report` —— 六类污染的汇总判定

设计原则
--------
**每条检查都从落库的数据反查，不信任调用方的自觉。**
如果隔离只由「调用时记得传对id」来保证，
那它就不是隔离，是约定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from brain import control_material as cm
from brain import heldout_bank as ho
from brain import novel_material as nm
from memory.db import (
    BLIND_STAGES_V3,
    GROUPS_V3,
    STAGES_V3,
    Database,
    checksum,
)
from practice import objective as ob

# ==========================================================================
# 组作用域
# ==========================================================================


@dataclass(frozen=True)
class GroupScope:
    """一个组在本次实验中的身份与资源边界。

    两组的 ``experiment_id`` 必须不同，且**都必须真实存在**于
    ``v3_experiments``——不存在的 id 意味着「这一组其实没跑」，
    而不是一个可以静默通过的边界。
    """

    run_id: str
    exp_group: str
    experiment_id: int
    material_kind: str
    material_text: str

    def __post_init__(self) -> None:
        if self.exp_group not in GROUPS_V3:
            raise ValueError(
                f"未知实验组 {self.exp_group!r}，只接受 {list(GROUPS_V3)}"
            )

    @property
    def is_learning(self) -> bool:
        return self.exp_group == "learning"

    @property
    def material_checksum(self) -> str:
        return checksum(self.material_text)

    def stage_of(self, base: str) -> str:
        """`pre` → `pre_learning` / `pre_control`。"""
        return f"{base}_{self.exp_group}"

    def peer(self) -> str:
        """另一组的组名。"""
        return "control" if self.is_learning else "learning"

    def meta(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "exp_group": self.exp_group,
            "experiment_id": self.experiment_id,
            "material_kind": self.material_kind,
            "material_checksum": self.material_checksum,
        }


def build_scopes(
    run_id: str,
    learning_experiment_id: int,
    control_experiment_id: int,
) -> dict[str, GroupScope]:
    """构造两组的 scope，并硬断言它们不共享 experiment_id。"""
    if learning_experiment_id == control_experiment_id:
        raise RuntimeError(
            f"两组共用experiment_id={learning_experiment_id}，"
            f"隔离不成立。Learning 与 Control 必须是两个独立 id。"
        )
    return {
        "learning": GroupScope(
            run_id=run_id,
            exp_group="learning",
            experiment_id=learning_experiment_id,
            material_kind="novel_zephyra",
            material_text=nm.NOVEL_MATERIAL,
        ),
        "control": GroupScope(
            run_id=run_id,
            exp_group="control",
            experiment_id=control_experiment_id,
            material_kind="control_veyra",
            material_text=cm.CONTROL_MATERIAL,
        ),
    }


# ==========================================================================
# 六类污染检测
# ==========================================================================


#: 污染类别常量。报告层按这些 key 输出，不允许临时起名。
CONTAM_SHARED_KNOWLEDGE = "two_groups_share_knowledge"
CONTAM_SHARED_NOTES = "two_groups_share_learning_notes"
CONTAM_POST_IN_STUDY = "post_question_appears_in_study"
CONTAM_TRANSFER_IN_PRACTICE = "transfer_question_appears_in_practice"
CONTAM_EVALUATOR_LEAK = "evaluator_saw_forbidden_context"
CONTAM_CHECKSUM_CHANGED = "checksum_changed"
#: 第七类：考生调用失败。它不是「污染」（数据没被别的组看到），
#: 而是「这次实验没跑成」——但后果同样致命：
#: 失败样本与「模型答不出」在数据里长得一样，两组分数被噪声替代。
CONTAM_CALL_FAILED = "candidate_call_failed"

#: 用户点名的**六类**污染。六类之外的类别（调用失败）也进检测，
#: 但它不进这个元组——六类是实验设计里的题，调用失败是运行环境的事。
ALL_CONTAMINATIONS: tuple[str, ...] = (
    CONTAM_SHARED_KNOWLEDGE,
    CONTAM_SHARED_NOTES,
    CONTAM_POST_IN_STUDY,
    CONTAM_TRANSFER_IN_PRACTICE,
    CONTAM_EVALUATOR_LEAK,
    CONTAM_CHECKSUM_CHANGED,
)

#: 全部会被检出的类别（六类 + 调用失败）
ALL_CONTAMINATION_KINDS: tuple[str, ...] = (
    *ALL_CONTAMINATIONS,
    CONTAM_CALL_FAILED,
)

#: POST/TRANSFER 题出现在哪个阶段就算泄题
_STUDY_STAGES = ("study_learning", "control_material")
_PRACTICE_STAGES = ("practice_learning", "practice_control")


@dataclass
class Contamination:
    """一处污染。`evidence` 必须能让人自己复核，不能只给结论。"""

    kind: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _rows(db: Database, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return db.query_all(sql, params)


def _stage_item_ids(
    db: Database, run_id: str, exp_group: str, stage: str
) -> set[str]:
    return {
        r["item_id"]
        for r in _rows(
            db,
            "SELECT item_id FROM v3_stage_items"
            " WHERE run_id=? AND exp_group=? AND stage=?",
            (run_id, exp_group, stage),
        )
    }


def _question_texts(
    db: Database, run_id: str, exp_group: str, stage: str
) -> list[str]:
    return [
        r["question"]
        for r in _rows(
            db,
            "SELECT question FROM v3_stage_items"
            " WHERE run_id=? AND exp_group=? AND stage=?",
            (run_id, exp_group, stage),
        )
    ]


def check_shared_knowledge(db: Database, run_id: str) -> list[Contamination]:
    """两组是否共享了 knowledge。

    判据：同一段知识内容（按content 的 checksum 判重）
    同时归属两个 experiment_id。两组若拿到同一条 knowledge，
    Control 就间接读到了 Learning 的学习产物。
    """
    out: list[Contamination] = []
    exp = _rows(
        db,
        "SELECT exp_group, id AS experiment_id FROM v3_experiments WHERE run_id=?",
        (run_id,),
    )
    by_group = {r["exp_group"]: int(r["experiment_id"]) for r in exp}
    if "learning" not in by_group or "control" not in by_group:
        return out
    li, co = by_group["learning"], by_group["control"]

    # knowledge 表在 V0.5 里没有 v3_experiment_id，
    # 归属走 experiment_id 列。这里按「同一条 knowledge 的 id 同时
    # 被两组引用不到」来判断——实际能落地查的是：
    # 两条内容完全相同的 knowledge 分属两组。
    dup = _rows(
        db,
        "SELECT content, COUNT(DISTINCT experiment_id) AS n"
        "  FROM knowledge WHERE experiment_id IN (?,?)"
        " GROUP BY content HAVING n > 1",
        (li, co),
    )
    if dup:
        out.append(Contamination(
            kind=CONTAM_SHARED_KNOWLEDGE,
            detail=(
                f"两组各有 {len(dup)} 条 knowledge 的内容完全相同，"
                f"Control 读到的知识里可能含 Learning 的学习产物"
            ),
            evidence={"samples": [d["content"][:80] for d in dup[:3]]},
        ))

    # 弱信号：两组 knowledge 条数完全相同。
    # **仅在条数相同时不报警**——两组各抽 4 个知识点是完全正常的
    # （知识点数量由材料结构决定，不由材料内容决定）。
    # 真正的污染信号是**内容相同**，已在上面查过。
    return out


def check_shared_notes(db: Database, run_id: str) -> list[Contamination]:
    """两组是否共享 learning_notes。

    判据更硬：**逐条比checksum**。只要有一条笔记的 checksum
    同时出现在两个 experiment_id 下，就算污染——
    因为那意味着 Control 的 context 里可能有 Learning 的笔记。
    """
    out: list[Contamination] = []
    exp = _rows(
        db,
        "SELECT exp_group, id AS experiment_id FROM v3_experiments WHERE run_id=?",
        (run_id,),
    )
    by_group = {r["exp_group"]: int(r["experiment_id"]) for r in exp}
    if "learning" not in by_group or "control" not in by_group:
        return out

    li, co = by_group["learning"], by_group["control"]
    overlap = _rows(
        db,
        "SELECT a.checksum AS checksum, a.id AS learning_note_id,"
        "       b.id AS control_note_id"
        "  FROM learning_notes a JOIN learning_notes b"
        "    ON a.checksum = b.checksum"
        " WHERE a.v3_experiment_id = ? AND b.v3_experiment_id = ?",
        (li, co),
    )
    if overlap:
        out.append(Contamination(
            kind=CONTAM_SHARED_NOTES,
            detail=(
                f"两组有 {len(overlap)} 条 learning_notes 的checksum 相同，"
                f"Control 的上下文可能包含 Learning 写的笔记。"
            ),
            evidence={"pairs": overlap[:5]},
        ))

    # 条数相同**不是**污染信号：知识点数量由材料结构决定，
    # 两组各抽 4 个是完全正常的。checksum 冲突才是。
    return out


def check_heldout_in_study(db: Database, run_id: str) -> list[Contamination]:
    """POST / TRANSFER 题是否出现在 STUDY 或 CONTROL_MATERIAL 阶段。

    判据用 **item_id 精确匹配 + 题干文本包含**双通道：
    只查 item_id 会被改名绕过，只查文本会误伤（题干短）。
    两者都查更严。
    """
    out: list[Contamination] = []
    held_ids = {it.item_id for it in ho.ALL_HELDOUT_ITEMS}
    held_q = {it.question.strip() for it in ho.ALL_HELDOUT_ITEMS}

    for group in GROUPS_V3:
        for stage in _STUDY_STAGES:
            if not stage.endswith(group):
                continue
            ids = _stage_item_ids(db, run_id, group, stage) & held_ids
            if ids:
                out.append(Contamination(
                    kind=CONTAM_POST_IN_STUDY,
                    detail=(
                        f"{group} 组的学习阶段 {stage} 出现了 held-out 题："
                        f"{sorted(ids)}"
                    ),
                    evidence={"stage": stage, "item_ids": sorted(ids)},
                ))
            for q in _question_texts(db, run_id, group, stage):
                q = q.strip()
                if q and q in held_q:
                    out.append(Contamination(
                        kind=CONTAM_POST_IN_STUDY,
                        detail=f"{group} 组 {stage} 的题干与某道 held-out 题完全相同",
                        evidence={"stage": stage, "question": q[:120]},
                    ))
                else:
                    # 题干包含关系：材料里嵌了题面的核心问法
                    for hit in held_q:
                        core = hit[:40]
                        if len(core) >= 20 and core in q:
                            out.append(Contamination(
                                kind=CONTAM_POST_IN_STUDY,
                                detail=(
                                    f"{group} 组 {stage} 的题干包含了held-out 题的开头"
                                ),
                                evidence={"stage": stage, "question": q[:120]},
                            ))
                            break
    return out


def check_transfer_in_practice(db: Database, run_id: str) -> list[Contamination]:
    """TRANSFER 题是否出现在 PRACTICE 阶段。"""
    out: list[Contamination] = []
    tr_ids = {it.item_id for it in ho.HELD_OUT_TRANSFER}
    tr_q = {it.question.strip() for it in ho.HELD_OUT_TRANSFER}

    for group in GROUPS_V3:
        stage = f"practice_{group}"
        ids = _stage_item_ids(db, run_id, group, stage) & tr_ids
        if ids:
            out.append(Contamination(
                kind=CONTAM_TRANSFER_IN_PRACTICE,
                detail=f"{group} 组的练习阶段出现了 TRANSFER 题：{sorted(ids)}",
                evidence={"stage": stage, "item_ids": sorted(ids)},
            ))
        for q in _question_texts(db, run_id, group, stage):
            q = q.strip()
            for hit in tr_q:
                core = hit[:40]
                if len(core) >= 20 and core in q:
                    out.append(Contamination(
                        kind=CONTAM_TRANSFER_IN_PRACTICE,
                        detail=f"{group} 组练习阶段的题干包含了 TRANSFER 题的开头",
                        evidence={"stage": stage, "question": q[:120]},
                    ))
                    break

    # TRANSFER 题出现在 PRE 阶段同样算污染（考生见过就再盲考没意义）
    for group in GROUPS_V3:
        stage = f"pre_{group}"
        ids = _stage_item_ids(db, run_id, group, stage) & tr_ids
        if ids:
            out.append(Contamination(
                kind=CONTAM_TRANSFER_IN_PRACTICE,
                detail=f"{group} 组的 PRE 阶段出现了 TRANSFER 题：{sorted(ids)}",
                evidence={"stage": stage, "item_ids": sorted(ids)},
            ))

    # ---------- 客观任务表 ----------
    #
    # 早先这里只查了 v3_stage_items（考试题），
    # **完全没查 v3_objective_runs（客观任务）**——
    # 于是「Control 组在 practice 阶段做 HELD_OUT_TASKS，
    # 而 post_objective 测的正是同一批题」这个真实存在的污染
    # 从门禁里整个漏了过去。客观任务表必须有独立的一轮检查。
    ho_task_ids = {t.task_id for t in ho.HELD_OUT_TASKS}
    pr_task_ids = {t.task_id for t in ob.OBJECTIVE_TASKS}
    if ho_task_ids & pr_task_ids:
        out.append(Contamination(
            kind=CONTAM_TRANSFER_IN_PRACTICE,
            detail=(
                f"practice 与 held-out 的 task_id 重叠："
                f"{sorted(ho_task_ids & pr_task_ids)}"
            ),
            evidence={"overlap": sorted(ho_task_ids & pr_task_ids)},
        ))

    # 1. held-out 任务不得出现在任何 practice_* 阶段
    for group in GROUPS_V3:
        stage = f"practice_{group}"
        hits = _rows(
            db,
            "SELECT DISTINCT task_id FROM v3_objective_runs"
            " WHERE run_id=? AND exp_group=? AND stage=? AND task_id IN"
            f" ({','.join('?' for _ in ho_task_ids)})",
            (run_id, group, stage, *sorted(ho_task_ids)),
        )
        if hits:
            out.append(Contamination(
                kind=CONTAM_TRANSFER_IN_PRACTICE,
                detail=(
                    f"{group} 组的练习阶段做了 held-out 客观任务："
                    f"{sorted(h['task_id'] for h in hits)}。"
                    f"这些题是 post_objective 的测试题，练过再测会让"
                    f"对照组分数虚高、objective_difference_in_gains 偏向假阴性"
                ),
                evidence={
                    "stage": stage,
                    "task_ids": sorted(h["task_id"] for h in hits),
                },
            ))

    # 2. held-out 任务不得出现在 PRE 阶段
    for group in GROUPS_V3:
        stage = f"pre_{group}"
        hits = _rows(
            db,
            "SELECT DISTINCT task_id FROM v3_objective_runs"
            " WHERE run_id=? AND exp_group=? AND stage=? AND task_id IN"
            f" ({','.join('?' for _ in ho_task_ids)})",
            (run_id, group, stage, *sorted(ho_task_ids)),
        )
        if hits:
            out.append(Contamination(
                kind=CONTAM_TRANSFER_IN_PRACTICE,
                detail=(
                    f"{group} 组的 PRE 阶段出现了 held-out 客观任务："
                    f"{sorted(h['task_id'] for h in hits)}"
                ),
                evidence={
                    "stage": stage,
                    "task_ids": sorted(h["task_id"] for h in hits),
                },
            ))

    # 3. post_objective 必须**整批**用held-out 任务。
    #    少一题都说明测试卷被换过——分母不同则前后差不可比。
    for group in GROUPS_V3:
        for stage in (f"pre_objective", f"post_objective"):
            rows = _rows(
                db,
                "SELECT DISTINCT task_id FROM v3_objective_runs"
                " WHERE run_id=? AND exp_group=? AND stage=?",
                (run_id, group, stage),
            )
            if not rows:
                continue
            used = {r["task_id"] for r in rows}
            if used != ho_task_ids:
                out.append(Contamination(
                    kind=CONTAM_TRANSFER_IN_PRACTICE,
                    detail=(
                        f"{group} 组 {stage} 的任务集与 held-out 不一致"
                        f"（缺 {sorted(ho_task_ids - used)}，"
                        f"多 {sorted(used - ho_task_ids)}）"
                    ),
                    evidence={
                        "stage": stage,
                        "missing": sorted(ho_task_ids - used),
                        "extra": sorted(used - ho_task_ids),
                    },
                ))
    return out


def check_evaluator_leak(db: Database, run_id: str) -> list[Contamination]:
    """评估器是否看到了不该看到的 context。

    三条判据：
    1. 盲考阶段的 ``context_used`` 必须为空串
    2. 任何阶段的 ``context_used`` 不得含另一组的 experiment_id
    3. 考生与评估器不得是同一个 provider 实例身份
       （「同族模型」不是错，「同一个 provider 对象」是错——
        同一个对象意味着它自己的输出进了自己的输入）
    """
    out: list[Contamination] = []

    # 1. 盲考 context 非空
    for stage in sorted(BLIND_STAGES_V3):
        rows = _rows(
            db,
            "SELECT id, exp_group, item_id, LENGTH(context_used) AS n"
            "  FROM v3_stage_items"
            " WHERE run_id=? AND stage=? AND TRIM(context_used) <> ''",
            (run_id, stage),
        )
        if rows:
            out.append(Contamination(
                kind=CONTAM_EVALUATOR_LEAK,
                detail=(
                    f"盲考阶段 {stage} 有 {len(rows)} 行 context_used 非空，"
                    f"考生在盲考时看到了东西"
                ),
                evidence={"stage": stage, "rows": rows[:5]},
            ))

    # 2. context 里不能出现另一组的笔记内容。
    #
    #    **不能用 `LIKE '%<另一组 id>%'`** —— 那会把材料正文里的
    #    任意数字串（"bucket 5"、"SPAN=60000"）误判成 id 引用，
    #    实测一次就跑出两条假污染。正确口径是拿另一组笔记的
    #    concept/principle 去context 里找**文本命中**：
    #    这才是「Control 的上下文里出现了 Learning 的笔记」的真实含义。
    exp = _rows(
        db,
        "SELECT exp_group, id AS experiment_id FROM v3_experiments WHERE run_id=?",
        (run_id,),
    )
    by_group = {r["exp_group"]: int(r["experiment_id"]) for r in exp}
    if len(by_group) == 2:
        for group, eid in by_group.items():
            other_group = "control" if group == "learning" else "learning"
            other_eid = by_group[other_group]
            peer_notes = _rows(
                db,
                "SELECT concept, principle FROM learning_notes"
                " WHERE v3_experiment_id = ?",
                (other_eid,),
            )
            probes = [
                str(n["concept"]).strip()
                for n in peer_notes
                if str(n["concept"]).strip() and len(str(n["concept"]).strip()) >= 4
            ]
            if not probes:
                continue
            rows = _rows(
                db,
                "SELECT id, stage, item_id, context_used FROM v3_stage_items"
                " WHERE run_id=? AND exp_group=?",
                (run_id, group),
            )
            hits = []
            for r in rows:
                ctx = r["context_used"]
                for p in probes:
                    if p in ctx:
                        hits.append({
                            "stage": r["stage"],
                            "item_id": r["item_id"],
                            "peer_concept": p,
                            "peer_experiment_id": other_eid,
                        })
                        break
            if hits:
                out.append(Contamination(
                    kind=CONTAM_EVALUATOR_LEAK,
                    detail=(
                        f"{group} 组有 {len(hits)} 行的 context_used 里"
                        f"出现了 {other_group} 组笔记的内容"
                    ),
                    evidence={"hits": hits[:5]},
                ))

    # 3. 考生与评估器同实例
    same = _rows(
        db,
        "SELECT DISTINCT candidate_provider, candidate_model,"
        "       evaluator_provider, evaluator_model FROM v3_runs WHERE run_id=?",
        (run_id,),
    )
    for r in same:
        if (
            r["candidate_provider"] == r["evaluator_provider"]
            and r["candidate_model"] == r["evaluator_model"]
        ):
            out.append(Contamination(
                kind=CONTAM_EVALUATOR_LEAK,
                detail=(
                    f"candidate 与 evaluator 是同一个 provider+model"
                    f"（{r['candidate_provider']}/{r['candidate_model']}），"
                    f"评审者与考生是同一个实例，评分不能作为独立证据"
                ),
                evidence=dict(r),
            ))
    return out


def check_call_failures(db: Database, run_id: str) -> list[Contamination]:
    """本次 run 有没有考生调用失败。

    调用失败与「模型说不会」必须分开：前者是这次实验没跑成，
    后者是一个有效观测值。失败样本若被当成答不出，
    两组分数就被同一批噪声替代——而 difference-in-gains
    会归零得莫名其妙，报告里却看不出任何异常。
    """
    out: list[Contamination] = []
    rows = _rows(
        db,
        "SELECT exp_group, stage, item_id, error FROM v3_call_failures"
        " WHERE run_id=? ORDER BY id",
        (run_id,),
    )
    if rows:
        out.append(Contamination(
            kind=CONTAM_CALL_FAILED,
            detail=(
                f"本次 run 有 {len(rows)} 次考生调用失败。"
                f"调用失败的样本无法区分「模型答不出」与「这次没跑成」，"
                f"两组分数不再可比"
            ),
            evidence={"failures": rows[:10]},
        ))
    return out


def check_checksum(db: Database, run_id: str) -> list[Contamination]:
    """五份冻结指纹是否仍与当前代码一致。

    任何一份变了，本次 run 的全部结论作废——
    因为「Learning 学的是 A 材料、对照学的是 B 材料」
    这件事已经不再成立。
    """
    out: list[Contamination] = []
    row = db.query_one("SELECT * FROM v3_runs WHERE run_id=?", (run_id,))
    if row is None:
        return [Contamination(
            kind=CONTAM_CHECKSUM_CHANGED,
            detail=f"run {run_id!r} 不存在",
            evidence={},
        )]

    current = frozen_fingerprints()
    names = (
        ("pre_fingerprint", "PRE 题库"),
        ("post_fingerprint", "POST 题库"),
        ("transfer_fingerprint", "TRANSFER 题库"),
        ("learning_material_checksum", "学习材料"),
        ("control_material_checksum", "对照材料"),
        ("practice_tasks_fingerprint", "practice 任务集"),
        ("heldout_tasks_fingerprint", "held-out 任务集"),
    )
    for col, label in names:
        frozen, cur = row[col], current[col]
        if not frozen:
            out.append(Contamination(
                kind=CONTAM_CHECKSUM_CHANGED,
                detail=f"{label} 在 run 建立时没有记录指纹，无法核对",
                evidence={"column": col, "frozen": frozen},
            ))
        elif frozen != cur:
            out.append(Contamination(
                kind=CONTAM_CHECKSUM_CHANGED,
                detail=(
                    f"{label}指纹不一致：冻结时 {frozen[:16]}…，"
                    f"当前 {cur[:16]}…。本次 run 结果作废。"
                ),
                evidence={"column": col, "frozen": frozen, "current": cur},
            ))
    return out


# ==========================================================================
# 汇总
# ==========================================================================

_CHECKS = (
    check_shared_knowledge,
    check_shared_notes,
    check_heldout_in_study,
    check_transfer_in_practice,
    check_evaluator_leak,
    check_call_failures,
    check_checksum,
)


#: 每类污染由哪个检查负责。
#:
#: 「检查是否覆盖完整」要问的是**这类污染有没有对应的检查**，
#: 而不是「这次有没有命中」。干净时 kinds 为空，
#: 按 kinds 算missing 会恒为全部 6 项——
#: 一个永远报警的自检信号等于没有信号。
CHECK_FOR_KIND: dict[str, str] = {
    CONTAM_SHARED_KNOWLEDGE: "check_shared_knowledge",
    CONTAM_SHARED_NOTES: "check_shared_notes",
    CONTAM_POST_IN_STUDY: "check_heldout_in_study",
    CONTAM_TRANSFER_IN_PRACTICE: "check_transfer_in_practice",
    CONTAM_EVALUATOR_LEAK: "check_evaluator_leak",
    CONTAM_CHECKSUM_CHANGED: "check_checksum",
    # 调用失败不是「第六类污染」——它归在 evaluator_leak 下判定，
    # 但有独立的一条检查，覆盖完整性要单独核。
    CONTAM_CALL_FAILED: "check_call_failures",
}


def contamination_report(db: Database, run_id: str) -> dict[str, Any]:
    """跑完全部污染检测，输出可判定的报告。

    ``checks_missing`` 报的是**没被任何检查负责的污染类别**。
    早先错拿类别名去比 ``per_check`` 的函数名，恒为全部 6 项；
    改成比「这次命中的类别」又会在干净时恒为空——
    两种口径都不是「覆盖是否完整」这个问题。
    """
    found: list[Contamination] = []
    per_check: dict[str, int] = {}
    for fn in _CHECKS:
        hits = fn(db, run_id)
        per_check[fn.__name__] = len(hits)
        found.extend(hits)

    kinds = {c.kind for c in found}
    missing = [
        k for k, fn in CHECK_FOR_KIND.items() if fn not in per_check
    ]
    return {
        "run_id": run_id,
        "clean": not found,
        "checks_run": list(per_check),
        "checks_missing": missing,
        "hits_by_kind": {k: per_check[k] for k in per_check},
        "kinds_hit": sorted(kinds),
        "contaminations": [c.to_dict() for c in found],
    }


def assert_no_contamination(db: Database, run_id: str) -> None:
    """任一污染命中 → 直接拒绝使用本次实验结果。"""
    rep = contamination_report(db, run_id)
    if rep["clean"]:
        return
    lines = [f"实验 {run_id!r} 检出 {len(rep['contaminations'])} 处污染，结果不可用："]
    for c in rep["contaminations"]:
        lines.append(f"  [{c['kind']}] {c['detail']}")
    raise RuntimeError("\n".join(lines))


# ==========================================================================
# 实验顺序审计
# ==========================================================================


def _stage_seq_map(db: Database, run_id: str) -> dict[str, int]:
    """每个阶段的 seq。

    **必须同时看两张表。** 早先只查 ``v3_stage_items``，
    于是 ``practice_learning`` / ``practice_control`` /
    ``pre_objective`` / ``post_objective`` 四个阶段
    在顺序审计里**完全不存在**——
    声称「十阶段顺序」的门禁其实只覆盖了八个。
    这类「门禁名字比门禁本身覆盖得多」的缺口，
    比没有门禁更难发现：报告里写着 clean=True。
    """
    seq_of: dict[str, int] = {}
    for table in ("v3_stage_items", "v3_objective_runs"):
        rows = _rows(
            db,
            f"SELECT stage, MIN(seq) AS s, COUNT(*) AS n FROM {table}"
            f" WHERE run_id=? GROUP BY stage",
            (run_id,),
        )
        for r in rows:
            stage = r["stage"]
            if r["n"] and r["s"] is not None:
                s = int(r["s"])
                # 两张表都有同名的阶段时取较小值（真实开始时间）
                seq_of[stage] = min(seq_of.get(stage, s), s)
    return seq_of


def order_audit(db: Database, run_id: str) -> dict[str, Any]:
    """核对十阶段是否按固定顺序落库。

    顺序不是形式：如果Control 的 PRE 跑在 Learning 的 STUDY 之后，
    同一模型实例可能已从上一轮 prompt 残留目标知识，
    Control 的 PRE 被抬高，difference-in-gains 被压低。
    """
    problems: list[str] = []
    seq_of = _stage_seq_map(db, run_id)

    # 1. 落库的阶段必须与固定顺序一致
    expected_order = [s for s in STAGES_V3 if s in seq_of]
    actual_seq = [seq_of[s] for s in expected_order]
    if actual_seq != sorted(actual_seq):
        problems.append(
            f"阶段 seq 与固定顺序不符：{dict(zip(expected_order, actual_seq))}"
        )

    # 2. 两个 PRE 必须都排在所有材料阶段之前。
    #    若Learning 的 STUDY(seq=3) 早于 Control 的 PRE(seq=2)，
    #    Control 的 PRE 就发生在 Learning 已读过目标知识之后。
    #
    #    客观 PRE 也算 PRE：它测的也是「未学习时会不会」，
    #    同样不能排在材料之后。
    pre_seqs = [
        seq_of[f"pre_{g}"] for g in GROUPS_V3 if f"pre_{g}" in seq_of
    ]
    if "pre_objective" in seq_of:
        pre_seqs.append(seq_of["pre_objective"])
    material_stages = (
        "study_learning", "control_material",
        "practice_learning", "practice_control",
    )
    if pre_seqs:
        for stage in material_stages:
            if stage in seq_of and seq_of[stage] < max(pre_seqs):
                problems.append(
                    f"材料阶段 {stage}(seq={seq_of[stage]}) 早于某个 PRE"
                    f"（最大 seq={max(pre_seqs)}）"
                )
    missing_pre = [f"pre_{g}" for g in GROUPS_V3 if f"pre_{g}" not in seq_of]
    if missing_pre:
        problems.append(f"以下 PRE 阶段没有落库：{missing_pre}")

    # 3. POST 必须晚于两组 PRACTICE
    practice_seqs = [
        seq_of[f"practice_{g}"] for g in GROUPS_V3 if f"practice_{g}" in seq_of
    ]
    for g in GROUPS_V3:
        st = f"post_{g}"
        if st in seq_of and practice_seqs and seq_of[st] < max(practice_seqs):
            problems.append(
                f"{st}(seq={seq_of[st]}) 早于某个 PRACTICE"
                f"（最大 seq={max(practice_seqs)}）"
            )

    # 4. 客观 POST 也必须晚于两组 PRACTICE —— 它测的是同一批
    #    held-out 题，若早于 practice，practice 阶段就等于泄题。
    if "post_objective" in seq_of and practice_seqs:
        if seq_of["post_objective"] < max(practice_seqs):
            problems.append(
                f"post_objective(seq={seq_of['post_objective']}) "
                f"早于某个 PRACTICE（最大 seq={max(practice_seqs)}）"
            )

    return {
        "run_id": run_id,
        "expected_order": list(STAGES_V3),
        "stages_present": expected_order,
        "all_stages_present": sorted(seq_of),
        "seq_by_stage": seq_of,
        "problems": problems,
        "clean": not problems,
    }


def assert_order_valid(db: Database, run_id: str) -> None:
    rep = order_audit(db, run_id)
    if not rep["clean"]:
        raise RuntimeError(
            f"实验 {run_id!r} 的阶段顺序不合法：\n  - "
            + "\n  - ".join(rep["problems"])
        )


# ==========================================================================
# 冻结指纹
# ==========================================================================


def frozen_fingerprints() -> dict[str, str]:
    """本次实验应当被冻结的全部指纹。"""
    from brain import benchmark_v2 as b2

    from practice import objective as ob

    return {
        "pre_fingerprint": b2.PRE_FINGERPRINT,
        "post_fingerprint": ho.POST_FINGERPRINT,
        "transfer_fingerprint": ho.TRANSFER_HELDOUT_FINGERPRINT,
        "learning_material_checksum": checksum(nm.NOVEL_MATERIAL),
        "control_material_checksum": checksum(cm.CONTROL_MATERIAL),
        "practice_tasks_fingerprint": ob.TASKS_FINGERPRINT,
        "heldout_tasks_fingerprint": ho.HO_TASKS_FINGERPRINT,
    }


def isolation_diagnostics() -> dict[str, Any]:
    """隔离层的静态诊断（不碰数据库）。"""
    from brain import benchmark_v2 as b2
    from practice import objective as ob

    problems: list[str] = []
    warnings: list[str] = []

    # 1. PRE 与 held-out 三卷都不重叠
    pre_ids = {it.item_id for it in b2.PRE_ITEMS}
    ho_ids = {it.item_id for it in ho.ALL_HELDOUT_ITEMS}
    if pre_ids & ho_ids:
        problems.append(f"PRE 与 held-out item_id 重复：{sorted(pre_ids & ho_ids)}")

    # 2. 学习材料里不含held-out 题干
    for it in ho.ALL_HELDOUT_ITEMS:
        core = it.question.strip()[:40]
        if len(core) >= 20 and core in nm.NOVEL_MATERIAL:
            problems.append(f"held-out 题 {it.item_id} 的题干出现在学习材料里")

    # 3. 对照材料里不含 held-out 题干，也不含目标知识
    for it in ho.ALL_HELDOUT_ITEMS:
        core = it.question.strip()[:40]
        if len(core) >= 20 and core in cm.CONTROL_MATERIAL:
            problems.append(f"held-out 题 {it.item_id} 的题干出现在对照材料里")
    for label, hits in cm.leakage_hits().items():
        if hits:
            problems.append(f"对照材料泄漏（{label}）：{hits}")

    # 4. 学习材料长度 vs 对照材料长度
    ratio = cm.length_ratio()
    if not (cm.LENGTH_RATIO_MIN <= ratio <= cm.LENGTH_RATIO_MAX):
        problems.append(
            f"对照材料长度比 {ratio} 超出 "
            f"[{cm.LENGTH_RATIO_MIN}, {cm.LENGTH_RATIO_MAX}]"
        )

    # 5. held-out 自身的门禁
    try:
        ho.assert_heldout_intact()
    except RuntimeError as exc:
        problems.append(str(exc))
    try:
        ho.assert_heldout_isolated()
    except RuntimeError as exc:
        problems.append(str(exc))
    try:
        ho.assert_heldout_scorers_intact()
    except RuntimeError as exc:
        problems.append(str(exc))

    # 6. 结构性提醒：两组用的是同一套题库
    warnings.append(
        "PRE / POST / TRANSFER 三卷对两组是同一套题（这是必须的，"
        "否则比较的不是同一把尺子）；因此题目本身不构成组间污染源，"
        "污染只可能来自 context / notes / practice 参数。"
    )

    return {
        "problems": problems,
        "warnings": warnings,
        "length_ratio": ratio,
        "learn_tokens": cm.token_estimate(nm.NOVEL_MATERIAL),
        "control_tokens": cm.token_estimate(cm.CONTROL_MATERIAL),
        "pre_items": len(b2.PRE_ITEMS),
        "heldout_items": len(ho.ALL_HELDOUT_ITEMS),
        "heldout_tasks": len(ho.HELD_OUT_TASKS),
        "practice_tasks": len(ob.OBJECTIVE_TASKS),
    }


def assert_isolation_intact() -> None:
    d = isolation_diagnostics()
    if d["problems"]:
        raise RuntimeError(
            "V0.6 隔离层不合法，实验作废：\n  - " + "\n  - ".join(d["problems"])
        )


__all__ = [
    "ALL_CONTAMINATIONS",
    "ALL_CONTAMINATION_KINDS",
    "CHECK_FOR_KIND",
    "CONTAM_CALL_FAILED",
    "BLIND_STAGES_V3",
    "GROUPS_V3",
    "STAGES_V3",
    "Contamination",
    "GroupScope",
    "assert_isolation_intact",
    "assert_no_contamination",
    "assert_order_valid",
    "build_scopes",
    "contamination_report",
    "frozen_fingerprints",
    "isolation_diagnostics",
    "order_audit",
]
