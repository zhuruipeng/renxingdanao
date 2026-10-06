"""brain/report.py —— V0.4 实验报告。

**V0.4 相对 V0.3 改了什么**
--------------------------
V0.3 报告的结构性缺陷：只有「两场 POST 同题」。

同题两场（OPEN_BOOK / BLIND_POST）能回答「有没有记住」，
但回答不了「有没有学会」——因为盲考时模型可能只是模糊地记得
「我刚才答过一道讲 B-Tree 的题」，靠的是题面里的线索，
而不是对原理的理解。**背答案和形成概念在这套实验里长得一样。**

所以 V0.4 把考试拆成三场，分工严格：

    PRE_BLIND        起点裸能力（不给任何资料）
    MEMORY  OPEN_BOOK  同一批新题，能查自己的笔记 -> 记忆可用性
    MEMORY  BLIND      同一批新题，收回笔记       -> 记忆保持度
    TRANSFER BLIND     完全不同题、同一知识点，不给笔记 -> 概念形成度

关键不对称：**TRANSFER 不开卷**。
开卷能查笔记就说明不了迁移能力，TRANSFER 要测的正是
「不依赖笔记能不能用同一个原理」。

**指标体系（老板指定的六个）**
------------------------------
    PRE               学习前能力
    OPEN_BOOK         外部记忆带来的提升
    BLIND             模型自身是否出现能力迁移
    Practice success  学习能否转化为实际任务能力
    Note quality      学习笔记本身是否可靠

严格说 BLIND 有两场（MEMORY_BLIND 与 TRANSFER_BLIND），
报告里必须分开列——它们的差值 `TRANSFER - MEMORY`
才是「记住了 vs 学会了」的分界。

**术语纪律**：即使 TRANSFER_GAIN > 0，也只能说「观察到了能力迁移迹象」，
绝不能说「模型训练成功」。本项目没做任何训练或微调。
"""

from __future__ import annotations

import json
from typing import Any

from brain import question_bank as qb
from brain.experiment import ExperimentRunner
from brain.replicate import SIGMA_K, aggregate_trials
from memory.gate import RESOURCE_EVALUATION, RESOURCE_MATERIAL, RESOURCE_REFERENCE
from practice.judge import USABLE_LINE
from learning_notes.quality import PASS_LINE as NOTE_PASS_LINE

#: 判定「学到外部记忆」与「观察到能力迁移」的阈值（单次实验用）
GAIN_THRESHOLD = 5.0

#: 天花板判定线：PRE 高于此值时，「提升空间」被压缩，增益判定的可靠性下降
CEILING_LINE = 85.0


def detect_ceiling(pre_mean: float) -> dict[str, Any]:
    """检测天花板效应，并说清它会怎样影响结论的可信度。

    **这是 v0.4.1 真实实验逼出来的东西。**
    DeepSeek 跑10 次，PRE 均值 90.0 —— 也就是说学习前它已经会 90% 了。
    OPEN_BOOK 的满分是 100，所以它最多只能「提升」10 分。

    后果很具体：OPEN_BOOK_GAIN 均值 +6.8看着是正的，
    但 σ=4.3 → mean - 2σ = -1.9 < 0，判定门槛把它归进「无法与噪声区分」。

    **这个「无法与噪声区分」不能被读成「学习没起作用」。**
    两种解释在数据上无法区分：
      A. 学习真的没带来提升
      B. 题目太简单，10 分的空间本来就被采样噪声吃掉了

    要区分只能改题库（把题改难/ 换更冷门的知识点），不是改分析代码。
    报告必须把这层不确定性写出来，否则读者会得到一个错误的否定结论。
    """
    if pre_mean >= CEILING_LINE:
        headroom = 100.0 - pre_mean
        return {
            "hit": True,
            "pre_mean": round(pre_mean, 1),
            "headroom": round(headroom, 1),
            "note": (
                f"PRE 均值 {pre_mean:.1f} 已达天花板线 {CEILING_LINE:.0f}，"
                f"开卷最多只剩 {headroom:.1f} 分的提升空间。"
                f"在这个空间里，采样噪声（σ 通常 3~5）足以吃掉全部增益——"
                f"「增益无法与噪声区分」既可能是「学习没用」，"
                f"也可能是「没有可测量的空间」，本实验分不出这两者。"
                f"要区分只能改题库（提高难度或换更冷门的知识点），"
                f"不是改判分或统计方法。"
            ),
        }
    return {
        "hit": False,
        "pre_mean": round(pre_mean, 1),
        "headroom": round(100.0 - pre_mean, 1),
        "note": (
            f"PRE 均值 {pre_mean:.1f}，距天花板还有 "
            f"{100.0 - pre_mean:.1f} 分空间，增益判定空间充足。"
        ),
    }


# ----------------------------------------------------------------------
# 单次实验的结论判读
# ----------------------------------------------------------------------
def interpret(
    pre: float,
    mem_open: float,
    mem_blind: float,
    transfer_blind: float,
    practice_usable_rate: float = 0.0,
    note_quality_avg: float = 0.0,
) -> dict[str, Any]:
    """把五个分数翻译成结论。

    这是整个 V0.4 最重要的一段逻辑——它决定读者会把实验理解成什么。

    判据分两层：
      1. **记住了** = MEMORY_BLIND > PRE（收回笔记后，同一批题还能答对）
      2. **学会了** = TRANSFER_BLIND > PRE（换一批题，同一个原理还能用）

    只有第 2 条为真才谈得上形成概念。
    """
    open_gain = round(mem_open - pre, 2)
    mem_gain = round(mem_blind - pre, 2)
    tr_gain = round(transfer_blind - pre, 2)
    # 迁移相对记忆的折损：正数 = 迁移不如记忆强，即「背住了但没内化」
    transfer_vs_memory = round(mem_gain - tr_gain, 2)

    used_memory = open_gain > GAIN_THRESHOLD
    remembered = mem_gain > GAIN_THRESHOLD
    formed = tr_gain > GAIN_THRESHOLD
    practice_ok = practice_usable_rate * 100 >= USABLE_LINE
    notes_ok = note_quality_avg >= NOTE_PASS_LINE

    # 支撑链：笔记可靠 + 实践可用 是「学会」的前置条件。
    # 笔记本身不靠谱时，开卷高分只说明「抄对了自己写的东西」。
    chain_intact = notes_ok and practice_ok

    if used_memory and formed and chain_intact:
        headline = "观察到了可能的能力迁移迹象（不是训练成功）"
        reading = (
            f"PRE={pre:.0f}，MEMORY_OPEN={mem_open:.0f}（{open_gain:+.0f}），"
            f"MEMORY_BLIND={mem_blind:.0f}（{mem_gain:+.0f}），"
            f"TRANSFER_BLIND={transfer_blind:.0f}（{tr_gain:+.0f}）。"
            "笔记质量与实践可用性都过关，同时换一批题、只考同一个知识点时分数仍高于起点。"
            "这比 V0.3 的证据强一档：不再只是「同一批题还答得对」。"
            "**注意：这仍然不等于模型被训练或微调成功了**——本项目没做任何参数更新。"
        )
        proves = [
            "在全新题面、同一知识点上，收回全部外部记忆后仍高于学习前基线",
            "学习产物本身可靠：笔记质量与实践可用性均达标（迁移的必要条件）",
        ]
        disproves = [
            "不等于模型权重改变——本项目无任何训练/微调",
            "不等于「学习算法有效」——单次实验无法排除噪声、题目泄漏与判分漂移",
        ]
    elif used_memory and remembered and not formed:
        headline = "背住了同一批题的话术，但没有形成概念"
        reading = (
            f"PRE={pre:.0f}，MEMORY_OPEN={mem_open:.0f}（{open_gain:+.0f}），"
            f"MEMORY_BLIND={mem_blind:.0f}（{mem_gain:+.0f}），"
            f"TRANSFER_BLIND={transfer_blind:.0f}（{tr_gain:+.0f}）。"
            f"迁移相对记忆折损 {transfer_vs_memory:+.0f} 分——"
            "同一批新题收回笔记后还能答对，但换个场景问同一个原理就答不出来。"
            "这是**背诵**的典型特征，不是学习的特征。"
            "V0.3 的两场 POST 同题设计会把这种情况误判成「学会了」。"
        )
        proves = [
            "学习流程能产出可用的外部记忆（结构化笔记确实提升了同题作答表现）",
            "该提升在收回笔记后仍部分保持——模型记住了这批题",
        ]
        disproves = [
            "没有形成可迁移的概念：换题面即失效",
            "模型的参数/权重没有因学习而改变",
        ]
    elif used_memory and not remembered:
        headline = "学会了利用外部记忆，但收回笔记就回落"
        reading = (
            f"PRE={pre:.0f}，MEMORY_OPEN={mem_open:.0f}（{open_gain:+.0f}），"
            f"MEMORY_BLIND={mem_blind:.0f}（{mem_gain:+.0f}），"
            f"TRANSFER_BLIND={transfer_blind:.0f}（{tr_gain:+.0f}）。"
            "开卷时能调用自己写的笔记拿到高分，收回笔记后分数回落到起点水平。"
            "这说明**学习确实发生了**——产物是外部记忆（knowledge + learning_notes），"
            "而不是模型参数的改变。Agent 变强了，但强在「有笔记可查」这件事上。"
        )
        proves = [
            "学习流程能产出可用的外部记忆（结构化笔记确实提升了答题表现）",
            "该提升依赖笔记存在，一旦撤走笔记即消失",
        ]
        disproves = [
            "模型的参数/权重没有因学习而改变",
            "BLIND 场分数与起点无差异，未观察到内化",
        ]
    elif not used_memory and not formed:
        headline = "既未观察到外部记忆利用，也未观察到能力迁移"
        reading = (
            f"PRE={pre:.0f}，MEMORY_OPEN={mem_open:.0f}（{open_gain:+.0f}），"
            f"MEMORY_BLIND={mem_blind:.0f}（{mem_gain:+.0f}），"
            f"TRANSFER_BLIND={transfer_blind:.0f}（{tr_gain:+.0f}）。"
            "基本持平。可能原因：笔记质量不足以支撑答题、"
            "评估器没有区分度、或该主题本身对模型太简单/太难。"
        )
        proves = ["本次实验未能观测到任何形式的提升"]
        disproves = ["无法支持任何「学习有效」的结论"]
    else:
        headline = "结果异常：需先排查实验设计"
        reading = (
            f"PRE={pre:.0f}，MEMORY_OPEN={mem_open:.0f}，"
            f"MEMORY_BLIND={mem_blind:.0f}，TRANSFER_BLIND={transfer_blind:.0f}。"
            "出现开卷未提升、盲考却提升（或反之）的组合，"
            "通常指向题目泄漏、评估器不一致或上下文残留，应先查数据再下结论。"
        )
        proves = []
        disproves = ["该数据组合不满足任何已知模式，结论不可靠"]

    # 替代解释：只要盲考有提升就必须列，避免过度解读
    alternatives: list[str] = []
    if mem_gain > 0 or tr_gain > 0:
        alternatives = [
            "上下文残留：同一进程内 PRE 的作答可能被带入后续调用（需跨进程复验）",
            f"评分噪声：单次 LLM 判分波动可达 ±10 分，样本仅 4 题/卷",
            "判分漂移：四场的评估 prompt 不同，可能存在系统性偏差"
            "（TRANSFER 的答案措辞与 MEMORY 不同，评估器可能整体偏松或偏紧）",
            "题目难度：MEMORY 卷与 TRANSFER 卷只做到主观配平，没有难度实测标定",
            "题目泄漏：TRANSFER 出自同一份材料，若材料本身暗示了答案路径会抬高分数",
        ]
    if formed and not notes_ok:
        alternatives.append(
            "笔记质量不达标却拿到高 TRANSFER 分：可能是判分偏松，"
            "而不是真的迁移——请先看 note_quality 明细"
        )

    return {
        "open_book_gain": open_gain,
        "memory_gain": mem_gain,
        "transfer_gain": tr_gain,
        "transfer_vs_memory": transfer_vs_memory,
        "used_external_memory": used_memory,
        "remembered": remembered,
        "formed_concept": formed,
        "chain_intact": chain_intact,
        "headline": headline,
        "reading": reading,
        "proves": proves,
        "disproves": disproves,
        "alternative_explanations": alternatives,
    }


# ----------------------------------------------------------------------
# 单次实验报告
# ----------------------------------------------------------------------
def _item_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": r["item_id"],
            "kp_label": qb.KP_LABELS.get(r["kp"], r["kp"]),
            "mode": r.get("mode"),
            "exam_kind": r.get("exam_kind"),
            "score": r["score"],
            "question": r["question"],
            "candidate_answer": r["candidate_answer"],
            "reference_answer": r["reference_answer"],
            "verdict": r["verdict"],
            "feedback": r["feedback"],
            "missing_points": json.loads(r["missing_points"] or "[]"),
            "context_used_len": len(r.get("context_used") or ""),
        }
        for r in rows
    ]


def _select(items: list[dict[str, Any]], kind: str, mode: str = "") -> list[dict[str, Any]]:
    """按 exam_kind + mode 选题。exam_kind 缺失的老数据回退到 phase。"""
    out = [i for i in items if i.get("exam_kind") == kind]
    if not out:
        out = [i for i in items if i.get("phase") == kind]
    if mode:
        out = [i for i in out if i.get("mode") == mode]
    return out


def _note_quality_rows(runner: ExperimentRunner, experiment_id: int) -> list[dict[str, Any]]:
    """笔记质量明细，带 note_id 供渲染时按笔记对齐。"""
    rows = runner.note_quality_store.for_experiment(experiment_id)
    out = []
    for r in rows:
        out.append({
            "note_id": r.get("note_id"),
            "accuracy": r["accuracy"],
            "completeness": r["completeness"],
            "transferable": r["transferable"],
            "no_hallucination": r["no_hallucination"],
            "total": r["total"],
            "verdict": r["verdict"],
            "feedback": r["feedback"],
            "problems": json.loads(r["problems"] or "[]"),
        })
    return out


def build_report(runner: ExperimentRunner, experiment_id: int) -> dict[str, Any]:
    exp = runner.store.get(experiment_id)
    if exp is None:
        raise ValueError(f"实验 #{experiment_id} 不存在")
    items = runner.store.items(experiment_id)
    practice = runner.store.practice(experiment_id)
    material = runner.materials.get(int(exp["material_id"])) if exp["material_id"] else None

    pre = _select(items, "pre")
    mem_open = _select(items, "memory", "open_book_memory")
    mem_blind = _select(items, "memory", "blind_model")
    transfer = _select(items, "transfer", "blind_model")

    def score_of(v: Any) -> float:
        return float(v) if v is not None else 0.0

    pre_score = score_of(exp["pre_score"])
    mem_open_score = score_of(exp["memory_open_score"] or exp["open_book_score"])
    mem_blind_score = score_of(exp["memory_blind_score"] or exp["post_score"])
    transfer_score = score_of(exp["transfer_blind_score"])
    usable_rate = score_of(exp["practice_usable_rate"])
    practice_score = score_of(exp["practice_avg_score"])
    nq_avg = score_of(exp["note_quality_avg"])

    interp = interpret(
        pre_score, mem_open_score, mem_blind_score, transfer_score,
        usable_rate, nq_avg,
    )

    success = [p for p in practice if p["outcome"] == "success"]
    failure = [p for p in practice if p["outcome"] == "failure"]
    cmp_result = runner.compare(experiment_id)
    note_rows = runner.notes.for_experiment(experiment_id)
    judge_rows = runner.practice_judge_store.for_experiment(experiment_id)

    # 一致性自检
    consistency = {
        "pre_count": len(pre),
        "memory_open_count": len(mem_open),
        "memory_blind_count": len(mem_blind),
        "transfer_count": len(transfer),
        "same_memory_items": (
            sorted(r["item_id"] for r in mem_open)
            == sorted(r["item_id"] for r in mem_blind)
        ),
        "transfer_is_new_set": not (
            {r["item_id"] for r in transfer}
            & ({r["item_id"] for r in pre} | {r["item_id"] for r in mem_blind})
        ),
        "blind_context_empty": all(
            not (r.get("context_used") or "").strip()
            for r in pre + mem_blind + transfer
        ),
        "open_book_context_present": any(
            (r.get("context_used") or "").strip() for r in mem_open
        ),
        "transfer_never_open_book": not any(
            r.get("mode") == "open_book_memory" for r in transfer
        ),
        "notes_verified": all(
            runner.notes.verify_checksum(n["id"]) for n in note_rows
        ),
    }

    caveats = [
        "**这是能力迁移实验，不是训练实验。** 本项目没有任何参数更新、微调或权重修改，"
        "TRANSFER_GAIN 再大也不能解释为模型本身被改造过。",
        "样本量极小（每卷 4 道题），单题 ±25 分就会让整卷波动 ±6 分。",
        "评估器与考生同属一个模型家族，存在共同偏差——"
        "「换个模型评委会不会判得不一样」这个问题本实验回答不了。",
        "MEMORY 卷与 TRANSFER 卷的难度只做到主观配平，没有做难度实测标定。",
        "同一进程内上下文残留尚未彻底排除，BLIND 侧结果需跨进程复验。",
    ]

    return {
        "experiment_id": experiment_id,
        "topic": exp["topic"],
        "status": exp["status"],
        "batch_id": exp.get("batch_id") or "",
        "trial_no": exp.get("trial_no") or 1,
        "bank": {
            "version": qb.BANK_VERSION,
            "fingerprint": qb.BANK_FINGERPRINT[:16] + "…",
            "transfer_fingerprint": qb.TRANSFER_FINGERPRINT[:16] + "…",
            "pre_count": len(qb.PRE_SET),
            "memory_count": len(qb.POST_SET),
            "transfer_count": len(qb.TRANSFER_ITEMS),
            "kps": [qb.KP_LABELS.get(k, k) for k in sorted(qb.kp_coverage(qb.PRE_SET))],
        },
        "material": {
            "id": material["id"] if material else None,
            "version": material["version"] if material else None,
            "checksum": (material["checksum"][:16] + "…") if material else None,
        },
        # ---- 老板指定的六个指标，单次实验视角 ----
        "metrics": {
            "pre": pre_score,
            "open_book": mem_open_score,
            "memory_blind": mem_blind_score,
            "transfer_blind": transfer_score,
            "practice_usable_rate": round(usable_rate * 100, 1),
            "practice_avg_score": practice_score,
            "note_quality_avg": nq_avg,
        },
        "scores": {
            "pre_blind": pre_score,
            "open_book": mem_open_score,
            "memory_blind": mem_blind_score,
            "transfer_blind": transfer_score,
            "open_book_gain": interp["open_book_gain"],
            "memory_gain": interp["memory_gain"],
            "transfer_gain": interp["transfer_gain"],
        },
        "interpretation": interp,
        "kp_comparison": {
            "improved": cmp_result["improved"],
            "flat": cmp_result["flat"],
            "regressed": cmp_result["regressed"],
        },
        "consistency_check": consistency,
        "learning_notes": [
            {
                "id": n["id"],
                "kp_label": qb.KP_LABELS.get(n["kp"], n["kp"]),
                "concept": n["concept"][:60],
                "principle_len": len(n["principle"]),
                "checksum": n["checksum"][:16] + "…",
                "verified": runner.notes.verify_checksum(n["id"]),
                "quality_total": n.get("quality_total"),
                "quality_verdict": n.get("quality_verdict") or "",
            }
            for n in note_rows
        ],
        "note_quality": _note_quality_rows(runner, experiment_id),
        "pre_items": _item_rows(pre),
        "memory_open_items": _item_rows(mem_open),
        "memory_blind_items": _item_rows(mem_blind),
        "transfer_items": _item_rows(transfer),
        "practice": {
            "total": len(practice),
            "success": len(success),
            "failure": len(failure),
            "usable_rate": round(usable_rate * 100, 1),
            "avg_score": practice_score,
            "records": [
                {
                    "kp_label": qb.KP_LABELS.get(p["kp"], p["kp"]),
                    "outcome": p["outcome"],
                    "score": p["score"],
                    "lesson": p["lesson"],
                }
                for p in practice
            ],
        },
        "practice_judgments": [
            {
                "kp_label": qb.KP_LABELS.get(r["kp"], r["kp"]),
                "usable": bool(r["usable"]),
                "score": r["score"],
                "verdict": r["verdict"],
                "feedback": r["feedback"],
            }
            for r in judge_rows
        ],
        "caveats": caveats,
        "forbidden_in_all_exams": [
            RESOURCE_REFERENCE,
            RESOURCE_EVALUATION,
            RESOURCE_MATERIAL,
        ],
    }


# ----------------------------------------------------------------------
# 跨次批次报告
# ----------------------------------------------------------------------
def build_batch_report(db: Any, batch_id: str = "") -> dict[str, Any]:
    """把一次 N 重复实验聚合成带统计量的报告。

    单次实验的分数是**一次抽样**。这里给的是分布——
    每个指标 mean/std/min/max，三个 gain 额外过判定门槛
    `mean - 2*std > 0`。
    """
    from brain.replicate import BatchStore

    store = BatchStore(db)
    row = store.get(batch_id) if batch_id else store.latest()
    if row is None:
        raise ValueError("还没有任何重复实验批次，请先运行 experiment --repeat。")
    bid = row["batch_id"]

    summary = json.loads(row["summary_json"] or "{}")
    if not summary:
        summary = aggregate_trials(store.trials(bid))
        summary.update({
            "batch_id": bid,
            "topic": row["topic"],
            "repeats": row["planned_trials"],
            "model": row["model"],
            "bank_fingerprint": row["bank_fingerprint"],
            "bank_version": qb.BANK_VERSION,
            "material_checksum": row["material_checksum"],
            "trials": store.trials(bid),
            "sigma_k": SIGMA_K,
        })
    summary["verdict"] = batch_verdict(summary)
    return summary


def batch_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    """跨次结论。

    三个 gain 各自的状态由 `gain_verdict` 给出（insufficient/positive/
    noisy/non_positive）。这里做的是**组合解读**：

        open_book 成立 + transfer 成立  -> 迁移迹象（仍不是训练成功）
        open_book 成立 + transfer 不成立 -> 背住了，没形成概念
        open_book 不成立                 -> 学习没起作用，或题目/评估器有问题
    """
    v = summary.get("verdicts", {})
    open_state = v.get("open_book_gain", {}).get("state", "insufficient")
    mem_state = v.get("memory_gain", {}).get("state", "insufficient")
    tr_state = v.get("transfer_gain", {}).get("state", "insufficient")
    n = summary.get("n_trials", 0)

    ok = lambda s: s == "positive"  # noqa: E731

    if ok(open_state) and ok(tr_state):
        headline = f"跨 {n} 次：外部记忆可用，且观察到可重复的能力迁移迹象"
        reading = (
            "OPEN_BOOK_GAIN 与 TRANSFER_GAIN 的「均值-2σ」都大于 0，"
            "说明这两个提升不是单次抽样的噪声。换题面、只考同一个知识点时仍高于起点。"
            "**仍然不是训练成功**——本项目无任何参数更新。"
        )
    elif ok(open_state) and not ok(tr_state):
        headline = f"跨 {n} 次：外部记忆可用，但没有可重复的概念形成"
        reading = (
            "OPEN_BOOK_GAIN 稳定为正，但 TRANSFER_GAIN 的「均值-2σ」不大于 0。"
            "同一批题能答对、换个场景答不对——这是背诵，不是学习。"
            "如果单次实验看起来 TRANSFER 涨了，重复几次就会露馅。"
        )
    elif not ok(open_state):
        headline = f"跨 {n} 次：未观察到稳定的外部记忆利用"
        reading = (
            "OPEN_BOOK_GAIN 未通过判定门槛。可能是学习没起作用，"
            "也可能是笔记质量不达标、题目偏易/偏难、或评估器没有区分度。"
        )
    else:
        headline = f"跨 {n} 次：结论不足"
        reading = "样本量或提升幅度都不足以支撑结论。"

    # 天花板检测必须在结论**之后**跑一遍：
    # 「增益无法与噪声区分」这句话在 PRE 90 分的前提下是有歧义的 ——
    # 可能是学习没用，也可能是根本没有可测量的空间。
    # 不写清楚，读者会把前者当成结论。
    # PRE 均值在 summary["metrics"]["pre_score"]["mean"]（aggregate_trials 的产物），
    # 不在 gains 里 —— 取错路径会静默得到 0，天花板检测形同虚设。
    pre_mean = summary.get("metrics", {}).get("pre_score", {}).get("mean", 0.0)
    ceiling = detect_ceiling(pre_mean)
    caveats: list[str] = []
    if ceiling["hit"]:
        caveats.append(ceiling["note"])
        if not ok(open_state):
            caveats.append(
                f"**因此「未观察到稳定的外部记忆利用」这个结论不可靠**："
                f"学习前已在 {pre_mean:.1f} 分，天花板下最多只剩 "
                f"{ceiling['headroom']:.1f} 分空间，"
                f"这个量级小于噪声，任何增益都测不出来。"
                f"要得到有效结论必须换更难的题库。"
            )

    # 题量诊断：每 kp 只1 题时逐 kp 结论没有第二个样本支撑
    diag = qb.bank_diagnostics()
    if diag["warnings"]:
        caveats.extend(diag["warnings"])

    return {
        "n_trials": n,
        "headline": headline,
        "reading": reading,
        "open_state": open_state,
        "memory_state": mem_state,
        "transfer_state": tr_state,
        "decided": any(ok(s) for s in (open_state, mem_state, tr_state)),
        "ceiling": ceiling,
        "caveats": caveats,
    }


# ----------------------------------------------------------------------
# 文本渲染
# ----------------------------------------------------------------------
def _wrap(text: str, width: int = 56) -> list[str]:
    """按显示宽度粗略折行（中文按 2 列计）。"""
    out: list[str] = []
    line, width_now = "", 0
    for ch in text:
        w = 2 if ord(ch) > 0x2E80 else 1
        if width_now + w > width:
            out.append(line)
            line, width_now = "", 0
        line += ch
        width_now += w
    if line:
        out.append(line)
    return out


_METRIC_ROWS = (
    ("PRE", "pre", "学习前裸能力（不给任何资料）"),
    ("OPEN_BOOK", "open_book", "外部记忆带来的提升（能查自己的笔记）"),
    ("MEMORY_BLIND", "memory_blind", "收回笔记，同一批题还答不对多少"),
    ("TRANSFER_BLIND", "transfer_blind", "换全新题面、同一知识点，不给笔记"),
    ("PRACTICE", "practice_usable_rate", "实践产出可执行的比例（%）"),
    ("NOTE_QUALITY", "note_quality_avg", "学习笔记本身的可靠性（0-100）"),
)


def format_text(report: dict[str, Any]) -> str:
    """人类可读的文本报告（单次实验）。"""
    s = report["scores"]
    m = report["metrics"]
    it = report["interpretation"]
    lines: list[str] = []
    add = lines.append

    add("=" * 66)
    add(f"Agent-01 V0.4 学习效果实验报告 · 实验 #{report['experiment_id']}")
    add(f"主题：{report['topic']}")
    if report.get("batch_id"):
        add(f"批次：{report['batch_id']}  第 {report['trial_no']} 次")
    add("=" * 66)
    add("")

    add("【一、六项指标】")
    for name, key, desc in _METRIC_ROWS:
        val = m.get(key)
        if key == "practice_usable_rate":
            txt = f"{val:.1f}%" if val is not None else "-"
        else:
            txt = f"{val:.1f} 分" if val is not None else "-"
        add(f"  {name:<14}{txt:>10}   {desc}")
    add(f"  {'（实践均分）':<14}{m['practice_avg_score']:>10.1f}   "
        "实践产出质量分，区别于「能不能用」")
    add("")

    add("【二、三个 Gain（语义不同，不可混用）】")
    add(f"  OPEN_BOOK_GAIN ：{s['open_book_gain']:+.1f}  <- 外部记忆的利用能力")
    add(f"  MEMORY_GAIN    ：{s['memory_gain']:+.1f}  <- 记忆保持（背住了）")
    add(f"  TRANSFER_GAIN  ：{s['transfer_gain']:+.1f}  <- 概念形成（学会了）")
    add(f"  TRANSFER - MEMORY：{it['transfer_vs_memory']:+.1f}"
        "  <- 正数=背住了但没内化")
    add("")

    add("【三、这个实验证明了什么】")
    add(f"  结论：{it['headline']}")
    add("")
    for line in _wrap(it["reading"]):
        add(f"  {line}")
    add("")
    add("  [证明了]")
    for p in it["proves"] or ["（无）"]:
        add(f"     - {p}")
    add("  [不能证明]")
    for p in it["disproves"] or ["（无）"]:
        add(f"     - {p}")
    if it["alternative_explanations"]:
        add("")
        add("  [!] 盲考侧有提升的替代解释（排除后才能下强结论）：")
        for a in it["alternative_explanations"]:
            for line in _wrap(a, 50):
                add(f"     - {line}" if line is a.splitlines()[0] else f"       {line}")
    add("")

    add("【四、逐知识点（PRE / MEMORY_BLIND / TRANSFER_BLIND）】")
    kp = report["kp_comparison"]
    for name, rows in (("迁移提升", kp["improved"]), ("持平", kp["flat"]),
                       ("迁移退步", kp["regressed"])):
        add(f"  {name}（{len(rows)} 项）：")
        if not rows:
            add("    （无）")
        for r in rows:
            add(
                f"    - {r['label']}：PRE {r['pre']:.0f} → "
                f"MEM {r['memory_blind']:.0f} → TRANSFER {r['transfer_blind']:.0f}"
                f"（迁移 {r['transfer_delta']:+.0f}）"
            )
    add("")

    add("【五、学习笔记与质量】")
    q_by_note = {q.get("note_id"): q for q in report["note_quality"]}
    if not report["learning_notes"]:
        add("  （无笔记）")
    for n in report["learning_notes"]:
        flag = "OK" if n["verified"] else "BAD"
        q = q_by_note.get(n["id"])
        qt = f"，质量 {q['total']:.1f}（{q['verdict']}）" if q else ""
        add(f"  [{flag}] [{n['kp_label']}]{qt} {n['concept']}…")
    add("")

    pr = report["practice"]
    add("【六、实践记录（独立评审）】")
    add(f"  共 {pr['total']} 次：成功 {pr['success']} / 失败 {pr['failure']}")
    add(f"  可执行率 {pr['usable_rate']:.1f}%   平均质量分 {pr['avg_score']:.1f}")
    for j in report["practice_judgments"]:
        u = "可用" if j["usable"] else "不可用"
        add(f"    - [{u} {j['score']:.0f}] {j['kp_label']}：{j['feedback'][:40]}")
    add("")

    add("【七、一致性自检】")
    c = report["consistency_check"]
    add(f"  PRE {c['pre_count']} 题 / MEMORY_OPEN {c['memory_open_count']} 题 / "
        f"MEMORY_BLIND {c['memory_blind_count']} 题 / TRANSFER {c['transfer_count']} 题")
    add(f"  MEMORY 两场同题：{'是' if c['same_memory_items'] else '否'}")
    add(f"  TRANSFER 是全新题：{'是' if c['transfer_is_new_set'] else '否'}")
    add(f"  TRANSFER 从未开卷：{'是' if c['transfer_never_open_book'] else '否'}")
    add(f"  三场盲考上下文为空：{'是' if c['blind_context_empty'] else '否'}")
    add(f"  OPEN_BOOK 场有上下文：{'是' if c['open_book_context_present'] else '否'}")
    add(f"  笔记 checksum 全部通过：{'是' if c['notes_verified'] else '否'}")
    add("")

    add("【八、局限】")
    for c2 in report["caveats"]:
        for line in _wrap(c2):
            add(f"  {line}")
    add("")
    b = report["bank"]
    add(f"题库：{b['version']}  指纹 {b['fingerprint']}  "
        f"TRANSFER 指纹 {b['transfer_fingerprint']}")
    add(f"卷数：PRE {b['pre_count']} / MEMORY {b['memory_count']} / "
        f"TRANSFER {b['transfer_count']}")
    add(f"材料：{report['material']['version']}  "
        f"校验 {report['material']['checksum']}")
    add("=" * 66)
    return "\n".join(lines)


def format_batch_text(summary: dict[str, Any]) -> str:
    """人类可读的跨次统计报告。"""
    lines: list[str] = []
    add = lines.append
    v = summary.get("verdict", {})

    add("=" * 74)
    add(f"Agent-01 V0.4 重复实验报告 · {summary.get('n_trials', 0)} 次")
    add(f"主题：{summary.get('topic', '')}   模型：{summary.get('model') or '(未标注)'}")
    add(f"批次：{summary.get('batch_id', '')}")
    add("=" * 74)
    add("")

    add("【一、六项指标 · 跨次分布】")
    add(f"  {'指标':<16}{'均值':>8}{'标准差':>9}{'最小':>8}{'最大':>8}   判定")
    metric_meta = (
        ("pre_score", "PRE", ""),
        ("memory_open_score", "OPEN_BOOK", ""),
        ("memory_blind_score", "MEMORY_BLIND", ""),
        ("transfer_blind_score", "TRANSFER_BLIND", ""),
        ("practice_usable_rate", "PRACTICE %", "×100"),
        ("note_quality_avg", "NOTE_QUALITY", ""),
    )
    for key, label, note in metric_meta:
        d = summary.get("metrics", {}).get(key, {})
        if not d or not d.get("n"):
            add(f"  {label:<16}{'—':>8}")
            continue
        scale = 100 if note else 1
        add(f"  {label:<16}{d['mean'] * scale:>8.1f}{d['std'] * scale:>9.1f}"
            f"{d['min'] * scale:>8.1f}{d['max'] * scale:>8.1f}")
    add("")

    add("【二、三个 Gain 的判定（门槛：均值 - 2σ > 0）】")
    state_cn = {
        "positive": "✔ 观察到提升",
        "noisy": "? 均值为正但小于波动",
        "non_positive": "✘ 未观察到提升",
        "insufficient": "- 样本不足",
    }
    for key, label in (
        ("open_book_gain", "OPEN_BOOK_GAIN  外部记忆利用"),
        ("memory_gain", "MEMORY_GAIN     记忆保持"),
        ("transfer_gain", "TRANSFER_GAIN   概念形成"),
    ):
        g = summary.get("verdicts", {}).get(key)
        if not g:
            continue
        add(f"  {label}")
        add(f"     {state_cn.get(g['state'], g['state'])}"
            f"   均值 {g['mean']:+.1f}  σ {g['std']:.1f}  "
            f"均值-2σ = {g['lower_bound']:+.1f}  (n={g['n']})")
        for line in _wrap(g["reason"], 50):
            add(f"     {line}")
    gap = summary.get("memory_vs_transfer_gap")
    if gap:
        add("")
        add(f"  背住了 vs 学会了（MEMORY_GAIN - TRANSFER_GAIN）")
        add(f"     {state_cn.get(gap['state'], gap['state'])}"
            f"   均值 {gap['mean']:+.1f}  均值-2σ = {gap['lower_bound']:+.1f}")
        for line in _wrap(gap["reason"], 50):
            add(f"     {line}")
    add("")

    add("【三、结论】")
    add(f"  {v.get('headline', '(未判定)')}")
    add("")
    for line in _wrap(v.get("reading", "")):
        add(f"  {line}")
    add("")

    nq_pass = summary.get("note_quality_pass_rate", 0.0)
    add("【四、辅助指标】")
    add(f"  笔记质量通过率：{nq_pass * 100:.1f}%")
    add("")

    # v0.4.1：结论的「但是」必须跟结论一起出现。
    # 少了这段，读者会把「增益无法与噪声区分」直接读成「学习没用」，
    # 而在 PRE 90 分的前提下它也可能是「根本没有可测量的空间」。
    caveats = v.get("caveats") or []
    add("【五、结论的限制（必读）】")
    if not caveats:
        add("  无特别限制。")
    else:
        for i, c in enumerate(caveats, 1):
            add(f"  ({i}) {c}")
            add("")
    add("")

    add("【六、术语纪律】")
    add("  本实验没有任何参数更新、微调或权重修改。")
    add("  TRANSFER_GAIN 为正只能表述为「观察到能力迁移迹象」，")
    add("  绝不能表述为「模型训练成功」。")
    add("")

    add(f"题库：{summary.get('bank_version', '')}  "
        f"指纹 {str(summary.get('bank_fingerprint', ''))[:16]}…")
    add(f"材料校验：{str(summary.get('material_checksum', ''))[:16]}…")
    add("=" * 74)
    return "\n".join(lines)
