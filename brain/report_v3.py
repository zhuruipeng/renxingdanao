"""brain/report_v3.py —— V0.6 报告层。

报告必须把五类口径**分开列**，不允许压缩成一句「提升了 X 分」：

    1. Observed gain       学习组自己的前后差
    2. Control gain        对照组自己的前后差
    3. Difference-in-gains 上面两者之差
    4. Confidence interval 这个差值有多不确定
    5. Effect size         相对于波动有多大

以及客观 held-out 口径的同五项（分数不同、口径相同）。

措辞硬门禁
----------
未经代码证据支持时，禁止出现「训练成功」「模型已经学会」
「模型能力已经改变」这类断言。这不是文字游戏——
这类话读起来像结论，实际上没有实验支撑，
一旦写进报告就会被当成事实引用。

放行与否由 :func:`brain.causal_v3.claim_permission` 决定：
CI 不跨 0、效应量达标、无污染检出、样本量≥2，四个条件全满足
才允许说「归因于学习」；否则只能陈述观测值。
"""

from __future__ import annotations

from typing import Any, Sequence

from brain import causal_v3 as c3
from brain import group_isolation as gi
from brain import heldout_bank as ho
from brain import control_material as cm
from brain import novel_material as nm
from memory.db import Database

#: 五类口径的中文标签
LENS_LABELS: dict[str, str] = {
    "observed_gain": "① 学习组前后差（Observed gain）",
    "control_gain": "② 对照组前后差（Control gain）",
    "difference_in_gains": "③ 差值的差值（Difference-in-gains）",
    "confidence_interval": "④ 95% 置信区间（Confidence interval）",
    "effect_size": "⑤ 效应量（Effect size）",
}

_EVAL_CN = {
    "pass": "通过",
    "fail": "未通过",
}


def _fmt_ci(ci: dict[str, Any]) -> str:
    if ci.get("ci_low") is None or ci.get("ci_high") is None:
        return f"不可用（{ci.get('method')}，n={ci.get('n')}）"
    return f"[{ci['ci_low']}, {ci['ci_high']}]"


def _fmt_gain(d: dict[str, Any]) -> str:
    return (
        f"mean={d.get('mean')} std={d.get('std')} n={d.get('n')} "
        f"CI={_fmt_ci(d)}"
    )


def _providers_text(rows: Sequence[dict[str, str]]) -> str:
    if not rows:
        return "（未记录）"
    return ", ".join(f"{r['provider']}/{r['model']}" for r in rows)


def _lens_block(summary: dict[str, Any]) -> list[str]:
    """五类口径逐行列出。"""
    lines = [LENS_LABELS["observed_gain"], f"   {_fmt_gain(summary['observed_gain'])}"]
    lines.append(LENS_LABELS["control_gain"])
    lines.append(f"   {_fmt_gain(summary['control_gain'])}")
    lines.append(LENS_LABELS["difference_in_gains"])
    lines.append(f"   {_fmt_gain(summary['difference_in_gains'])}")
    lines.append(LENS_LABELS["confidence_interval"])
    ci = summary["confidence_interval"]
    lines.append(
        f"   {_fmt_ci(ci)}  method={ci.get('method')} level={ci.get('level')}"
    )
    if ci.get("caveat"):
        lines.append(f"   ⚠ {ci['caveat']}")
    lines.append(LENS_LABELS["effect_size"])
    es = summary["effect_size"]
    if es.get("d") is None:
        lines.append(f"   不可计算（{es.get('method')}）：{es.get('note', '')}")
    else:
        lines.append(
            f"   Cohen's d = {es['d']}（{es['label']}，阈值 {es['threshold']}）"
            f"  学习组 gain 均值={es.get('treatment_mean')}"
            f"  对照组 gain 均值={es.get('control_mean')}"
            f"  合并 SD={es.get('pooled_sd')}"
        )
    return lines


def build_report(
    *,
    batch_id: str,
    db: Database,
    trials: Sequence[dict[str, Any]],
    objective_trials: Sequence[dict[str, Any]],
    run_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """汇总一份 V0.6 报告。

    ``trials`` / ``objective_trials`` 是各次 run 的成对指标。
    """
    diff = c3.aggregate_difference(list(trials))
    obj_diff = c3.objective_difference_summary(list(objective_trials))

    # 污染与顺序
    contamination: list[dict[str, Any]] = []
    order: list[dict[str, Any]] = []
    for rid in run_ids or ():
        contamination.append(gi.contamination_report(db, rid))
        order.append(gi.order_audit(db, rid))
    contam_clean = all(c["clean"] for c in contamination) if contamination else True
    order_clean = all(o["clean"] for o in order) if order else True

    perm = c3.claim_permission(diff, contam_clean, len(trials))
    obj_perm = c3.claim_permission(obj_diff, contam_clean, len(objective_trials))

    report: dict[str, Any] = {
        "batch_id": batch_id,
        "n_trials": len(trials),
        "providers": {
            "candidate": _provider_block(db, run_ids),
            "evaluator": _provider_block(db, run_ids, key="evaluator"),
        },
        "materials": {
            "learning": {
                "topic": nm.NOVEL_TOPIC,
                "tokens": cm.token_estimate(nm.NOVEL_MATERIAL),
                "checksum": gi.frozen_fingerprints()[
                    "learning_material_checksum"
                ],
            },
            "control": {
                "topic": cm.CONTROL_TOPIC,
                "tokens": cm.token_estimate(cm.CONTROL_MATERIAL),
                "checksum": gi.frozen_fingerprints()[
                    "control_material_checksum"
                ],
            },
            "length_ratio": cm.length_ratio(),
        },
        "exam_metrics": diff,
        "objective_metrics": obj_diff,
        "contamination": contamination,
        "order_audit": order,
        "call_failures": _call_failures(db, run_ids),
        "contamination_clean": contam_clean,
        "order_clean": order_clean,
        "claim_permission": perm,
        "objective_claim_permission": obj_perm,
        "residual_contamination": ho.residual_contamination(),
        "methodology_warnings": _methodology_warnings(
            len(trials), contam_clean, order_clean
        ),
    }
    return report


def _call_failures(db: Database, run_ids: Sequence[str]) -> list[dict[str, str]]:
    """本次实验里所有考生调用失败。

    调用失败率必须印在报告上：它是「两组分数可比」的底线指标。
    失败样本若被当成「模型答不出」，两组就被同一批噪声污染，
    而 difference-in-gains 会归零得莫名其妙。
    """
    if not run_ids:
        return []
    marks = ",".join("?" for _ in run_ids)
    rows = db.query_all(
        f"SELECT run_id, exp_group, stage, item_id, error"
        f"  FROM v3_call_failures WHERE run_id IN ({marks}) ORDER BY id",
        tuple(run_ids),
    )
    return [dict(r) for r in rows]   # type: ignore[arg-type]


def _provider_block(
    db: Database, run_ids: Sequence[str], key: str = "candidate"
) -> list[dict[str, str]]:
    if not run_ids:
        return []
    marks = ",".join("?" for _ in run_ids)
    rows = db.query_all(
        f"SELECT DISTINCT {key}_provider AS provider, {key}_model AS model"
        f"  FROM v3_runs WHERE run_id IN ({marks})",
        tuple(run_ids),
    )
    return [{"provider": r["provider"], "model": r["model"]} for r in rows]


def _methodology_warnings(
    n_trials: int, contam_clean: bool, order_clean: bool
) -> list[str]:
    out: list[str] = []
    if n_trials < 2:
        out.append(
            f"本次只有 {n_trials} 次 trial。样本量 < 2 时置信区间退化为点、"
            f"效应量不可估，**不能给出任何统计结论**。"
        )
    else:
        out.append(
            f"本次 {n_trials} 次 trial。t 分布近似在自由度 "
            f"{max(1, n_trials - 1)} 处取值，n<10 时区间本身很不稳定。"
        )
    if not contam_clean:
        out.append("检出污染，本次结果不可用于归因。")
    if not order_clean:
        out.append("阶段顺序不合规，Control 的 PRE 可能已被污染。")
    out.append(
        "PRE / POST / TRANSFER 三卷对两组是同一套题——这是必须的，"
        "否则比较的不是同一把尺子；但也意味着题目本身无法提供组间区分度。"
    )
    out.append(
        "两组做的是**同一套** practice 任务集（参数与 held-out 全部错开）。"
        "早先让对照组做 held-out 任务集是错的：那批题正是 post_objective 的"
        "测试题，对照组练过再测会让 objective_difference_in_gains "
        "系统性偏向假阴性。同题的另一面是：两组 practice 的**分数**"
        "可以直接相减，但这道差值同时包含「学了什么」与「会不会做题」，"
        "仍然不能当作因果结论。"
    )
    out.append(
        "对照组练习的是 Zephyra 的题（它没学过，大概率做不对），"
        "却没做 Veyra 的题。也就是说两组在 practice 阶段"
        "承受的**挫败感与题面难度并不对等**——"
        "若模型的情绪/尝试次数受挫败影响，这会成为一个未控制的混淆变量。"
    )
    out.append(
        "对照组材料是另一个虚构系统（Veyra），与目标知识无交集。"
        "它能控制「读了多少字」这个变量，但不能完全等价于"
        "「学了同样难度、同样抽象层级的知识」——"
        "材料难度差异仍是未控制的混淆变量。"
    )
    out.append(
        "评估器与考生是不同 provider 实例，但**同属一个模型家族**。"
        "共享的先验知识可能让评估器对两组答案给出系统性偏好。"
    )
    return out


def format_text(report: dict[str, Any]) -> str:
    """渲染成可读文本。所有措辞过越权门禁。"""
    L: list[str] = []
    n = report["n_trials"]

    L.append("=" * 68)
    L.append(f"Agent-01 V0.6 双组实验报告  batch={report['batch_id']}")
    L.append("=" * 68)
    L.append("")
    L.append(f"trial 次数：{n}")
    cand = report["providers"]["candidate"]
    ev = report["providers"]["evaluator"]
    L.append(f"考生provider：{_providers_text(cand)}")
    L.append(f"评估 provider：{_providers_text(ev)}")
    same = bool(cand and ev and cand[0] == ev[0])
    L.append(
        "考生与评估是否同一实例："
        + ("是 —— 污染检测会判不合格" if same else "否")
    )
    L.append("")

    m = report["materials"]
    L.append("-" * 68)
    L.append("材料")
    L.append("-" * 68)
    L.append(
        f"  学习材料：{m['learning']['topic']}  "
        f"约 {m['learning']['tokens']} token"
    )
    L.append(
        f"  对照材料：{m['control']['topic']}  "
        f"约 {m['control']['tokens']} token"
    )
    L.append(f"  长度比：{m['length_ratio']}（目标 0.90~1.10）")
    L.append("")

    L.append("-" * 68)
    L.append("考试口径（五类指标分开列）")
    L.append("-" * 68)
    L += _lens_block(report["exam_metrics"])
    L.append("")

    L.append("-" * 68)
    L.append("客观 held-out 口径（机器判分，参数与 practice 全部错开）")
    L.append("-" * 68)
    L += _lens_block(report["objective_metrics"])
    L.append("")

    L.append("-" * 68)
    L.append("污染检测")
    L.append("-" * 68)
    if not report["contamination"]:
        L.append("  （无 run 记录）")
    for c in report["contamination"]:
        L.append(f"  run={c['run_id']}  clean={c['clean']}")
        for hit in c["contaminations"]:
            L.append(f"    ✗ [{hit['kind']}] {hit['detail']}")
    L.append("")

    L.append("-" * 68)
    L.append("实验顺序")
    L.append("-" * 68)
    for o in report["order_audit"]:
        L.append(f"  run={o['run_id']}  clean={o['clean']}")
        for p in o["problems"]:
            L.append(f"    ✗ {p}")
    L.append("")

    L.append("-" * 68)
    L.append("考生调用失败")
    L.append("-" * 68)
    fails = report["call_failures"]
    if not fails:
        L.append("  0 次（本次实验的调用全部成功）")
    else:
        L.append(f"  {len(fails)} 次失败 —— 数据已被污染，不可用：")
        for f in fails:
            L.append(
                f"    ✗ {f['exp_group']}/{f['stage']}/{f['item_id']}: "
                f"{f['error'][:80]}"
            )
    L.append("")

    L.append("-" * 68)
    L.append("结论许可")
    L.append("-" * 68)
    for label, perm in (
        ("考试口径", report["claim_permission"]),
        ("客观口径", report["objective_claim_permission"]),
    ):
        # 标签里绝不能出现归因类的字面量。
        # 门禁比的是**字面子串**，不是语义：
        # 「是否允许把差异归因于学习」是元信息，不是结论，
        # 但它包含精确禁用子串「归因于学习」，于是报告把自己拦下了。
        # 报门禁拦的是**陈述**，不是关于「能不能陈述」的元信息。
        allowed = perm["may_attribute_to_learning"]
        L.append(
            f"  {label}：学习效应归因许可 = {'允许' if allowed else '不允许'}"
        )
        for cond, ok in perm["conditions"].items():
            L.append(f"    {'✓' if ok else '✗'} 条件 {cond}")
        for b in perm["blocked_by"]:
            L.append(f"    ✗ {b}")
        L.append(f"    措辞：{perm['required_phrasing']}")
    L.append("")

    L.append("-" * 68)
    L.append("残留污染（参数错开了，但表达层仍重合）")
    L.append("-" * 68)
    for k, v in report["residual_contamination"].items():
        L.append(f"  {k}：{v}")
    L.append("")

    L.append("-" * 68)
    L.append("方法学提醒")
    L.append("-" * 68)
    for w in report["methodology_warnings"]:
        L.append(f"  · {w}")
    L.append("")
    L.append("=" * 68)

    text = "\n".join(L)
    # 措辞硬门禁：拿到归因许可才放行归因类措辞
    may = (
        report["claim_permission"]["may_attribute_to_learning"]
        and report["objective_claim_permission"]["may_attribute_to_learning"]
    )
    c3.assert_claims_allowed(text, may_attribute=may, where="V0.6 报告")
    return text


__all__ = ["LENS_LABELS", "build_report", "format_text"]
