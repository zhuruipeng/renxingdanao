"""brain/report_v2.py —— V0.5 报告层。

这一层唯一不可协商的规则
------------------------
**任何指标都不得被称为「模型训练成功」。**

本项目没有训练、没有微调、没有梯度更新。跑的是一次前向推理 + 一次
提示工程。把它叫「训练成功」是**事实性错误**，不是措辞问题：
它会让读者以为存在一个可累积的参数状态，而实际上每次实验之间
没有任何东西被持久化到模型里。

所以本模块里所有 gain 都被拆成四个互不替代的观测口径：

===================== ====================================================
指标口径               它回答的问题
===================== ====================================================
external_memory_gain  「能不能查自己写的笔记答对题」
memory_retention_gain「笔记收走以后还记得多少」
transfer_gain「换场景还能不能做对」
objective_practice_gain「能不能写出机器能验的产出」
===================== ====================================================

四者都是**行为观测**，不是能力状态。哪一项高都不等于「学会了」，
更不等于「训练成功」。

机器分与 LLM 分必须分表
------------------------
`practice_runs.objective_score` 是机器判分，
`practice_evaluations.evaluator_score` 是 LLM 评审。
两者回答不同问题，合并成一个「实践分」会同时丢掉信息：
机器 100 / LLM 30 说明产出正确但工程上不可用；
机器 30 / LLM 90 说明 LLM 被表面格式骗了。
所以本模块有两个独立表，绝不出现在同一行做加权。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from brain import benchmark_v2 as b2
from brain import novel_material as nm
from brain import stats_v2 as st
from memory.db import checksum
from practice.objective import tasks_meta, task_diagnostics

#: 全文禁止出现的说法。这些短语一旦出现在报告里，就是在越权下结论。
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "训练成功",
    "训练完成",
    "训练有效",
    "学会了规则",
    "已掌握该知识",
    "模型学会了",
    "习得了",
)

#: 允许出现的**否定式引用**形式。
#:
#: 报告必须能写「本指标不代表训练成功」——否则就没法自我限定。
#: 所以检查的不是简单的子串出现，而是「去掉否定前缀后是否仍是断言」。
#: 命中这些前缀的片段会被剔除再检查。
NEGATION_PREFIXES: tuple[str, ...] = (
    "不得", "不许", "不能", "不会", "并非", "不是", "非", "没有",
    "无", "禁止", "拒绝", "不代表", "不等于", "谈不上", "谈不上说",
)

#: 否定词回看窗口。命中禁词时向前找这么多个字符，
#: 窗口内出现否定词就视为「否定式引用」而非断言。
#:
#: 用窗口而不是紧邻匹配，是因为中文免责句式多种多样：
#:   不能称为「训练成功」   ← 中间隔了「称为」
#:   并不代表模型训练成功
#:   谈不上训练成功
#: 紧邻匹配只能覆盖其中一种，剩下两种会误报——
#: 而误报的代价是报告层被迫闭嘴，连免责声明都不敢写。
NEGATION_LOOKBACK = 10

#: 允许 transfer_gain 使用的最强措辞上限
TRANSFER_MAX_CLAIM = "四者中最接近「概念形成」的观测口径"


# ==========================================================================
# 越权措辞检查
# ==========================================================================


def _is_negated(text: str, idx: int) -> bool:
    """禁词出现在 `idx` 时，判断它前面是否被否定词覆盖。"""
    window = text[max(0, idx - NEGATION_LOOKBACK):idx]
    return any(p in window for p in NEGATION_PREFIXES)


def scan_forbidden(text: str) -> list[str]:
    """找出文本里所有越权**断言**。

    否定式引用（「本指标不代表训练成功」）不算越权——
    报告必须能自我限定，否则唯一的选择就是闭嘴。
    """
    hits: list[str] = []
    for phrase in FORBIDDEN_PHRASES:
        start = text.find(phrase)
        while start != -1:
            if not _is_negated(text, start):
                hits.append(phrase)
                break
            start = text.find(phrase, start + 1)
    return hits


def assert_no_forbidden(text: str, where: str = "报告") -> None:
    """硬门禁：报告里出现越权断言就直接抛异常。

    放在生成路径上而不是事后检查，是因为越权结论一旦写进报告
    就会被下游当成事实传下去——事后靠人读是拦不住的。
    """
    hits = scan_forbidden(text)
    if hits:
        raise AssertionError(
            f"{where}出现越权措辞{hits}。"
            f"本项目没有训练或微调，任何指标都不许被表述为「训练成功」。"
        )


# ==========================================================================
# 单次 trial 报告
# ==========================================================================


def build_trial_report(
    trial: dict[str, Any],
    practice_runs: Sequence[dict[str, Any]] = (),
    evaluator_rows: Sequence[dict[str, Any]] = (),
    stage_items: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """一次 trial 的完整报告结构（不格式化，便于测试断言）。

    `stage_items` 来自 `v2_stage_items` 表，是隔离自检的**唯一**依据。
    不从 `detail` 里取，因为 detail 是实验结束时才落的快照，
    拿快照自检等于自己验自己签的名。
    """
    s = trial["scores"]
    detail = trial.get("detail") or {}

    bank = b2.bank_diagnostics()
    tasks = task_diagnostics()
    cd = trial.get("diagnostics") or {}

    # 单次 trial 的 gap 分析：不是 gain 判定（那是跨 trial 的事）
    pre = float(s.get("pre_blind") or 0.0)
    deltas = {
        "external_memory_gain": round(float(s.get("open_book") or 0.0) - pre, 2),
        "memory_retention_gain": round(float(s.get("memory_blind") or 0.0) - pre, 2),
        "transfer_gain": round(float(s.get("transfer_blind") or 0.0) - pre, 2),
        "objective_practice_gain": round(
            float(s.get("objective_practice_avg") or 0.0) - pre, 2
        ),
    }

    # transfer - memory：区分「背住了」与「会用」
    mem = float(s.get("memory_blind") or 0.0)
    tr = float(s.get("transfer_blind") or 0.0)

    consistency = _consistency(stage_items)

    # 隔离违规一律并入 caveats，不允许藏在别处
    caveats: list[str] = []
    for item in practice_runs:
        if item.get("parse_error"):
            caveats.append(
                f"任务 {item.get('task_id')} 产出无法解析：{item['parse_error']}；"
                f"该任务按 0 分计入"
            )
    if not practice_runs:
        caveats.append("本次实验没有任何 objective practice 记录")
    if not evaluator_rows:
        caveats.append("本次实验没有 LLM 评审记录，evaluator_practice_avg 不可用")
    caveats.extend(consistency["problems"])

    report = {
        "version": "V0.5",
        "experiment_id": trial.get("experiment_id"),
        "batch_id": trial.get("batch_id", ""),
        "trial_no": trial.get("trial_no", 1),
        "topic": nm.NOVEL_TOPIC,
        "scores": s,
        "gains": deltas,
        "practice": _practice_tables(practice_runs, evaluator_rows),
        "consistency_check": consistency,
        "bank": {
            "version": b2.BANK_VERSION,
            "fingerprint": b2.BANK_FINGERPRINT,
            "total": bank["total_items"],
            "items_per_kp": bank["items_per_kp"],
            "problems": bank["problems"],
        },
        "material": {
            "topic": nm.NOVEL_TOPIC,
            "checksum": checksum(nm.NOVEL_MATERIAL),
        },
        "tasks": {
            "fingerprint": tasks_meta()["fingerprint"],
            "total": tasks["total"],
            "kinds": tasks["kinds"],
            "per_kp": tasks["per_kp"],
            "problems": tasks["problems"],
        },
        "stage_order": list(detail.get("stage_order") or []),
        "transfer_vs_memory": round(tr - mem, 2),
        "caveats": caveats,
        "diagnostics": cd,
    }
    # 自检：拼好的报告里不许含越权措辞
    assert_no_forbidden(str(report), "单次 trial 报告")
    return report


def _practice_tables(
    runs: Sequence[dict[str, Any]], evals: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """机器判分与 LLM 评审**分表**返回，绝不合并。"""
    obj_rows = [
        {
            "task_id": r.get("task_id"),
            "kp": r.get("kp"),
            "kind": r.get("kind"),
            "title": r.get("title"),
            "objective_score": r.get("objective_score"),
            "objective_passed": bool(r.get("objective_passed")),
            "parse_error": r.get("parse_error") or "",
            "objective_result": r.get("objective_result"),
        }
        for r in runs
    ]
    llm_rows = [
        {
            "task_id": e.get("task_id") or "",
            "kp": e.get("kp"),
            "evaluator_score": e.get("score"),
            "usable": bool(e.get("usable")),
            "verdict": e.get("verdict"),
            "feedback": e.get("feedback", ""),
        }
        for e in evals
    ]
    obj_scores = [float(r["objective_score"]) for r in obj_rows
                  if r["objective_score"] is not None]
    llm_scores = [float(r["evaluator_score"]) for r in llm_rows
                  if r["evaluator_score"] is not None]
    return {
        "objective": {
            "rows": obj_rows,
            "total": len(obj_rows),
            "avg": st.mean(obj_scores),
            "pass_rate": round(
                sum(1 for r in obj_rows if r["objective_passed"])
                / (len(obj_rows) or 1),
                4,
            ),
            "note": (
                "机器判分：算法输出/SQL 结果/单测/JSON schema 四类硬校验。"
                "产出结构不对就是不对，不受措辞影响。"
            ),
        },
        "evaluator": {
            "rows": llm_rows,
            "total": len(llm_rows),
            "avg": st.mean(llm_scores),
            "note": (
                "LLM 评审：回答「工程上能不能用」，与机器判分回答不同问题。"
                "两者不一致本身就是结论，不做加权平均。"
            ),
        },
        "divergence": _divergence(obj_rows, llm_rows),
    }


def _divergence(
    obj_rows: Sequence[dict[str, Any]], llm_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """机器分与 LLM 分的不一致清单。"""
    llm_by_task = {r["task_id"]: r for r in llm_rows}
    out: list[dict[str, Any]] = []
    for r in obj_rows:
        e = llm_by_task.get(r["task_id"])
        if e is None:
            continue
        o = float(r["objective_score"] or 0.0)
        l = float(e["evaluator_score"] or 0.0)
        if abs(o - l) >= 30.0:
            out.append({
                "task_id": r["task_id"],
                "objective_score": o,
                "evaluator_score": l,
                "direction": "llm_higher" if l > o else "objective_higher",
            })
    return {
        "rows": out,
        "count": len(out),
        "unlinked": sum(1 for r in llm_rows if not r["task_id"]),
        "note": (
            "差值 ≥30 分的任务。出现分歧说明「机器认为对」和「人认为可用」"
            "是两件事，必须分别看，不要合成一个实践分。"
        ),
    }


BLIND_STAGE_NAMES = ("pre_blind", "memory_blind", "transfer_blind")

#: 泄漏探针的最小长度。短于此值的答案不做子串判定——
#: 「是」「对」这类答案在任何文本里都命中，全报泄漏等于没报。
_MIN_PROBE_LEN = 8


def _answer_in(text: str, answer: str, probe_len: int = 24) -> bool:
    """判断参考答案是否出现在给定文本里。

    用**前缀子串**判定，不用完全相等：LLM 不会一字不差地复制答案，
    但只要它把答案开头的一段复述进 prompt，就已经是泄漏。

    探针长度有下限：答案是「是」这类单字/双字短语时，
    任何正常文本都会包含它，逐条报泄漏等于把真泄漏埋掉。
    探针太短会误报，太长会漏报（改写了开头），所以取两头兼顾的窗口。
    """
    probe = (answer or "").strip()[:probe_len]
    if len(probe) < _MIN_PROBE_LEN:
        return False
    return probe in (text or "")


def _consistency(stage_items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """隔离与流程自检。数据源是 `v2_stage_items` 的实际行。

    **注意 `reference_answer` 字段本身不是泄漏。**
    它是判分基准，必须落库，否则事后无法复核判分是否合理。
    泄漏的定义是「参考答案进入了送给模型的 prompt」，
    即 `context_used` 或 `question` 里出现了答案文本。
    """
    problems: list[str] = []
    blind_empty = True
    open_present = False
    leaks: list[str] = []
    seen_stages: dict[str, int] = {}

    for row in stage_items:
        stage = row.get("stage") or "?"
        ctx = (row.get("context_used") or "").strip()
        q = row.get("question") or ""
        seen_stages[stage] = seen_stages.get(stage, 0) + 1

        if stage in BLIND_STAGE_NAMES and ctx:
            blind_empty = False
            problems.append(f"{stage} 阶段 context_used 非空，盲考被污染")
        if stage == "open_book" and ctx:
            open_present = True
        # 答案是否泄进 prompt：context_used 或题面本身
        if _answer_in(ctx, row.get("reference_answer") or ""):
            leaks.append(f"{row.get('item_id')}@context")
        if _answer_in(q, row.get("reference_answer") or ""):
            leaks.append(f"{row.get('item_id')}@question")

    if leaks:
        problems.append(
            f"参考答案泄进 prompt：{leaks}；"
            f"考生看到的文本里出现了答案，该题分数作废"
        )

    for stage in BLIND_STAGE_NAMES:
        if stage not in seen_stages:
            problems.append(f"缺少 {stage} 阶段记录，实验流程不完整")
    if "open_book" not in seen_stages:
        problems.append("缺少 open_book 阶段记录，实验流程不完整")

    return {
        "blind_context_empty": blind_empty,
        "open_book_context_present": open_present,
        "answer_leak": bool(leaks),
        "leaked": leaks,
        "stage_counts": seen_stages,
        "stage_row_count": len(stage_items),
        "problems": problems,
    }


# ==========================================================================
# 批次报告（跨 trial）
# ==========================================================================


def build_batch_report(
    summary: dict[str, Any],
    batch_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """跨 trial 统计报告。`summary` 来自 `stats_v2.aggregate_trials`。"""
    diag = st.diagnose_batch(summary)
    gains = summary.get("gains") or {}
    report = {
        "version": "V0.5",
        "batch_id": (batch_meta or {}).get("batch_id", ""),
        "topic": nm.NOVEL_TOPIC,
        "model": (batch_meta or {}).get("model", ""),
        "trials": summary.get("trials", 0),
        "metrics": summary.get("metrics") or {},
        "gains": gains,
        "per_kp": summary.get("per_kp") or {},
        "ceiling": summary.get("ceiling") or {},
        "floor": summary.get("floor") or {},
        "sample_size_diagnostics": summary.get("sample_size_diagnostics") or {},
        "diagnosis": diag,
        "frozen": _frozen_block(batch_meta or {}),
        "claims": _claim_table(gains),
    }
    assert_no_forbidden(str(report), "批次报告")
    return report


def build_trial_report_from_store(
    store: Any, experiment_id: int
) -> dict[str, Any]:
    """从 `V2Store` 直接拉数据构建单次报告（不依赖内存里的 result 对象）。"""
    row = store.experiment(experiment_id)
    if row is None:
        raise KeyError(f"V2 experiment {experiment_id} 不存在")
    detail = json.loads(row.get("detail_json") or "{}")
    trial = {
        "experiment_id": int(row["id"]),
        "batch_id": row.get("batch_id") or "",
        "trial_no": int(row.get("trial_no") or 1),
        "scores": {
            "pre_blind": row.get("pre_score"),
            "open_book": row.get("open_book_score"),
            "memory_blind": row.get("memory_blind_score"),
            "transfer_blind": row.get("transfer_blind_score"),
            "objective_practice_avg": row.get("objective_practice_score"),
            "objective_practice_pass_rate": row.get("objective_practice_pass_rate"),
            "evaluator_practice_avg": row.get("evaluator_practice_score"),
            "note_quality_avg": row.get("note_quality_avg"),
        },
        "detail": detail,
        "diagnostics": json.loads(row.get("diagnostics_json") or "{}"),
    }
    items = store.items(experiment_id)
    runs = store.practice_runs(experiment_id)
    evals = store.evaluations_for(experiment_id)
    return build_trial_report(trial, runs, evals, items)


def build_batch_report_from_store(store: Any, batch_id: str) -> dict[str, Any]:
    """从库读一个批次的全部 trial，统计后出报告。"""
    meta = store.batch_meta(batch_id) or {"batch_id": batch_id}
    trials = st.load_trials(store, batch_id)
    summary = st.aggregate_trials(trials)
    store.finish_batch(batch_id, summary)
    return build_batch_report(summary, meta)


def _frozen_block(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "bank_fingerprint": meta.get("bank_fingerprint", ""),
        "material_checksum": meta.get("material_checksum", ""),
        "tasks_fingerprint": meta.get("tasks_fingerprint", ""),
        "note": (
            "三份指纹在每次实验开始与结束时各校验一次。"
            "任一变化 → 该批次全部结果作废（模型不能改题目/材料/任务）。"
        ),
    }


def _claim_table(gains: dict[str, Any]) -> list[dict[str, Any]]:
    """每个 gain 允许说到什么程度。

    这一张表是「禁止称训练成功」的落地形式：不靠措辞自律，
    而是给每个指标预先绑定一句它能支撑的话和一句它不能支撑的话。
    """
    allowed = {
        "external_memory_gain": (
            "能利用自己写的外部记忆答题。",
            "不能说明规则被内化；也不能说明换场景仍可用。",
        ),
        "memory_retention_gain": (
            "收回全部外部记忆后仍能答对同一批题。",
            "不能区分「理解规则」与「记住了笔记措辞」。",
        ),
        "transfer_gain": (
            TRANSFER_MAX_CLAIM,
            "仍不等于「模型被训练过」：本项目没有训练或微调，"
            "且单次上下文内的表现不等于参数里的持久状态。",
        ),
        "objective_practice_gain": (
            "能写出机器可验证的正确产出。",
            "与考试卷不同题型，属可观测代理指标，不能当同题对比读。",
        ),
    }
    rows: list[dict[str, Any]] = []
    for label, g in gains.items():
        ok, not_ok = allowed.get(label, ("（未登记）", "（未登记）"))
        rows.append({
            "metric": label,
            "mean": g.get("mean"),
            "ci": [g.get("ci_low"), g.get("ci_high")],
            "verdict": g.get("verdict"),
            "description": g.get("description", ""),
            "supports": ok,
            "does_not_support": not_ok,
            "provisional": bool(g.get("provisional")),
        })
    return rows


# ==========================================================================
# 文本格式化
# ==========================================================================


def _wrap(text: str, width: int = 58) -> list[str]:
    """按显示宽度折行（中文按 2 列算）。"""
    out: list[str] = []
    line = ""
    w = 0
    for ch in text:
        cw = 2 if ord(ch) > 0x2E80 else 1
        if w + cw > width:
            out.append(line)
            line, w = "", 0
        line += ch
        w += cw
    if line:
        out.append(line)
    return out


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "-"
    if pct:
        return f"{float(v) * 100:.1f}%"
    return f"{float(v):.1f}"


def _ci(d: dict[str, Any]) -> str:
    lo, hi = d.get("ci_low"), d.get("ci_high")
    if lo is None or hi is None:
        return "(不可用)"
    return f"[{lo}, {hi}]"


_VERDICT_CN = {
    "positive": "正向",
    "non_positive": "非正向",
    "noisy": "证据不足",
}


def format_batch_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 76)
    add(f"Agent-01 V0.5 / Benchmark V2 报告 · {report['trials']} 次重复")
    add(f"主题：{report['topic']}")
    add(f"模型：{report.get('model') or '(未标注)'}")
    if report.get("batch_id"):
        add(f"批次：{report['batch_id']}")
    add("=" * 76)
    add("")

    add("【一、口径声明（先读这一段）】")
    for ln in _wrap(
        "本项目没有训练、没有微调、没有梯度更新，每次实验都是一次前向推理。"
        "下面四项指标全部是行为观测口径，任何一项都不构成"
        "「模型被训练过」这一结论——那不是措辞保守，是事实错误。"
        "报告的越权措辞门禁会在生成时直接拦下这类表述。"
    ):
        add(f"  {ln}")
    add("")

    add("【二、四项观测指标· 跨次分布】")
    add(f"  {'指标':<22}{'均值':>7}{'σ':>7}{'最小':>7}{'最大':>7}  {'95% CI':>16}")
    labels = [
        ("pre_blind", "PRE_BLIND"),
        ("open_book", "OPEN_BOOK"),
        ("memory_blind", "MEMORY_BLIND"),
        ("transfer_blind", "TRANSFER_BLIND"),
        ("objective_practice_avg", "机器判分实践"),
        ("objective_practice_pass_rate", "机器判分通过率"),
        ("evaluator_practice_avg", "LLM 评审实践"),
        ("note_quality_avg", "笔记质量"),
    ]
    m = report["metrics"]
    for key, name in labels:
        d = m.get(key)
        if not d:
            add(f"  {name:<22}{'-':>7}{'-':>7}{'-':>7}{'-':>7}  {'-':>16}")
            continue
        pct = key.endswith("_rate")
        add(
            f"  {name:<22}{_fmt(d['mean'], pct):>7}{_fmt(d['std'], pct):>7}"
            f"{_fmt(d['min'], pct):>7}{_fmt(d['max'], pct):>7}  {_ci(d):>16}"
        )
        if d.get("caveat"):
            for ln in _wrap("! " + d["caveat"], 62):
                add(f"      {ln}")
    add("")

    add("【三、四类 Gain · 语义严格分离，禁止混称】")
    for row in report["claims"]:
        add(f"  {row['metric']}")
        add(f"    均值 {row['mean']}   95% CI {_ci(row)}   "
            f"判定 {_VERDICT_CN.get(row['verdict'], row['verdict'])}")
        if row.get("provisional"):
            add("    [暂定] 样本方差为 0，该判定不可当作统计确证")
        for ln in _wrap("口径：" + row["description"], 62):
            add(f"    {ln}")
        for ln in _wrap("能说明：" + row["supports"], 62):
            add(f"    {ln}")
        for ln in _wrap("不能说明：" + row["does_not_support"], 62):
            add(f"    {ln}")
        add("")

    add("【四、天花板 / 地板诊断】")
    c, f = report["ceiling"], report["floor"]
    add(f"  天花板：{'命中' if c.get('hit') else '未命中'}"
        f"  PRE 均值 {_fmt(c.get('pre_mean'))}  剩余空间 {_fmt(c.get('headroom'))}")
    for ln in _wrap(c.get("note", ""), 66):
        add(f"    {ln}")
    add(f"  地板　：{'命中' if f.get('hit') else '未命中'}")
    for ln in _wrap(f.get("note", ""), 66):
        add(f"    {ln}")
    add("")

    add("【五、样本量与方差诊断】")
    ssd = report["sample_size_diagnostics"]
    add(f"  trial 数 {ssd.get('n_trials', 0)}"
        f"（≥3 为{'达标' if ssd.get('sufficient_trials') else '不达标'}）")
    if ssd.get("all_zero_variance"):
        add("  结果离散度：无（各次结果完全相同 → 重复次数不提供额外信息）")
    elif ssd.get("has_variation") is False:
        add("  结果离散度：无")
    if ssd.get("zero_variance_gains"):
        add(f"  方差为 0 的增益：{ssd['zero_variance_gains']}")
    if ssd.get("thin_kps"):
        add(f"  样本不足的知识点：{ssd['thin_kps']}")
    if ssd.get("high_variance_gains"):
        add(f"  方差过大的增益：{ssd['high_variance_gains']}")
    for p in ssd.get("problems") or []:
        for ln in _wrap("! " + p, 66):
            add(f"    {ln}")
    if not ssd.get("problems"):
        add("    （无诊断问题）")
    add("")

    add("【六、逐知识点】")
    per_kp = report["per_kp"]
    if not per_kp:
        add("  （无逐题数据）")
    for kp, stages in per_kp.items():
        add(f"  {kp}")
        add(f"    {'阶段':<16}{'n':>4}{'均值':>8}{'σ':>7}  {'95% CI':>16}  备注")
        for stage, d in stages.items():
            notes: list[str] = []
            if d.get("insufficient_sample"):
                notes.append("样本不足")
            if d.get("method") == "zero_variance":
                # 确定性 provider 下这一列必然是 0，
                # 不标出来的话 [70, 70] 会被读成「结果非常精确」
                notes.append("方差为0·区间退化")
            add(
                f"    {stage:<16}{d['n']:>4}{_fmt(d['mean']):>8}{_fmt(d['std']):>7}"
                f"  {_ci(d):>16}  {'·'.join(notes)}"
            )
    add("")

    add("【七、结论可用性】")
    d = report["diagnosis"]
    add(f"  可直接下结论：{'是' if d['reliable'] else '否'}")
    if d["caveats"]:
        for i, cav in enumerate(d["caveats"], 1):
            for j, ln in enumerate(_wrap(cav, 66)):
                add(f"  {i}. {ln}" if j == 0 else f"     {ln}")
    else:
        add("    无附加限制")
    add("")

    add("【八、冻结指纹（模型不得修改题目/材料/任务）】")
    fz = report["frozen"]
    add(f"  题库　{fz['bank_fingerprint']}")
    add(f"  材料　{fz['material_checksum']}")
    add(f"  任务集{fz['tasks_fingerprint']}")
    for ln in _wrap(fz["note"], 66):
        add(f"  {ln}")
    add("=" * 76)
    return "\n".join(lines)


def format_trial_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    s = report["scores"]
    add("=" * 72)
    add(f"Agent-01 V0.5 单次实验报告 · 实验 #{report['experiment_id']}"
        f"（第 {report['trial_no']} 次）")
    add(f"主题：{report['topic']}")
    add("=" * 72)
    add("")
    add("【阶段顺序】" + " → ".join(report["stage_order"]))
    add("")
    add("【分数】")
    for key, name in (
        ("pre_blind", "PRE_BLIND"),
        ("open_book", "OPEN_BOOK"),
        ("memory_blind", "MEMORY_BLIND"),
        ("transfer_blind", "TRANSFER_BLIND"),
        ("objective_practice_avg", "机器判分实践"),
        ("evaluator_practice_avg", "LLM 评审实践"),
        ("note_quality_avg", "笔记质量"),
    ):
        add(f"  {name:<18}{_fmt(s.get(key)):>8}")
    add("")
    add("【四类 gain（禁止称训练成功）】")
    for k, v in report["gains"].items():
        add(f"  {k:<26}{v:+.1f}")
    add(f"  {'transfer - memory':<26}{report['transfer_vs_memory']:+.1f}"
        f"   正数 = 同场景能答，换场景掉分")
    add("")
    p = report["practice"]
    add("【机器判分（objective）】")
    for r in p["objective"]["rows"]:
        mark = "PASS" if r["objective_passed"] else "FAIL"
        add(f"  [{mark}] {r['objective_score']:>6}  {r['task_id']}  ({r['kind']})")
    add(f"  均分 {_fmt(p['objective']['avg'])}   "
        f"通过率 {p['objective']['pass_rate'] * 100:.1f}%")
    add("")
    add("【LLM 评审（evaluator）】")
    for r in p["evaluator"]["rows"]:
        u = "可用" if r["usable"] else "不可用"
        add(f"  [{u}] {r['evaluator_score']:>6}  {r['task_id']}")
    add(f"  均分 {_fmt(p['evaluator']['avg'])}")
    if p["divergence"]["count"]:
        add(f"  ! 机器分与 LLM 分相差 ≥30 的任务 {p['divergence']['count']} 个：")
        for r in p["divergence"]["rows"]:
            add(f"    {r['task_id']}: 机器 {r['objective_score']} / "
                f"LLM {r['evaluator_score']}（{r['direction']}）")
    if p["divergence"]["unlinked"]:
        add(f"  ! {p['divergence']['unlinked']} 条评审记录缺 task_id，"
            f"无法与机器判分对齐，分歧定位不到具体任务")
    add("")
    add("【隔离自检】")
    c = report["consistency_check"]
    add(f"  盲考 context 恒空：{'是' if c['blind_context_empty'] else '否'}")
    add(f"  开卷有 context　：{'是' if c['open_book_context_present'] else '否'}")
    add(f"  参考答案未泄题　：{'是' if not c['answer_leak'] else '否'}")
    add(f"  阶段记录行数　：{c['stage_row_count']}")
    add(f"  各阶段行数　　　：{c['stage_counts']}")
    add("")
    if report["caveats"]:
        add("【限制】")
        for cav in report["caveats"]:
            for ln in _wrap("- " + cav, 66):
                add(f"  {ln}")
        add("")
    add("=" * 72)
    return "\n".join(lines)


__all__ = [
    "FORBIDDEN_PHRASES",
    "TRANSFER_MAX_CLAIM",
    "assert_no_forbidden",
    "build_batch_report",
    "build_trial_report",
    "format_batch_text",
    "format_trial_text",
    "scan_forbidden",
]
