"""brain/stats_v2.py —— V0.5 统计层：把多次 trial 变成能下结论的东西。

V0.4.1 的统计缺陷（全部在本模块被强制暴露）
--------------------------------------------
1. **天花板**：PRE 已 90 分 → 开卷最多剩 10 分空间，
   而噪声 σ=3~5装不下10 分。「增益测不出来」有两种不可区分解释：
   学习没用 / 没有可测量的空间。`detect_ceiling()` 把这层限制
   强制印在结论旁边，不允许悄悄省略。

2. **逐 kp 无样本**：V0.4 每 kp 每卷只 1 题，单题判分误差
   = 整个 kp 的分数。V2 每 kp 每卷 2 题、`aggregate_trials` 会报
   `per_kp_sample_count`，n<`MIN_SAMPLES_FOR_KP` 时结论打上
   `insufficient_sample` 标记。

3. **单次结果当结论**：`verdict_from_stats()` 要求
   mean-2σ 与 0 的关系明确，否则判`noisy` 而不是 `positive`。

为什么用 t 分布而不是正态
------------------------
n 通常是 5~10，样本标准差本身的误差很大，
正态近似的置信区间会**明显偏窄**——给出虚假的确定性。
所以 n < 30 时走 t 分布（查表法，无第三方依赖）。

**t 表覆盖不到的自由度**：t_{0.975} 只列到 df=30。
df>30 时用正态近似 1.96，并在返回值里标 `method`，
让读报告的人知道这段区间是怎么来的。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

#: 天花板判定线：PRE 高于此值 → 后续增益没有足够测量空间
CEILING_LINE = 85.0

#: 地板判定线：PRE 低于此值 → 后续增益可能被「本来就不会」放大
FLOOR_LINE = 15.0

#: 逐 kp 结论需要的最小样本数（每 kp 每卷题数 × trial 数）
MIN_SAMPLES_FOR_KP = 4

#: 判定 positive 所需的最小效应量（分）
MIN_EFFECT = 2.0

#: t 分布 97.5 分位数（双尾 95% CI），df = n - 1
_T_TABLE: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

#: df > 30 时用正态近似
_Z_975 = 1.96


# ==========================================================================
# 基础统计
# ==========================================================================


def mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    """样本标准差（n-1）。n<2 返回 0.0。"""
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return round(math.sqrt(var), 2)


def t_critical(df: int) -> tuple[float, str]:
    """返回 (t 临界值, 方法说明)。"""
    if df <= 0:
        return _Z_975, "normal_approx(df<=0)"
    if df in _T_TABLE:
        return _T_TABLE[df], "t_table"
    if df > 30:
        return _Z_975, "normal_approx(df>30)"
    return _Z_975, "normal_approx"


def confidence_interval(
    values: Sequence[float], level: float = 0.95
) -> dict[str, Any]:
    """均值的置信区间。

    n<2 时**不返回区间**而是返回 `insufficient`——
    单个样本算出的「区间」是 [x, x]，看起来精确，实际什么都没说。
    """
    n = len(values)
    if n < 2:
        return {
            "n": n, "mean": mean(values), "std": 0.0,
            "ci_low": None, "ci_high": None,
            "method": "insufficient", "level": level,
            "note": f"样本数 {n} < 2，无法给出置信区间（单样本的区间退化为点）",
        }
    m = mean(values)
    sd = stdev(values)
    df = n - 1
    t, method = t_critical(df)
    half = t * sd / math.sqrt(n)
    out = {
        "n": n, "mean": m, "std": sd,
        "ci_low": round(m - half, 2), "ci_high": round(m + half, 2),
        "method": method, "level": level,
    }
    if sd == 0.0:
        # σ=0 时区间退化成 [m, m]，**看起来**比任何真实区间都精确，
        # 实际上只说明这批样本恰好一样，不说明真实值没有波动。
        # ScriptedProvider 下这是常态（确定性 provider），
        # 若不标注，报告会输出「CI = [70, 70]」这种误导性的完美区间。
        out["method"] = "zero_variance"
        out["caveat"] = (
            f"样本方差为 0，置信区间退化为点 [{m}, {m}]。"
            f"这只说明这 {n} 次结果完全相同，不说明真实波动为零；"
            f"若 provider 是确定性的（如 ScriptedProvider），该区间不可解读为精度。"
        )
    return out


def describe(values: Sequence[float], level: float = 0.95) -> dict[str, Any]:
    """一次算全 mean / std / CI / min / max。"""
    if not values:
        return {
            "n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "ci_low": None, "ci_high": None, "method": "empty", "level": level,
        }
    ci = confidence_interval(values, level)
    out = {
        "n": len(values),
        "mean": ci["mean"],
        "std": ci["std"],
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "method": ci["method"],
        "level": level,
    }
    # 零方差caveat 必须一路传到报告，不能在describe 这一层被吃掉
    if ci.get("caveat"):
        out["caveat"] = ci["caveat"]
    return out


# ==========================================================================
# 天花板 / 地板
# ==========================================================================


def detect_ceiling(pre_mean: float, line: float = CEILING_LINE) -> dict[str, Any]:
    """天花板检测。

    **这个函数存在的唯一理由**是让「学习没用」和「没有测量空间」
    这两种解释无法被混为一谈。V0.4.1 的 PRE 是 90 分，
    于是 OPEN_BOOK_GAIN 只有 10 分空间，而 σ=3~5——
    增益测不出来时，报告如果说「学习没有帮助」，那是**越权结论**。
    """
    hit = pre_mean >= line
    return {
        "hit": hit,
        "pre_mean": round(pre_mean, 2),
        "line": line,
        "headroom": round(100.0 - pre_mean, 2),
        "note": (
            f"PRE 均值 {pre_mean:.1f} ≥ 天花板线 {line}，"
            f"剩余可测量空间只有 {100.0 - pre_mean:.1f} 分。"
            f"此时任何增益都难以与噪声区分——「测不出增益」既可能是「学习没用」，"
            f"也可能是「没有可测量的空间」，本实验分不开这两者。"
            if hit else
            f"PRE 均值 {pre_mean:.1f} < 天花板线 {line}，"
            f"剩余空间 {100.0 - pre_mean:.1f} 分，增益有被测出来的可能。"
        ),
    }


def detect_floor(pre_mean: float, line: float = FLOOR_LINE) -> dict[str, Any]:
    """地板检测。

    PRE 太低时，后续阶段的涨幅会被「本来就不会」放大：
    从 2 分涨到 40 分是+38，但那只说明基线近乎为零，
    不说明学到了东西。所以地板与天花板必须一起看。
    """
    hit = pre_mean <= line
    return {
        "hit": hit,
        "pre_mean": round(pre_mean, 2),
        "line": line,
        "note": (
            f"PRE 均值 {pre_mean:.1f} ≤ 地板线 {line}，基线近乎为零。"
            f"此后的涨幅会被「本来就不会」放大，"
            f"高分不代表学到了规则，只代表从不会到会。"
            if hit else
            f"PRE 均值 {pre_mean:.1f} > 地板线 {line}，基线有一定水平，涨幅有解释空间。"
        ),
    }


# ==========================================================================
# 增益判定
# ==========================================================================


def verdict_from_gain(values: Sequence[float], min_effect: float = MIN_EFFECT) -> dict[str, Any]:
    """把一个增益的多次测量变成判定。

    三档：
      positive      mean - 2σ > 0 且 mean > min_effect
      non_positive mean + 2σ <= 0
      noisy         其余（含 CI 跨 0、效应量过小、样本不足）

    「CI 跨 0 就说没用」是 V0.4.1 犯过的错：
    跨 0 只能说明**证据不足**，不能说明**效应为零**。
    所以这里叫 `noisy` 而不是 `negative`。
    """
    stats = describe(values)
    n = stats["n"]
    m = stats["mean"]
    sd = stats["std"]
    lo, hi = stats["ci_low"], stats["ci_high"]

    if n < 2:
        verdict = "noisy"
        reason = f"只有 {n} 次测量，无法区分信号与噪声"
    elif lo is None:
        verdict = "noisy"
        reason = "置信区间不可用"
    elif lo > 0 and m >= min_effect:
        verdict = "positive"
        reason = (
            f"95% CI 下界 {lo} > 0 且均值 {m} ≥ 最小效应量 {min_effect}："
            f"增益在统计上可区分于噪声"
        )
    elif hi is not None and hi <= 0:
        verdict = "non_positive"
        reason = f"95% CI 上界 {hi} ≤ 0：增益在统计上可区分于零或负"
    elif abs(m) < min_effect:
        verdict = "noisy"
        reason = (
            f"均值 {m} 的绝对值小于最小效应量 {min_effect}，"
            f"即使方向明确也没有实际意义"
        )
    else:
        verdict = "noisy"
        reason = f"95% CI [{lo}, {hi}] 跨越 0：证据不足，不能断言有效或无效"

    return {
        **stats,
        "verdict": verdict,
        "reason": reason,
        "min_effect": min_effect,
        "mean_minus_2sd": round(m - 2 * sd, 2),
        "mean_plus_2sd": round(m + 2 * sd, 2),
        # σ=0 的positive 不代表「效应被确证」，只代表「这一批确定性结果一致」。
        # 保留 verdict 原样（口径统一），但用 provisional 标出让报告必须打折。
        "provisional": stats.get("method") == "zero_variance",
    }


# ==========================================================================
# 跨 trial 聚合
# ==========================================================================


def load_trials(store: Any, batch_id: str) -> list[dict[str, Any]]:
    """从库里读一个批次的全部 trial，并补上逐 kp 明细。

    `store` 是 `V2Store`。逐 kp 数据来自 `v2_stage_items`，
    每次考试 8 题 × 4 kp × 2 题= 逐 kp 每次只有 2 个样本点，
    所以 `per_kp` 的 n 靠 trial 数累积——这正是 V2 相对 V0.4 的关键改进。
    """
    trials: list[dict[str, Any]] = []
    for row in store.batch_trials(batch_id):
        items = store.items(int(row["id"]))
        trials.append({
            "experiment_id": int(row["id"]),
            "trial_no": int(row["trial_no"]),
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
            "per_kp_rows": [
                {
                    "kp": it["knowledge_point"],
                    "knowledge_point": it["knowledge_point"],
                    "stage": it["stage"],
                    "score": it["score"],
                }
                for it in items
                if it["stage"] != "study"
            ],
        })
    return trials


def aggregate_trials(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把多次 trial 聚合成统计报告。

    `trials` 每项须含 `scores` 字典，键与 `V2Store` 的分数列同名。
    """
    if not trials:
        return {"trials": 0, "metrics": {}, "per_kp": {}, "errors": ["没有任何 trial"]}

    metric_keys: list[str] = []
    for t in trials:
        for k in (t.get("scores") or {}):
            if k not in metric_keys:
                metric_keys.append(k)

    metrics: dict[str, Any] = {}
    for k in metric_keys:
        series = [
            float(t["scores"][k])
            for t in trials
            if t.get("scores") and t["scores"].get(k) is not None
        ]
        metrics[k] = describe(series)
        if k in _GAIN_BASE:
            metrics[k]["verdict"] = verdict_from_gain(series)["verdict"]

    # 天花板 / 地板
    pre_series = [
        float(t["scores"]["pre_blind"])
        for t in trials
        if t.get("scores", {}).get("pre_blind") is not None
    ]
    pre_mean = mean(pre_series)

    # 四类 gain 严格分开，一个都不合并
    gains: dict[str, Any] = {}
    for label, (key, base) in _GAIN_BASE.items():
        series = []
        for t in trials:
            sc = t.get("scores") or {}
            if sc.get(key) is None or sc.get(base) is None:
                continue
            series.append(float(sc[key]) - float(sc[base]))
        gains[label] = verdict_from_gain(series)
        gains[label]["description"] = _GAIN_DESC[label]

    per_kp = aggregate_per_kp(trials)

    return {
        "trials": len(trials),
        "metrics": metrics,
        "gains": gains,
        "ceiling": detect_ceiling(pre_mean),
        "floor": detect_floor(pre_mean),
        "per_kp": per_kp,
        "sample_size_diagnostics": sample_size_diagnostics(trials, per_kp),
        "trials_detail": [
            {"experiment_id": t.get("experiment_id"), "trial_no": t.get("trial_no"),
             "scores": t.get("scores")}
            for t in trials
        ],
    }


#: gain 名 → (被减数指标, 减数指标)
_GAIN_BASE: dict[str, tuple[str, str]] = {
    "external_memory_gain": ("open_book", "pre_blind"),
    "memory_retention_gain": ("memory_blind", "pre_blind"),
    "transfer_gain": ("transfer_blind", "pre_blind"),
    "objective_practice_gain": ("objective_practice_avg", "pre_blind"),
}

_GAIN_DESC: dict[str, str] = {
    "external_memory_gain": (
        "OPEN_BOOK − PRE_BLIND：能利用自己写的外部记忆（笔记/知识点）答题。"
        "高只说明「能查笔记」，不说明「学会了」。"
    ),
    "memory_retention_gain": (
        "MEMORY_BLIND − PRE_BLIND：收回全部外部记忆后仍能答对同一批题。"
        "测的是记忆保持——可能只是记住了笔记里的措辞。"
    ),
    "transfer_gain": (
        "TRANSFER_BLIND − PRE_BLIND：全新场景、同一知识点，只盲考。"
        "这是四者里唯一接近「概念形成」的指标，"
        "但仍不等于「模型被训练过」。"
    ),
    "objective_practice_gain": (
        "机器判分实践均分 − PRE_BLIND：能否写出机器可验证的正确产出。"
        "与 LLM 实践分**分开统计**，因为两者回答不同问题。"
    ),
}


def aggregate_per_kp(
    trials: Sequence[dict[str, Any]], min_samples: int = MIN_SAMPLES_FOR_KP
) -> dict[str, Any]:
    """逐知识点聚合。

    **样本数不足时结论必须带标记。**
    V0.4 每 kp 每卷只 1 题，单题判分误差直接等于该 kp 的分数，
    却照样和 8 题的总体均值并排印在报告里——那是不确定的。
    """
    per_kp: dict[str, Any] = {}
    for t in trials:
        for row in t.get("per_kp_rows") or []:
            kp = row.get("knowledge_point") or row.get("kp")
            if not kp:
                continue
            bucket = per_kp.setdefault(kp, {})
            bucket.setdefault(row.get("stage", "?"), []).append(
                float(row.get("score") or 0.0)
            )

    out: dict[str, Any] = {}
    for kp, stages in sorted(per_kp.items()):
        entry: dict[str, Any] = {}
        for stage, series in sorted(stages.items()):
            stats = describe(series)
            stats["insufficient_sample"] = stats["n"] < min_samples
            stats["min_samples_required"] = min_samples
            if stats["insufficient_sample"]:
                stats["caveat"] = (
                    f"样本数 {stats['n']} < {min_samples}，"
                    f"该知识点结论不具备统计意义，仅供观察"
                )
            entry[stage] = stats
        out[kp] = entry
    return out


def sample_size_diagnostics(
    trials: Sequence[dict[str, Any]], per_kp: dict[str, Any]
) -> dict[str, Any]:
    """样本量与方差的诊断结论。

    单独拎出来是因为它是**前置条件**：样本不够或方差太大时，
    后面所有 gain 的判定都只能当参考。
    """
    n_trials = len(trials)
    thin_kps = [
        kp for kp, stages in per_kp.items()
        if any(s.get("insufficient_sample") for s in stages.values())
    ]
    # 高噪声指标：σ 相对均值过大。绝对阈值 5 分是拍的，
    # 但「σ 比效应量还大」这件事本身就是诊断结论，
    # 报告里必须出现，不能只靠看 CI 跨不跨 0。
    high_variance: list[str] = []
    gain_stats = aggregate_gain_series(trials) or {}
    for label, g in gain_stats.items():
        if g["n"] >= 2 and g["std"] > max(5.0, abs(g["mean"]) * 0.5):
            high_variance.append(label)

    # 零方差：全部 trial 结果完全相同。
    # 在确定性 provider（ScriptedProvider）下这是必然的，
    # 此时任何「5 次重复」都只是同一次结果抄了 5 遍——
    # 天花板/地板/方差诊断全部退化成废话，必须显式说明。
    zero_variance = [
        label for label, g in gain_stats.items()
        if g["n"] >= 2 and g["std"] == 0.0
    ]

    problems: list[str] = []
    if n_trials < 3:
        problems.append(
            f"只跑了 {n_trials} 次，样本量不足以判断稳定性（建议 ≥5）"
        )
    if len(zero_variance) == len(gain_stats) and gain_stats:
        problems.append(
            "全部增益的方差都是 0：多次实验结果完全相同。"
            "若 provider 是确定性的（如 ScriptedProvider），"
            "重复次数不提供任何额外信息，"
            "所有区间与判定都不能解读为统计证据。"
        )
    elif zero_variance:
        problems.append(
            f"以下增益方差为 0（多次结果完全相同）：{sorted(zero_variance)}；"
            f"其区间退化为点，判定不可当作统计确证"
        )
    if thin_kps:
        problems.append(
            f"以下知识点样本不足：{sorted(thin_kps)}；"
            f"逐 kp 结论不可用，只能看总体"
        )
    if high_variance:
        problems.append(
            f"以下增益方差过大（σ > max(5, 0.5×|mean|)）：{sorted(high_variance)}；"
            f"判定为 noisy 时应优先怀疑实验噪声而非效应"
        )
    return {
        "n_trials": n_trials,
        # 数量达标≠信息量达标。确定性 provider 下跑 100 次与跑 5 次
        # 提供的信息量完全一样，所以两个标记必须分开看。
        "sufficient_trials": n_trials >= 3,
        "has_variation": not (
            bool(gain_stats) and len(zero_variance) == len(gain_stats)
        ),
        "thin_kps": sorted(thin_kps),
        "high_variance_gains": sorted(high_variance),
        "zero_variance_gains": sorted(zero_variance),
        "all_zero_variance": bool(gain_stats) and len(zero_variance) == len(gain_stats),
        "problems": problems,
    }


def aggregate_gain_series(trials: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """四类 gain 的原始序列（供诊断与作图）。"""
    out: dict[str, list[float]] = {label: [] for label in _GAIN_BASE}
    for t in trials:
        sc = t.get("scores") or {}
        for label, (key, base) in _GAIN_BASE.items():
            if sc.get(key) is None or sc.get(base) is None:
                continue
            out[label].append(float(sc[key]) - float(sc[base]))
    return {k: describe(v) for k, v in out.items() if v}


def diagnose_batch(summary: dict[str, Any]) -> dict[str, Any]:
    """把一次批次统计的诊断信息汇总成给人看的结论。"""
    ceiling = summary.get("ceiling") or {}
    floor = summary.get("floor") or {}
    ssd = summary.get("sample_size_diagnostics") or {}
    problems: list[str] = []
    if ceiling.get("hit"):
        problems.append(ceiling.get("note", ""))
    if floor.get("hit"):
        problems.append(floor.get("note", ""))
    problems.extend(ssd.get("problems") or [])
    return {
        "has_ceiling": bool(ceiling.get("hit")),
        "has_floor": bool(floor.get("hit")),
        "caveats": problems,
        "reliable": not problems,
    }


__all__ = [
    "CEILING_LINE",
    "FLOOR_LINE",
    "MIN_EFFECT",
    "MIN_SAMPLES_FOR_KP",
    "aggregate_gain_series",
    "aggregate_per_kp",
    "aggregate_trials",
    "confidence_interval",
    "describe",
    "detect_ceiling",
    "detect_floor",
    "diagnose_batch",
    "load_trials",
    "mean",
    "sample_size_diagnostics",
    "stdev",
    "t_critical",
    "verdict_from_gain",
]
