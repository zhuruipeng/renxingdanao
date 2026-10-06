"""brain/causal_v3.py —— V0.6 因果指标层：差异中的差异。

V0.5 的问题（用户对 V0.5 的直接纠正）
--------------------------------------
V0.5 的四类 gain 都是「自己的前后差」：

    gain = POST - PRE

这个差值同时混着两样东西：

    (1) 因学习而获得的能力
    (2) 任何一组都会发生的自然变化
        （被测效应、格式熟悉、题面措辞被记住、模型自身的随机漂移）

只报这个差，等于把 (1) 和 (2) 加在一起当成 (1)。
而 (2) 到底有多大**无法估计**——
因为没有对照组，就没有「不学习时的同样变化是多少」这个参照。

V0.6 的做法
-----------
引入对照组，把差值变成**差值的差值**：

    learning_gain = learning_post - learning_pre
    control_gain  = control_post  - control_pre
    difference_in_gains = learning_gain - control_gain
                      = (learning_post - learning_pre)
                      - (control_post  - control_pre)

只有当 `learning_gain` 明显大于 `control_gain` 时，
才能说提升归因于学习。反过来，如果两者差不多，
那么观察到的那点提升就只是「做过一遍这件事」的正常变化。

两份 gain 必须都报，且必须**明确标注它们是两个不同的口径**：
一个是「学习组的前后差」，一个是「对照组的前后差」，
只有第三个（差值的差值）才是可归因的部分。

效应量
------
除了「提升了多少分」，还要回答「这个提升相对于波动有多大」。
单次实验的 raw gain 再大，若跨 trial 的 std 更大，它就没有意义。
所以给三个：

- ``difference_in_gains``   差值的均值（点估计）
- ``ci``                    95% 置信区间（不确定性）
- ``cohens_d``              标准化效应量（相对波动）

**不允许**在没有区间的情况下说「提升了」——
CI 跨0 就是「与零不可区分」。

客观 held-out 口径
-----------------
用户明确要求：**不要用「实践机器分 - 考试分」当gain**。
两者量纲不同（一个是任务通过率，一个是考试均分），
相减得到的东西没有解释。

所以客观口径是**同一套 held-out 任务在 PRE 与 POST 上各跑一遍**：

    learning_objective_gain = learning_post_obj - learning_pre_obj
    control_objective_gain  = control_post_obj  - control_pre_obj
    objective_difference_in_gains = 上面两者之差

这样 objective 分数和考试分数**永远不混在一起减**，
两组用的是**同一把尺子**。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from brain import stats_v2 as st

#: 结论门槛：|effect_size| 达到这个值才值得说「有可观测的效应」
MIN_EFFECT_SIZE = 0.5

#: CI 跨 0 时禁止的措辞
NO_CLAIM = "difference_in_gains 的 95% 置信区间跨 0，观测差异与「无差异」不可区分"

#: 四类 gain 的名字，报告层按这些 key 输出
KEY_LEARNING_PRE = "learning_pre"
KEY_LEARNING_POST = "learning_post"
KEY_CONTROL_PRE = "control_pre"
KEY_CONTROL_POST = "control_post"
KEY_LEARNING_GAIN = "learning_gain"
KEY_CONTROL_GAIN = "control_gain"
KEY_DIFFERENCE = "difference_in_gains"

KEY_OBJ_LEARNING_PRE = "learning_objective_pre"
KEY_OBJ_LEARNING_POST = "learning_objective_post"
KEY_OBJ_CONTROL_PRE = "control_objective_pre"
KEY_OBJ_CONTROL_POST = "control_objective_post"
KEY_OBJ_LEARNING_GAIN = "learning_objective_gain"
KEY_OBJ_CONTROL_GAIN = "control_objective_gain"
KEY_OBJ_DIFFERENCE = "objective_difference_in_gains"

#: 报告必须分清的五类口径
FIVE_LENSES: tuple[str, ...] = (
    "observed_gain",       # 学习组自己的前后差
    "control_gain",        # 对照组自己的前后差
    "difference_in_gains", # 差值的差值
    "confidence_interval", # 这个差值有多不确定
    "effect_size",         # 相对于波动有多大
)


# ==========================================================================
# 单次 trial 的成对指标
# ==========================================================================


def paired_trial(
    learning_pre: float,
    learning_post: float,
    control_pre: float,
    control_post: float,
) -> dict[str, Any]:
    """一次 trial 的全部成对指标。

    输入是四个分数（都是百分制），输出是**三层分离**的结构：

        raw    —— 四个原始分 + 两个各自的 gain + 差值的差值
        obj    —— 同上，但走客观 held-out 口径（结构完全平行）
        order  —— 四个数在字典里的固定呈现顺序（报告层按它输出）
    """
    learning_gain = round(learning_post - learning_pre, 2)
    control_gain = round(control_post - control_pre, 2)
    diff = round(learning_gain - control_gain, 2)
    return {
        KEY_LEARNING_PRE: learning_pre,
        KEY_LEARNING_POST: learning_post,
        KEY_CONTROL_PRE: control_pre,
        KEY_CONTROL_POST: control_post,
        KEY_LEARNING_GAIN: learning_gain,
        KEY_CONTROL_GAIN: control_gain,
        KEY_DIFFERENCE: diff,
        "order": (
            KEY_LEARNING_PRE, KEY_LEARNING_POST,
            KEY_LEARNING_GAIN,
            KEY_CONTROL_PRE, KEY_CONTROL_POST,
            KEY_CONTROL_GAIN,
            KEY_DIFFERENCE,
        ),
    }


def objective_trial(
    learning_objective_pre: float,
    learning_objective_post: float,
    control_objective_pre: float,
    control_objective_post: float,
) -> dict[str, Any]:
    """一次 trial 的客观 held-out 成对指标。结构与 :func:`paired_trial` 平行。"""
    lg = round(learning_objective_post - learning_objective_pre, 2)
    cg = round(control_objective_post - control_objective_pre, 2)
    return {
        KEY_OBJ_LEARNING_PRE: learning_objective_pre,
        KEY_OBJ_LEARNING_POST: learning_objective_post,
        KEY_OBJ_CONTROL_PRE: control_objective_pre,
        KEY_OBJ_CONTROL_POST: control_objective_post,
        KEY_OBJ_LEARNING_GAIN: lg,
        KEY_OBJ_CONTROL_GAIN: cg,
        KEY_OBJ_DIFFERENCE: round(lg - cg, 2),
        "order": (
            KEY_OBJ_LEARNING_PRE, KEY_OBJ_LEARNING_POST,
            KEY_OBJ_LEARNING_GAIN,
            KEY_OBJ_CONTROL_PRE, KEY_OBJ_CONTROL_POST,
            KEY_OBJ_CONTROL_GAIN,
            KEY_OBJ_DIFFERENCE,
        ),
    }


# ==========================================================================
# 跨 trial 聚合
# ==========================================================================


def cohens_d(treatment: Sequence[float], control: Sequence[float]) -> dict[str, Any]:
    """Cohen's d。

    用合并标准差（pooled SD）而不是「除以其中一个的 std」——
    后者在两组样本量不等或方差差得远时会给出偏向大组的假象。
    """
    n1, n2 = len(treatment), len(control)
    if n1 < 2 or n2 < 2:
        return {
            "d": None,
            "method": "insufficient",
            "note": f"两组样本量不足（treatment={n1}, control={n2}），"
                    f"样本量<2 时标准差不可估，效应量不给出。",
        }
    m1 = sum(treatment) / n1
    m2 = sum(control) / n2
    v1 = sum((v - m1) ** 2 for v in treatment) / (n1 - 1)
    v2 = sum((v - m2) ** 2 for v in control) / (n2 - 1)
    pooled = math.sqrt(
        ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    )
    if pooled == 0:
        return {
            "d": None,
            "method": "zero_pooled_variance",
            "note": (
                f"合并标准差为 0（两组各自方差都是 0），效应量不可定义。"
                f"这在确定性 provider 下是常态，此时不能用 d 说明任何问题。"
            ),
            "treatment_mean": round(m1, 2),
            "control_mean": round(m2, 2),
        }
    return {
        "d": round((m1 - m2) / pooled, 3),
        "method": "pooled_sd",
        "treatment_mean": round(m1, 2),
        "control_mean": round(m2, 2),
        "pooled_sd": round(pooled, 3),
        "n_treatment": n1,
        "n_control": n2,
    }


def paired_effect_size(
    treatment_gains: Sequence[float], control_gains: Sequence[float]
) -> dict[str, Any]:
    """对 gain 做效应量（这才是本实验真正要问的量）。

    注意是对 **gain** 做 d，不是对 POST 分数做 d。
    「学习组考得比对照组高」和「学习带来的提升比对照组大」
    是两件不同的事；前者可能纯粹因为两组 PRE 基线不同。
    """
    return cohens_d(treatment_gains, control_gains)


def effect_size_label(d: float | None) -> str:
    """Cohen 惯例的定性标签。"""
    if d is None:
        return "不可计算"
    a = abs(d)
    if a < 0.2:
        return "可忽略"
    if a < 0.5:
        return "小"
    if a < 0.8:
        return "中等"
    return "大"


def aggregate_difference(
    trials: Sequence[dict[str, Any]],
    gain_key: str = KEY_DIFFERENCE,
    learning_gain_key: str = KEY_LEARNING_GAIN,
    control_gain_key: str = KEY_CONTROL_GAIN,
) -> dict[str, Any]:
    """把N 次 trial 的成对指标聚合成一份可判定的结论。

    输出刻意分成五块（对应报告必须分清的五类口径），
    并且**在 CI 跨 0 时不给出「有提升」的判定**——
    这不是保守，是数学：区间跨 0 就是与零不可区分。
    """
    diffs = [float(t[gain_key]) for t in trials if gain_key in t]
    lgains = [float(t[learning_gain_key]) for t in trials if learning_gain_key in t]
    cgains = [float(t[control_gain_key]) for t in trials if control_gain_key in t]

    ci = st.confidence_interval(diffs)
    es = paired_effect_size(lgains, cgains)
    d_val = es.get("d")

    # 三个条件分别判定，避免把「样本量不足」与「区间跨 0」混成一条：
    # 前者是「还不到能算的阶段」，后者是「算了，结果与零不可区分」。
    ci_has_bounds = ci.get("ci_low") is not None and ci.get("ci_high") is not None
    ci_excludes_zero = ci_has_bounds and (
        ci["ci_low"] > 0 or ci["ci_high"] < 0
    )
    effect_big_enough = d_val is not None and abs(d_val) >= MIN_EFFECT_SIZE
    enough_trials = len(diffs) >= 2

    conditions = {
        "ci_excludes_zero": ci_excludes_zero,
        "effect_size_reaches_threshold": effect_big_enough,
        "has_at_least_two_trials": enough_trials,
    }
    observable = all(conditions.values())

    reasons: list[str] = []
    if not enough_trials:
        reasons.append(
            f"只有 {len(diffs)} 次 trial，样本量<2 时置信区间退化为点，"
            f"无法支撑任何统计判断"
        )
    if not ci_excludes_zero:
        if ci_has_bounds:
            reasons.append(
                f"95% CI = [{ci['ci_low']}, {ci['ci_high']}] 跨 0，"
                f"观测差异与「无差异」不可区分"
            )
        elif enough_trials:
            # 有足够样本却给不出区间，属于不该发生的异常
            reasons.append(f"置信区间不可用：{ci.get('method')}。{NO_CLAIM}")
    if not effect_big_enough:
        if d_val is None:
            reasons.append(
                f"效应量不可计算（{es.get('method')}）：{es.get('note', '')}"
            )
        else:
            reasons.append(
                f"|Cohen's d| = {abs(d_val):.2f} < 阈值 {MIN_EFFECT_SIZE}，"
                f"提升相对于波动太小"
            )

    ci_text = (
        f"[{ci.get('ci_low')}, {ci.get('ci_high')}]"
        if ci_has_bounds
        else f"不可用（{ci.get('method')}）"
    )
    d_text = str(d_val) if d_val is not None else "不可计算"

    return {
        # 1. 观测到的学习组前后差
        "observed_gain": st.describe(lgains),
        # 2. 对照组自己的前后差
        "control_gain": st.describe(cgains),
        # 3. 差值的差值
        "difference_in_gains": st.describe(diffs),
        # 4. 不确定性
        "confidence_interval": ci,
        # 5. 相对波动
        "effect_size": {
            **es,
            "label": effect_size_label(d_val),
            "threshold": MIN_EFFECT_SIZE,
        },
        "n_trials": len(diffs),
        "conditions": conditions,
        "observable_effect": observable,
        "reasons_not_observable": reasons,
        # 单一结论句：可观测时给正面判定，否则只陈述观测到什么
        "verdict": (
            f"difference-in-gains = {st.mean(diffs)}，"
            f"95% CI = {ci_text}，"
            f"Cohen's d = {d_text}（{effect_size_label(d_val)}）"
            + (
                "；三个条件全部满足，可报告为可观测效应"
                if observable
                else "；" + "；".join(reasons)
            )
        ),
    }


# ==========================================================================
# 结论许可
# ==========================================================================


#: 未经代码证据支持时禁止的断言。
#: 注意写成**短核心词**而不是完整句子：
#: 早先只列「学习确实提升了」，结果「学习提升了 5 分」整句绕过了门禁——
#: 因为门禁比的是字面，不是语义。短词覆盖面大，误报由否定词回看窗口消化。
FORBIDDEN_CLAIMS_V3: tuple[str, ...] = (
    "训练成功",
    "已经学会",
    "已学会",
    "学会了",
    "掌握",
    "能力已经改变",
    "能力已改变",
    "确实有效",
    "确实提升了",
    "证明了",
    "学会了并",
)

#: 这些词只在**未获归因许可**时才是越权的。
#: 拿到许可后报告本来就要说「归因于学习」，否则门禁会自伤
#: ——和 V0.5 里免责声明被自己拦下是同一类错误。
ATTRIBUTION_TERMS: tuple[str, ...] = (
    "归因于学习",
    "学习造成了",
    "学习确实提升了",
    # 「学习确实提升了」拦不住「学习提升了 5 分」，
    # 所以更短的核心也要列进来。
    "学习提升了",
)

#: 反向的、同样未经证据支持的断言
FORBIDDEN_NEGATIVE_CLAIMS_V3: tuple[str, ...] = (
    "训练失败",
    "没有学到任何东西",
    "模型完全没提升",
    "学习毫无作用",
)

#: 否定词回看窗口（沿用 V0.5 的做法）
NEGATION_LOOKBACK = 10

_NEGATORS = ("不", "未", "没", "非", "无", "尚未", "并非", "不能", "无法")


def _is_negated(text: str, idx: int) -> bool:
    window = text[max(0, idx - NEGATION_LOOKBACK):idx]
    return any(n in window for n in _NEGATORS)


def scan_forbidden_claims(text: str, may_attribute: bool = False) -> list[str]:
    """扫描未经证据支持的断言。

    只在**不是否定语境**时报。
    「不能声称训练成功」这句话本身含有「训练成功」，
    但它是一个禁止声明，不是声明。

    ``may_attribute=True`` 时放行 :data:`ATTRIBUTION_TERMS` ——
    那是拿到归因许可后本来就应该写的话，
    不放行等于门禁把合规报告也拦下来（V0.5 踩过这个坑）。
    """
    phrases = list(FORBIDDEN_CLAIMS_V3)
    if not may_attribute:
        phrases += list(ATTRIBUTION_TERMS)
    phrases += list(FORBIDDEN_NEGATIVE_CLAIMS_V3)

    hits: list[str] = []
    for phrase in phrases:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            if not _is_negated(text, idx):
                hits.append(phrase)
            start = idx + 1
    return hits


def claim_permission(
    diff_summary: dict[str, Any],
    contamination_clean: bool,
    n_trials: int,
) -> dict[str, Any]:
    """能不能说「学习造成了提升」——**由数据决定，不由措辞决定**。

    三个必要条件，缺一即降级为「只报告观测，不作归因」：

    1. 污染检测全部通过
    2. difference-in-gains 的 95% CI 不跨 0
    3. |Cohen's d| 达到阈值
    """
    conditions = {
        "no_contamination": bool(contamination_clean),
        "ci_excludes_zero": bool(
            diff_summary.get("conditions", {}).get("ci_excludes_zero")
        ),
        "effect_size_reaches_threshold": bool(
            diff_summary.get("conditions", {}).get("effect_size_reaches_threshold")
        ),
        "enough_trials": n_trials >= 2,
    }
    allowed = all(conditions.values())
    blocked: list[str] = []
    labels = {
        "no_contamination": "存在污染检出，两组差异无法归因于学习",
        "ci_excludes_zero": "95% 置信区间跨 0，与「无差异」不可区分",
        "effect_size_reaches_threshold": (
            f"效应量未达阈值 {MIN_EFFECT_SIZE}"
        ),
        "enough_trials": f"只有 {n_trials} 次 trial，不足以给出统计判断",
    }
    for k, ok in conditions.items():
        if not ok:
            blocked.append(labels[k])
    return {
        "may_attribute_to_learning": allowed,
        "conditions": conditions,
        "blocked_by": blocked,
        "required_phrasing": (
            "可表述为：本次实验中学习组的能力提升幅度大于对照组，"
            "该差异的 95% 置信区间不包含 0。"
            if allowed
            else "只能表述为：观测到学习组与对照组的差值为 "
                 f"{diff_summary.get('difference_in_gains', {}).get('mean')} 分，"
                 "但本次实验尚不具备归因条件，不得表述为学习造成了提升。"
        ),
    }


def assert_claims_allowed(
    text: str,
    may_attribute: bool,
    where: str = "报告",
) -> None:
    """越权措辞硬门禁。

    没有归因许可却在断言「学会了 / 训练成功 / 能力已改变」——
    这类话是**最危险的一类错误**：读起来像结论，实际上没有实验支撑。
    """
    hits = scan_forbidden_claims(text, may_attribute=may_attribute)
    if hits:
        raise RuntimeError(
            f"{where} 出现未经证据支持的断言：{sorted(set(hits))}。\n"
            f"  归因许可：may_attribute_to_learning = {may_attribute}\n"
            f"  这些话必须等到 CI 不跨 0 且效应量达标且无污染检出之后才可以说。"
        )


def objective_difference_summary(
    trials: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """客观 held-out 口径的聚合。与 :func:`aggregate_difference` 结构平行。"""
    return aggregate_difference(
        trials,
        gain_key=KEY_OBJ_DIFFERENCE,
        learning_gain_key=KEY_OBJ_LEARNING_GAIN,
        control_gain_key=KEY_OBJ_CONTROL_GAIN,
    )


__all__ = [
    "ATTRIBUTION_TERMS",
    "FIVE_LENSES",
    "FORBIDDEN_CLAIMS_V3",
    "FORBIDDEN_NEGATIVE_CLAIMS_V3",
    "KEY_CONTROL_GAIN",
    "KEY_CONTROL_POST",
    "KEY_CONTROL_PRE",
    "KEY_DIFFERENCE",
    "KEY_LEARNING_GAIN",
    "KEY_LEARNING_POST",
    "KEY_LEARNING_PRE",
    "KEY_OBJ_CONTROL_GAIN",
    "KEY_OBJ_CONTROL_POST",
    "KEY_OBJ_CONTROL_PRE",
    "KEY_OBJ_DIFFERENCE",
    "KEY_OBJ_LEARNING_GAIN",
    "KEY_OBJ_LEARNING_POST",
    "KEY_OBJ_LEARNING_PRE",
    "MIN_EFFECT_SIZE",
    "aggregate_difference",
    "assert_claims_allowed",
    "claim_permission",
    "cohens_d",
    "effect_size_label",
    "objective_difference_summary",
    "objective_trial",
    "paired_effect_size",
    "paired_trial",
    "scan_forbidden_claims",
]
