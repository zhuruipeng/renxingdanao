"""重复实验与跨次统计 —— V0.4 的第二根支柱。

**为什么单次结果不作数**
----------------------
V0.3 的整个实验只跑一次。这在方法论上有个致命问题：
LLM 判分本身有波动（同一份答案，两次判可能差 ±10 分），
样本又只有 4 题。于是单次跑出TRANSFER +15 分，你无法区分：

    (a) 真的迁移了一点
    (b) 判分噪声
    (c) 这 4 道题恰好偏易

所以 V0.4 强制重复 N 次（默认 5），看**分布**而不是看单点。

**判定门槛**
------------
跨次标准差（std）是这次实验的噪声底线。只有当

    mean(gain) - 2 * std(gain) > 0

时才敢说「观察到了提升」。这条线比 `mean > 0` 严格得多——
它要求提升幅度大于波动本身。这是本模块的核心判据。

**成对检验**
------------
同一个 trial 内的 PRE / TRANSFER 是成对的（同一批题、同一模型实例配置）。
所以 gain 用成对差值再取均值，比把两组独立分数相减更抗噪：
每个 trial 自己就是自己的对照，trial 间的难度波动被差值抵消掉一部分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from brain import question_bank as qb
from brain.experiment import ExperimentRunner
from memory.db import Database, checksum

#: 默认重复次数。用户要求至少 5～10 次，这里取下限。
DEFAULT_REPEATS = 5

#: 判定门槛系数：mean - K * std > 0 才算观察到提升
SIGMA_K = 2.0

#: 低于这个次数不给结论 —— 3 次以内的 std 估计太不稳
MIN_TRIALS_FOR_VERDICT = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# 基础统计（纯标准库，不引 numpy）
# ----------------------------------------------------------------------
def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    """样本标准差（n-1）。样本量 < 2 时返回 0。"""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def describe(values: Sequence[float]) -> dict[str, Any]:
    """一组数的完整描述。报告里所有统计都出自这里。"""
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "sem": 0.0, "values": []}
    m = mean(values)
    sd = stdev(values)
    return {
        "n": len(values),
        "mean": round(m, 2),
        "std": round(sd, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "sem": round(sd / math.sqrt(len(values)), 2) if values else 0.0,
        "values": [round(v, 2) for v in values],
    }


def gain_verdict(values: Sequence[float], label: str = "") -> dict[str, Any]:
    """判定一组 gain 是否真的为正。

    判据：mean - 2*std > 0，且 n >= MIN_TRIALS_FOR_VERDICT。

    这条门槛的意义：单看均值会把噪声读成信号；
    要求均值超过波动幅度的两倍，才值得当成观察到的现象。
    """
    d = describe(values)
    lower_bound = round(d["mean"] - SIGMA_K * d["std"], 2)
    enough = d["n"] >= MIN_TRIALS_FOR_VERDICT
    if not enough:
        state = "insufficient"
        reason = f"只有 {d['n']} 次，低于 {MIN_TRIALS_FOR_VERDICT} 次门槛，不给结论"
    elif lower_bound > 0:
        state = "positive"
        reason = (
            f"均值 {d['mean']:+.2f}，标准差 {d['std']:.2f}，"
            f"均值-2σ = {lower_bound:+.2f} > 0，提升幅度超过波动"
        )
    elif d["mean"] > 0:
        state = "noisy"
        reason = (
            f"均值 {d['mean']:+.2f} 为正，但均值-2σ = {lower_bound:+.2f} ≤ 0，"
            f"提升幅度小于波动幅度，无法与噪声区分"
        )
    else:
        state = "non_positive"
        reason = f"均值 {d['mean']:+.2f} ≤ 0，未观察到提升"
    return {
        "label": label,
        **d,
        "lower_bound": lower_bound,
        "state": state,
        "reason": reason,
    }


# ----------------------------------------------------------------------
# 批次存储
# ----------------------------------------------------------------------
@dataclass
class BatchStore:
    """重复实验批次存储。"""

    db: Database

    def create(
        self,
        batch_id: str,
        topic: str,
        planned_trials: int,
        bank_fingerprint: str,
        model: str,
        material_checksum: str,
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO experiment_batches(batch_id, topic, planned_trials,
                                               bank_fingerprint, model,
                                               material_checksum, status, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (batch_id, topic, planned_trials, bank_fingerprint, model,
             material_checksum, "running", _now()),
        )
        return int(cur.lastrowid)

    def finish(self, batch_id: str, summary: dict[str, Any]) -> None:
        import json

        self.db.execute(
            "UPDATE experiment_batches SET status = 'done', summary_json = ?,"
            " finished_at = ? WHERE batch_id = ?",
            (json.dumps(summary, ensure_ascii=False), _now(), batch_id),
        )

    def get(self, batch_id: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM experiment_batches WHERE batch_id = ?", (batch_id,)
        )

    def latest(self) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM experiment_batches ORDER BY id DESC LIMIT 1"
        )

    def trials(self, batch_id: str) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM experiments WHERE batch_id = ? ORDER BY trial_no",
            (batch_id,),
        )


# ----------------------------------------------------------------------
# 跨次聚合
# ----------------------------------------------------------------------
METRIC_KEYS = (
    "pre_score",
    "memory_open_score",
    "memory_blind_score",
    "transfer_blind_score",
    "practice_usable_rate",
    "practice_avg_score",
    "note_quality_avg",
)


def aggregate_trials(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把多次实验聚合成一份带统计量的报告。

    每个指标给 mean/std/min/max；三个 gain 额外过
    `gain_verdict` 给判定——**这是「有没有真提升」的答案所在**。
    """
    n = len(trials)
    metrics: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = [float(t[key]) for t in trials if t.get(key) is not None]
        metrics[key] = describe(values)

    # 成对差值：每个 trial 自己做自己的对照
    open_gains = [
        round(float(t["memory_open_score"]) - float(t["pre_score"]), 2)
        for t in trials
        if t.get("memory_open_score") is not None
    ]
    memory_gains = [
        round(float(t["memory_blind_score"]) - float(t["pre_score"]), 2)
        for t in trials
        if t.get("memory_blind_score") is not None
    ]
    transfer_gains = [
        round(float(t["transfer_blind_score"]) - float(t["pre_score"]), 2)
        for t in trials
        if t.get("transfer_blind_score") is not None
    ]

    verdicts = {
        "open_book_gain": gain_verdict(open_gains, "OPEN_BOOK_GAIN（能否用外部记忆）"),
        "memory_gain": gain_verdict(memory_gains, "MEMORY_GAIN（记忆保持）"),
        "transfer_gain": gain_verdict(transfer_gains, "TRANSFER_GAIN（概念形成）"),
    }

    return {
        "n_trials": n,
        "metrics": metrics,
        "verdicts": verdicts,
        "note_quality_pass_rate": _avg(
            [t["note_quality_pass_rate"] for t in trials
             if t.get("note_quality_pass_rate") is not None]
        ),
        "memory_vs_transfer_gap": gain_verdict(
            [round(m - t, 2) for m, t in zip(memory_gains, transfer_gains)],
            "MEMORY_GAIN - TRANSFER_GAIN（背住了 vs 学会了）",
        ),
    }


def _avg(values: Sequence[float]) -> float:
    return round(mean(values), 4) if values else 0.0


# ----------------------------------------------------------------------
# 重复实验编排
# ----------------------------------------------------------------------
def make_batch_id(topic: str, trial: int = 0) -> str:
    """批次 id。带时间戳，保证同一配置的多次跑能区分开。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{topic}-{stamp}" if trial == 0 else f"{topic}-{stamp}-{trial}"


@dataclass
class ReplicateRunner:
    """把同一配置的实验重复跑 N 次并聚合。

    **每次都新建 provider 实例**，否则 ScriptedProvider 的 calls 列表
    会跨 trial 累积（测试里能查出来），真实模型侧也会把上一轮的
    作答上下文带进下一轮——那会污染 BLIND_GAIN。
    """

    db: Database
    provider_factory: Any  # Callable[[], LLMProvider]
    evaluator_provider_factory: Any | None = None
    batch: BatchStore = field(init=False)

    def __post_init__(self) -> None:
        self.batch = BatchStore(self.db)

    def run_repeated(
        self,
        topic: str,
        material_content: str,
        repeats: int = DEFAULT_REPEATS,
        model_label: str = "",
        batch_id: str = "",
        progress: Any = None,
    ) -> dict[str, Any]:
        """跑 N 次并返回聚合报告。

        progress: 可选回调 progress(trial_no, result)，用于 CLI 打进度。
        """
        qb.assert_bank_intact()
        if repeats < 1:
            raise ValueError("重复次数至少为 1")
        bid = batch_id or make_batch_id(topic)
        material_checksum = checksum(material_content)
        self.batch.create(
            bid, topic, repeats, qb.BANK_FINGERPRINT, model_label, material_checksum
        )

        trials: list[dict[str, Any]] = []
        results: list[Any] = []
        for i in range(1, repeats + 1):
            # 每次全新 provider —— 避免上一轮的上下文/调用记录渗进来
            runner = ExperimentRunner(
                self.db,
                self.provider_factory(),
                evaluator_provider=(
                    self.evaluator_provider_factory()
                    if self.evaluator_provider_factory
                    else None
                ),
            )
            res = runner.run(topic, material_content, batch_id=bid, trial_no=i)
            results.append(res)
            trials.append({
                "trial_no": i,
                "experiment_id": res.experiment_id,
                "pre_score": res.pre_score,
                "memory_open_score": res.memory_open_score,
                "memory_blind_score": res.memory_blind_score,
                "transfer_blind_score": res.transfer_blind_score,
                "practice_usable_rate": res.practice_usable_rate,
                "practice_avg_score": res.practice_avg_score,
                "note_quality_avg": res.note_quality_avg,
                "note_quality_pass_rate": (
                    res.note_quality_detail.get("pass_rate")
                    if res.note_quality_detail else 0.0
                ),
                "open_book_gain": res.open_book_gain,
                "memory_gain": res.memory_gain,
                "transfer_gain": res.transfer_gain,
            })
            if progress:
                progress(i, res)

        summary = aggregate_trials(trials)
        summary.update({
            "batch_id": bid,
            "topic": topic,
            "repeats": repeats,
            "model": model_label,
            "bank_fingerprint": qb.BANK_FINGERPRINT,
            "bank_version": qb.BANK_VERSION,
            "material_checksum": material_checksum,
            "trials": trials,
            "sigma_k": SIGMA_K,
        })
        self.batch.finish(bid, summary)
        return summary


__all__ = [
    "DEFAULT_REPEATS",
    "METRIC_KEYS",
    "MIN_TRIALS_FOR_VERDICT",
    "SIGMA_K",
    "BatchStore",
    "ReplicateRunner",
    "aggregate_trials",
    "describe",
    "gain_verdict",
    "make_batch_id",
    "mean",
    "stdev",
]
