"""brain/experiment_v3.py —— V0.6 双组实验编排。

一次 run 的形状
--------------
    Learning Group:  PRE → STUDY → PRACTICE → POST → TRANSFER
    Control  Group:  PRE → CONTROL_MATERIAL → CONTROL_PRACTICE → POST → TRANSFER

两次 PRE 都在任何材料介入之前跑完，然后才交替给材料。
**这个顺序不是形式**：
若 Learning 先跑完 POST 再给 Control 跑 PRE，
Control 的 PRE 就发生在同一模型实例已读过目标知识之后——
虽然两个组用的是不同 context，但 prompt 里可能残留着上一轮的知识，
Control 的 PRE 被抬高，difference-in-gains 被压低，
而且这种压低**在数据里看不出来**（它看起来就是「对照组提升小」）。

十阶段固定顺序（:data:`memory.db.STAGES_V3`）::

      1 pre_learning        2 pre_control
      3 study_learning      4 control_material
      5 practice_learning   6 practice_control
      7 post_learning       8 post_control
      9 transfer_learning   10 transfer_control

两套 provider
-------------
V0.5 只有一个 provider，candidate 与 evaluator 是同一个——
于是「评委就是考生」这件事在数据里查不出来。
V0.6 把两者拆开：

    candidate_provider / candidate_model   —— 答题的
    evaluator_provider / evaluator_model   —— 打分的

两者**必须分别记录**。若配置成同一个实例，
污染检测会直接判不合格：同一个对象意味着它自己的输出
进了自己的输入，评分不能作为独立证据。

客观 held-out 口径
-----------------
**不使用「实践机器分 - 考试分」作为 gain**（用户对 V0.5 的直接纠正）。
两把尺子量纲不同，相减得到的东西没有解释。

本模块的做法是：同一套 held-out 客观任务在 PRE 与 POST 上**各跑一遍**：

    pre_objective  → 8 个 held-out 任务，盲考
    post_objective → 同一批8 个任务，仍盲考

于是 objective 的前后差和考试的前后差是同一种东西，
两组也用同一把尺子，差值的差值才有意义。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from brain import benchmark_v2 as b2
from brain import causal_v3 as c3
from brain import control_material as cm
from brain import group_isolation as gi
from brain import heldout_bank as ho
from brain import novel_material as nm
from brain.llm import LLMProvider
from memory.db import (
    BLIND_STAGES_V3,
    GROUPS_V3,
    STAGES_V3,
    Database,
    checksum,
    create_v3_sentinel,
    v3_experiment_key,
)

# ==========================================================================
# 提示词
# ==========================================================================

#: 考生系统提示。两组**必须完全一致**——
# 唯一区别是 user 里的材料，system 若不同就成了第二个自变量。
CANDIDATE_SYSTEM = (
    "你是一个正在学习一个全新虚构存储系统技术细节的工程师。"
    "你只能依据对话中给出的信息作答，不要使用你可能记得的同类系统知识。"
    "作答要给出推理过程和结论。"
)

#: 评估器系统提示
EVALUATOR_SYSTEM = (
    "你是评分员。给出这道题的标准答案，然后对比考生答案，"
    "给出 0-100 的分数、结论与缺失要点。"
    "只输出 JSON：{\"score\": 数字, \"verdict\": \"pass\"|\"fail\","
    " \"feedback\": \"...\", \"missing_points\": [\"...\"]}。"
    "评分要严格：只答对一部分就不能给满分。"
)

#: 客观任务提交系统提示
OBJECTIVE_SYSTEM = (
    "你是一个 Python 工程师。严格按照题目要求，只输出一个 JSON 对象，"
    "不要有任何解释文字或代码块标记。"
)

#: 实践阶段系统提示
PRACTICE_SYSTEM = (
    "你是一个正在练习的工程师。题目要求你提交一个 JSON 对象。"
    "只输出 JSON，不要输出解释。"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: 客观任务阶段的 seq。
#:
#: 客观任务与考试卷**并行**跑（都在同一批题目上），
#: 但它们共用``v3_objective_runs.seq`` 落进顺序审计，
#: 所以必须给一个确定的位置，否则``index()`` 会直接抛错。
#: 口径：客观 PRE 紧随两个考试 PRE 之后（seq=2.5→ 取 3 的位置），
#: 客观 POST 紧随两个考试 POST 之后。
_OBJECTIVE_SEQ: dict[str, int] = {
    "pre_objective": 3,
    "post_objective": 9,
}


def _seq_for(stage: str) -> int:
    """任何阶段的 seq。未知阶段直接抛错，不给默认值 0。

    默认 0 会让一个拼错的 stage 静默排在最前面——
    而顺序审计恰恰是按 seq 排序的，
    这种错误在数据里完全看不出来。
    """
    if stage in _OBJECTIVE_SEQ:
        return _OBJECTIVE_SEQ[stage]
    if stage in STAGES_V3:
        return STAGES_V3.index(stage) + 1
    raise ValueError(f"未知阶段 {stage!r}")


def stage_marker(stage: str, exp_group: str) -> str:
    """把阶段与组别写进 prompt。

    这不是给真实模型看的——真实模型不需要知道自己在第几阶段。
    它是给**离线脚本化 provider** 与事后审计用的锚点：

    - 离线 provider 靠它区分 PRE 与 POST。早先靠「prompt 里有没有
      材料」来判断，但盲考阶段的 PRE 与 POST prompt **结构完全相同**
      （都不带材料），于是 POST 被误判成 PRE，两组都没提升。
    - 审计时能从落库的 answer 里回读出这条标记，
      确认某个答案确实是那个阶段产出的。

    标记写在 user prompt 的**末尾**，不污染题目本身。
    """
    return f"\n[[V3_STAGE:{stage}]] [[V3_GROUP:{exp_group}]]"


# ==========================================================================
# 存储
# ==========================================================================


@dataclass
class ProviderConfig:
    """一套 provider 配置。candidate 与 evaluator 各一份。"""

    name: str
    model: str
    provider: LLMProvider

    def meta(self) -> dict[str, str]:
        return {"provider": self.name, "model": self.model}


@dataclass
class RunSpec:
    """一次 run 的全部配置。"""

    run_id: str
    trial_no: int
    topic: str
    candidate: ProviderConfig
    evaluator: ProviderConfig
    #: 每个阶段可覆盖 provider（默认走 spec 里的两套）
    stage_overrides: dict[str, tuple[ProviderConfig, ProviderConfig]] = field(
        default_factory=dict
    )

    def providers_for(self, stage: str) -> tuple[ProviderConfig, ProviderConfig]:
        return self.stage_overrides.get(stage, (self.candidate, self.evaluator))


@dataclass
class V3Store:
    """V0.6 的五张表读写。

    ``spec`` 可选：只在 :meth:`record_call_failure` 里用到——
    失败记录要带上当时的 provider/model，
    否则「哪一套 provider 崩了」这条信息就丢了。
    """

    db: Database
    spec: RunSpec | None = None

    # ---------- run ----------
    def create_run(self, spec: RunSpec) -> None:
        fp = gi.frozen_fingerprints()
        self.db.execute(
            """INSERT INTO v3_runs(run_id, topic, trial_no,
                 candidate_provider, candidate_model,
                 evaluator_provider, evaluator_model,
                 pre_fingerprint, post_fingerprint, transfer_fingerprint,
                 learning_material_checksum, control_material_checksum,
                 practice_tasks_fingerprint, heldout_tasks_fingerprint,
                 status, summary_json, diagnostics_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'running','{}','{}',?)""",
            (
                spec.run_id, spec.topic, spec.trial_no,
                spec.candidate.name, spec.candidate.model,
                spec.evaluator.name, spec.evaluator.model,
                fp["pre_fingerprint"], fp["post_fingerprint"],
                fp["transfer_fingerprint"],
                fp["learning_material_checksum"], fp["control_material_checksum"],
                fp["practice_tasks_fingerprint"], fp["heldout_tasks_fingerprint"],
                _now(),
            ),
        )

    def create_group(
        self, run_id: str, exp_group: str, material_kind: str, material_text: str
    ) -> int:
        cur = self.db.execute(
            """INSERT INTO v3_experiments(run_id, exp_group, material_kind,
                 material_checksum, status, detail_json, created_at)
               VALUES (?,?,?,?,'running','{}',?)""",
            (run_id, exp_group, material_kind, checksum(material_text), _now()),
        )
        eid = int(cur.lastrowid)
        # 老experiments 表里建哨兵行：learning_notes.experiment_id
        # 是 NOT NULL REFERENCES experiments(id)，不建就外键失败。
        create_v3_sentinel(self.db.conn, eid, f"{run_id}/{exp_group}")
        return eid

    def finish_group(
        self, experiment_id: int, scores: dict[str, Any], detail: dict[str, Any]
    ) -> None:
        self.db.execute(
            """UPDATE v3_experiments
               SET pre_score=?, post_score=?, transfer_score=?,
                   pre_objective_score=?, post_objective_score=?,
                   pre_objective_pass_rate=?, post_objective_pass_rate=?,
                   status='done', detail_json=?, finished_at=?
               WHERE id=?""",
            (
                scores.get("pre"), scores.get("post"), scores.get("transfer"),
                scores.get("objective_pre"), scores.get("objective_post"),
                scores.get("objective_pre_pass_rate"),
                scores.get("objective_post_pass_rate"),
                json.dumps(detail, ensure_ascii=False), _now(), experiment_id,
            ),
        )

    def finish_run(self, run_id: str, summary: dict[str, Any],
                   diagnostics: dict[str, Any]) -> None:
        self.db.execute(
            "UPDATE v3_runs SET status='done', summary_json=?,"
            " diagnostics_json=?, finished_at=? WHERE run_id=?",
            (
                json.dumps(summary, ensure_ascii=False),
                json.dumps(diagnostics, ensure_ascii=False),
                _now(), run_id,
            ),
        )

    # ---------- 落库 ----------
    def add_stage_item(
        self,
        *,
        run_id: str,
        experiment_id: int,
        exp_group: str,
        stage: str,
        item: b2.ExamItem,
        answer: str,
        score: float | None,
        context_used: str,
        candidate: ProviderConfig,
        evaluator: ProviderConfig,
        mode: str = "",
        reference_answer: str = "",
    ) -> None:
        self.db.execute(
            """INSERT INTO v3_stage_items(run_id, experiment_id, exp_group, stage,
                 seq, item_id, knowledge_point, question_type, question, answer,
                 score, context_used, model, provider, evaluator,
                 reference_answer, mode, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, experiment_id, exp_group, stage, _seq_for(stage),
                item.item_id, item.kp, item.question_type, item.question, answer,
                score, context_used,
                candidate.model, candidate.name, evaluator.name,
                reference_answer, mode, _now(),
            ),
        )

    def add_objective(
        self,
        *,
        run_id: str,
        experiment_id: int,
        exp_group: str,
        stage: str,
        task_id: str,
        kp: str,
        kind: str,
        candidate_output: str,
        result: Any,
        candidate: ProviderConfig,
        context_used: str = "",
    ) -> None:
        self.db.execute(
            """INSERT INTO v3_objective_runs(run_id, experiment_id, exp_group,
                 stage, seq, task_id, kp, kind, candidate_output, checks_json,
                 objective_score, objective_passed, parse_error,
                 model, provider, context_used, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id, experiment_id, exp_group, stage,
                _seq_for(stage),
                task_id, kp, kind,
                candidate_output,
                json.dumps(result.checks, ensure_ascii=False),
                result.objective_score, 1 if result.passed else 0,
                result.parse_error or "",
                candidate.model, candidate.name, context_used, _now(),
            ),
        )

    # ---------- 读回 ----------
    def group_row(self, run_id: str, exp_group: str) -> dict[str, Any] | None:
        return self.db.query_one(
            "SELECT * FROM v3_experiments WHERE run_id=? AND exp_group=?",
            (run_id, exp_group),
        )

    def stage_items(
        self, run_id: str, exp_group: str, stage: str
    ) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM v3_stage_items WHERE run_id=? AND exp_group=? AND stage=?"
            " ORDER BY id",
            (run_id, exp_group, stage),
        )

    def objective_runs(
        self, run_id: str, exp_group: str, stage: str
    ) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM v3_objective_runs"
            " WHERE run_id=? AND exp_group=? AND stage=? ORDER BY id",
            (run_id, exp_group, stage),
        )

    # ---------- 调用失败 ----------
    def record_call_failure(
        self,
        run_id: str,
        exp_group: str,
        stage: str,
        item_id: str,
        error: str,
    ) -> None:
        """结构化记录一次考生调用失败。

        调用失败**不是**「模型答不出」：前者是这次实验没跑成，
        后者是一个有效的观测值。混为一谈会让两组分数被
        随机噪声替代，而报告里看起来一切正常。
        """
        provider = model = ""
        if self.spec is not None:
            provider, model = (
                self.spec.candidate.name, self.spec.candidate.model
            )
        self.db.execute(
            """INSERT INTO v3_call_failures(run_id, exp_group, stage,
                 item_id, error, provider, model, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (run_id, exp_group, stage, item_id, error,
             provider, model, _now()),
        )

    def call_failures(self, run_id: str) -> list[dict[str, Any]]:
        return self.db.query_all(
            "SELECT * FROM v3_call_failures WHERE run_id=? ORDER BY id",
            (run_id,),
        )


# ==========================================================================
# 判分
# ==========================================================================


def score_answer(
    evaluator: ProviderConfig,
    question: str,
    candidate_answer: str,
    reference_answer: str,
) -> tuple[float, str, str]:
    """用**评估器 provider** 给一份答案打分。

    返回 (score, verdict, feedback)。LLM 返回不了合法 JSON 时
    走规则兜底——但兜底必须**记下来**，
    因为「LLM 判分失败率」本身是评估器可靠性的一个指标。
    """
    user = (
        f"题目：{question}\n"
        f"参考答案：{reference_answer}\n"
        f"考生答案：{candidate_answer}"
    )
    data: dict[str, Any] | None = None
    try:
        data = evaluator.provider.ask_json(
            EVALUATOR_SYSTEM, user, temperature=0.2
        )
    except ValueError:
        data = None
    if not isinstance(data, dict) or "score" not in data:
        # 规则兜底：只按「是否提到参考答案的关键片段」给一个保守分
        overlap = _overlap_ratio(candidate_answer, reference_answer)
        score = round(overlap * 60.0, 2)
        verdict = "pass" if score >= 60 else "fail"
        return score, verdict, "（评估器未返回合法 JSON，已降级为规则兜底）"
    score = max(0.0, min(100.0, float(data.get("score", 0))))
    verdict = str(data.get("verdict") or ("pass" if score >= 60 else "fail"))
    return score, verdict, str(data.get("feedback", ""))


def _overlap_ratio(answer: str, reference: str, probe_len: int = 6) -> float:
    """参考答案的关键片段在考生答案里的命中率。"""
    ref = "".join(reference.split())
    ans = "".join(answer.split())
    if not ref:
        return 0.0
    probes = [
        ref[i:i + probe_len]
        for i in range(0, max(1, len(ref) - probe_len), probe_len)
    ]
    probes = [p for p in probes if len(p) == probe_len]
    if not probes:
        return 0.0
    hit = sum(1 for p in probes if p in ans)
    return hit / len(probes)


# ==========================================================================
# 编排
# ==========================================================================


class DualGroupRunner:
    """跑一次完整的双组实验。"""

    def __init__(
        self,
        store: V3Store,
        spec: RunSpec,
        call_candidate: Callable[[ProviderConfig, str, str], str],
        score_fn: Callable[
            [ProviderConfig, str, str, str], tuple[float, str, str]
        ] = score_answer,
    ) -> None:
        self.store = store
        self.spec = spec
        # store 借一次 spec，好让失败记录带上 provider/model
        if store.spec is None:
            store.spec = spec
        self.call_candidate = call_candidate
        self.score_fn = score_fn
        self._experiment_ids: dict[str, int] = {}

    # ---------- 建立 ----------
    def setup(self) -> dict[str, gi.GroupScope]:
        gi.assert_isolation_intact()
        ho.assert_heldout_intact()
        ho.assert_heldout_isolated()
        ho.assert_heldout_scorers_intact()
        cm.assert_control_intact()

        self.store.create_run(self.spec)
        ids = {
            g: self.store.create_group(
                self.spec.run_id, g, kind, text
            )
            for g, kind, text in (
                ("learning", "novel_zephyra", nm.NOVEL_MATERIAL),
                ("control", "control_veyra", cm.CONTROL_MATERIAL),
            )
        }
        self._experiment_ids = ids
        return gi.build_scopes(
            self.spec.run_id, ids["learning"], ids["control"]
        )

    # ---------- 单个阶段 ----------
    def _run_exam(
        self,
        scope: gi.GroupScope,
        stage_base: str,
        items: Sequence[b2.ExamItem],
        context_used: str = "",
        mode: str = "",
    ) -> list[dict[str, Any]]:
        stage = scope.stage_of(stage_base)
        cand, ev = self.spec.providers_for(stage)
        rows: list[dict[str, Any]] = []
        for item in items:
            user = (
                (f"以下是本阶段可见的材料：\n{context_used}\n\n" if context_used else "")
                + f"题目：{item.question}\n"
                f"请给出你的推理与最终答案。"
                + stage_marker(stage, scope.exp_group)
            )
            try:
                answer = self.call_candidate(
                    cand, CANDIDATE_SYSTEM, user
                )
            except Exception as exc:  # noqa: BLE001
                # **考生崩溃与「模型说不会」是两件完全不同的事。**
                #
                # 早先这里把崩溃也记成一句文本，然后照样送进评估器——
                # 于是「（考生调用失败：RuntimeError）」被当成一份答案打分，
                # 两组都拿到同一个中间分，difference-in-gains 凭空变成 0。
                # 排查时最难发现的就是这个：分数有值、报告能出、
                # 只是数据是假的。
                #
                # 所以崩溃直接记账并中止本次 run：
                # 调用失败率本身是实验可靠性的指标，不能被抹平成一个分数。
                self.store.record_call_failure(
                    self.spec.run_id, scope.exp_group, stage,
                    item.item_id, f"{type(exc).__name__}: {exc}",
                )
                raise RuntimeError(
                    f"{stage} 的 {item.item_id} 调用考生失败："
                    f"{type(exc).__name__}: {exc}\n"
                    f"已记入 v3_call_failures。本次 run 作废——"
                    f"调用失败的样本被当成「模型答不出」，"
                    f"会让两组分数被随机噪声替代。"
                ) from exc
            score, _verdict, _fb = self.score_fn(
                ev, item.question, answer, item.answer
            )
            self.store.add_stage_item(
                run_id=self.spec.run_id,
                experiment_id=scope.experiment_id,
                exp_group=scope.exp_group,
                stage=stage,
                item=item,
                answer=answer,
                score=score,
                context_used=context_used,
                candidate=cand,
                evaluator=ev,
                mode=mode or ("blind" if stage in BLIND_STAGES_V3 else "open"),
                reference_answer=item.answer,
            )
            rows.append({"item_id": item.item_id, "score": score, "answer": answer})
        return rows

    def _run_objective(
        self,
        scope: gi.GroupScope,
        stage: str,
        tasks: Sequence[Any],
        context_used: str = "",
    ) -> list[dict[str, Any]]:
        cand, _ev = self.spec.providers_for(stage)
        rows: list[dict[str, Any]] = []
        for task in tasks:
            try:
                out = self.call_candidate(
                    cand, OBJECTIVE_SYSTEM,
                    f"题目：{task.prompt}\n提交格式：{task.submit_hint}"
                    + stage_marker(stage, scope.exp_group),
                )
            except Exception as exc:  # noqa: BLE001
                # 同 _run_exam：调用失败不是「模型答不出」。
                # 客观任务上更危险——判分器对空/非法产出照样给分，
                # 失败会被算成「做了但做错」，而不是「根本没做」。
                self.store.record_call_failure(
                    self.spec.run_id, scope.exp_group, stage,
                    task.task_id, f"{type(exc).__name__}: {exc}",
                )
                raise RuntimeError(
                    f"{stage} 的 {task.task_id} 调用考生失败："
                    f"{type(exc).__name__}: {exc}\n"
                    f"已记入 v3_call_failures。本次 run 作废。"
                ) from exc
            # held-out 任务用held-out 自己的判分器
            #（practice 的判分器把practice 的参数写死在检查项里）
            res = ho.evaluate_heldout(task.task_id, out)
            self.store.add_objective(
                run_id=self.spec.run_id,
                experiment_id=scope.experiment_id,
                exp_group=scope.exp_group,
                stage=stage,
                task_id=task.task_id,
                kp=task.kp,
                kind=task.kind,
                candidate_output=out,
                result=res,
                candidate=cand,
                context_used=context_used,
            )
            rows.append({
                "task_id": task.task_id,
                "score": res.objective_score,
                "passed": res.passed,
            })
        return rows

    def _run_material(
        self, scope: gi.GroupScope, stage: str
    ) -> dict[str, Any]:
        """材料阶段：给模型材料，让它产出自己的学习产物。

        两组都做**同样的动作**（读材料 → 抽知识点 → 写笔记），
        唯一区别是材料内容不同。这样「两组最终分数之差」才只由
        材料内容解释，而不是由「做没做这一步」解释。
        """
        cand, _ev = self.spec.providers_for(stage)
        material = (
            nm.NOVEL_MATERIAL if scope.is_learning else cm.CONTROL_MATERIAL
        )
        user = (
            f"以下是本阶段的学习材料：\n{material}\n\n"
            f"请通读全文，写下你理解的关键知识点。"
            f"输出 JSON：{{\"points\": [{{\"kp\": \"知识点id\","
            f" \"concept\": \"概念\", \"principle\": \"原理\"}}]}}"
            + stage_marker(stage, scope.exp_group)
        )
        try:
            raw = self.call_candidate(cand, CANDIDATE_SYSTEM, user)
        except Exception as exc:  # noqa: BLE001
            self.store.record_call_failure(
                self.spec.run_id, scope.exp_group, stage,
                f"{stage}_material_read", f"{type(exc).__name__}: {exc}",
            )
            raise RuntimeError(
                f"{stage} 的材料阶段调用考生失败："
                f"{type(exc).__name__}: {exc}\n"
                f"已记入 v3_call_failures。本次 run 作废。"
            ) from exc

        points: list[dict[str, Any]] = []
        parsed = _extract_json(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("points"), list):
            points = [p for p in parsed["points"] if isinstance(p, dict)]

        # 落学习产物：带本组自己的 v3_experiment_id
        self._persist_learning_products(scope, stage, material, points)

        # 材料阶段也记一行 stage_item，让「这一步做没做」在数据里可见
        probe = b2.ExamItem(
            item_id=f"{stage}_material_read",
            kp="__material__",
            question=f"（{scope.exp_group} 组材料阶段）",
            answer=raw,
            question_type="material",
        )
        self.store.add_stage_item(
            run_id=self.spec.run_id,
            experiment_id=scope.experiment_id,
            exp_group=scope.exp_group,
            stage=stage,
            item=probe,
            answer=raw,
            score=None,
            context_used=material,
            candidate=cand,
            evaluator=_ev,
            mode="open",
            reference_answer="",
        )
        return {"points": len(points), "raw": raw}

    def _persist_learning_products(
        self,
        scope: gi.GroupScope,
        stage: str,
        material: str,
        points: Sequence[dict[str, Any]],
    ) -> None:
        """把知识点 / 笔记 / 材料写进库，全部带本组的 v3_experiment_id。

        这是组间隔离的**物理层**：两组写的是同一个 topic，
        但 experiment_id 不同，所以任何按 experiment_id 的读取
        天然分得开。
        """
        now = _now()
        for p in points:
            kp = str(p.get("kp") or f"kp_{len(points)}")
            concept = str(p.get("concept") or "")
            principle = str(p.get("principle") or "")
            # knowledge
            self.store.db.execute(
                """INSERT INTO knowledge(topic, title, content, keywords, source,
                     confidence, created_at, updated_at, experiment_id)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    self.spec.topic, concept, principle, kp, f"v6:{stage}",
                    0.5, now, now, scope.experiment_id,
                ),
            )
            # learning_notes（checksum 逐条算，污染检测靠它）
            note_text = f"{kp}\n{concept}\n{principle}"
            self.store.db.execute(
                """INSERT INTO learning_notes(experiment_id, topic, kp, concept,
                     principle, example, common_mistake, application_rule,
                     checksum, created_at, verified, v2_experiment_id,
                     v3_experiment_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    v3_experiment_key(scope.experiment_id), self.spec.topic,
                    kp, concept, principle,
                    "", "", "", checksum(note_text), now, 0, None,
                    scope.experiment_id,
                ),
            )
        # materials：对照组材料必须一眼能认出是 control
        self.store.db.execute(
            """INSERT INTO materials(topic, version, title, content, checksum,
                 created_at, v3_experiment_id, exp_group)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                self.spec.topic, "v6.0", f"{scope.exp_group} 材料",
                material, checksum(material), now,
                scope.experiment_id, scope.exp_group,
            ),
        )

    def _run_practice(
        self, scope: gi.GroupScope, tasks: Sequence[Any]
    ) -> list[dict[str, Any]]:
        """实践阶段。

        **两组做同一套 practice 任务集（参数与 held-out 全部错开）。**

        早先这里让 Control 组做 ``ho.HELD_OUT_TASKS``，理由是
        「两组都要经历做题」。但那是错的：
        ``post_objective`` 测的正是 ``HELD_OUT_TASKS``，
        Control 组在 practice 里做过同一批题，
        它的 post_objective 分数会因「练过」而虚高，
        ``objective_difference_in_gains`` 被系统性压低——
        实验会偏向**假阴性**，而且这个偏差在任何统计量里都看不出来。

        所以两组统一做 practice 任务集：
        两组都经历了「做题 → 拿到反馈」，但**谁都没碰过测试题**。
        """
        from practice import objective as pob

        cand, _ev = self.spec.providers_for(scope.stage_of("practice"))
        rows: list[dict[str, Any]] = []

        for task in tasks:
            try:
                out = self.call_candidate(
                    cand, PRACTICE_SYSTEM,
                    f"题目：{task.prompt}\n提交格式：{task.submit_hint}"
                    + stage_marker(scope.stage_of("practice"), scope.exp_group),
                )
            except Exception as exc:  # noqa: BLE001
                self.store.record_call_failure(
                    self.spec.run_id, scope.exp_group,
                    scope.stage_of("practice"), task.task_id,
                    f"{type(exc).__name__}: {exc}",
                )
                raise RuntimeError(
                    f"practice 的 {task.task_id} 调用考生失败："
                    f"{type(exc).__name__}: {exc}\n"
                    f"已记入 v3_call_failures。本次 run 作废。"
                ) from exc
            # 两组都是 practice 任务集，判分器统一用 practice 的
            res = pob.evaluate_objective(task, out)
            self.store.add_objective(
                run_id=self.spec.run_id,
                experiment_id=scope.experiment_id,
                exp_group=scope.exp_group,
                stage=scope.stage_of("practice"),
                task_id=task.task_id,
                kp=task.kp,
                kind=task.kind,
                candidate_output=out,
                result=res,
                candidate=cand,
                context_used="",
            )
            rows.append({
                "task_id": task.task_id,
                "score": res.objective_score,
                "passed": res.passed,
            })
        return rows

    def _finish_groups(
        self,
        scopes: dict[str, gi.GroupScope],
        l_pre_obj: list[dict[str, Any]],
        c_pre_obj: list[dict[str, Any]],
        l_post_obj: list[dict[str, Any]],
        c_post_obj: list[dict[str, Any]],
        trial: dict[str, Any],
        obj_trial: dict[str, Any],
    ) -> None:
        """把两组的分数落库。

        分数**从库里读回**而不是从内存变量拿——
        落库这一环要是坏了（列名写错、条件写错），
        读回时会立刻发现，而不是让报告拿着一份没落库的内存值讲结论。
        """
        obj_pre = {"learning": l_pre_obj, "control": c_pre_obj}
        obj_post = {"learning": l_post_obj, "control": c_post_obj}
        for group, scope in scopes.items():
            def scores_of(base: str) -> float:
                rows = self.store.stage_items(
                    self.spec.run_id, group, f"{base}_{group}"
                )
                return _avg([{"score": r["score"]} for r in rows])

            self.store.finish_group(
                scope.experiment_id,
                {
                    "pre": scores_of("pre"),
                    "post": scores_of("post"),
                    "transfer": scores_of("transfer"),
                    "objective_pre": _avg(obj_pre[group]),
                    "objective_post": _avg(obj_post[group]),
                    "objective_pre_pass_rate": _pass_rate(obj_pre[group]),
                    "objective_post_pass_rate": _pass_rate(obj_post[group]),
                },
                {"trial": trial, "objective_trial": obj_trial},
            )

    # ---------- 完整流程 ----------
    def run(self) -> dict[str, Any]:
        from practice import objective as pob

        scopes = self.setup()
        L, C = scopes["learning"], scopes["control"]

        # --- 1. Learning PRE（两组的 PRE 都先跑完，再给材料）---
        l_pre = self._run_exam(L, "pre", b2.PRE_ITEMS)
        # --- 2. Control PRE ---
        c_pre = self._run_exam(C, "pre", b2.PRE_ITEMS)

        # 客观 held-out 的 PRE：两组同一批任务，全部盲考
        l_pre_obj = self._run_objective(L, "pre_objective", ho.HELD_OUT_TASKS)
        c_pre_obj = self._run_objective(C, "pre_objective", ho.HELD_OUT_TASKS)

        # --- 3. Learning STUDY ---
        self._run_material(L, "study_learning")
        # --- 4. Control CONTROL_MATERIAL ---
        self._run_material(C, "control_material")

        # --- 5. Learning PRACTICE（practice 任务集）---
        self._run_practice(L, pob.OBJECTIVE_TASKS)
        # --- 6. Control PRACTICE（**同一套** practice 任务集）---
        #    绝不能用 HELD_OUT_TASKS：那批题是 post_objective 的测试题，
        #    Control 组练过再测，objective_difference_in_gains 会被压成假阴性。
        self._run_practice(C, pob.OBJECTIVE_TASKS)

        # --- 7. Learning POST（held-out 卷，盲）---
        l_post = self._run_exam(L, "post", ho.HELD_OUT_POST)
        # --- 8. Control POST（同一批题，盲）---
        c_post = self._run_exam(C, "post", ho.HELD_OUT_POST)

        l_post_obj = self._run_objective(L, "post_objective", ho.HELD_OUT_TASKS)
        c_post_obj = self._run_objective(C, "post_objective", ho.HELD_OUT_TASKS)

        # --- 9. Learning TRANSFER ---
        l_tr = self._run_exam(L, "transfer", ho.HELD_OUT_TRANSFER)
        # --- 10. Control TRANSFER ---
        c_tr = self._run_exam(C, "transfer", ho.HELD_OUT_TRANSFER)

        # ---------- 汇总 ----------
        trial = c3.paired_trial(
            _avg(l_pre), _avg(l_post), _avg(c_pre), _avg(c_post)
        )
        obj_trial = c3.objective_trial(
            _avg(l_pre_obj), _avg(l_post_obj),
            _avg(c_pre_obj), _avg(c_post_obj),
        )
        self._finish_groups(scopes, l_pre_obj, c_pre_obj, l_post_obj, c_post_obj,
                            trial, obj_trial)

        diff = c3.aggregate_difference([trial])
        obj_diff = c3.objective_difference_summary([obj_trial])
        contam = gi.contamination_report(self.store.db, self.spec.run_id)
        order = gi.order_audit(self.store.db, self.spec.run_id)
        perm = c3.claim_permission(diff, contam["clean"], 1)

        summary = {
            "run_id": self.spec.run_id,
            "trial": trial,
            "objective_trial": obj_trial,
            "difference_summary": diff,
            "objective_difference_summary": obj_diff,
            "claim_permission": perm,
        }
        self.store.finish_run(
            self.spec.run_id, summary,
            {"contamination": contam, "order": order},
        )
        return {
            **summary,
            "contamination": contam,
            "order": order,
            "experiment_ids": dict(self._experiment_ids),
        }


def _avg(rows: Sequence[dict[str, Any]]) -> float:
    vals = [float(r["score"]) for r in rows if r.get("score") is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _pass_rate(rows: Sequence[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    hit = sum(1 for r in rows if r.get("passed"))
    return round(100.0 * hit / len(rows), 2)


def _extract_json(text: str) -> Any:
    """从可能包裹着代码块或前后废话的文本里提取 JSON。"""
    import re

    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i = t.find(opener)
        j = t.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except ValueError:
                continue
    return None


__all__ = [
    "CANDIDATE_SYSTEM",
    "EVALUATOR_SYSTEM",
    "DualGroupRunner",
    "OBJECTIVE_SYSTEM",
    "PRACTICE_SYSTEM",
    "ProviderConfig",
    "RunSpec",
    "V3Store",
    "score_answer",
]
