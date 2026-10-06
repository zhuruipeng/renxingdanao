"""实践能力评估 —— V0.4 指标之一：学习能否转化为实际任务能力。

**为什么 V0.3 的实践评分不够**
----------------------------
V0.3 的 `_practice_score` 只看两件事：模型自报的 outcome（success/failure），
以及结果文本的长度。这测的是「模型自己觉得做成了吗」和「写得多不多」，
不是「产出到底能不能用」。

一个模型完全可以写一大段看起来很像样的话，内容全是空话，
而V0.3 的打分会给它 85 分。

**V0.4 的做法**
--------------
用**独立评估器**判两件事：
1. `usable`  —— 产出是不是可执行的（有没有具体步骤、命令、SQL、验证方法）
2. `score`   —— 00-100，按可执行性/正确性/是否解决真实问题打分

评估器**看不到**学习材料，也看不到笔记。理由和评估笔记时一样：
「这个产出好不好」和「它有没有用上材料」是两件事，混在一起就分不清了。
如果连材料都看不到，模型本来就答不出材料细节，那它靠先验知识
能写对的东西，恰好就是我们要问的「能力是否内化」。

**这一项独立于笔记**：笔记质量高不代表实践能力强。
反之亦然——实践中露出的错误往往比笔记里的错误更致命。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from brain.llm import LLMProvider
from memory.db import Database, v2_experiment_key
from memory.gate import RESOURCE_EXPERIENCES, AccessPolicy

PRACTICE_JUDGE_SYSTEM = (
    "你是严格的技术实践评审。你会拿到一个知识点名称和一段学习者的实践产出。"
    "请判断这段产出是否**真的可执行**，而不是描述性文字。"
    "标准：\n"
    "- 必须有可操作的具体内容：命令、SQL、步骤、检查方法、可观测的判据\n"
    "- 仅有概念复述、没有任何操作路径的，usable判false\n"
    "- 有步骤但存在事实错误（比如给出的命令在该数据库上不成立），"
    "按错误严重程度扣分\n"
    "只输出 JSON："
    '{"usable":true,"score":70,"verdict":"pass|fail","feedback":"简短评语"}'
)

PRACTICE_JUDGE_TAG = "[[TASK:PRACTICE_JUDGE]]"

#: 判定「产出可执行」的分数线
USABLE_LINE = 55.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: 这些词出现说明产出停留在「我要做什么」而不是「我做了什么/怎么做」
_VAGUE_MARKERS = (
    "我将会",
    "我会先",
    "接下来我会",
    "计划先",
    "准备先",
    "将要",
    "打算",
)


class PracticeJudgeStore:
    def __init__(self, db: Database, policy: AccessPolicy | None = None) -> None:
        self.db = db
        self.policy = policy or AccessPolicy.open_mode()

    def add(
        self,
        experiment_id: int,
        kp: str,
        outcome: str,
        task_prompt: str,
        candidate_answer: str,
        score: float,
        usable: bool,
        verdict: str,
        feedback: str,
        evaluator: str = "llm_judge",
    ) -> int:
        self.policy.guard(RESOURCE_EXPERIENCES)
        cur = self.db.execute(
            """INSERT INTO practice_evaluations(experiment_id, kp, outcome, task_prompt,
                                                 candidate_answer, score, usable, verdict,
                                                 feedback, evaluator, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (experiment_id, kp, outcome, task_prompt, candidate_answer,
             float(score), 1 if usable else 0, verdict, feedback, evaluator, _now()),
        )
        return int(cur.lastrowid)

    def for_experiment(self, experiment_id: int) -> list[dict[str, Any]]:
        self.policy.guard(RESOURCE_EXPERIENCES)
        return self.db.query_all(
            "SELECT * FROM practice_evaluations WHERE experiment_id = ? ORDER BY id",
            (experiment_id,),
        )

    def add_v2(
        self,
        v2_experiment_id: int,
        kp: str,
        outcome: str,
        task_prompt: str,
        candidate_answer: str,
        score: float,
        usable: bool,
        verdict: str,
        feedback: str,
        evaluator: str = "llm_judge",
        task_id: str = "",
    ) -> int:
        """V0.5 写入：**experiment_id 写负数哨兵**，归属靠符号判断。

        为什么不能直接写 V2 的 id：那会挂到同号的 V0.4 实验名下，
        读回来就是别人的数据——数字撞上，且不会报任何错。
        为什么不用 NULL：三列都是 NOT NULL，SQLite 改不掉。

        `task_id` 是 V2 新增：机器判分在 `practice_runs` 里有 task_id，
        评审侧没有的话两侧**无法对齐**——看到「一个55 分一个 70 分」，
        却不知道分别是哪个任务，分歧就成了无法定位的噪声。
        """
        self.policy.guard(RESOURCE_EXPERIENCES)
        cur = self.db.execute(
            """INSERT INTO practice_evaluations(experiment_id, v2_experiment_id, kp,
                                                 outcome, task_prompt, candidate_answer,
                                                 score, usable, verdict, feedback,
                                                 evaluator, task_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (v2_experiment_key(v2_experiment_id), v2_experiment_id, kp, outcome,
             task_prompt, candidate_answer, float(score), 1 if usable else 0,
             verdict, feedback, evaluator, task_id, _now()),
        )
        return int(cur.lastrowid)

    def for_v2_experiment(self, v2_experiment_id: int) -> list[dict[str, Any]]:
        self.policy.guard(RESOURCE_EXPERIENCES)
        return self.db.query_all(
            "SELECT * FROM practice_evaluations WHERE v2_experiment_id = ? ORDER BY id",
            (v2_experiment_id,),
        )

    def usable_rate(self, experiment_id: int) -> float:
        """实践产出可执行的比例。空实验返回 0.0。"""
        rows = self.for_experiment(experiment_id)
        if not rows:
            return 0.0
        return round(sum(1 for r in rows if r["usable"]) / len(rows), 4)

    def average_score(self, experiment_id: int) -> float:
        rows = self.for_experiment(experiment_id)
        if not rows:
            return 0.0
        return round(sum(float(r["score"]) for r in rows) / len(rows), 2)


class PracticeJudge:
    """独立实践评审器。"""

    def __init__(
        self, db: Database, store: PracticeJudgeStore, provider: LLMProvider
    ) -> None:
        self.db = db
        self.store = store
        self.provider = provider

    def judge(
        self,
        experiment_id: int,
        kp: str,
        kp_label: str,
        task_prompt: str,
        candidate_answer: str,
        *,
        v2: bool = False,
        task_id: str = "",
    ) -> dict[str, Any]:
        """判一次实践产出。

        注意：**不传学习材料，也不传笔记**。
        这样能判的是「模型自身现在能不能做出可执行的动作」，
        而不是「它有没有把笔记用上」。

        `v2=True` 时走 V0.5 的写入路径（只填 v2_experiment_id）。
        """
        user = (
            f"{PRACTICE_JUDGE_TAG}\n知识点：{kp_label or kp}\n"
            f"实践任务：{task_prompt}\n"
            f"学习者的实践产出：\n{candidate_answer}"
        )
        data: dict[str, Any] | None = None
        try:
            data = self.provider.ask_json(PRACTICE_JUDGE_SYSTEM, user, temperature=0.1)
        except ValueError:
            data = None

        if isinstance(data, dict) and "score" in data:
            score = _clamp(data.get("score"))
            usable = bool(data.get("usable")) and score >= USABLE_LINE
            verdict = str(data.get("verdict") or ("pass" if usable else "fail"))
            feedback = str(data.get("feedback", ""))
            evaluator = "llm_judge"
        else:
            score, usable, verdict, feedback = self._rule_fallback(candidate_answer)
            # 必须留痕：兜底只查了可操作性，没查正确性，
            # 报告里若把它和真判分混为一谈，「实践可用率」就失去意义了。
            evaluator = "rule_fallback"

        if v2:
            self.store.add_v2(
                v2_experiment_id=experiment_id,
                kp=kp,
                outcome="success" if usable else "failure",
                task_prompt=task_prompt,
                candidate_answer=candidate_answer,
                score=score,
                usable=usable,
                verdict=verdict,
                feedback=feedback,
                evaluator=evaluator,
                task_id=task_id,
            )
        else:
            self.store.add(
                experiment_id=experiment_id,
                kp=kp,
                outcome="success" if usable else "failure",
                task_prompt=task_prompt,
                candidate_answer=candidate_answer,
                score=score,
                usable=usable,
                verdict=verdict,
                feedback=feedback,
                evaluator=evaluator,
            )
        return {
            "kp": kp,
            "task_id": task_id,
            "score": score,
            "usable": usable,
            "verdict": verdict,
            "evaluator": evaluator,
            "feedback": feedback,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _rule_fallback(answer: str) -> tuple[float, bool, str, str]:
        """LLM 不可用时的规则兜底：只检查「有没有可操作内容」。

        **刻意不检查正确性**——没有材料对照就判不了对错，
        与笔记质量兜底同样的理由：宁可给低分不给假高分。
        """
        text = answer.strip()
        has_command = any(
            t in text for t in ("SELECT", "INSERT", "CREATE", "EXPLAIN",
                                "PRAGMA", "index", "VACUUM", "sqlite3")
        )
        has_step = any(t in text for t in ("1.", "2.", "3.", "首先", "然后", "最后"))
        vague_only = any(t in text for t in _VAGUE_MARKERS) and not has_command

        score = 20.0
        if has_command:
            score += 35.0
        if has_step:
            score += 20.0
        if len(text) >= 120:
            score += 15.0
        if vague_only:
            score = min(score, 35.0)
        score = min(score, 90.0)  # 规则兜底永远不给满分

        usable = score >= USABLE_LINE and not vague_only
        return (
            round(score, 2),
            usable,
            "pass" if usable else "fail",
            "规则兜底：未做正确性核验，仅按可操作性打分",
        )


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "PRACTICE_JUDGE_SYSTEM",
    "PRACTICE_JUDGE_TAG",
    "USABLE_LINE",
    "PracticeJudge",
    "PracticeJudgeStore",
]
