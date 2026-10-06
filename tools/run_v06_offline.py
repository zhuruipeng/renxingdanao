"""tools/run_v06_offline.py —— 用 ScriptedProvider 跑一次完整 V0.6 双组实验。

**不联网、不烧 token、不调真实模型。**
目的是验证整条链路能跑通，并且**两组能产生不同结果**——
如果两组结果一样，说明实验设计没起作用，链路是空的。

脚本钦定分数，跑出来是多少都不构成能力结论。

用法:
    python tools/run_v06_offline.py            # 跑 1 次
    python tools/run_v06_offline.py --trials 3# 跑 3 次看统计
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import causal_v3 as c3  # noqa: E402
from brain import control_material as cm  # noqa: E402
from brain import experiment_v3 as ev3  # noqa: E402
from brain import group_isolation as gi  # noqa: E402
from brain import heldout_bank as ho  # noqa: E402
from brain import novel_material as nm  # noqa: E402
from brain import report_v3 as r3  # noqa: E402
from brain.llm import LLMResponse, ScriptedProvider  # noqa: E402
from memory.db import Database  # noqa: E402
from practice.objective import OBJECTIVE_TASKS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"


# ==========================================================================
# 8 个 practice 任务 + 8 个 held-out 任务的标准产出
# ==========================================================================


def _practice_correct() -> dict[str, str]:
    from practice import objective as ob

    return {
        "obj_hash_01": json.dumps({
            "key": "abc", "steps": ob.HASH_ABC_STEPS,
            "lumen_hash": ob.HASH_ABC_VALUE, "bucket": ob.HASH_ABC_BUCKET,
        }),
        "obj_hash_02": json.dumps({
            "code": "def lumen_hash(key):\n    h = 7\n"
                    "    for c in key:\n        h = (h * 131 + ord(c)) % 65537\n"
                    "    return h"}),
        "obj_bucket_01": json.dumps({
            "assignments": [{"key": n, "from_bucket": 0, "to_bucket": 0}
                            for n, _ in ob.SPLIT_KEYS],
            "migrated_count": ob.SPLIT_MIGRATED_COUNT,
            "still_over_limit": ob.SPLIT_STILL_OVER_LIMIT,
        }),
        "obj_bucket_02": json.dumps({
            "sql": "SELECT bucket FROM zephyra_buckets"
                   " WHERE live_count > 4 ORDER BY bucket"}),
        "obj_compact_01": json.dumps({
            "compress": ob.COMPACT_TRIGGERED,
            "epoch_after": ob.COMPACT_EPOCH_AFTER,
        }),
        "obj_compact_02": json.dumps({
            "code": "def should_compact(tombstones, live):\n"
                    "    return tombstones >= 2 and live <= 1"}),
        "obj_consist_01": json.dumps({
            "readable": ob.CONSIST_READABLE,
            "need_retag": [k["key"] for k in ob.CONSIST_KEYS],
            "new_tags": ob.CONSIST_EXPECTED_TAGS,
        }),
        "obj_consist_02": json.dumps({
            "sql": "SELECT key FROM zephyra_keys WHERE ztag != (lumen_hash +"
                   " (SELECT value FROM zephyra_meta WHERE name='epoch')) % 65537"
                   " ORDER BY key"}),
    }


def _heldout_correct() -> dict[str, str]:
    out: dict[str, str] = {}
    for task in ho.HELD_OUT_TASKS:
        out[task.task_id] = ho._reference_answer(task)
    return out


PRACTICE_CORRECT = _practice_correct()
HELDOUT_CORRECT = _heldout_correct()


# ==========================================================================
# 题面片段
# ==========================================================================

#: practice 任务的题面片段（必须互不重叠，否则会静默错配）
_PRACTICE_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("obj_hash_01", "LumenHash 值，并给出它所属的桶号"),
    ("obj_hash_02", "def lumen_hash(key)"),
    ("obj_bucket_01", "0、8、16、24、32"),
    ("obj_bucket_02", "每个桶最多容纳 4 个活跃键"),
    ("obj_compact_01", "tombstones=9"),
    ("obj_compact_02", "def should_compact(tombstones, live)"),
    ("obj_consist_01", "4000、6000、8200"),
    ("obj_consist_02", "zephyra_meta(name, value)"),
)

#: held-out 任务的题面片段
_HELDOUT_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("ho_hash_01", "LumenHash 值，并给出它所属的桶号"),
    ("ho_hash_02", "def lumen_hash(key)"),
    ("ho_bucket_01", "5、13、21、29、37"),
    ("ho_bucket_02", "每个桶最多容纳 4 个活跃键"),
    ("ho_compact_01", "桶 5：live=5, tombstones=0"),
    ("ho_compact_02", "def should_compact(tombstones, live)"),
    ("ho_consist_01", "5 个键的 LumenHash 与写入时的全局 epoch"),
    ("ho_consist_02", "zephyra_meta(name, value)"),
)


def selfcheck_fragments() -> list[str]:
    """跑之前先确认片段都还能命中且互不重叠。

    片段匹配不上时**不会报错**，只会静默走 fallback，
    于是全部任务拿 0 分——看上去像「模型全错」，
    真相是脚本没匹配上。V0.5 开发时踩过一次，这里做成硬门禁。
    """
    problems: list[str] = []
    for tasks, frags, label in (
        (OBJECTIVE_TASKS, _PRACTICE_FRAGMENTS, "practice"),
        (ho.HELD_OUT_TASKS, _HELDOUT_FRAGMENTS, "held-out"),
    ):
        by_id = {t.task_id: t for t in tasks}
        for task_id, frag in frags:
            task = by_id.get(task_id)
            if task is None:
                problems.append(f"{label}/{task_id}: 任务不存在")
            elif frag not in task.prompt:
                problems.append(
                    f"{label}/{task_id}: 题面已改，片段 {frag!r} 不再命中"
                )
        for i, (tid_a, frag_a) in enumerate(frags):
            for tid_b, frag_b in frags[i + 1:]:
                tb = by_id.get(tid_b)
                if tb is not None and frag_a in tb.prompt:
                    problems.append(
                        f"片段冲突：{label}/{tid_a} 的片段 {frag_a!r} "
                        f"也出现在 {tid_b} 的题面里，会静默错配"
                    )
    return problems


# ==========================================================================
# 考生脚本
# ==========================================================================

#: 考试题片段 → 阶段。
#:
#: PRE 阶段两组都答不出（模拟「未学习时不会」）；
#: POST/TRANSFER 阶段学习组能答出、对照组答不出。
#: 这个分叉正是 V0.6 要测的东西。
_ANSWER_UNKNOWN = "我不知道这个虚构系统的规则，无法作答。"
_ANSWER_LEARNING = (
    "根据 Zephyra 材料：LumenHash 初值 7，逐字符 h=(h*131+ord(c)) mod 65537；"
    "bucket = hash mod 8；一级分裂迁移到 (b+8) mod 8 = b，是恒等变换、不减负载；"
    "压缩条件是墓碑>=2 **且** 活跃键<=1；"
    "Z-Tag=(LumenHash+epoch) mod 65537，epoch 一变全库存量键失配，"
    "所以触发是局部的、影响是全局的。"
)
_ANSWER_CONTROL = (
    "根据 Veyra 材料：区间跨度 SPAN=60000，seg = T // SPAN，"
    "活跃数超过 8 就按 2*seg / 2*seg+1 分裂，父节点是 seg//2；"
    "校验码 check = (len(payload)*31 + seq) mod 8191。"
    "这些与题目问的分桶/压缩/标签规则无关。"
)


def _marker(text: str) -> tuple[str | None, str | None]:
    """从 prompt 里读出阶段与组别标记。

    早先靠「prompt 里有没有『本阶段可见的材料』」区分 PRE 与 POST，
    但盲考阶段的 PRE 与 POST prompt **结构完全相同**（都不带材料），
    于是 POST 被当成 PRE，两组都没提升、difference 恒等于 0。
    这类静默失配比报错更危险：报告照样出得来，只是两组一模一样。

    用正则而不是 split：手写 split 会把 ``[[V3_STAGE:x]]`` 切成
    ``STAGE:x]]``，key 少个前缀就静默匹配不上——
    第一次就是这么写的，结果又变成两组都答不出。
    """
    stage = _MARK_STAGE_RE.search(text)
    group = _MARK_GROUP_RE.search(text)
    return (
        stage.group(1) if stage else None,
        group.group(1) if group else None,
    )


_MARK_STAGE_RE = re.compile(r"\[\[V3_STAGE:([a-z_]+)\]\]")
_MARK_GROUP_RE = re.compile(r"\[\[V3_GROUP:([a-z]+)\]\]")


#: 「材料介入之后」的阶段前缀。客观任务的 stage 形如
#: ``pre_objective`` / ``post_objective``（**不带组名**），
#: 考试题则带组名（``post_learning`` / ``post_control``）。
#: 早先把它们混在一个白名单里，结果 post_objective 不在名单中，
#: 学习组客观 POST 也被判成答不出——四个阶段全是 15.83 分。
_POST_LIKE_EXAM = frozenset({
    "post_learning", "transfer_learning",
    "post_control", "transfer_control",
})
_POST_LIKE_OBJECTIVE = frozenset({"post_objective"})


def _is_pre(stage: str) -> bool:
    """这个阶段是不是「任何材料介入之前」。"""
    return stage.startswith("pre_")


def _can_solve(stage: str | None, group: str | None) -> bool:
    """这次调用能不能答对题。

    只有「学习组 + 材料介入之后」才答得出。
    其余一律答不出——包括对照组的所有阶段：
    它学的是 Veyra，本来就不该会做 Zephyra 的题。
    """
    if stage is None or group is None:
        return False
    if group != "learning":
        return False
    return stage in _POST_LIKE_EXAM or stage in _POST_LIKE_OBJECTIVE


#: 答不出时提交的占位 JSON。
#: **必须是合法 JSON** —— 判分器先解析再打分，
#: 塞一句自然语言会走 parse_error 分支，
#: 那样测的是「有没有按格式提交」，不是「会不会做」。
_UNSOLVED_JSON = json.dumps({
    "key": "", "steps": [0], "lumen_hash": 0, "bucket": 0,
    "assignments": [], "migrated_count": 0, "still_over_limit": False,
    "compress": [], "epoch_after": 0,
    "readable": [], "need_retag": [], "new_tags": {},
}, ensure_ascii=False)


class V3Scripted(ScriptedProvider):
    """离线考生。

    按**题面片段 + 阶段标记**分叉：

    - PRE 阶段：一律答不出（两组相同 → 基线可比）
    - POST / TRANSFER：Learning 组答得出、Control 组答不出
    - 材料阶段：返回固定知识点 JSON

    这是**钦定**的行为，不是能力测量。脚本在这里的作用是
    「证明链路能把两组分开」，不是「证明模型学会了什么」。
    """

    def complete(self, messages, *, temperature=0.7, max_tokens=None,
                 json_mode=False):
        joined = "\n".join(m.get("content", "") for m in messages)
        stage, group = _marker(joined)

        # 1. 客观任务：按片段返回标准答案
        #
        #    **但 PRE 必须答不出，且对照组任何阶段都答不出。**
        #    客观任务的 pre_objective 与 post_objective 用的是同一批 prompt，
        #    唯一区别就是阶段标记。若这里无条件返回标准答案，
        #    两次都会满分，objective gain 恒等于 0，客观口径整条链路是空的。
        for task_id, frag in _HELDOUT_FRAGMENTS:
            if frag in joined:
                if not _can_solve(stage, group):
                    return LLMResponse(text=_UNSOLVED_JSON, model="scripted-v3")
                return LLMResponse(text=HELDOUT_CORRECT[task_id], model="scripted-v3")
        for task_id, frag in _PRACTICE_FRAGMENTS:
            if frag in joined:
                return LLMResponse(text=PRACTICE_CORRECT[task_id], model="scripted-v3")

        # 2. 材料阶段
        if "请通读全文，写下你理解的关键知识点" in joined:
            if "Zephyra" in joined:
                pts = [
                    {"kp": "kp_lumen_hash", "concept": "LumenHash",
                     "principle": "h=7 起逐字符 (h*131+ord(c)) mod 65537"},
                    {"kp": "kp_lumen_bucket", "concept": "Z-8 分桶",
                     "principle": "bucket=hash mod 8；分裂 (b+8) mod 8 = b 恒等"},
                    {"kp": "kp_lumen_compact", "concept": "压缩",
                     "principle": "墓碑>=2 且活跃<=1"},
                    {"kp": "kp_lumen_consist", "concept": "Epoch 标签",
                     "principle": "ZTag=(hash+epoch) mod 65537，epoch 变则全库失配"},
                ]
            else:
                pts = [
                    {"kp": "kp_span", "concept": "区间跨度",
                     "principle": "SPAN=60000，seg = T // SPAN"},
                    {"kp": "kp_split", "concept": "分裂阈值",
                     "principle": "活跃数 > 8 触发，left=2seg, right=2seg+1"},
                    {"kp": "kp_parent", "concept": "父节点",
                     "principle": "parent = seg // 2"},
                    {"kp": "kp_check", "concept": "校验码",
                     "principle": "check = (len(payload)*31 + seq) mod 8191"},
                ]
            return LLMResponse(
                text=json.dumps({"points": pts}, ensure_ascii=False),
                model="scripted-v3",
            )

        # 3. 考试题：按阶段标记分叉
        #    没有标记 → 宁可报错，也不猜。
        #    猜错的话两组又变成同一个分数，
        #    而「两组一样」在报告里看起来像「没差异」，
        #    实际上只是脚本没接上。
        if stage is None:
            raise RuntimeError(
                f"prompt 里没有阶段标记，离线考生无法判断该答什么。"
                f"prompt 前 200 字：{joined[:200]!r}"
            )
        # PRE 两组都答不出，基线才可比。
        # 早先这里对control 一律返回 _ANSWER_CONTROL，
        # 于是 Control 的 PRE 拿了 72 分——它的「基线」其实是
        # 读着材料答出来的，等于把对照组凭空抬高了 62 分，
        # difference-in-gains 被压缩到一个偏小的值上。
        if not _is_pre(stage):
            if group == "control":
                return LLMResponse(text=_ANSWER_CONTROL, model="scripted-v3")
            return LLMResponse(text=_ANSWER_LEARNING, model="scripted-v3")
        return LLMResponse(text=_ANSWER_UNKNOWN, model="scripted-v3")


#: 目标知识（Zephyra）的核心规则要点。评估器只按**这些要点有没有被答到**给分，
#: 不看答案里出现了哪个系统名——
#: 早先判「含 Zephyra 或含 Veyra 就给 72」，
#: 结果对照组答 Veyra 也拿满分，两组 POST 一模一样。
_TARGET_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("哈希初值与递推", ("131", "65537")),
    ("桶号=哈希 mod 8", ("mod 8",)),
    ("分裂是恒等变换", ("(b+8) mod 8", "(b + 8) mod 8")),
    ("压缩条件是「且」", ("且",)),
    ("Z-Tag 随 epoch 变化", ("Z-Tag", "ZTag", "epoch")),
)


class V3Evaluator(ScriptedProvider):
    """离线评估器。**与考生是不同的 provider 实例**。

    这一点很重要：如果考生和评估器是同一个对象，
    污染检测会判它「评委就是考生」，实验直接不合格。
    所以这里必须是两个独立实例，名字也不同。

    判分口径：**只按答案里有没有目标知识的核心规则要点**。
    出现系统名（Zephyra / Veyra）本身不得分——
    对照组答的正是 Veyra 的规则，按「提到系统名就高分」判，
    两组 POST 会拿到同一个分数，链路测不出任何东西。
    """

    def complete(self, messages, *, temperature=0.7, max_tokens=None,
                 json_mode=False):
        joined = "\n".join(m.get("content", "") for m in messages)
        # 参考答案本身就在 joined 里，不能拿它当考生答案的一部分来判。
        answer = joined.rsplit("考生答案：", 1)[-1]

        if "我不知道" in answer or "无法作答" in answer or not answer.strip():
            score, hit_names = 10.0, []
        else:
            hits = [
                name for name, probes in _TARGET_PROBES
                if any(p in answer for p in probes)
            ]
            hit_names = hits
            score = round(
                10.0 + 62.0 * len(hits) / len(_TARGET_PROBES), 2
            )
        return LLMResponse(
            text=json.dumps({
                "score": score,
                "verdict": "pass" if score >= 60 else "fail",
                "feedback": f"离线评估器：命中目标规则要点 {hit_names}",
                "missing_points": [
                    n for n, _ in _TARGET_PROBES if n not in hit_names
                ],
            }, ensure_ascii=False),
            model="scripted-v3-eval",
        )


# ==========================================================================
# 主流程
# ==========================================================================


def run_once(db: Database, run_id: str, trial_no: int) -> dict:
    store = ev3.V3Store(db)
    spec = ev3.RunSpec(
        run_id=run_id,
        trial_no=trial_no,
        topic="Zephyra 分布式哈希存储 vs Veyra 区间树日志索引",
        candidate=ev3.ProviderConfig(
            name="scripted-v3", model="scripted-v3", provider=V3Scripted({})
        ),
        evaluator=ev3.ProviderConfig(
            name="scripted-v3-eval", model="scripted-v3-eval",
            provider=V3Evaluator({}),
        ),
    )
    runner = ev3.DualGroupRunner(store, spec, _call_candidate)
    return runner.run()


def _call_candidate(cfg: ev3.ProviderConfig, system: str, user: str) -> str:
    return cfg.provider.complete(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        json_mode=True,
    ).text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--batch", default="v06-offline")
    ap.add_argument("--db", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # ---- 门禁：片段必须还能命中 ----
    frag_problems = selfcheck_fragments()
    if frag_problems:
        print("片段自检失败，脚本与题面已脱节：", file=sys.stderr)
        for p in frag_problems:
            print(f"  - {p}", file=sys.stderr)
        return 2

    # ---- 门禁：隔离层必须成立 ----
    gi.assert_isolation_intact()
    cm.assert_control_intact()
    ho.assert_heldout_intact()
    ho.assert_heldout_isolated()
    ho.assert_heldout_scorers_intact()

    # ---- 门禁：报告措辞必须合规（用「无归因许可」模式自检）----
    c3.assert_claims_allowed(
        "本次实验的 difference-in-gains 观测值如下。", may_attribute=False
    )

    db_path = args.db or os.path.join(tempfile.gettempdir(), "agent_v06.db")
    if os.path.exists(db_path) and not args.db:
        os.remove(db_path)
    db = Database(db_path)
    db.init_schema()

    trials: list[dict] = []
    obj_trials: list[dict] = []
    run_ids: list[str] = []
    for i in range(1, args.trials + 1):
        run_id = f"{args.batch}-t{i}"
        run_ids.append(run_id)
        res = run_once(db, run_id, i)
        trials.append(res["trial"])
        obj_trials.append(res["objective_trial"])
        print(
            f"[trial {i}] learning {res['trial']['learning_pre']}→"
            f"{res['trial']['learning_post']}  "
            f"control {res['trial']['control_pre']}→"
            f"{res['trial']['control_post']}  "
            f"diff={res['trial']['difference_in_gains']}  "
            f"obj_diff={res['objective_trial']['objective_difference_in_gains']}  "
            f"污染={len(res['contamination']['contaminations'])}"
        )
        if not res["contamination"]["clean"]:
            print("  污染明细：", file=sys.stderr)
            for c in res["contamination"]["contaminations"]:
                print(f"    [{c['kind']}] {c['detail']}", file=sys.stderr)
        if not res["order"]["clean"]:
            print("  顺序问题：", file=sys.stderr)
            for p in res["order"]["problems"]:
                print(f"    - {p}", file=sys.stderr)

    # ---- 门禁：两组必须真的不同 ----
    # 两组分数一样时链路是空的：difference-in-gains 恒等于 0，
    # 报告照样能出得来，只是里面没有任何东西。
    # 早先就踩过一次——ScriptedProvider 把 POST 误判成 PRE，
    # 两组都答不出、分数一模一样，而控制台只打印了一行看起来正常的输出。
    _assert_groups_differ(trials, obj_trials)

    report = r3.build_report(
        batch_id=args.batch, db=db, trials=trials,
        objective_trials=obj_trials, run_ids=run_ids,
    )
    text = r3.format_text(report)
    print()
    print(text)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"{args.batch}.txt"
    out.write_text(text, encoding="utf-8")
    (out.with_suffix(".json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已写入 {out}")
    db.close()
    return 0


def _assert_groups_differ(
    trials: list[dict], obj_trials: list[dict]
) -> None:
    """两组必须产生不同结果，否则实验链路是空的。"""
    same_exam = [t for t in trials if t["difference_in_gains"] == 0.0]
    same_obj = [t for t in obj_trials if t["objective_difference_in_gains"] == 0.0]
    if same_exam or same_obj:
        lines = ["两组结果完全相同，链路没起作用："]
        if same_exam:
            lines.append(
                f"  {len(same_exam)}/{len(trials)} 次 trial 的考试口径 "
                f"difference-in-gains = 0"
            )
        if same_obj:
            lines.append(
                f"  {len(same_obj)}/{len(obj_trials)} 次 trial 的客观口径 "
                f"difference-in-gains = 0"
            )
        lines.append(
            "  两组一模一样通常意味着：阶段标记没传进prompt、"
            "题面片段没命中，或 provider 没按组分叉。"
        )
        raise RuntimeError("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
