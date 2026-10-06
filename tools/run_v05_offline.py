"""tools/run_v05_offline.py —— 用 ScriptedProvider 跑一次完整 V0.5 链路。

**不联网、不烧 token、不调真实模型。**
目的是验证整条实验链路（流程 / 落库 / 隔离 / 统计 / 报告）能跑通，
而不是测分数高低——分数由脚本钦定，跑出来是多少都不构成能力结论。

用法:
    python tools/run_v05_offline.py            # 跑 1 次
    python tools/run_v05_offline.py --trials 5 # 跑 5 次看统计
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import experiment_v2 as ev2  # noqa: E402
from brain import report_v2 as r2  # noqa: E402
from brain import stats_v2 as st  # noqa: E402
from brain.llm import LLMResponse, ScriptedProvider  # noqa: E402
from memory.db import Database, v2_experiment_key  # noqa: E402
from practice.objective import OBJECTIVE_TASKS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "reports"

# 8 个实践任务的正确产出（离线脚本钦定）
_CORRECT: dict[str, str] = {
    "obj_hash_01": json.dumps(
        {"key": "abc", "steps": [7, 1014, 1858, 46886],
         "lumen_hash": 46886, "bucket": 6}),
    "obj_hash_02": json.dumps(
        {"code": "def lumen_hash(key):\n    h = 7\n"
                 "    for c in key:\n        h = (h * 131 + ord(c)) % 65537\n"
                 "    return h"}),
    "obj_bucket_01": json.dumps(
        {"assignments": [{"key": n, "from_bucket": 0, "to_bucket": 0}
                         for n in ("k0", "k8", "k16", "k24", "k32")],
         "migrated_count": 0, "still_over_limit": True}),
    "obj_bucket_02": json.dumps(
        {"sql": "SELECT bucket FROM zephyra_buckets"
                " WHERE live_count > 4 ORDER BY bucket"}),
    "obj_compact_01": json.dumps({"compress": [0, 3], "epoch_after": 3}),
    "obj_compact_02": json.dumps(
        {"code": "def should_compact(tombstones, live):\n"
                 "    return tombstones >= 2 and live <= 1"}),
    "obj_consist_01": json.dumps(
        {"readable": [], "need_retag": ["x1", "x2", "x3", "x4"],
         "new_tags": {"x1": 2502, "x2": 4002, "x3": 6002, "x4": 8202}}),
    "obj_consist_02": json.dumps(
        {"sql": "SELECT key FROM zephyra_keys WHERE ztag != (lumen_hash +"
                " (SELECT value FROM zephyra_meta WHERE name='epoch')) % 65537"
                " ORDER BY key"}),
}

_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("obj_hash_01", "LumenHash 值，并给出它所属的桶号"),
    ("obj_hash_02", "def lumen_hash(key)"),
    ("obj_bucket_01", "0、8、16、24、32"),
    # 片段必须**每个任务独有**。obj_consist_02 的题面也列了
    # zephyra_buckets(...) 与 zephyra_keys(...)，若用表名做片段会被
    # 靠前的 obj_bucket_02 抢走，静默返回错误答案。
    ("obj_bucket_02", "每个桶最多容纳 4 个活跃键"),
    ("obj_compact_01", "tombstones=9"),
    ("obj_compact_02", "def should_compact(tombstones, live)"),
    ("obj_consist_01", "4000、6000、8200"),
    ("obj_consist_02", "zephyra_meta(name, value)"),
)

_SCRIPT: dict[str, object] = {
    "ANSWER": {
        "default": "我不知道这个虚构系统的规则，无法作答。",
        "参考资料": (
            "根据我自己的笔记：LumenHash 初值 7，(h*131+ord(c)) mod 65537；"
            "桶号 = hash mod 8；分裂是恒等变换 (b+8) mod 8 = b；"
            "压缩需墓碑>=2 且活跃<=1；Z-Tag=(hash+epoch) mod 65537，"
            "epoch 变了全库失配。"
        ),
    },
    "EVALUATE": '{"score":70,"verdict":"pass","feedback":"覆盖了部分要点",'
                '"missing_points":[]}',
    "STUDY": json.dumps({"items": [
        {"title": "LumenHash", "content": "h=7 起逐字符 (h*131+ord(c)) mod 65537",
         "keywords": ["hash"]},
        {"title": "Z-8 分桶", "content": "bucket=hash mod 8，分裂是恒等变换",
         "keywords": ["bucket"]},
        {"title": "压缩", "content": "墓碑>=2 且活跃<=1 时触发", "keywords": ["compact"]},
        {"title": "Epoch", "content": "ZTag=(hash+epoch) mod 65537", "keywords": ["epoch"]},
    ]}, ensure_ascii=False),
    "LEARNING_NOTES": json.dumps({"notes": [
        {"kp": "kp_lumen_hash", "concept": "LumenHash",
         "principle": "h=7 起逐字符 (h*131+ord(c)) mod 65537",
         "example": 'LumenHash("ab")=1858', "common_mistake": "初值写成0",
         "application_rule": "算哈希再对8取模"},
        {"kp": "kp_lumen_bucket", "concept": "Z-8 分桶", "principle": "分裂是恒等变换",
         "example": "(b+8) mod 8 = b", "common_mistake": "以为分裂有效",
         "application_rule": "容量按8桶规划"},
        {"kp": "kp_lumen_compact", "concept": "压缩", "principle": "墓碑>=2 且活跃<=1",
         "example": "3墓碑2活跃不触发", "common_mistake": "把且看成或",
         "application_rule": "两条件同时满足才压"},
        {"kp": "kp_lumen_consist", "concept": "Epoch 标签",
         "principle": "ZTag=(hash+epoch) mod 65537", "example": "epoch+1 全库失配",
         "common_mistake": "以为标签自动更新", "application_rule": "压缩后必须批量重算"},
    ]}, ensure_ascii=False),
    "QUALITY": '{"accuracy":85,"completeness":80,"transferable":75,'
               '"no_hallucination":90,"feedback":"笔记基本准确","problems":[]}',
    "PRACTICE_JUDGE": '{"usable":true,"score":75,"verdict":"pass",'
                     '"feedback":"产出可执行"}',
}


class V2Scripted(ScriptedProvider):
    def complete(self, messages, *, temperature=0.7, max_tokens=None,
                 json_mode=False):
        joined = "\n".join(m.get("content", "") for m in messages)
        for task_id, frag in _FRAGMENTS:
            if frag in joined:
                return LLMResponse(text=_CORRECT[task_id], model="scripted")
        return super().complete(messages, temperature=temperature,
                                max_tokens=max_tokens, json_mode=json_mode)


def selfcheck_fragments() -> list[str]:
    """跑之前先确认 8 个题面片段都还能命中，且**互不重叠**。

    片段匹配对不上时**不会报错**，只会静默走到fallback，
    于是 8 个任务全部拿 0 分——看上去像「模型全错」，
    真相是脚本没匹配上。开发时踩过一次，所以这里做成硬门禁。
    """
    problems: list[str] = []
    for task_id, frag in _FRAGMENTS:
        task = next((t for t in OBJECTIVE_TASKS if t.task_id == task_id), None)
        if task is None:
            problems.append(f"{task_id}: 任务不存在")
        elif frag not in task.prompt:
            problems.append(f"{task_id}: 题面已改，片段 {frag!r} 不再命中")

    # 重叠检查：片段 A 若同时出现在任务 B 的题面里，
    # 靠前的 A 会抢走 B 的请求，返回错误答案而不报错。
    for i, (tid_a, frag_a) in enumerate(_FRAGMENTS):
        for tid_b, frag_b in _FRAGMENTS[i + 1:]:
            task_b = next((t for t in OBJECTIVE_TASKS if t.task_id == tid_b), None)
            if task_b is not None and frag_a in task_b.prompt:
                problems.append(
                    f"片段冲突：{tid_a} 的片段 {frag_a!r} 也出现在 {tid_b} 的题面里，"
                    f"会静默错配"
                )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--batch", default="v05-offline")
    ap.add_argument("--db", default="")
    args = ap.parse_args()

    problems = selfcheck_fragments()
    if problems:
        print("片段自检失败，拒绝运行：", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2

    tmp = tempfile.mkdtemp(prefix="wb_v05_")
    db = Database(args.db or os.path.join(tmp, "v05.db"))
    db.init_schema()

    prov = V2Scripted(dict(_SCRIPT))
    runner = ev2.V2ExperimentRunner(db, prov, prov)

    results = []
    for i in range(1, args.trials + 1):
        res = runner.run(batch_id=args.batch, trial_no=i)
        results.append(res)
        print(f"[trial {i}] experiment_id={res.experiment_id} "
              f"pre={res.scores['pre_blind']} ob={res.scores['open_book']} "
              f"mb={res.scores['memory_blind']} tr={res.scores['transfer_blind']} "
              f"obj={res.scores['objective_practice_avg']}", flush=True)

    store = ev2.V2Store(db)
    last = results[-1]

    # ---------- 单次报告 ----------
    trial_rep = r2.build_trial_report_from_store(store, last.experiment_id)
    trial_txt = r2.format_trial_text(trial_rep)
    print()
    print(trial_txt)

    # ---------- 落库统计 ----------
    print("=" * 72)
    print("落库统计")
    print("=" * 72)
    for t in ("v2_experiments", "v2_stage_items", "practice_runs",
              "v2_batches", "learning_notes", "note_quality",
              "practice_evaluations", "experiments"):
        print(f"  {t:<24}{db.count(t):>4} 行")
    print("  stage 分布：")
    for row in db.query_all(
        "SELECT stage, COUNT(*) n FROM v2_stage_items WHERE experiment_id=?"
        " GROUP BY stage ORDER BY stage", (last.experiment_id,)
    ):
        print(f"    {row['stage']:<16}{row['n']:>3}")
    print("  污染检查（V2 数据不得出现在正 id 查询里，应全为 0）：")
    for t in ("learning_notes", "note_quality", "practice_evaluations"):
        n = db.scalar(f"SELECT COUNT(*) FROM {t} WHERE experiment_id > 0")
        print(f"    {t:<24}{n:>3}")

    # ---------- 批次报告 ----------
    batch_txt = ""
    if args.trials > 1:
        batch_rep = r2.build_batch_report_from_store(store, args.batch)
        batch_txt = r2.format_batch_text(batch_rep)
        print()
        print(batch_txt)

    # ---------- 落盘 ----------
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "v05-offline-trial.txt").write_text(trial_txt, encoding="utf-8")
    summary = {
        "trials": len(results),
        "experiment_ids": [r.experiment_id for r in results],
        "scores": {str(r.experiment_id): r.scores for r in results},
        "gains": {str(r.experiment_id): r.gains for r in results},
        "objective_detail": last.objective_detail,
        "evaluator_practice": last.evaluator_practice,
        "consistency_check": trial_rep["consistency_check"],
        "caveats": trial_rep["caveats"],
    }
    if args.trials > 1:
        summary["batch_stats"] = st.aggregate_trials(
            st.load_trials(store, args.batch))
    (OUT_DIR / "v05-offline-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if batch_txt:
        (OUT_DIR / "v05-offline-batch.txt").write_text(batch_txt, encoding="utf-8")

    print()
    print(f"报告已写入 {OUT_DIR}/v05-offline-trial.txt")
    if batch_txt:
        print(f"批次报告 {OUT_DIR}/v05-offline-batch.txt")
    print(f"结构化摘要 {OUT_DIR}/v05-offline-summary.json")
    print(f"数据库 {db.path if hasattr(db, 'path') else '(临时)'}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
