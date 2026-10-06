#!/usr/bin/env python3
"""Agent-01 V0.4 CLI。

V0.4 的实验命令固定为这条链路，重复 N 次：

    PRE → 阅读材料 → 自己整理学习笔记 → 实践
        → MEMORY OPEN_BOOK → MEMORY BLIND → TRANSFER BLIND

用法示例：
    python main.py experiment --provider openai --repeat 10
    python main.py experiment --provider scripted --repeat 5
    python main.py batch-report
    python main.py learn --topic "SQLite 索引原理"
    python main.py cycle --topic "SQLite 索引原理"
    python main.py bank
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 允许 `python main.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain.agent import Agent  # noqa: E402
from brain.llm import LLMProvider, OpenAIProvider, ScriptedProvider  # noqa: E402
from brain.replicate import DEFAULT_REPEATS  # noqa: E402
from memory.db import Database  # noqa: E402
from memory.gate import AccessPolicy, ExamAccessViolation  # noqa: E402


def _load_dotenv() -> None:
    """极简 .env 加载，避免额外依赖 python-dotenv。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def demo_script() -> dict:
    """--provider scripted 使用的内置演示脚本，让离线也能跑出完整闭环。

    V0.4 要求六个阶段返回**不同**的模拟结果：
        PRE / STUDY / NOTES / PRACTICE / OPEN_BOOK / BLIND / TRANSFER / EVALUATION
    全阶段返回同一个值的话，离线实验永远测不出东西——那是自欺欺人。

    下面这套脚本刻意模拟了三种不同的作答水平：
      - 裸答/盲答（PRE、MEMORY_BLIND）-> 只泛泛而谈，命中要点少
      - 开卷（MEMORY_OPEN）-> 覆盖全部关键要点
      - 迁移答（TRANSFER_BLIND）-> **答错**新场景

    TRANSFER 刻意给低分，是有意的设计验证：
    它模拟「背住了同一批题的话术，但没形成概念」这种最常见的情况。
    如果离线脚本让 TRANSFER 也拿高分，那说明管线分不清
    「记住了」和「学会了」——那才是真问题。

    ⚠️ **再次强调：这些分数是脚本写死的，不是测出来的。**
    离线实验只验证「管线能正确区分四个阶段」，
    真实学习效果必须 --provider openai 跑。
    """
    # 裸答 / 盲答（PRE 与 MEMORY_BLIND 共用）：泛泛而谈，命中要点少
    blind_answer = (
        "SQLite 大概是用树结构存索引，比全表扫快一些，"
        "具体细节记不太清了。"
    )
    # 迁移答（TRANSFER_BLIND）：把新场景当成老题来答，答非所问
    # 模拟「记住了话术但没形成概念」——这正是 TRANSFER 卷要识别的情况
    transfer_answer = (
        "这个问题应该从平衡多路树的角度来回答：B-Tree 的节点分叉数固定，"
        "所以叶子节点都在同一层。这样查询路径就稳定了，"
        "查找次数不随数据量增长。覆盖索引也能减少回表。"
    )
    # 开卷（带笔记）：覆盖全部关键要点
    open_book_answer = (
        "SQLite 用 B-Tree 组织索引：所有叶子节点在同一层，"
        "内部节点分叉数固定，因此查找路径长度不随数据量增长，"
        "查找次数被树高（对数级）限制住。\n"
        "索引项存「索引列值 + rowid」，与表B-Tree 是两棵独立的树，"
        "拿到 rowid 还要回表；这与 InnoDB 把整行存在聚簇索引叶子节点里不同。\n"
        "当查询所需的全部列都在索引中时即为覆盖索引，无需回表；"
        "缺一列就会退化成回表，产生一次随机 IO。\n"
        "代价是写放大：每次增删改都要同步维护所有索引，"
        "所以批量写入为主的表应减少索引数量、推迟建索引。"
    )
    # 实践产出：必须含可执行内容，否则实践评审会正确判为不可用
    practice_answer = (
        "验证步骤：\n"
        "1. 建表并导入十万行数据：sqlite3 test.db \"CREATE TABLE t(id INTEGER PRIMARY KEY, "
        "name TEXT, price REAL);\"\n"
        "2. 建索引后用 EXPLAIN QUERY PLAN 观察是否走索引："
        "sqlite3 test.db \"EXPLAIN QUERY PLAN SELECT name, price FROM t WHERE name='x';\"\n"
        "3. 判断标准：输出里出现 SEARCH t USING INDEX 才算命中；"
        "若出现 SCAN t 则说明索引没被用上。\n"
        "4. 对比回表代价：先建 (name) 单列索引，再建 (name, price) 复合索引，"
        "比较两者的 EXPLAIN 输出差异。"
    )
    return {
        "LEARN": json.dumps(
            {
                "items": [
                    {
                        "title": "B-Tree 索引结构",
                        "content": "SQLite 用 B-Tree 组织索引，叶子节点同深度，查询路径稳定。",
                        "keywords": ["B-Tree", "索引"],
                    },
                    {
                        "title": "覆盖索引",
                        "content": "查询所需列全在索引中时无需回表，显著减少随机 IO。",
                        "keywords": ["覆盖索引", "回表"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        "STUDY": json.dumps(
            {
                "items": [
                    {
                        "title": "B-Tree 平衡多路树",
                        "content": "所有叶子节点同层，查找次数不随数据量增长。",
                        "keywords": ["B-Tree", "平衡"],
                    },
                    {
                        "title": "覆盖索引省回表",
                        "content": "查询列全在索引中时无需回表，减少随机 IO。",
                        "keywords": ["覆盖索引", "回表"],
                    },
                    {
                        "title": "索引与行分属两棵树",
                        "content": "索引存 rowid，需回表；与 InnoDB 聚簇索引不同。",
                        "keywords": ["rowid", "回表"],
                    },
                    {
                        "title": "写放大代价",
                        "content": "每次增删改都要维护所有索引，批量写入应少建索引。",
                        "keywords": ["写放大", "索引策略"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        "LEARNING_NOTES": json.dumps(
            {
                "notes": [
                    {
                        "kp": "kp_btree",
                        "concept": "B-Tree 是一种平衡多路查找树，SQLite 用它组织索引。",
                        "principle": "所有叶子节点位于同一层，查找路径长度固定；节点内按键有序可二分，"
                                     "因此查找次数被树高限制在对数级，不随数据量线性增长。",
                        "example": "表有一万行还是一亿行，查一个键的I/O 次数几乎不变。",
                        "common_mistake": "误以为索引查找是 O(n) 扫描，或以为树深会随数据线性加深。",
                        "application_rule": "需要按列快速定位时优先用 B-Tree 索引，不用手动优化树结构。",
                    },
                    {
                        "kp": "kp_covering",
                        "concept": "查询所需的全部列都在索引中时，称为覆盖索引。",
                        "principle": "索引叶子只存「索引列+rowid」，缺列就必须拿 rowid 回表，"
                                     "多一次随机 IO；列全命中则可省掉这次回表。",
                        "example": "SELECT name, price WHERE sku=? 有索引 (sku,name,price) 即覆盖。",
                        "common_mistake": "只想着用索引过滤，忘了确认 SELECT 列是否也在索引里。",
                        "application_rule": "把高频查询的 SELECT 列尽量和WHERE 列放进同一个复合索引。",
                    },
                    {
                        "kp": "kp_clustered",
                        "concept": "SQLite 的索引与表数据是两棵独立的 B-Tree。",
                        "principle": "索引存索引列加rowid，行数据在表 B-Tree 里；"
                                     "InnoDB 则把整行存在聚簇索引叶子节点，所以行是否与索引同树是本质差别。",
                        "example": "SQLite 查非索引列要rowid 回表；InnoDB 二级索引回聚簇索引取整行。",
                        "common_mistake": "把 SQLite 当成 InnoDB，认为二级索引叶子就有整行数据。",
                        "application_rule": "设计索引时按 SQLite 的两棵树模型估算回表代价。",
                    },
                    {
                        "kp": "kp_write",
                        "concept": "索引会带来写放大：每次增删改都要同步维护所有相关索引。",
                        "principle": "索引越多，维护成本与额外存储线性上升，且涉及随机页写入，"
                                     "所以读多写少适合多索引，批量写入适合少索引。",
                        "example": "日志表每天追加两百万行，只建一个时间戳单列索引。",
                        "common_mistake": "给每列都建索引，认为查询快就一定好。",
                        "application_rule": "批量写入场景先灌数据最后建索引，或用分区替代索引过滤。",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        "PRACTICE": json.dumps(
            {
                "outcome": "success",
                "result": practice_answer,
                "kp": "kp_btree",
            },
            ensure_ascii=False,
        ),
        "PRACTICE_JUDGE": json.dumps(
            {
                "usable": True,
                "score": 78,
                "verdict": "pass",
                "feedback": "有具体命令和可观测的判据（EXPLAIN 输出），可执行。",
            },
            ensure_ascii=False,
        ),
        "NOTE_QUALITY": json.dumps(
            {
                "accuracy": 82,
                "completeness": 88,
                "transferable": 74,
                "no_hallucination": 92,
                "total": 81.6,
                "verdict": "pass",
                "feedback": "五要素齐全、机制描述与材料一致；"
                            "但部分笔记偏结论，解释「为什么」的深度不足。",
                "problems": ["个别笔记的原理部分偏向复述结论，未讲透前提条件"],
            },
            ensure_ascii=False,
        ),
        "ANSWER": {
            # 分叉顺序即优先级。TRANSFER 标记必须排在最前，
            # 因为 TRANSFER 也走 ANSWER 接口，用的是 transfer_answer。
            # 标记串取 transfer_answer 里独有的措辞，
            # 保证不会误命中 memory 卷或参考答案是它的题目。
            "平衡多路树的角度来回答": transfer_answer,
            # 带笔记 -> 开卷作答；否则 -> 盲答
            "以下是你自己此前整理的学习笔记": open_book_answer,
            "default": blind_answer,
        },
        "EVALUATE": {
            # 分叉标记的坑：只能取**考生答案里的**特征串，不能取学习阶段的
            # 知识点标题，也不能取知识点内部的措辞——因为评估 prompt 里有
            # 「参考答案」，标记若同时出现在参考答案里，盲答也会被误判成开卷。
            # （曾用「内部节点分叉数固定」做标记，它正好在 post_btree_01 的
            #   参考答案里，导致盲考首题拿了 92 分。test_evaluate_fork_marker
            #   会锁住这个坑。）
            # TRANSFER 分支必须在最前：它的特征串「从平衡多路树的角度」
            # 也要用「用 B-Tree 组织索引」来兜底匹配优先级顺序。
            "平衡多路树的角度": json.dumps(
                {
                    "score": 38,
                    "verdict": "fail",
                    "feedback": "答非所问：把新场景当成老题来答，"
                                "没有针对问题里的具体约束做分析。",
                    "missing_points": ["未针对新场景", "未说明前提约束", "结论与问题无关"],
                },
                ensure_ascii=False,
            ),
            # 盲答特征串：用「大概」这种只可能出现在含糊作答里的措辞，
            # 它在题库参考答案中不存在（测试会验证）。
            "大概是用树结构存索引": json.dumps(
                {
                    "score": 45,
                    "verdict": "fail",
                    "feedback": "只泛泛提到树结构和更快，缺少关键机制。",
                    "missing_points": ["覆盖索引", "写放大", "回表"],
                },
                ensure_ascii=False,
            ),
            "default": json.dumps(
                {
                    "score": 92,
                    "verdict": "pass",
                    "feedback": "四个知识点全部覆盖，机制讲到了底层。",
                    "missing_points": [],
                },
                ensure_ascii=False,
            ),
        },
        "REFLECT": json.dumps(
            {"lesson": "先查笔记再作答，命中率明显更高", "action": "考试时优先用自己的笔记"},
            ensure_ascii=False,
        ),
        "GEN_SAMPLES": json.dumps(
            {
                "samples": [
                    {
                        "question": "什么是覆盖索引？",
                        "answer": "查询所需列全在索引中，无需回表。",
                        "difficulty": "easy",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        "MAKE_QUESTION": json.dumps(
            {
                "items": [
                    {
                        "question": "简述 SQLite 索引的工作原理。",
                        "reference_answer": "B-Tree 结构；聚簇索引；覆盖索引减少回表。",
                        "difficulty": "medium",
                    }
                ]
            },
            ensure_ascii=False,
        ),
    }


def build_provider(name: str = "openai", model: str = "") -> LLMProvider:
    """构造模型提供方。

    scripted  = 离线演示脚本，不联网，分数是写死的（只能验证管线）
    openai    = 真实模型，key 从环境变量 OPENAI_API_KEY 读
    """
    _load_dotenv()
    if name == "scripted":
        return ScriptedProvider(demo_script())  # 离线演示，不联网
    return OpenAIProvider(model=model or None)


def _print(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-01", description="Agent-01 V0.4 AI 学习实验室（能力迁移实验）"
    )
    parser.add_argument("--db", default="agent.db", help="SQLite 路径（默认 agent.db）")
    parser.add_argument(
        "--provider", default="openai", choices=["openai", "scripted"],
        help="模型提供方：openai=真实模型（需 OPENAI_API_KEY），scripted=离线演示",
    )
    parser.add_argument(
        "--model", default="", help="模型名（覆盖 OPENAI_MODEL 环境变量）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="初始化数据库表结构")
    sub.add_parser("stats", help="查看各表数据量")

    p_learn = sub.add_parser("learn", help="学习模式：生成知识点并落库")
    p_learn.add_argument("--topic", required=True)
    p_learn.add_argument("--max-items", type=int, default=3)

    p_prac = sub.add_parser("practice", help="实践模式：带知识库作答并评估")
    p_prac.add_argument("--topic", required=True)
    p_prac.add_argument("--count", type=int, default=2)

    p_exam = sub.add_parser("exam", help="严格考试模式（禁读知识与训练样本）")
    p_exam.add_argument("--topic", required=True)
    p_exam.add_argument("--count", type=int, default=3)
    p_exam.add_argument(
        "--stage", default="exam", choices=["pre_test", "post_test", "exam"]
    )

    p_cycle = sub.add_parser("cycle", help="一键跑完整闭环")
    p_cycle.add_argument("--topic", required=True)
    p_cycle.add_argument("--count", type=int, default=3)

    p_ask = sub.add_parser("ask", help="单题问答（可加 --blind 走严格考试模式）")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--topic", default="")
    p_ask.add_argument("--blind", action="store_true", help="禁止读取知识库")

    p_exp = sub.add_parser(
        "experiment",
        help="跑完整学习效果实验（PRE → 学 → 笔记 → 实践 → MEMORY 开卷/盲考 → TRANSFER），"
             "默认重复 5 次",
    )
    p_exp.add_argument("--topic", default="SQLite 索引")
    p_exp.add_argument("--material-file", default="", help="学习材料文件（默认用内置材料）")
    p_exp.add_argument(
        "--repeat", type=int, default=DEFAULT_REPEATS,
        help=f"重复次数（默认 {DEFAULT_REPEATS}，建议 5-10；1 表示只跑一次不聚合）",
    )
    p_exp.add_argument(
        "--format", default="text", choices=["text", "json"], help="报告输出格式"
    )
    p_exp.add_argument(
        "--batch-id", default="", help="自定义批次 id（默认按主题+时间戳生成）"
    )

    p_batch = sub.add_parser("batch-report", help="查看最近一次重复实验批次的跨次统计")
    p_batch.add_argument("--id", default="", help="批次 id（默认取最近一次）")
    p_batch.add_argument("--format", default="text", choices=["text", "json"])

    p_trials = sub.add_parser("trials", help="列出某批次下每次实验的分数")
    p_trials.add_argument("--id", default="", help="批次 id（默认取最近一次）")

    p_report = sub.add_parser("report", help="查看某次实验的单次报告")
    p_report.add_argument("--id", type=int, default=0, help="实验 id（默认取最近一次）")
    p_report.add_argument("--format", default="text", choices=["text", "json"])

    p_bank = sub.add_parser("bank", help="查看固定题库信息与指纹")

    args = parser.parse_args(argv)

    db = Database(args.db)
    db.init_schema()

    if args.command == "init":
        _print({"db": args.db, "tables": db.table_names()})
        db.close()
        return 0

    if args.command == "stats":
        # stats 只读数据库，不需要模型，也就不需要 API key
        tables = db.table_names()
        _print({t: db.count(t) for t in tables})
        db.close()
        return 0

    try:
        agent = Agent(db, build_provider(args.provider, args.model))
    except RuntimeError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        print("提示：可用 --provider scripted 进行离线冒烟测试。", file=sys.stderr)
        db.close()
        return 2

    if args.command == "learn":
        _print(agent.learn(args.topic, max_items=args.max_items).as_dict())
    elif args.command == "practice":
        _print(agent.practice(args.topic, count=args.count).as_dict())
    elif args.command == "exam":
        _print(agent.exam(args.topic, count=args.count, stage=args.stage).as_dict())
    elif args.command == "cycle":
        _print(agent.full_cycle(args.topic, count=args.count))
    elif args.command == "ask":
        try:
            if args.blind:
                policy = AccessPolicy(stage="blind", blind=True)
                from knowledge.store import KnowledgeStore

                KnowledgeStore(db, policy).list_by_topic(args.topic or "*", limit=1)
                context = ""
            else:
                context = agent._build_context(args.topic) if args.topic else ""
            answer = agent.builder.answer(args.question, context=context)
            _print({"question": args.question, "blind": args.blind, "answer": answer})
        except ExamAccessViolation as exc:
            print(f"[闸门拦截] {exc}", file=sys.stderr)
            db.close()
            return 3

    elif args.command == "bank":
        from brain import question_bank

        _print(question_bank.bank_meta())

    elif args.command == "experiment":
        from brain.experiment import ExperimentRunner
        from brain.materials import DEFAULT_MATERIAL
        from brain.replicate import ReplicateRunner
        from brain.report import (
            batch_verdict, build_report, format_batch_text, format_text,
        )

        if args.material_file:
            with open(args.material_file, "r", encoding="utf-8") as fh:
                content = fh.read()
        else:
            content = DEFAULT_MATERIAL

        # 每次 trial 都新建 provider：真实模型侧会把上一轮作答上下文
        # 带进下一轮，污染 BLIND_GAIN（见 brain/replicate.py 文档）
        def factory():
            return build_provider(args.provider, args.model)

        model_label = (
            args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            if args.provider != "scripted" else "scripted"
        )

        # --repeat 1：只跑一次，直接出单次报告（不聚合，避免 std=0 的假确定性）
        if args.repeat <= 1:
            runner = ExperimentRunner(db, factory())
            result = runner.run(args.topic, content)
            report = build_report(runner, result.experiment_id)
            if args.format == "json":
                _print(report)
            else:
                print(format_text(report))
            db.close()
            return 0

        def progress(no: int, res: object) -> None:
            print(
                f"  [trial {no}/{args.repeat}] PRE {res.pre_score:.1f} | "
                f"OPEN {res.memory_open_score:.1f} | MEM_BLIND {res.memory_blind_score:.1f} | "
                f"TRANSFER {res.transfer_blind_score:.1f} | "
                f"练习 {res.practice_usable_rate * 100:.0f}% | "
                f"笔记 {res.note_quality_avg:.1f}",
                file=sys.stderr,
            )

        rep = ReplicateRunner(db, factory, evaluator_provider_factory=None)
        print(
            f"跑 {args.repeat} 次实验（provider={args.provider}, model={model_label}）…",
            file=sys.stderr,
        )
        summary = rep.run_repeated(
            args.topic, content,
            repeats=args.repeat,
            model_label=model_label,
            batch_id=args.batch_id,
            progress=progress,
        )
        summary["verdict"] = batch_verdict(summary)

        if args.format == "json":
            _print(summary)
        else:
            print(format_batch_text(summary))
            last_eid = summary["trials"][-1]["experiment_id"]
            print()
            print("=" * 74)
            print(f"附：最后一次（trial #{args.repeat}）的单次明细")
            print("=" * 74)
            runner2 = ExperimentRunner(db, factory())
            print(format_text(build_report(runner2, int(last_eid))))
        db.close()
        return 0

    elif args.command == "batch-report":
        from brain.report import build_batch_report, format_batch_text

        try:
            summary = build_batch_report(db, args.id)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            db.close()
            return 4
        if args.format == "json":
            _print(summary)
        else:
            print(format_batch_text(summary))
        db.close()
        return 0

    elif args.command == "trials":
        from brain.replicate import BatchStore

        store = BatchStore(db)
        row = store.get(args.id) if args.id else store.latest()
        if row is None:
            print("还没有任何批次，请先运行 experiment --repeat。", file=sys.stderr)
            db.close()
            return 4
        _print({
            "batch_id": row["batch_id"],
            "status": row["status"],
            "model": row["model"],
            "trials": [
                {
                    "trial_no": t["trial_no"],
                    "experiment_id": t["id"],
                    "pre": t["pre_score"],
                    "open_book": t["memory_open_score"],
                    "memory_blind": t["memory_blind_score"],
                    "transfer_blind": t["transfer_blind_score"],
                    "practice_use": t["practice_usable_rate"],
                    "note_quality": t["note_quality_avg"],
                }
                for t in store.trials(row["batch_id"])
            ],
        })
        db.close()
        return 0

    elif args.command == "report":
        from brain.experiment import ExperimentRunner
        from brain.report import build_report, format_text

        runner = ExperimentRunner(db, build_provider(args.provider, args.model))
        if args.id:
            eid = args.id
        else:
            latest = runner.store.latest()
            if latest is None:
                print("还没有任何实验，请先运行 experiment。", file=sys.stderr)
                db.close()
                return 4
            eid = int(latest["id"])
        report = build_report(runner, eid)
        if args.format == "json":
            _print(report)
        else:
            print(format_text(report))
        db.close()
        return 0

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
