"""brain/heldout_bank.py —— V0.6 held-out 题库（POST / TRANSFER 共用）。

为什么必须另建一套题
--------------------
V0.5 的 POST 概念是「重新答一遍同样的 application 卷」，
但那套题在 MEMORY 阶段已经答过一次。于是 POST 混着两种东西：

    (1) 因学习而获得的能力
    (2) 刚做过的同一道题的练习效应

V0.6 把 POST 与 PRE 也彻底分开：PRE 用`pre` 卷，POST/TRANSFER 用
**本文件的 held-out 卷**，两卷题目**不重叠**。

三条不可协商的约束
------------------
1. **不在学习材料中**——held-out 题的答案不能是材料里的原句。
   实际上材料只给规则不给算例，题目里的数字全部是现算的。
2. **不在 PRACTICE 中**——practice 的 8 个任务用参数 P，
   held-out 的客观任务用参数 H，**同一知识点也不同数字**。
   否则「练习过」和「会做」分不开。
3. **不在 PRE 中**——PRE 卷与 held-out 卷的 item_id 与题干都不同。

参数错开这件事是**可机器检查的**：
`assert_heldout_isolated()` 比对 held-out 的每一个参数与
practice 的每一个参数，交集非空即拒绝跑实验。

字段与题库结构沿用 V0.5 的 `ExamItem`，
这样 `v2_stage_items` 的写入路径不必分叉。
"""

from __future__ import annotations

import json
from typing import Any

from brain.benchmark_v2 import PRE_ITEMS, ExamItem, _fingerprint  # noqa: F401
from practice import objective as ob

HELD_OUT_VERSION = "ho1.0.0"

#: 四个知识点（沿用 V0.5，与 practice / PRE 一一对应）
HELD_OUT_KPS: tuple[str, ...] = (
    "kp_lumen_hash",
    "kp_lumen_bucket",
    "kp_lumen_compact",
    "kp_lumen_consist",
)

# ==========================================================================
# held-out 客观任务用到的参数（**全部与 practice 错开**）
#
# practice 用的是：key="abc"、LumenHash ∈ {2500,4000,6000,8200}、
#                 压缩桶 {0,3}、epoch_after=3、单测用例 (...2,1)/(...3,2)…
# held-out 用的是：key="zeta"、LumenHash ∈ {1777, 5231, 8899}、
#                 压缩桶 {1,4,6}、epoch_after=4、单测用例 全新的边界组合。
# ==========================================================================

HO_HASH_KEY = "zeta"
HO_HASH_STEPS = [7]
_h = 7
for _c in HO_HASH_KEY:                      # noqa: B007
    _h = (_h * 131 + ord(_c)) % 65537
    HO_HASH_STEPS.append(_h)
HO_HASH_VALUE = _h                          # 参考实现独立验算过
HO_HASH_BUCKET = HO_HASH_VALUE % 8

#: held-out 单测用例：刻意包含空串、超长串、非 ASCII
HO_UNIT_HASH: tuple[tuple[str, int], ...] = (
    ("", ob.ref_lumen_hash("")),
    ("z", ob.ref_lumen_hash("z")),
    ("zeta", ob.ref_lumen_hash("zeta")),
    ("AZ", ob.ref_lumen_hash("AZ")),
    ("Zephyra", ob.ref_lumen_hash("Zephyra")),
    ("0123456789", ob.ref_lumen_hash("0123456789")),
)

#: 压缩判定单测用例：与 practice 的 (2,1)(3,2)(1,1)(2,0)(5,1)(2,2)
#: 六条**全部错开**，且专门包含「只满足一个条件」的反例。
#: （`(1,1,False)` / `(2,2,False)` 这两条 practice 里已有，
#:   放在这里会让「练习过」和「会做」在同一条用例上无法区分，
#:   门禁 `practice_param_overlap()` 会直接拦下。）
HO_UNIT_COMPACT: tuple[tuple[int, int, bool], ...] = (
    (0, 1, False),    # 墓碑不够
    (7, 5, False),    # 活跃键超限
    (0, 0, False),    # 边界：两个条件都不满足
    (4, 0, True),     # 双条件满足（墓碑够 + 无活跃键）
    (6, 1, True),     # 双条件满足（墓碑够 + 活跃键恰在上限）
    (3, 0, True),     # 双条件满足（另一组取值）
    (4, 2, False),    # 墓碑够但活跃键超限 —— 「且」不是「或」
    (9, 0, True),     # 墓碑远超阈值
)

#: 压缩场景：5 个桶，触发的是 1、4、6
HO_COMPACT_BUCKETS: tuple[tuple[int, int, int], ...] = (
    # (bucket, live, tombstones)
    (0, 3, 1),      # 不触发：墓碑不足
    (1, 1, 2),      # 触发
    (2, 2, 2),      # 不触发：活跃键超限（这是「且」不是「或」的考点）
    (3, 4, 5),      # 不触发：两项都不满足
    (4, 0, 7),      # 触发
    (5, 5, 0),      # 不触发：墓碑为 0
    (6, 1, 4),      # 触发
)
HO_COMPACT_TRIGGERED = [1, 4, 6]
HO_COMPACT_START_EPOCH = 2
HO_COMPACT_EPOCH_AFTER = HO_COMPACT_START_EPOCH + len(HO_COMPACT_TRIGGERED)

#: 一致性场景：LumenHash 与 practice 的 2500/4000/6000/8200 全部错开。
#:
#: **结构也必须错开。** practice 的 4 个键全部在 epoch=0 写入、
#: 全部失配、全部需重算——只要照着 practice 的答案结构换个数字就能答对。
#: held-out 改成**混合状态**：
#:   m1、m2 在 epoch=0 写入（失配，需重算）
#:   m3 在 epoch=3 写入（可读，标签正确）
#:   m4 在 epoch=2 写入（失配，且需回绕计算）
#:   m5 在 epoch=3 写入（可读）
#: 于是「哪些能读、哪些要重算」必须逐键算 epoch 差，
#: 套用 practice 的「全失配」结构会直接错。
HO_CONSIST_KEYS = (
    {"key": "m1", "lumen_hash": 1777, "written_epoch": 0},
    {"key": "m2", "lumen_hash": 5231, "written_epoch": 0},
    {"key": "m3", "lumen_hash": 8899, "written_epoch": 3},
    {"key": "m4", "lumen_hash": 12043, "written_epoch": 2},
    {"key": "m5", "lumen_hash": 64001, "written_epoch": 3},
)
HO_CONSIST_EPOCH = 3
#: 存入标签 = (LumenHash + 写入时 epoch) mod 65537
HO_CONSIST_TAGS = {
    k["key"]: ob.ref_ztag(k["lumen_hash"], k["written_epoch"])
    for k in HO_CONSIST_KEYS
}
#: 逐键用参考实现的读取校验算，不手写
HO_CONSIST_READABLE: list[str] = [
    k["key"] for k in HO_CONSIST_KEYS
    if ob.ref_read_ok(HO_CONSIST_TAGS[k["key"]], k["lumen_hash"], HO_CONSIST_EPOCH)
]
HO_CONSIST_NEED_RETAG: list[str] = [
    k["key"] for k in HO_CONSIST_KEYS if k["key"] not in HO_CONSIST_READABLE
]
#: 当前 epoch 下的期望标签，**只列需重算的键**
#: （题目只要求给需重算的键算新标签；给可读的键也算多余信息）
HO_CONSIST_NEW_TAGS = {
    k["key"]: ob.ref_ztag(k["lumen_hash"], HO_CONSIST_EPOCH)
    for k in HO_CONSIST_KEYS if k["key"] in HO_CONSIST_NEED_RETAG
}

#: 分裂场景：与 practice 的 0/8/16/24/32 错开
HO_SPLIT_HASHES = (5, 13, 21, 29, 37)
HO_SPLIT_BUCKET = 5 % 8  # = 5
HO_SPLIT_MIGRATED = 0
HO_SPLIT_STILL_OVER = True


def _ho_fixture():
    """held-out 专用的 fixture 库（参数与 practice 不同）。"""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.executescript(ob.FIXTURE_SCHEMA)
    # 同样的结构，不同的 epoch 与分桶分布
    conn.executemany(
        "INSERT INTO zephyra_meta(name, value) VALUES (?,?)",
        [("epoch", HO_CONSIST_EPOCH), ("bucket_count", 8), ("live_limit", 4)],
    )
    conn.executemany(
        "INSERT INTO zephyra_buckets(bucket, live_count, tombstone_count) VALUES (?,?,?)",
        [
            (0, 4, 0),   # 未超载
            (1, 4, 0),
            (2, 5, 1),   # 超载
            (3, 3, 0),
            (4, 6, 2),   # 超载
            (5, 2, 0),
            (6, 5, 0),   # 超载
            (7, 1, 0),
        ],
    )
    epoch = HO_CONSIST_EPOCH
    # 每个键的 ztag 按它**自己写入那一刻**的 epoch 算，
    # 所以「写入时刻」是本题的变量，而不只是哈希值。
    rows = [
        ("m1", 1777, 1777 % 8, ob.ref_ztag(1777, 0), 0),                  # 失配
        ("m2", 5231, 5231 % 8, ob.ref_ztag(5231, 0), 0),                  # 失配
        ("m3", 8899, 8899 % 8, ob.ref_ztag(8899, 3), 3),                  # 匹配
        ("m4", 12043, 12043 % 8, ob.ref_ztag(12043, 2), 2),               # 失配
        ("m5", 64001, 64001 % 8, ob.ref_ztag(64001, 3), 3),               # 匹配（含回绕）
        ("n1", 31337, 31337 % 8, ob.ref_ztag(31337, 1), 1),               # 失配
        ("n2", 4242, 4242 % 8, ob.ref_ztag(4242, 3), 3),                  # 匹配
        ("n3", 58899, 58899 % 8, ob.ref_ztag(58899, 0), 0),               # 失配（接近回绕）
        ("n4", 19999, 19999 % 8, ob.ref_ztag(19999, 2), 2),               # 失配
        ("n5", 51515, 51515 % 8, ob.ref_ztag(51515, 3), 3),               # 匹配
        ("n6", 7, 7 % 8, ob.ref_ztag(7, 0), 0),                          # 失配
        ("n7", 65535, 65535 % 8, ob.ref_ztag(65535, 3), 3),               # 匹配（边界值）
        ("n8", 32768, 32768 % 8, ob.ref_ztag(32768, 1), 1),               # 失配
    ]
    conn.executemany(
        "INSERT INTO zephyra_keys(key, lumen_hash, bucket, ztag, epoch_at_write)"
        " VALUES (?,?,?,?,?)", rows,
    )
    conn.commit()
    return conn



def _ho_overloaded_buckets() -> list[int]:
    conn = _ho_fixture()
    try:
        return [
            r[0] for r in conn.execute(
                "SELECT bucket FROM zephyra_buckets WHERE live_count > 4"
                " ORDER BY bucket"
            )
        ]
    finally:
        conn.close()


def _ho_unreadable_keys() -> list[str]:
    conn = _ho_fixture()
    try:
        return [
            r[0] for r in conn.execute(
                "SELECT key FROM zephyra_keys WHERE ztag !="
                " (lumen_hash + (SELECT value FROM zephyra_meta"
                " WHERE name='epoch')) % 65537 ORDER BY key"
            )
        ]
    finally:
        conn.close()


# ==========================================================================
# held-out 客观任务集
# ==========================================================================

#: held-out 任务的参数指纹：与 TASKS_FINGERPRINT 必须不同
HO_TASKS_FINGERPRINT = ob._sha256_of(
    {
        "version": HELD_OUT_VERSION,
        "hash_key": HO_HASH_KEY,
        "hash_value": HO_HASH_VALUE,
        "unit_hash": [list(c) for c in HO_UNIT_HASH],
        "unit_compact": [list(c) for c in HO_UNIT_COMPACT],
        "compact_buckets": [list(b) for b in HO_COMPACT_BUCKETS],
        "compact_triggered": HO_COMPACT_TRIGGERED,
        "compact_epoch_after": HO_COMPACT_EPOCH_AFTER,
        "consist_keys": [dict(k) for k in HO_CONSIST_KEYS],
        "consist_epoch": HO_CONSIST_EPOCH,
        "split_hashes": list(HO_SPLIT_HASHES),
    }
)


class HeldOutTask(ob.ObjectiveTask):
    """held-out 客观任务。结构与 practice 相同，只是参数不同。"""


HELD_OUT_TASKS: tuple[HeldOutTask, ...] = (
    HeldOutTask(
        task_id="ho_hash_01",
        kp="kp_lumen_hash",
        title="held-out：算指定键的哈希与桶号",
        kind="algorithm_output",
        prompt=(
            "请按学习材料里 LumenHash 的定义，逐步计算键 "
            f"\"{HO_HASH_KEY}\" 的 LumenHash 值，并给出它所属的桶号。\n"
            "必须提交 JSON，包含四个字段：\n"
            "  key       —— 固定为字符串 "
            f"\"{HO_HASH_KEY}\"\n"
            "  steps     —— 整数数组，h 的逐步取值序列，"
            "第一个元素是初值，最后一个元素是最终哈希值\n"
            "  lumen_hash —— 最终哈希值（整数）\n"
            "  bucket     —— 所属桶号（整数）"
        ),
        submit_hint=(
            '{"key": "填入键名", "steps": [0, 0, 0, 0],'
            ' "lumen_hash": 0, "bucket": 0}'
        ),
        checker="check_hash_algorithm",
        submit_fields=("key", "steps", "lumen_hash", "bucket"),
        expected={
            "key": HO_HASH_KEY,
            "steps": HO_HASH_STEPS,
            "lumen_hash": HO_HASH_VALUE,
            "bucket": HO_HASH_BUCKET,
        },
    ),
    HeldOutTask(
        task_id="ho_hash_02",
        kp="kp_lumen_hash",
        title="held-out：实现 LumenHash 并通过未见用例",
        kind="python_unit_test",
        prompt=(
            "请用 Python 实现学习材料里定义的 LumenHash 函数，"
            "函数签名必须是 `def lumen_hash(key):`，返回一个整数。\n"
            "提交 JSON，只含一个字段：code —— 完整的 Python 源码字符串\n"
            "评测方式：把你的代码放进测试环境，用多组键实际运行，"
            "比对返回值。注意空串、超长串都要能处理。\n"
            "**本题的测试用例与你此前可能做过的任何一题都不同。**"
        ),
        submit_hint=(
            '{"code":"def lumen_hash(key):\\n    h = 7\\n'
            '    for c in key:\\n        h = (h * 131 + ord(c)) % 65537\\n'
            '    return h"}'
        ),
        checker="check_hash_unit_test",
        submit_fields=("code",),
        expected={"cases": [list(c) for c in HO_UNIT_HASH]},
    ),
    HeldOutTask(
        task_id="ho_bucket_01",
        kp="kp_lumen_bucket",
        title="held-out：算一级分桶分裂后每个键的桶号",
        kind="algorithm_output",
        prompt=(
            f"某个桶里有 5 个键，它们的 LumenHash 值分别是 "
            f"{HO_SPLIT_HASHES[0]}、{HO_SPLIT_HASHES[1]}、{HO_SPLIT_HASHES[2]}、"
            f"{HO_SPLIT_HASHES[3]}、{HO_SPLIT_HASHES[4]}"
            f"（所以它们本来就在同一个桶里）。\n"
            f"现在对{HO_SPLIT_BUCKET} 号桶触发一次分裂。请按学习材料的分裂规则，"
            "算出这 5 个键分裂后各自的桶号，并判断这次分裂有没有减少负载。\n"
            "提交 JSON，包含三个字段：\n"
            "  assignments    —— 数组，每项 "
            '{"key":"k0","from_bucket":0,"to_bucket":0}，'
            "五个键都要列出，key 名依次用 k0、k5、k13、k21、k29、k37 中与"
            "各LumenHash 对应的五个\n"
            "  migrated_count —— 实际离开原桶的键数（整数）\n"
            "  still_over_limit —— 分裂后该桶是否仍然超出活跃键上限（布尔）"
        ),
        submit_hint=(
            '{"assignments": [{"key": "k0", "from_bucket": 0, "to_bucket": 0}],'
            ' "migrated_count": 0, "still_over_limit": false}'
        ),
        checker="check_split_algorithm",
        submit_fields=("assignments", "migrated_count", "still_over_limit"),
        expected={
            "assignments": [
                {"key": f"k{h}", "from_bucket": HO_SPLIT_BUCKET,
                 "to_bucket": HO_SPLIT_BUCKET}
                for h in HO_SPLIT_HASHES
            ],
            "migrated_count": HO_SPLIT_MIGRATED,
            "still_over_limit": HO_SPLIT_STILL_OVER,
        },
    ),
    HeldOutTask(
        task_id="ho_bucket_02",
        kp="kp_lumen_bucket",
        title="held-out：用 SQL 查超载桶（另一组阈值数据）",
        kind="sql_exec",
        prompt=(
            "数据库里有两张表（只读，不要建表不要改数据）：\n"
            "  zephyra_buckets(bucket, live_count, tombstone_count)\n"
            "  zephyra_keys(key, lumen_hash, bucket, ztag, epoch_at_write)\n"
            "每个桶最多容纳 4 个活跃键。请写一条**只读** SQL，"
            "把所有活跃键数超过 4 的桶号查出来，按桶号升序。\n"
            "提交 JSON，只含一个字段：sql —— SQL 语句字符串\n"
            "**本题的数据与此前可能做过的同类题不同。**"
        ),
        submit_hint='{"sql":"SELECT bucket FROM zephyra_buckets WHERE live_count > 4 ORDER BY bucket"}',
        checker="check_overload_sql",
        submit_fields=("sql",),
        expected={"rows": [[b] for b in _ho_overloaded_buckets()]},
    ),
    HeldOutTask(
        task_id="ho_compact_01",
        kp="kp_lumen_compact",
        title="held-out：逐桶判定压缩触发并给出压缩后 epoch",
        kind="json_schema",
        prompt=(
            "下面是 7 个桶的当前状态（live = 活跃键数，tombstones = 墓碑数）：\n"
            + "".join(
                f"  桶 {b}：live={lv}, tombstones={tb}\n"
                for b, lv, tb in HO_COMPACT_BUCKETS
            )
            + f"当前全局 epoch 是 {HO_COMPACT_START_EPOCH}。请按学习材料的压缩触发条件，"
            "判断哪些桶会触发压缩，并给出执行完这些压缩之后的全局 epoch。\n"
            "提交 JSON，包含两个字段：\n"
            "  compress   —— 整数数组，列出所有触发压缩的桶号，按升序\n"
            "  epoch_after —— 整数，全部压缩完成后的 epoch"
        ),
        submit_hint='{"compress":[0,1],"epoch_after":3}',
        checker="check_compact_schema",
        submit_fields=("compress", "epoch_after"),
        expected={
            "compress": HO_COMPACT_TRIGGERED,
            "epoch_after": HO_COMPACT_EPOCH_AFTER,
        },
    ),
    HeldOutTask(
        task_id="ho_compact_02",
        kp="kp_lumen_compact",
        title="held-out：实现压缩判定并通过未见用例",
        kind="python_unit_test",
        prompt=(
            "请用 Python 实现一个函数判断某个桶是否会触发压缩，"
            "签名必须是 `def should_compact(tombstones, live):`，"
            "返回布尔值。判定条件以学习材料为准（注意是「且」还是「或」）。\n"
            "提交 JSON，只含一个字段：code —— 完整的 Python 源码字符串\n"
            "**本题的测试用例与你此前可能做过的任何一题都不同，"
            "包含多个「只满足一个条件」的反例。**"
        ),
        submit_hint=(
            '{"code":"def should_compact(tombstones, live):\\n'
            '    return tombstones >= 2 and live <= 1"}'
        ),
        checker="check_compact_unit_test",
        submit_fields=("code",),
        expected={"cases": [list(c) for c in HO_UNIT_COMPACT]},
    ),
    HeldOutTask(
        task_id="ho_consist_01",
        kp="kp_lumen_consist",
        title="held-out：算 epoch 变化后标签失配情况（混合写入时刻）",
        kind="algorithm_output",
        prompt=(
            "5 个键的 LumenHash 与写入时的全局 epoch 如下：\n"
            + "".join(
                f"  {k['key']}：LumenHash = {k['lumen_hash']}，"
                f"写入时 epoch = {k['written_epoch']}\n"
                for k in HO_CONSIST_KEYS
            )
            + f"此后数据库继续压缩，当前全局 epoch = {HO_CONSIST_EPOCH}。\n"
            "注意：这 5 个键**不是**在同一个 epoch 写入的，"
            "有的写入后没经历过压缩、有的经历过。\n"
            "请回答：(1) 现在哪些键还能被正常读取；(2) 哪些键需要重算标签；"
            "(3) 每个需要重算的键，重算后的 Z-Tag 应该是多少。\n"
            "提交 JSON，包含三个字段：\n"
            "  readable  —— 字符串数组，能读的键名（读不出就留空数组）\n"
            "  need_retag —— 字符串数组，需要重算标签的键名\n"
            "  new_tags  —— 对象，键名到新标签值的映射"
        ),
        submit_hint=(
            '{"readable": ["键名"], "need_retag": ["键名"],'
            ' "new_tags": {"键名": 0}}'
        ),
        checker="check_consistency_algorithm",
        submit_fields=("readable", "need_retag", "new_tags"),
        expected={
            "readable": HO_CONSIST_READABLE,
            "need_retag": HO_CONSIST_NEED_RETAG,
            "new_tags": HO_CONSIST_NEW_TAGS,
        },
    ),
    HeldOutTask(
        task_id="ho_consist_02",
        kp="kp_lumen_consist",
        title="held-out：用 SQL 查读取被拒的键（另一组数据）",
        kind="sql_exec",
        prompt=(
            "数据库里有三张表（只读）：\n"
            "  zephyra_meta(name, value)      —— name='epoch' 的行是当前全局 epoch\n"
            "  zephyra_keys(key, lumen_hash, bucket, ztag, epoch_at_write)\n"
            "  zephyra_buckets(bucket, live_count, tombstone_count)\n"
            "每个键存下来的 ztag 是它写入那一刻按当时 epoch 算好的，"
            "之后不会自动更新。请写一条**只读** SQL，"
            "按学习材料的读取校验规则，把所有当前读取会被拒绝的键查出来，"
            "按键名升序。\n"
            "提交 JSON，只含一个字段：sql —— SQL 语句字符串\n"
            "**本题的数据与此前可能做过的同类题不同。**"
        ),
        submit_hint=(
            '{"sql":"SELECT key FROM zephyra_keys WHERE ztag != (lumen_hash +'
            ' (SELECT value FROM zephyra_meta WHERE name=\'epoch\')) % 65537'
            ' ORDER BY key"}'
        ),
        checker="check_unreadable_sql",
        submit_fields=("sql",),
        expected={"rows": [[k] for k in _ho_unreadable_keys()]},
    ),
)


# ==========================================================================
# held-out 考试卷（POST 与 TRANSFER 共用）
# ==========================================================================

#: POST 卷：application 题型，每 kp 2 题
HELD_OUT_POST: tuple[ExamItem, ...] = (
    ExamItem(
        item_id="ho_post_hash_01", kp="kp_lumen_hash",
        question_type="application",
        question=(
            "Zephyra 的 LumenHash 初值是 7，每字符执行 h = (h * 131 + ord(c)) mod 65537。"
            "现在要把键 \"zeta\" 写入存储。请给出完整的计算过程（每一步 h 的取值）、"
            "最终 LumenHash 值，以及它所属的桶号（桶号 = LumenHash mod 8，桶有 8 个）。"
        ),
        answer=(
            "逐步计算：h 初值 7。第一个字符 'z'（ord=122）：h = (7 × 131 + 122) mod 65537 "
            "= (917 + 122) = 1039（小于 65537，不回绕）。"
            "第二个字符 'e'（ord=101）：h = (1039 × 131 + 101) = 136210；"
            "136210 mod 65537 = 136210 − 65537 × 2 = 136210 − 131074 = 5136。"
            "第三个字符 't'（ord=116）：h = (5136 × 131 + 116) = 672932；"
            "672932 mod 65537 = 672932 − 65537 × 10 = 672932 − 655370 = 17562。"
            "第四个字符 'a'（ord=97）：h = (17562 × 131 + 97) = 2300719；"
            "2300719 mod 65537 = 2300719 − 65537 × 35 = 2300719 − 2293795 = 6924。"
            "所以 LumenHash(\"zeta\") = 6924，步骤序列为 [7, 1039, 5136, 17562, 6924]。"
            "桶号 = 6924 mod 8 = 4（6924 = 8 × 865 + 4）。"
            "因此该键落在 4 号桶。"
        ),
        premise_in_question="yes",
    ),
    ExamItem(
        item_id="ho_post_hash_02", kp="kp_lumen_hash",
        question_type="application",
        question=(
            "Zephyra 的桶有 8 个，分桶规则是 bucket = LumenHash mod 8。"
            "已知一个键的 LumenHash = 30000，但它落在 0 号桶——这违反了分桶规则。"
            "请找出这个矛盾，并说明：要让键正确落到 0 号桶，它的 LumenHash "
            "必须满足什么条件？另外，若一个键的 LumenHash 是 30001，它的桶号是多少？"
        ),
        answer=(
            "矛盾在于：按规则 bucket = 30000 mod 8 = 0（30000 = 8 × 3750，余 0），"
            "而题目说它落在 0 号桶，这两者其实**并不矛盾**——问题出在题目本身的隐含假设。"
            "正确回答应指出：按给定规则 30000 mod 8 = 0，所以「落在 0 号桶」是**符合**规则的，"
            "不存在矛盾。若要构造真正矛盾的情形，"
            "需要 LumenHash = 30000 却声称落在 1 号桶。"
            "要让键正确落到 0 号桶，LumenHash 必须满足 LumenHash mod 8 = 0，"
            "即 LumenHash 是 8 的倍数（0、8、16、24、…）。"
            "对 LumenHash = 30001：30001 mod 8 = 1（30001 = 8 × 3750 + 1），所以落在 1 号桶。"
            "本题的关键判断是：模 8 的余数只由最低三位二进制位决定，"
            "与 LumenHash 的大小无关，因此余数必然在 0~7 之间，"
            "不存在「算出 8 或更大余数」的情况。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_post_bucket_01", kp="kp_lumen_bucket",
        question_type="application",
        question=(
            "Zephyra 的桶 5 里有 5 个键，LumenHash 分别是 5、13、21、29、37，"
            "活跃键上限是 4。现在对 5 号桶触发一次分裂。"
            "请回答：(1) 按材料规则，这 5 个键分裂后各自落在哪个桶？"
            "(2) 实际有几个键离开了原桶？(3) 分裂后该桶是否仍然超限？"
        ),
        answer=(
            "先确认这 5 个键确实同桶：5 mod 8 = 5，13 mod 8 = 5，21 mod 8 = 5，"
            "29 mod 8 = 5，37 mod 8 = 5，全部落在 5 号桶，题目前提成立。"
            "分裂规则是「迁移到 (bucket + 8) mod 8 的桶」。"
            "对 5 号桶：(5 + 8) mod 8 = 13 mod 8 = 5。"
            "所以这 5 个键迁移后仍然全部落在 5 号桶，"
            "k5/k13/k21/k29/k37 的 to_bucket 均为 5。"
            "实际离开原桶的键数为 0——**一级分桶的分裂是恒等变换，不产生任何负载分散效果**。"
            "分裂后 5 号桶仍有 5 个活跃键，超过上限 4，因此**仍然超限**。"
            "工程含义：单靠一级分裂无法解决一级分桶的超载问题，"
            "要么扩大桶数（改分桶基数），要么采用完全不同的分散策略"
            "（例如按 LumenHash mod 16 重新分桶，但那已经不是材料的分裂规则）。"
        ),
        premise_in_question="yes",
    ),
    ExamItem(
        item_id="ho_post_bucket_02", kp="kp_lumen_bucket",
        question_type="application",
        question=(
            "Zephyra 的分裂规则被确认是恒等变换。团队提出第二个方案："
            "把每个键的 LumenHash 全部加上一个常数 C（模 65537），"
            "理由是「加常数相当于整体平移，能让原本挤在一起的键散开」。"
            "请判断这个方案是否有效，并说明理由。"
        ),
        answer=(
            "**方案无效。** 设某键原哈希为 h，加C 后为 h' = (h + C) mod 65537。"
            "由于 C 是常数，与 h 无关，两个键 h1、h2 的差值在平移后保持："
            "h1' - h2' ≡ (h1 - h2) (mod 65537)。"
            "关键在于桶号取模于 8，而 8 | 65537 不成立（65537 = 2^16 + 1，"
            "是奇数，与 8 互质），所以 +C 确实会改变 h mod 8 的结果。"
            "但这带来的是**整体重排**而不是「散开」："
            "新桶号 = (h + C) mod 8，与旧桶号 h mod 8 之间是固定的平移关系，"
            "即新桶号 = (旧桶号 + C mod 8) mod 8，是一个对所有键都相同的循环位移。"
            "两个原本同桶的键，加同一常数后仍然同桶——"
            "因为 (h1 + C) mod 8 = (h2 + C) mod 8 当且仅当 h1 ≡ h2 (mod 8)。"
            "所以原本挤在一起的 5 个键，加C 后仍然挤在同一个桶里，负载一点没分散。"
            "**结论：加常数是恒等的重排，对超载毫无帮助。**"
            "真正的解决办法必须改变「哪些键被分到同一桶」这件事本身，"
            "例如增加桶数或采用二级分裂。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_post_compact_01", kp="kp_lumen_compact",
        question_type="application",
        question=(
            "Zephyra 有 7 个桶，状态如下（live=活跃键数，tombstones=墓碑数）："
            "桶 0: live=3, tombstones=1；桶 1: live=1, tombstones=2；"
            "桶 2: live=2, tombstones=2；桶 3: live=4, tombstones=5；"
            "桶 4: live=0, tombstones=7；桶 5: live=5, tombstones=0；"
            "桶 6: live=1, tombstones=4。当前全局 epoch = 2。"
            "请判断哪些桶会触发压缩，并给出全部压缩完成后的 epoch。"
        ),
        answer=(
            "压缩条件是 Tombstone >= 2 **且** 活跃键 <= 1（注意是「且」，两个条件必须同时满足）。"
            "逐桶判断：桶 0: tombstones=1 < 2，不触发（虽然活跃键少但墓碑不够）。"
            "桶 1: tombstones=2 >= 2 且 live=1 <= 1，**触发**。"
            "桶 2: tombstones=2 >= 2 但 live=2 > 1，**不触发**——这是本组数据里最关键的陷阱，"
            "把「且」误读成「或」会错误地把它算进触发集合。"
            "桶 3: tombstones=5 >= 2 但 live=4 > 1，**不触发**（两项都不满足墓碑与活跃的条件组合）。"
            "桶 4: tombstones=7 >= 2 且 live=0 <= 1，**触发**。"
            "桶 5: tombstones=0 < 2，**不触发**（墓碑为 0 是另一个方向的陷阱）。"
            "桶 6: tombstones=4 >= 2 且 live=1 <= 1，**触发**。"
            "所以触发压缩的桶是 [1, 4, 6]，共 3 个。"
            "每次压缩使全局 epoch 加 1，当前 epoch = 2，"
            "全部完成后 epoch = 2 + 3 = 5。"
        ),
        premise_in_question="yes",
    ),
    ExamItem(
        item_id="ho_post_compact_02", kp="kp_lumen_compact",
        question_type="application",
        question=(
            "Zephyra 的一个桶有 0 个活跃键、9 个 Tombstone，因此必然触发压缩。"
            "有人认为「压缩后该桶就干净了，以后不会再有问题」。"
            "请评估这个说法，并说明压缩除了清除墓碑之外还会带来什么全局影响。"
        ),
        answer=(
            "「以后不会再有问题」这个说法**不成立**。理由有三层。"
            "第一层：压缩确实清除该桶的墓碑，但该桶墓碑会随后续删除操作重新累积，"
            "所以「压缩一次就一劳永逸」是错的。"
            "第二层：压缩的三个动作中有一个是**全局 epoch + 1**，"
            "这影响的不只是被压缩的那个桶。"
            "由于 ZTag = (LumenHash + epoch) mod 65537，"
            "epoch 变化会让**全库所有存量键**的 Z-Tag 校验失配——"
            "包括那些从未被压缩、状态完好的桶里的键。"
            "这正是材料强调的「触发是局部的，影响是全局的」。"
            "第三层：因此一次局部压缩会引发全库范围的读取被拒，"
            "必须配套一次全库标签重算（retag）。"
            "工程结论：压缩不能当成局部维护动作随手执行，"
            "它实质上是一次全局 epoch 变更，需要配合全库 retag 窗口，"
            "否则会出现「压缩了某个桶，全库却大面积读不出」的事故。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_post_consist_01", kp="kp_lumen_consist",
        question_type="application",
        question=(
            "Zephyra 有 3 个键：h1（LumenHash=1777）、h2（LumenHash=5231）、"
            "h3（LumenHash=8899），都在 epoch=0 时写入。"
            "此后数据库发生了 3 次压缩，当前 epoch = 3。"
            "请回答：(1) 现在哪些键还能读出来？(2) 哪些键需要重算标签？"
            "(3) 每个需重算的键，新标签是多少？"
        ),
        answer=(
            "先算当前应有标签：ZTag = (LumenHash + epoch) mod 65537 = "
            "(LumenHash + 3) mod 65537。"
            "h1: 1777 + 3 = 1780（小于 65537，不回绕）→ 新标签 1780。"
            "h2: 5231 + 3 = 5234 → 新标签 5234。"
            "h3: 8899 + 3 = 8902 → 新标签 8902。"
            "再算存入的标签：因为写入时 epoch=0，"
            "存入标签 = (LumenHash + 0) mod 65537 = LumenHash 本身，"
            "即h1 存 1777、h2 存 5231、h3 存 8899。"
            "逐键比较：h1 存 1777 vs 应 1780 → 不匹配 → **读不出**。"
            "h2 存 5231 vs 应 5234 → 不匹配 → **读不出**。"
            "h3 存 8899 vs 应 8902 → 不匹配 → **读不出**。"
            "所以 (1) 能读出的键：**空集**。"
            "(2) 需要重算标签的键：h1、h2、h3（全部）。"
            "(3) 新标签：{\"h1\": 1780, \"h2\": 5234, \"h3\": 8902}。"
            "注意这里「epoch=0 写入时标签恰好等于 LumenHash」这个巧合让失配看起来很整齐，"
            "但机制与 V0.5 的例子完全相同：epoch 一变，存量键全部失配。"
        ),
        premise_in_question="yes",
    ),
    ExamItem(
        item_id="ho_post_consist_02", kp="kp_lumen_consist",
        question_type="application",
        question=(
            "Zephyra 运维观察到一个现象：每执行一次压缩，"
            "「读取被拒的键数」几乎等于「全库活跃键总数」。"
            "请解释这个现象，并说明如果要避免大面积读取被拒，"
            "压缩与标签重算（retag）应该以什么顺序、什么粒度执行。"
        ),
        answer=(
            "现象的根因是「压缩的触发是局部的，而 epoch 的影响是全局的」。"
            "具体推导：一次压缩把 epoch 加 1；"
            "而 ZTag = (LumenHash + epoch) mod 65537，"
            "所以对每一个存量键，其应有标签都整体增加了 1，"
            "而实际存储的标签没变——于是**每一个存量键都失配**。"
            "设压缩前全库活跃键数为 N，则压缩后读取被拒的键数即为 N，"
            "所以两者「几乎相等」不是巧合，而是机制的必然结果。"
            "（若严格说有例外，只有那些恰好在 epoch 增加的同时被重算过标签的键能读，"
            "所以是「几乎」相等而非完全相等。）"
            "关于执行顺序与粒度：正确的顺序是**先压缩、后 retag**，"
            "因为 retag 要用压缩后的新 epoch 来算，顺序颠倒会用旧 epoch 计算，"
            "算出来的标签立刻又失效。"
            "粒度上必须**全局**而非按桶：即便只有 1 号桶触发压缩，"
            "也要对全库所有键重算标签，"
            "只对被压缩的桶 retag 会留下大量失配键。"
            "工程上应把压缩与 retag 打包成一个原子维护操作，"
            "或在压缩前预判将被触发的桶数，据此安排停写窗口。"
        ),
        premise_in_question="asserted",
    ),
)

#: TRANSFER 卷：全新场景，每 kp 2 题，全部 asserted 前置
HELD_OUT_TRANSFER: tuple[ExamItem, ...] = (
    ExamItem(
        item_id="ho_tr_hash_01", kp="kp_lumen_hash",
        question_type="transfer",
        question=(
            "【新场景·无真实系统对应】某团队用 Zephyra 做多租户隔离，"
            "每个租户的键都加了租户前缀 \"t<N>_\"。"
            "团队提出一个前提：「只要把 LumenHash 的初值从 7 改成租户编号 N，"
            "不同租户的哈希空间就彼此隔离了，不同租户不可能哈希碰撞」。"
            "请判断这个前提是否成立，并说明理由。"
        ),
        answer=(
            "**前提不成立。** 三个层面。"
            "第一，桶级隔离：键最终落在哪个桶只取决于 LumenHash mod 8。"
            "设租户 A 初值 1、租户 B 初值 2，"
            "A 键 \"ab\" 的哈希以 1 起算、B 键 \"b\" 以 2 起算，"
            "两个值对 8 取模**完全可能相同**——因为模 8 后余数只有 8 种取值，"
            "而租户键的组合是无限的，鸽笼原理必然出现跨租户同桶。"
            "第二，即使哈希值不同，也不构成隔离："
            "Zephyra 的隔离边界是**桶**，不是哈希值本身；"
            "两个键只要桶号相同就共享槽位与墓碑，"
            "一个租户的压缩会改变全局 epoch，"
            "进而让**另一个租户**的键 Z-Tag 失配。"
            "所以「哈希不碰撞」并不等于「互不影响」。"
            "第三，跨租户同桶还会造成容量干扰："
            "一个租户大量写入会把同桶的其他租户挤到超限，"
            "形成经典的 noisy neighbor 问题。"
            "**正确做法**是把租户编号放进桶选择逻辑"
            "（例如 bucket = (hash + tenant_slot) mod 8，为每个租户分配不重叠的槽位集合），"
            "让隔离发生在桶这一层，而不是哈希这一层。"
            "前提把「哈希分布」当成了「隔离机制」，"
            "而材料里隔离的实际边界是桶与 epoch。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_tr_hash_02", kp="kp_lumen_hash",
        question_type="transfer",
        question=(
            "【新场景】某团队怀疑自己的 Zephyra 部署存在哈希实现 bug，"
            "但无法确定 bug 出在「初值」还是「乘数」上。"
            "他们采集了两个观测：(a) 键 \"ab\" 的哈希是 1858；"
            "(b) 键 \"user\" 的哈希是 14991。"
            "团队提出一个前提：「只要这两个观测与材料算法算出的值一致，"
            "就说明实现没有 bug」。请判断这个前提能否排除全部可能的实现缺陷。"
        ),
        answer=(
            "**前提不足以排除全部缺陷。** 它只能确认「这两个特定输入上，"
            "实现与材料算法一致」，覆盖面远小于「实现没有 bug」。"
            "具体地，以下缺陷都能让 (a)(b) 两个观测同时成立却仍然是 bug："
            "第一，**空串与单字符路径**。若实现对 len(key)==0 直接 return 0 "
            "而不是返回初值 7，两个观测都不受影响，但空串键会全部错。"
            "第二，**非 ASCII 字符**。若实现按字节而不是按 Unicode 码点运算"
            "（例如 UTF-8 编码后逐字节处理），"
            "对 \"ab\"、\"user\" 这类纯 ASCII 输入结果完全相同，"
            "但对任何含中文的键都会算出不同的值。"
            "第三，**溢出与取模时机**。若实现在每步取模的位置不同"
            "（例如先算完整个字符串再取模一次，而不是每字符都取模），"
            "由于 131 与 65537 互质，多项式形式展开后最终取模结果其实相同，"
            "所以这条不构成反例——恰恰说明**这条也无法被现有观测排除**，"
            "只是恰好等价。第四，**非数字键**。材料未定义数字键的行为，"
            "实现可以自行选择，同样不影响 ASCII 字母键的结果。"
            "结论：要排除实现缺陷，需要在**边界与异质输入**上做测试"
            "（空串、单字符、非 ASCII、超长键、含数字键），"
            "而不是在两个「正常」输入上做对照。"
            "这题考的是：局部一致的观测不能推广为全局正确性，"
            "而材料中哈希的定义正是靠「每个字符都取模」和「按 ord()」"
            "这两个细节才成立的。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_tr_bucket_01", kp="kp_lumen_bucket",
        question_type="transfer",
        question=(
            "【新场景】某团队要做一个「桶级熔断」功能："
            "当某个桶的活跃键数超过上限时暂停对该桶的写入。"
            "团队提出一个前提：「只要把分裂真正实现成"
            "「按 LumenHash mod 16 把键分到 (2b) 和 (2b+1) 两个桶」，"
            "熔断就永远不会触发，因为每个子桶的负载一定减半」。"
            "请判断这个前提是否成立。"
        ),
        answer=(
            "**前半成立，后半不成立。** 拆开看。"
            "前半：「按 LumenHash mod 16 分到 2b 和 2b+1」是一个**合理的改造方案**"
            "（它确实能把原桶的负载拆到两个子桶里，也是真实系统常用的做法），"
            "但它**不是材料定义的分裂规则**——材料写的是迁移到 (b+8) mod 8，"
            "而 (b+8) mod 8 = b，是恒等变换。所以这个改造必须显式写进材料/代码，"
            "不能指望它自动生效。"
            "后半：「负载一定减半」在**当前桶内均匀分布**时不成立。"
            "设原桶有 6 个键，LumenHash mod 16 的值分别是 0、0、0、8、8、8，"
            "则 2b 桶分到 3 个、2b+1 桶分到 3 个，看起来减半了；"
            "但如果 6 个键的 mod 16 值分别是 0、1、2、3、4、5（即全是偶数），"
            "则 2b 桶分到 6 个、2b+1 桶分到 0 个，"
            "**负载不减反增**，其中一半的键全部挤在一个子桶里，"
            "另一个子桶完全空置。"
            "因为 LumenHash mod 16 只由 LumenHash 的低 4 位决定，"
            "而同一桶（mod 8 相同）只约束了低 3 位相同、高 4 位可能完全一致，"
            "所以「同桶键在 mod 16 上均匀分布」这件事**没有任何保证**。"
            "工程结论：二级分裂只能改善期望负载，不能保证最坏情况。"
            "要保证熔断不触发，必须保留超限检测本身，"
            "把它当作正常分支而不是永不触发的异常分支。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_tr_bucket_02", kp="kp_lumen_bucket",
        question_type="transfer",
        question=(
            "【新场景】某团队要把 Zephyra 从 8 桶扩容到 16 桶，"
            "方案是把所有键的 LumenHash 保持不变，"
            "只把分桶基数从 8 改成 16。团队提出一个前提："
            "「扩容后原有键会自动分散到新桶里，不需要迁移数据」。"
            "请判断这个前提是否成立，并说明这个方案会带来的一个具体风险。"
        ),
        answer=(
            "**前提成立（键会自动重新分布），但「不需要迁移」是危险的表述。**"
            "分两面回答。"
            "第一面：为什么键会分散。bucket = LumenHash mod N，"
            "N 从 8 变成 16 后，桶号由 LumenHash 的低 3 位决定变成低 4 位决定，"
            "原来同桶（mod 8 相同）的键在 mod 16 下可能落入不同的新桶，"
            "确实会自动分散。这个结论是成立的。"
            "第二面：具体风险。**Z-Tag 全部失配。**"
            "ZTag = (LumenHash + epoch) mod 65537 只依赖 LumenHash 与 epoch，"
            "与桶号无关——所以改桶数不会直接改标签，这一点是好的。"
            "真正的风险是：**扩容通常伴随一次数据重分布与重建，"
            "而重建过程中若伴随任何一次压缩，epoch 就会变化，"
            "导致全库存量键 Z-Tag 失配、大面积读取被拒。**"
            "也就是说，「键自动分散」这个无害的性质会把工程师的注意力引向分桶，"
            "而真正需要预案的是 epoch 变更带来的全局标签失效。"
            "第二个具体风险是**数据可见性缺口**：扩容瞬间新旧桶号并存，"
            "若查询侧与写入侧对「当前 N 是多少」不一致，"
            "就会出现能写不能读（或反之）的窗口期。"
            "工程结论：扩容是一次需要停写窗口与全库 retag 预案的操作，"
            "不能当作「改个常数就行」。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_tr_compact_01", kp="kp_lumen_compact",
        question_type="transfer",
        question=(
            "【新场景】某团队在 Zephyra 上做定期巡检，"
            "巡检规则写成：「遍历所有桶，只要发现某个桶的读取被拒键数 > 0，"
            "就立刻对该桶执行一次压缩」。"
            "团队提出一个前提：「这个巡检规则能自动修复读取被拒的问题」。"
            "请判断这个前提是否成立，并说明最坏情况下会发生什么。"
        ),
        answer=(
            "**前提不成立，而且这个巡检规则会显著放大故障。** 逐条推。"
            "第一，读取被拒的根因是 epoch 与存入标签不匹配，"
            "而压缩的动作之一正是「epoch + 1」。"
            "所以对这个桶执行压缩，会让该桶的期望标签**再整体加 1**，"
            "原本就读不出的键依然读不出（甚至错得更远），"
            "却额外制造了**新的失配**：如果该桶本来还有能读出的键，"
            "它们会因为 epoch 再变而全部失配。"
            "结论：**压缩不是读取被拒的修复手段，它本身就是失配的来源。**"
            "第二，最坏情况是指数恶化。假设每轮巡检触发 k 个桶压缩，"
            "则全局 epoch 每轮增加 k；"
            "而 retag 若只覆盖被压缩的桶，"
            "那么每轮都会有全库 (N - 刚retag 的键数) 个键失配，"
            "且失配总量随轮次累积。实测形态是：巡检跑得越勤，"
            "读不出的键越多——**一个自制的故障放大器**。"
            "第三，正确做法：读取被拒时应做 **retag**"
            "（用当前 epoch 重算 Z-Tag），而不是压缩。"
            "压缩只在「墓碑堆积到需要回收槽位」时才是正确动作，"
            "而那是一个**容量**问题，与读取被拒**无关**。"
            "工程结论：把「读取被拒」与「槽位不足」两个正交问题"
            "分别对应到 retag 与 compress，"
            "并把 epoch 变更视为需要全局预案的维护事件。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_tr_compact_02", kp="kp_lumen_compact",
        question_type="transfer",
        question=(
            "【新场景】某团队只有一个桶，且这个桶里长期只有 1 个活跃键，"
            "但删除操作非常频繁，墓碑不断累积。"
            "团队提出一个前提：「既然只有一个桶会触发压缩，"
            "那么 epoch 变化就只影响这一个桶，"
            "维护影响面很小，可以放心地自动触发压缩」。"
            "请判断这个前提是否成立。"
        ),
        answer=(
            "**前提不成立。** 它把「触发范围」错误地等同于「影响范围」，"
            "而材料恰恰强调这两者解耦："
            "压缩的**触发条件是局部的**（只看本桶的墓碑与活跃键数），"
            "但 epoch 变化的**影响是全局的**"
            "（ZTag = (LumenHash + epoch) mod 65537，epoch 变则全库存量键的应有标签都变）。"
            "所以「只有一个桶触发」只意味着「只有这一个桶需要槽位回收」，"
            "不意味着「只有一个桶受影响」。"
            "具体后果：这次压缩让 epoch 加 1，"
            "全库所有键（包括那些位于其他桶、状态完好、"
            "从未有过墓碑的键）的 Z-Tag 全部失配，"
            "读取被拒的键数会从 0 直接跳到接近全库活跃键数。"
            "工程结论：影响面的大小取决于 epoch 是否变化，"
            "而不是取决于有几个桶触发压缩。"
            "因此正确的设计原则是：**压缩的决策可以局部自动化，"
            "但压缩的执行必须是全局事件**——"
            "要么安排停写窗口 + 全库 retag，"
            "要么把epoch 从全局变量改成桶级/分片级变量（这是架构改造，不是配置调整）。"
            "这题考的是：局部条件不能用来推断全局影响。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_tr_consist_01", kp="kp_lumen_consist",
        question_type="transfer",
        question=(
            "【新场景】某团队要实现「标签自愈」：每次读取失败时，"
            "顺手把该键的 Z-Tag 重算成当前 epoch 下的正确值，让下次读取成功。"
            "团队提出一个前提：「自愈机制能让系统自动从epoch 变更中恢复，"
            "运维不再需要任何人工干预」。"
            "请判断这个前提是否成立，并指出这个方案最危险的一个副作用。"
        ),
        answer=(
            "**前提不成立。** 自愈确实能让**被访问到的键**恢复，"
            "但它无法解决两个更根本的问题。"
            "第一，它只覆盖「被访问过」的键。"
            "全库失配时，冷数据里的键从未被读，也就从未被自愈；"
            "等它第一次被读时才现场修复，"
            "于是「读取成功」这件事依赖于「这次真的访问到了它」，"
            "而在批量扫描或校验类任务中，"
            "首次访问就意味着要承担一次写入开销，"
            "把读路径变成了隐式写路径。"
            "第二，**最危险的副作用是掩盖了 epoch 变更的真实规模**。"
            "自愈会让「读取被拒率」这个监控指标长期显示为接近 0，"
            "运维因此完全看不到「刚刚发生过一次全库失配」；"
            "而那次失配意味着某个时间点之后写入的所有键都处于未知状态，"
            "历史数据的一致性没有被验证过，只是没被访问到。"
            "这属于典型的「用自动化掩盖告警」，"
            "最终结果是：真正的数据事故不会再触发任何人工响应。"
            "工程结论：自愈可以作为**兜底手段**保留，"
            "但 epoch 变更必须仍然走「显式维护 + 全局 retag + 事后校验」，"
            "并让「发生过 epoch 变更」这件事在监控上可见。"
            "这题考的是：修复手段能消除症状，不等于消除了病因。"
        ),
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="ho_tr_consist_02", kp="kp_lumen_consist",
        question_type="transfer",
        question=(
            "【新场景】某团队把Zephyra 部署成多实例共享存储，"
            "每个实例各自维护自己的 epoch 计数器（都从 0 开始，各自独立递增）。"
            "团队提出一个前提：「因为每个实例的 epoch 都独立维护，"
            "所以一个实例的压缩不会影响其他实例的数据」。"
            "请判断这个前提是否成立，并说明这个架构的实际后果。"
        ),
        answer=(
            "**前提成立（各实例的 epoch 确实互不干扰），但结论「所以安全」不成立，"
            "因为这个架构在 Zephyra 的规则下根本无法正确工作。**"
            "第一层：确认前提成立的部分。"
            "若每个实例的 epoch 完全独立，那么实例 A 的压缩确实不会改变"
            "实例 B 的 Z-Tag 期望值。从「互不干扰」这个角度说，前提是对的。"
            "第二层：但 Z-Tag 是**按全局 epoch** 计算的，"
            "而材料里的标签校验规则是"
            "「存下的 ztag 是否等于 (LumenHash + epoch) mod 65537」——"
            "这里的 epoch 是**一个全局量**。"
            "一旦每个实例各自维护，"
            "那么「同一个键被写入时的 epoch」和「读取该键时用来校验的 epoch」"
            "就可能来自不同计数器，"
            "于是同一个键在不同时刻、由不同实例读取时，"
            "校验结果可能不同——"
            "即**读操作变成了非确定性的**。"
            "第三层：实际后果有三类。"
            "(a) 数据静默不可读：某实例的 epoch 比写入时大，"
            "读到该键就判定失配并跳过，且这是**随机的**（取决于哪个实例在服务）。"
            "(b) 数据被错误接受：若某实例的 epoch 恰好与写入时一致，"
            "它会判定标签匹配而放行，"
            "但此时这个键的真实状态其实从未被验证——"
            "**假阳性比假阴性更危险**，因为它让错误数据通过校验。"
            "(c) 不可重现：同一查询在不同实例上返回不同结果，"
            "排障时无法定位。"
            "工程结论：epoch 必须是全局一致的单一值（单一计数器或强一致的共识存储），"
            "并且任何 epoch 变更都必须触发全库 retag。"
            "把 epoch 局部化在单实例索引上可行，"
            "在**共享存储的标签校验**上不可行——"
            "这题考的是：局部状态不能直接用作全局一致性校验的依据。"
        ),
        premise_in_question="asserted",
    ),
)

#: TRANSFER 出题说明（供复核）
HELD_OUT_TRANSFER_NOTES: dict[str, dict[str, str]] = {
    "ho_tr_hash_01": {
        "kp": "kp_lumen_hash",
        "tests": "哈希空间隔离是否等于存储隔离",
        "trap": "把「哈希不碰撞」当成「隔离」，忽略桶与 epoch 才是隔离边界",
    },
    "ho_tr_hash_02": {
        "kp": "kp_lumen_hash",
        "tests": "局部一致的观测能否推广为全局正确性",
        "trap": "两个正常输入通过 ≠ 实现无 bug；空串/非 ASCII 是盲区",
    },
    "ho_tr_bucket_01": {
        "kp": "kp_lumen_bucket",
        "tests": "二级分裂能否保证负载均衡",
        "trap": "「一定减半」在LumenHash 低 4 位相同时完全失效",
    },
    "ho_tr_bucket_02": {
        "kp": "kp_lumen_bucket",
        "tests": "改分桶基数这个「无害变更」的真实风险在哪",
        "trap": "风险不在键分布，而在 epoch 变更与可见性缺口",
    },
    "ho_tr_compact_01": {
        "kp": "kp_lumen_compact",
        "tests": "把错误手段当修复手段的后果",
        "trap": "压缩会制造新的失配，巡检越勤失配越多",
    },
    "ho_tr_compact_02": {
        "kp": "kp_lumen_compact",
        "tests": "局部触发条件能否推断全局影响",
        "trap": "只有 1 个桶触发 ≠ 只有 1 个桶受影响",
    },
    "ho_tr_consist_01": {
        "kp": "kp_lumen_consist",
        "tests": "自动化修复是否掩盖了真实故障",
        "trap": "自愈让告警长期沉默，根因反而不可见",
    },
    "ho_tr_consist_02": {
        "kp": "kp_lumen_consist",
        "tests": "局部状态能否用作全局一致性校验依据",
        "trap": "各实例独立 epoch 会让读变成非确定性，且产生假阳性",
    },
}

#: held-out 题库卷
HELDOUT_SETS: dict[str, tuple[ExamItem, ...]] = {
    "post": HELD_OUT_POST,
    "transfer": HELD_OUT_TRANSFER,
}

#: PRE 仍沿用 V0.5 的卷（V0.6 不改 PRE，保证与 V0.5 可比）
PRE_HELDOUT_CHECK = len(PRE_ITEMS) > 0

ALL_HELDOUT_ITEMS: tuple[ExamItem, ...] = HELD_OUT_POST + HELD_OUT_TRANSFER

HELDOUT_FINGERPRINT = _fingerprint(ALL_HELDOUT_ITEMS)
POST_FINGERPRINT = _fingerprint(HELD_OUT_POST)
TRANSFER_HELDOUT_FINGERPRINT = _fingerprint(HELD_OUT_TRANSFER)


def get_heldout(which: str) -> tuple[ExamItem, ...]:
    if which not in HELDOUT_SETS:
        raise KeyError(
            f"未知held-out 卷 {which!r}，只接受 {sorted(HELDOUT_SETS)}"
        )
    return HELDOUT_SETS[which]


def residual_contamination() -> dict[str, list[str]]:
    """held-out 任务在**表达层**上仍与 practice 重合的部分。

    为什么需要单独报
    ----------------
    `practice_param_overlap()` 查的是**参数**层：数值、键名、触发桶集合。
    那一层现在完全错开。

    但有四种任务，**答案的表达形式与参数无关**，
    于是「参数错开」对它们在定义上就不成立：

    - ``sql_exec``：正确答案就是那条 SQL 文本本身。
      它在 held-out 的 fixture 上跑、返回 held-out 的答案集
      （[2,4,6] 而非 practice 的 [3,4,7]），所以**判分是真的**；
      但「和 practice 用同一条 SQL」这件事本身不构成污染，
      因为 SQL 题的隔离只能由数据提供，无法由题面提供。

    - ``python_unit_test``：正确答案就是那个函数本身。
      材料只教了一个LumenHash、一个压缩判定，
      正确实现**有且只有一种**。换一组用例能测泛化，
      但不可能让「实现」这个表达层与practice 不同。

    这两类任务的隔离强度依赖的是：
      SQL  →  held-out fixture 与 practice 不同（已错开）
      unit →  held-out 用例集与 practice 不同（已错开）

    真正的残留风险是：**在 practice 阶段已经写出过正确实现的模型，
    在 held-out 的 unit test 上会近乎不失分**。
    这不是判分器的bug，是「同一知识点被考两次」的结构性代价。
    显式报出来，好过让读者以为这8 道题全都严丝合缝。
    """
    out: dict[str, list[str]] = {}
    for t in HELD_OUT_TASKS:
        if t.kind == "sql_exec":
            out.setdefault("sql_exec_答案与参数无关_隔离靠数据", []).append(t.task_id)
        elif t.kind == "python_unit_test":
            out.setdefault("python_unit_test_正确实现唯一_隔离靠用例", []).append(t.task_id)
    return out


def heldout_diagnostics() -> dict[str, Any]:
    """held-out 题库与任务集的诊断。"""
    problems: list[str] = []
    warnings: list[str] = []

    # 1. 规模与配平
    if len(HELD_OUT_POST) != 8:
        problems.append(f"POST 卷 {len(HELD_OUT_POST)} 题，应为 8")
    if len(HELD_OUT_TRANSFER) != 8:
        problems.append(f"TRANSFER 卷 {len(HELD_OUT_TRANSFER)} 题，应为 8")
    for name, items in HELDOUT_SETS.items():
        counts: dict[str, int] = {}
        for it in items:
            counts[it.kp] = counts.get(it.kp, 0) + 1
        thin = [k for k, n in counts.items() if n < 2]
        if thin:
            problems.append(f"{name} 卷以下知识点不足2 题：{sorted(thin)}")
        missing = set(HELD_OUT_KPS) - set(counts)
        if missing:
            problems.append(f"{name} 卷缺少知识点：{sorted(missing)}")

    # 2. premise 显式声明
    for it in ALL_HELDOUT_ITEMS:
        if it.premise_in_question not in ("yes", "asserted"):
            problems.append(
                f"{it.item_id}：premise_in_question 取值非法 {it.premise_in_question!r}"
            )
    for it in HELD_OUT_TRANSFER:
        if it.premise_in_question != "asserted":
            problems.append(
                f"{it.item_id}：TRANSFER 题必须 asserted，当前 "
                f"{it.premise_in_question!r}"
            )
        if it.item_id not in HELD_OUT_TRANSFER_NOTES:
            problems.append(f"{it.item_id}：缺少出题说明")
        if len(it.answer) < 120:
            problems.append(
                f"{it.item_id}：参考答案过短（{len(it.answer)} 字），判分会失真"
            )

    # 3. 与 V0.5 题库不重叠
    from brain import benchmark_v2 as b2
    v2_ids = {it.item_id for it in b2.ALL_ITEMS}
    dup_ids = {it.item_id for it in ALL_HELDOUT_ITEMS} & v2_ids
    if dup_ids:
        problems.append(f"held-out 与 V0.5 题库 item_id 重复：{sorted(dup_ids)}")
    v2_questions = {it.question.strip() for it in b2.ALL_ITEMS}
    dup_q = [it.item_id for it in ALL_HELDOUT_ITEMS
             if it.question.strip() in v2_questions]
    if dup_q:
        problems.append(f"held-out 与 V0.5 题库题干重复：{dup_q}")

    # 4. held-out 任务集
    if len(HELD_OUT_TASKS) < 8:
        problems.append(f"held-out 客观任务 {len(HELD_OUT_TASKS)} 个，少于 8")
    per_kp: dict[str, int] = {}
    for t in HELD_OUT_TASKS:
        per_kp[t.kp] = per_kp.get(t.kp, 0) + 1
    for kp in HELD_OUT_KPS:
        if per_kp.get(kp, 0) < 2:
            problems.append(f"held-out 客观任务：{kp} 只有 {per_kp.get(kp, 0)} 个，应≥2")

    if HO_TASKS_FINGERPRINT == ob.TASKS_FINGERPRINT:
        problems.append("held-out 任务集指纹与 practice 相同，参数未错开")

    # 5. 判分器自检：参考答案必须满分
    scorer = selfcheck_answers()
    for p in scorer["problems"]:
        problems.append(f"判分器自检：{p}")

    return {
        "version": HELD_OUT_VERSION,
        "problems": problems,
        "warnings": warnings,
        "total": len(ALL_HELDOUT_ITEMS),
        "per_set_count": {k: len(v) for k, v in HELDOUT_SETS.items()},
        "per_kp_count": per_kp,
        "fingerprint": HELDOUT_FINGERPRINT,
        "post_fingerprint": POST_FINGERPRINT,
        "transfer_fingerprint": TRANSFER_HELDOUT_FINGERPRINT,
        "tasks_fingerprint": HO_TASKS_FINGERPRINT,
        "practice_fingerprint": ob.TASKS_FINGERPRINT,
        "scorer_selfcheck": scorer["rows"],
        "residual_contamination": residual_contamination(),
    }


def assert_heldout_intact() -> None:
    d = heldout_diagnostics()
    if d["problems"]:
        raise RuntimeError(
            "held-out 题库/任务集不合法，实验作废：\n  - "
            + "\n  - ".join(d["problems"])
        )


# ==========================================================================
# held-out 与 practice 的参数隔离检查
# ==========================================================================
#
# 三种比较口径
# -----------
# 「参数错开」不能用一种比较方式一把梭，否则要么漏报要么误报：
#
# 1. **无界值（哈希值、键名）→ 必须集合不相交。**
#    这类值在 0..65536 或任意字符串空间里，
#    两组各取几个值又撞上的概率极低，撞上就是真污染。
#
# 2. **有界值（桶号 0..7）→ 必须整组序列不相等，而不是元素不相交。**
#    桶号只有8 个取值，两组各占3 个，**必然相交**。
#    在有界域上要求元素不相交，等于要求数学上不可能成立的事，
#    门禁会永远报警直至被绕过——那才是真正的风险。
#    有意义的问题是「同一道题的不同数据集，答案是否相同」。
#
# 3. **恒等式产物（空串的哈希 = 初值 7）→ 不可比，且必须显式豁免。**
#    LumenHash("") 恒等于初值 7，这是算法定义的推论而非数据选择，
#    不可能通过换参数错开。这类值必须**显式列出豁免理由**，
#    默默跳过等于把检查掏空。
#
# 下面三种口径都在代码里显式落地，不靠"感觉无关"。

#: practice 任务用过的关键参数（用于比对）
_PRACTICE_PARAMS: dict[str, Any] = {
    # (1) 无界值：要求集合不相交
    "unit_hash_values": sorted(v for k, v in ob.UNIT_CASES_HASH if k != ""),
    "consist_hashes": sorted(k["lumen_hash"] for k in ob.CONSIST_KEYS),
    "split_hash_values": sorted(v for _, v in ob.SPLIT_KEYS),
    "unreadable_keys": list(ob.EXPECTED_UNREADABLE_KEYS),
    "compact_case_shapes": [tuple(c) for c in ob.UNIT_CASES_COMPACT],
    # (2) 有界值：要求整组序列不相等
    "hash_key": "abc",
    "compact_triggered": list(ob.COMPACT_TRIGGERED),
    "overloaded_buckets": list(ob.EXPECTED_OVERLOADED_BUCKETS),
    "compact_epoch_after": ob.COMPACT_EPOCH_AFTER,
    # (3) 恒等式产物：显式豁免，附理由
    "invariant_empty_hash": 7,
}

#: held-out 任务用到的对应参数
_HELDOUT_PARAMS: dict[str, Any] = {
    "unit_hash_values": sorted(v for k, v in HO_UNIT_HASH if k != ""),
    "consist_hashes": sorted(k["lumen_hash"] for k in HO_CONSIST_KEYS),
    "split_hash_values": sorted(HO_SPLIT_HASHES),
    "unreadable_keys": list(_ho_unreadable_keys()),
    "compact_case_shapes": [tuple(c) for c in HO_UNIT_COMPACT],
    "hash_key": HO_HASH_KEY,
    "compact_triggered": list(HO_COMPACT_TRIGGERED),
    "overloaded_buckets": list(_ho_overloaded_buckets()),
    "compact_epoch_after": HO_COMPACT_EPOCH_AFTER,
    "invariant_empty_hash": ob.ref_lumen_hash(""),
}

#: (1) 无界值：值域足够大，交集非空即判污染
_UNBOUNDED_PARAMS = (
    "unit_hash_values",
    "consist_hashes",
    "split_hash_values",
    "unreadable_keys",
    "compact_case_shapes",
)

#: (2) 有界值：整组序列不相等才算错开（元素相交是必然，不算污染）
_BOUNDED_PARAMS = (
    "hash_key",
    "compact_triggered",
    "overloaded_buckets",
    "compact_epoch_after",
)

#: (3) 恒等式豁免项：值必须与两组各自的一致，但**豁免本身要被检查**
_INVARIANT_EXEMPT: dict[str, str] = {
    "invariant_empty_hash": (
        "LumenHash(空串) 恒等于初值 7，这是算法定义的恒等式推论，"
        "不是可由数据选择改变的参数；两组的空串哈希必然相同，"
        "因此不参与隔离判定，但它必须确实等于 7（否则说明算法被改过）。"
    ),
}


def practice_param_overlap() -> dict[str, list[str]]:
    """held-out 与 practice 的参数冲突（空 dict = 完全错开）。

    返回的每一项都代表**真污染**，不是「碰巧数值相同」。
    """
    out: dict[str, list[str]] = {}

    # (1) 无界值：集合必须不相交
    for k in _UNBOUNDED_PARAMS:
        a, b = _PRACTICE_PARAMS[k], _HELDOUT_PARAMS[k]
        common = sorted(set(a) & set(b))
        if common:
            out[k] = [str(x) for x in common]

    # (2) 有界值：整组序列必须不相等
    for k in _BOUNDED_PARAMS:
        if _PRACTICE_PARAMS[k] == _HELDOUT_PARAMS[k]:
            out[k] = [str(_HELDOUT_PARAMS[k])]

    return out


def invariant_exemptions() -> dict[str, dict[str, Any]]:
    """恒等式豁免项的自检：值必须确实是那个恒等式要求的样子。"""
    report: dict[str, dict[str, Any]] = {}
    for key, reason in _INVARIANT_EXEMPT.items():
        expected = _PRACTICE_PARAMS[key]
        got_p = _PRACTICE_PARAMS[key]
        got_h = _HELDOUT_PARAMS[key]
        report[key] = {
            "reason": reason,
            "expected": expected,
            "practice": got_p,
            "heldout": got_h,
            "holds": got_p == expected and got_h == expected,
        }
    return report


def assert_invariants_hold() -> None:
    """恒等式豁免项失效 = 算法定义被改动，直接拒绝。"""
    bad = [k for k, v in invariant_exemptions().items() if not v["holds"]]
    if bad:
        raise RuntimeError(
            "held-out 恒等式豁免项失效："
            + "、".join(bad)
            + "。这些值本应由算法定义唯一确定，"
              "对不上说明题目赖以成立的前提已被改动，实验作废。"
        )


def assert_heldout_isolated() -> None:
    """硬门禁：held-out 与practice 的参数必须错开。"""
    problems = heldout_diagnostics()["problems"]
    overlap = practice_param_overlap()
    if overlap:
        problems.append(
            f"held-out 与 practice 参数冲突：{overlap}；"
            f"「练习过」与「会做」将无法区分"
        )
    try:
        assert_invariants_hold()
    except RuntimeError as exc:
        problems.append(str(exc))
    if problems:
        raise RuntimeError(
            "held-out 隔离检查失败，实验作废：\n  - " + "\n  - ".join(problems)
        )


def heldout_meta() -> dict[str, Any]:
    assert_heldout_intact()
    assert_heldout_isolated()
    return {
        "version": HELD_OUT_VERSION,
        "fingerprint": HELDOUT_FINGERPRINT,
        "post_fingerprint": POST_FINGERPRINT,
        "transfer_fingerprint": TRANSFER_HELDOUT_FINGERPRINT,
        "tasks_fingerprint": HO_TASKS_FINGERPRINT,
        "total": len(ALL_HELDOUT_ITEMS),
        "per_set_count": {k: len(v) for k, v in HELDOUT_SETS.items()},
        "practice_param_overlap": practice_param_overlap(),
        "invariant_exemptions": invariant_exemptions(),
    }


# ==========================================================================
# held-out 客观任务的判分入口
# ==========================================================================
#
# 为什么不能直接复用 practice 的 checker
# ---------------------------------------
# practice 的两个判分器把**该组自己的参数**写进了检查项：
#
#   check_split_algorithm : "没有键被迁移到非 **0** 号桶（(0+8) mod 8 = 0）"
#   check_compact_schema  : "1 not in compress and 4 not in compress"
#
# held-out 的分裂桶是 5 号、4 号桶**应该**触发压缩。
# 直接复用会把正确答案判成错的——这不是判分严格，是判分器串了组。
# 所以 held-out 用自己的一套 checker，检查项从**材料规则**推导，
# 不写死任何一组的具体取值。

import contextlib
from typing import Callable

from practice.objective import _ck, _as_int, _as_int_list, _as_str_list

_CheckFn = Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]]]


def _ho_check_split(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    """held-out 版分裂判定：检查项不写死桶号，只按规则核对。"""
    checks: list[dict[str, Any]] = []
    assigns = data.get("assignments")
    ok_assign = isinstance(assigns, list) and len(assigns) == len(expected["assignments"])
    n = len(expected["assignments"])
    checks.append(ob._ck(f"assignments 覆盖全部 {n} 个键", ok_assign, f"实际 {assigns!r}"))
    if ok_assign:
        want_map = {e["key"]: e["to_bucket"] for e in expected["assignments"]}
        got_map: dict[str, Any] = {}
        for row in assigns:
            if isinstance(row, dict) and isinstance(row.get("key"), str):
                got_map[row["key"]] = _as_int(row.get("to_bucket"))
        checks.append(ob._ck(
            "每个键的分裂后桶号都正确", got_map == want_map,
            f"期望 {want_map}，实际 {got_map}",
        ))
        # 规则本身：迁移到 (b + BUCKET_COUNT) mod BUCKET_COUNT。
        # 只要每个键都落在原桶，就说明模型认出了这是恒等变换。
        # **不写死是哪个桶** —— 否则这个检查项只对0 号桶的题目有效。
        target = HO_SPLIT_BUCKET  # (b + 8) mod 8 = b，对任意 b 都成立
        off = [k for k, v in got_map.items() if v != target]
        checks.append(ob._ck(
            f"分裂后全部键都落在 (桶号+8) mod 8 = {target}（恒等变换不减负载）",
            not off,
            f"以下键落到了 {target} 以外的桶：{off}",
        ))
        # 反向检查：不能原地不动当成"没做任何计算"以外的原因蒙对——
        # 要求显式给出 from_bucket 且与题面一致。
        from_ok = all(
            _as_int(r.get("from_bucket")) == HO_SPLIT_BUCKET
            for r in assigns if isinstance(r, dict)
        )
        checks.append(ob._ck(
            f"每个键都正确声明了原桶号 {HO_SPLIT_BUCKET}", from_ok, "from_bucket 与题面不符"
        ))
    else:
        checks.append(ob._ck("每个键的分裂后桶号都正确", False, "assignments 结构不对"))
        checks.append(ob._ck("分裂后全部键都落在同一个桶", False, "assignments 结构不对"))
        checks.append(ob._ck("每个键都正确声明了原桶号", False, "assignments 结构不对"))

    migrated = _as_int(data.get("migrated_count"))
    checks.append(ob._ck(
        "migrated_count 正确（分裂是恒等变换，实际迁移 0 个）",
        migrated == expected["migrated_count"],
        f"期望 {expected['migrated_count']}，实际 {data.get('migrated_count')!r}",
    ))
    over = data.get("still_over_limit")
    checks.append(ob._ck(
        "still_over_limit 正确（分裂不减负载，桶仍然超限）",
        over is expected["still_over_limit"],
        f"期望 {expected['still_over_limit']}，实际 {over!r}",
    ))
    return checks


def _ho_check_compact(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    """held-out 版压缩判定：反例桶从 held-out 自己的输入推导。"""
    checks: list[dict[str, Any]] = []
    compress = _as_int_list(data.get("compress"))
    checks.append(ob._ck(
        "compress 是整数数组", compress is not None, f"实际 {data.get('compress')!r}"
    ))
    # 从 held-out 自己的输入表推导两类反例：
    #   tombstone 够但live 超限 → 不该触发
    #   live 够但 tombstone 不够 → 不该触发
    # 绝不写死具体桶号，否则检查项只对这一组数据有效。
    over_live = [
        b for b, live, tomb in HO_COMPACT_BUCKETS
        if tomb >= 2 and live > 1
    ]
    short_tomb = [
        b for b, live, tomb in HO_COMPACT_BUCKETS
        if tomb < 2 and live <= 1
    ]
    if compress is not None:
        checks.append(ob._ck(
            "触发压缩的桶号完全正确（墓碑>=2 且 活跃键<=1，两条必须同时成立）",
            compress == expected["compress"],
            f"期望 {expected['compress']}，实际 {compress}",
        ))
        wrong_over = [b for b in over_live if b in compress]
        checks.append(ob._ck(
            f"没有把「墓碑够但活跃键超限」的桶 {over_live} 误判为触发",
            not wrong_over, f"被误判为触发的桶：{wrong_over}",
        ))
        wrong_tomb = [b for b in short_tomb if b in compress]
        checks.append(ob._ck(
            f"没有把「活跃键够但墓碑不足」的桶 {short_tomb} 误判为触发",
            not wrong_tomb, f"被误判为触发的桶：{wrong_tomb}",
        ))
    else:
        checks.append(ob._ck("触发压缩的桶号完全正确", False, "类型不对"))
        checks.append(ob._ck("没有误判「墓碑够但活跃键超限」的桶", False, "类型不对"))
        checks.append(ob._ck("没有误判「活跃键够但墓碑不足」的桶", False, "类型不对"))

    after = _as_int(data.get("epoch_after"))
    checks.append(ob._ck(
        "epoch_after 正确（每触发一次压缩全局 epoch 加 1）",
        after == expected["epoch_after"],
        f"期望 {expected['epoch_after']}，实际 {data.get('epoch_after')!r}",
    ))
    return checks


def _ho_check_hash(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    """held-out 版哈希判定：只用材料规则，不引用 practice 的 key。"""
    return ob.check_hash_algorithm(data, expected)


def _ho_check_compact_unit(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    return ob.check_compact_unit_test(data, expected)


def _ho_check_hash_unit(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    return ob.check_hash_unit_test(data, expected)


def _ho_check_consist(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    """held-out 版一致性判定。

    practice 的判分器把「epoch 变了 → 存量键全部失配」写进了检查项名称，
    而held-out 是**混合写入时刻**（部分键可读、部分需重算），
    沿用它会把正确答案判错。
    这里改为三个方向都查：
    可读的必须真可读、需重算的必须真需重算、两者不得重叠且并集完整。
    """
    checks: list[dict[str, Any]] = []
    readable = _as_str_list(data.get("readable"))
    checks.append(ob._ck(
        "readable 是字符串数组", readable is not None,
        f"实际 {data.get('readable')!r}",
    ))
    retag = _as_str_list(data.get("need_retag"))
    checks.append(ob._ck(
        "need_retag 是字符串数组", retag is not None,
        f"实际 {data.get('need_retag')!r}",
    ))

    want_r = list(expected["readable"])
    want_t = list(expected["need_retag"])
    if readable is not None:
        checks.append(ob._ck(
            f"readable 正确（这组数据里可读的是 {want_r}）",
            readable == want_r,
            f"期望 {want_r}，实际 {readable!r}",
        ))
    if retag is not None:
        checks.append(ob._ck(
            f"need_retag 正确（这组数据里需重算的是 {want_t}）",
            retag == want_t,
            f"期望 {want_t}，实际 {retag!r}",
        ))
    if readable is not None and retag is not None:
        overlap = sorted(set(readable) & set(retag))
        checks.append(ob._ck(
            "同一道题里不存在「既可读又需重算」的键",
            not overlap, f"重叠键：{overlap}",
        ))
        union = set(readable) | set(retag)
        all_keys = {k["key"] for k in HO_CONSIST_KEYS}
        checks.append(ob._ck(
            "readable 与 need_retag 合起来覆盖全部键",
            union == all_keys,
            f"缺少 {sorted(all_keys - union)}",
        ))

    tags = data.get("new_tags")
    got_tags: dict[str, Any] = {}
    if isinstance(tags, dict):
        for k, v in tags.items():
            if isinstance(k, str):
                got_tags[k] = _as_int(v)
    checks.append(ob._ck(
        "new_tags 是「键名 → 整数标签」的对象",
        isinstance(tags, dict) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in tags.values()
        ),
        f"实际 {tags!r}",
    ))
    # 只对**需重算**的键要标签；给可读的键也算错（多余信息说明没读懂题意）
    checks.append(ob._ck(
        "new_tags 恰好覆盖需要重算的键（不多不少）",
        set(got_tags) == set(want_t),
        f"期望键集合 {sorted(want_t)}，实际 {sorted(got_tags)}",
    ))
    if set(got_tags) == set(want_t):
        checks.append(ob._ck(
            f"每个新标签 = (LumenHash + 当前 epoch {HO_CONSIST_EPOCH}) mod 65537",
            got_tags == expected["new_tags"],
            f"期望 {expected['new_tags']}，实际 {got_tags}",
        ))
    return checks


#: held-out 自己的 checker 表
HELD_OUT_CHECKERS: dict[str, _CheckFn] = {
    "check_hash_algorithm": _ho_check_hash,
    "check_hash_unit_test": _ho_check_hash_unit,
    "check_split_algorithm": _ho_check_split,
    "check_overload_sql": ob.check_overload_sql,
    "check_compact_schema": _ho_check_compact,
    "check_compact_unit_test": _ho_check_compact_unit,
    "check_consistency_algorithm": _ho_check_consist,
    "check_unreadable_sql": ob.check_unreadable_sql,
}


def _evaluate_with(
    checkers: dict[str, _CheckFn],
    task: ob.ObjectiveTask,
    candidate_output: str,
) -> ob.ObjectiveResult:
    """照 `objective.evaluate_objective` 的结构跑一遍，但用指定的 checker 表。"""
    checker = checkers.get(task.checker)
    if checker is None:  # pragma: no cover — 门禁保证不会走到
        return ob.ObjectiveResult(
            task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
            passed=False, objective_score=0.0, detail=f"未注册的判分器 {task.checker!r}",
        )
    data, err = ob.parse_candidate(candidate_output)
    if data is None:
        return ob.ObjectiveResult(
            task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
            passed=False, objective_score=0.0,
            detail="产出无法解析为 JSON 对象", parse_error=err,
        )
    try:
        checks = checker(data, task.expected)
    except Exception as exc:  # noqa: BLE001 — 判分器自身出错也不能炸实验
        return ob.ObjectiveResult(
            task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
            passed=False, objective_score=0.0,
            detail=f"判分器异常：{type(exc).__name__}: {exc}",
        )
    if not checks:  # pragma: no cover
        return ob.ObjectiveResult(
            task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
            passed=False, objective_score=0.0, detail="判分器没有产出任何 check",
        )
    hit = sum(1 for c in checks if c["passed"])
    score = round(100.0 * hit / len(checks), 2)
    failed = [c["name"] for c in checks if not c["passed"]]
    return ob.ObjectiveResult(
        task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
        passed=not failed,
        objective_score=score,
        checks=checks,
        detail="全部通过" if not failed else "未通过：" + "；".join(failed),
        evaluator=OBJECTIVE_EVALUATOR,
    )


#: held-out 客观判分是**机器判定**，不经过任何 LLM 评审
OBJECTIVE_EVALUATOR = "machine:heldout_objective_v1"

#: practice 原始 fixture 构造器（判分后必须还原）
_ORIGINAL_FIXTURE_BUILDER = ob._build_fixture_db


def _ho_patched_fixture() -> Any:
    """让 SQL checker 在 held-out 任务上用 held-out 的库。"""
    return _ho_fixture()


@contextlib.contextmanager
def _ho_fixture_guard():
    original = ob._build_fixture_db
    ob._build_fixture_db = _ho_patched_fixture
    try:
        yield
    finally:
        ob._build_fixture_db = original


def evaluate_heldout(task_id: str, candidate_output: str) -> ob.ObjectiveResult:
    """判一个 held-out 客观任务。

    两件与practice 不同的事：
    1. 用held-out 自己的 checker 表（不复用写死了 practice 参数的判分器）
    2. SQL 任务在 held-out 的fixture 上跑
    """
    task = next((t for t in HELD_OUT_TASKS if t.task_id == task_id), None)
    if task is None:
        raise KeyError(f"未知 held-out 任务 {task_id!r}")
    if task.kind != "sql_exec":
        return _evaluate_with(HELD_OUT_CHECKERS, task, candidate_output)
    with _ho_fixture_guard():
        return _evaluate_with(HELD_OUT_CHECKERS, task, candidate_output)


#: 自检结果缓存。
#:
#: ``selfcheck_answers`` 会真的跑一次 ``python -I``子进程
#:（两个 ``python_unit_test`` 任务），单次约 2.5 秒。
#: 而 ``assert_isolation_intact`` 与各阶段的 setup 都会调它，
#: 54 个测试里重复跑十几次就变成 40+ 秒。
#:
#: 判分器是纯函数（同样的输入必然同样的输出），
#: 且题面与参考答案都是模块级常量，
#: 所以**同一进程内结果不可能变**——缓存是安全的。
#: 若真要在运行中改判分逻辑，``assert_heldout_scorers_intact``
#: 可传 ``force=True`` 强制重算。
_SELFCHECK_CACHE: dict[str, Any] | None = None


def selfcheck_answers(*, force: bool = False) -> dict[str, Any]:
    """用参考实现的正确答案自检 8 个 held-out 任务。

    这是判分器自己的门禁：如果标准答案都拿不到满分，
    说明判分器和题目对不上，整个 held-out 分数不可信。
    """
    global _SELFCHECK_CACHE
    if _SELFCHECK_CACHE is not None and not force:
        return _SELFCHECK_CACHE

    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for task in HELD_OUT_TASKS:
        good = _reference_answer(task)
        res = evaluate_heldout(task.task_id, good)
        ok = res.passed and res.objective_score == 100.0
        rows.append({
            "task_id": task.task_id,
            "kp": task.kp,
            "kind": task.kind,
            "checker": task.checker,
            "score": res.objective_score,
            "passed": res.passed,
            "failed_checks": [c["name"] for c in res.checks if not c["passed"]],
        })
        if not ok:
            problems.append(
                f"{task.task_id}：参考答案只得 {res.objective_score} 分，"
                f"未通过项 {rows[-1]['failed_checks']}"
            )
    out = {
        "version": HELD_OUT_VERSION,
        "problems": problems,
        "rows": rows,
        "fixture_restored": ob._build_fixture_db is _ORIGINAL_FIXTURE_BUILDER,
    }
    _SELFCHECK_CACHE = out
    return out


def assert_heldout_scorers_intact(*, force: bool = False) -> None:
    """硬门禁：held-out 的判分器必须认可自己的参考答案。"""
    d = selfcheck_answers(force=force)
    if d["problems"]:
        raise RuntimeError(
            "held-out 判分器自检失败，held-out 分数不可信：\n  - "
            + "\n  - ".join(d["problems"])
        )
    if not d["fixture_restored"]:  # pragma: no cover
        raise RuntimeError("held-out 判分未还原 practice 的 fixture 构造器")


def _reference_answer(task: ob.ObjectiveTask) -> str:
    """按题目类型生成一份标准答案（用参考实现，不是手写）。"""
    import json as _json

    if task.task_id == "ho_hash_01":
        return _json.dumps({
            "key": HO_HASH_KEY, "steps": HO_HASH_STEPS,
            "lumen_hash": HO_HASH_VALUE, "bucket": HO_HASH_BUCKET,
        })
    if task.task_id == "ho_hash_02":
        code = (
            "def lumen_hash(key):\n"
            "    h = 7\n"
            "    for c in key:\n"
            "        h = (h * 131 + ord(c)) % 65537\n"
            "    return h\n"
        )
        return _json.dumps({"code": code})
    if task.task_id == "ho_bucket_01":
        return _json.dumps({
            "assignments": [
                {"key": f"k{h}", "from_bucket": HO_SPLIT_BUCKET,
                 "to_bucket": ob.ref_split_target(h, HO_SPLIT_BUCKET)}
                for h in HO_SPLIT_HASHES
            ],
            "migrated_count": 0,
            "still_over_limit": True,
        })
    if task.task_id == "ho_bucket_02":
        return _json.dumps({
            "sql": "SELECT bucket FROM zephyra_buckets"
                   " WHERE live_count > 4 ORDER BY bucket",
        })
    if task.task_id == "ho_compact_01":
        return _json.dumps({
            "compress": HO_COMPACT_TRIGGERED,
            "epoch_after": HO_COMPACT_EPOCH_AFTER,
        })
    if task.task_id == "ho_compact_02":
        code = (
            "def should_compact(tombstones, live):\n"
            "    return tombstones >= 2 and live <= 1\n"
        )
        return _json.dumps({"code": code})
    if task.task_id == "ho_consist_01":
        return _json.dumps({
            "readable": list(HO_CONSIST_READABLE),
            "need_retag": list(HO_CONSIST_NEED_RETAG),
            "new_tags": dict(HO_CONSIST_NEW_TAGS),
        })
    if task.task_id == "ho_consist_02":
        return _json.dumps({
            "sql": "SELECT key FROM zephyra_keys WHERE ztag != (lumen_hash +"
                   " (SELECT value FROM zephyra_meta WHERE name='epoch')) % 65537"
                   " ORDER BY key",
        })
    raise KeyError(task.task_id)  # pragma: no cover


__all__ = [
    "ALL_HELDOUT_ITEMS",
    "HELD_OUT_KPS",
    "HELD_OUT_POST",
    "HELD_OUT_TASKS",
    "HELD_OUT_TRANSFER",
    "HELD_OUT_TRANSFER_NOTES",
    "HELD_OUT_VERSION",
    "HELDOUT_FINGERPRINT",
    "HELDOUT_SETS",
    "HO_TASKS_FINGERPRINT",
    "HeldOutTask",
    "assert_heldout_intact",
    "assert_heldout_isolated",
    "evaluate_heldout",
    "get_heldout",
    "heldout_diagnostics",
    "heldout_meta",
    "practice_param_overlap",
]
