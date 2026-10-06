"""practice/objective.py —— V0.5 客观实践任务：判分不靠 LLM 说了算。

V0.4.1 暴露的第三个统计缺陷
-----------------------------
V0.4 的实践分由 `PracticeJudge`（LLM）给。LLM 判分有三个硬伤：

1. **自评虚高**：同一个 provider 既写产出又判产出，天然偏高。
2. **不可重放**：判分结论没有一个可以事后重新执行的「事实」；
   模型说「我算出 1858」，除了相信它，没法验证。
3. **阈值任意**：`USABLE_LINE = 55` 这个分数线没有客观依据，
   调高调低都能让结论翻转。

V0.5 的解法：把实践产出交给**机器**判
-------------------------------------
每个任务都要求模型提交一份**结构化产出**，判分器用确定性方式核对：

    algorithm_output  与预先算好的期望值逐字段比对
    json_schema       结构、类型、枚举、必填项全部机器校验
    sql_exec真的把 SQL 丢进 sqlite 跑，比对结果集
    python_unit_test  真的把模型写的函数抽出来执行，喂多组用例断言

`python_unit_test` 是这里最硬的一类：模型必须真的实现算法，
因为测试用例包含它没见过的键。**写死答案骗不过去**——
这正是「机器可验证」相对「LLM 自评」的本质差别。

安全边界（必须说清楚）
----------------------
`sql_exec` 与 `python_unit_test` **会真的执行模型产出的代码**。
这是本地一次性实验环境，不是对外服务；执行走
`python -I`（隔离模式，不读 PYTHONPATH 与用户 site）、独立临时目录、
15 秒超时、最小环境变量。

即便如此，执行任意代码仍有风险，因此本模块**不允许**候选 SQL
包含写操作（DROP/DELETE/UPDATE/INSERT/ATTACH/PRAGMA 写），
只接受只读查询；违规本身就是一个 objective check 的失败项。
这不只是安全考虑——「只读」正是这道题要考的能力。

期望值为什么必须硬编码
----------------------
期望值写死成常量，**不在运行时从参考实现现算**。
如果期望值随参考实现一起变，改判分逻辑就会自动改答案，
门禁形同虚设。改期望值必须在 diff 里看得见。

指纹把判分逻辑本身也算了进去
----------------------------
`_fingerprint` 会把每个 checker 函数的**源码**（`inspect.getsource`）
一起哈希。改了判分逻辑却没改期望值 → 指纹变化 → 实验作废。
V0.4.1 踩过「改了题面指纹没变」的坑，这里把闸门扩大。
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from memory.db import Database

# ==========================================================================
# Zephyra 参考实现
#
# 用途仅限：生成候选答案时对答案、跑预置 fixture。
# **不参与判分**——判分只看模型产出与硬编码期望值是否一致。
# ==========================================================================


def ref_lumen_hash(key: str) -> int:
    """LumenHash：h=7 起，逐字符 h=(h*131+ord(c)) mod 65537。"""
    h = 7
    for ch in key:
        h = (h * 131 + ord(ch)) % 65537
    return h


def ref_bucket(key: str) -> int:
    return ref_lumen_hash(key) % 8


def ref_split_target(lumen_hash: int, bucket: int) -> int:
    """一级分桶的分裂：mod 16 >= 8 则迁到 (b+8) mod 8。"""
    if lumen_hash % 16 >= 8:
        return (bucket + 8) % 8
    return bucket


def ref_should_compact(tombstones: int, live: int) -> bool:
    return tombstones >= 2 and live <= 1


def ref_ztag(lumen_hash: int, epoch: int) -> int:
    return (lumen_hash + epoch) % 65537


def ref_read_ok(stored_ztag: int, lumen_hash: int, epoch: int) -> bool:
    return stored_ztag == ref_ztag(lumen_hash, epoch)


# ==========================================================================
# 预置数据库（sql_exec 任务用）
#
# fixture 由上面的参考实现生成，但**写死成常量**：
# 数据集变了就必须在 diff 里看得见，不能悄悄跟着参考实现漂移。
# ==========================================================================

FIXTURE_SCHEMA = """
CREATE TABLE zephyra_meta (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE zephyra_buckets (
    bucket          INTEGER PRIMARY KEY,
    live_count      INTEGER NOT NULL,
    tombstone_count INTEGER NOT NULL
);
CREATE TABLE zephyra_keys (
    key             TEXT PRIMARY KEY,
    lumen_hash      INTEGER NOT NULL,
    bucket          INTEGER NOT NULL,
    ztag            INTEGER NOT NULL,
    epoch_at_write  INTEGER NOT NULL
);
"""

#: epoch = 2（已被压缩过两次）
FIXTURE_META = [("epoch", 2), ("bucket_count", 8), ("live_limit", 4)]

#: 每个桶的活跃键数：[2, 3, 4, 5, 6, 1, 0, 7] → 超载（>4）的桶是 3、4、7
FIXTURE_BUCKETS = [
    (0, 2, 0), (1, 3, 1), (2, 4, 2), (3, 5, 1),
    (4, 6, 3), (5, 1, 4), (6, 0, 2), (7, 7, 0),
]

#: 12 个键，epoch_at_write 分布：0×8、1×2、2×2
#: 在epoch=2 下读取被拒的是 epoch_at_write != 2 的那 10 个。
#: **bucket 列必须等于 lumen_hash mod 8**（材料的第一条分桶规则），
#: 这里逐行按参考实现算过——fixture 自己违反材料规则的话，
#: obj_consist_02 就是在考一个不存在的世界。
FIXTURE_KEYS: tuple[tuple[str, int, int, int, int], ...] = (
    ("k01", 40000, 0, 40000, 0),
    ("k02", 40001, 1, 40001, 0),
    ("k03", 40002, 2, 40002, 0),
    ("k04", 40003, 3, 40003, 0),
    ("k05", 40004, 4, 40004, 0),
    ("k06", 40005, 5, 40005, 0),
    ("k07", 40006, 6, 40006, 0),
    ("k08", 40007, 7, 40007, 0),
    ("k09", 30000, 0, 30001, 1),
    ("k10", 30001, 1, 30002, 1),
    ("k11", 20000, 0, 20002, 2),
    ("k12", 20001, 1, 20003, 2),
)

#: 超载桶号（live_count > 4）
EXPECTED_OVERLOADED_BUCKETS = [3, 4, 7]

#: 在 epoch=2 下读取被拒的键（stored ztag != (h + 2) mod 65537）
EXPECTED_UNREADABLE_KEYS = [
    "k01", "k02", "k03", "k04", "k05", "k06", "k07", "k08", "k09", "k10",
]

#: LumenHash 单元测试用例（模型没见过这些键，写死答案必挂）
UNIT_CASES_HASH = [("a", 1014), ("ab", 1858), ("abc", 46886), ("", 7), ("user", 14991)]

#: should_compact(tombstones, live) 的期望真值表
UNIT_CASES_COMPACT = [
    (2, 1, True),    # A、B 同时成立
    (3, 2, False),   # B 不成立
    (1, 1, False),   # A 不成立
    (2, 0, True),    # A、B 同时成立
    (5, 1, True),
    (2, 2, False),
]

#: LumenHash("abc") 的逐步中间值
HASH_ABC_STEPS = [7, 1014, 1858, 46886]
HASH_ABC_VALUE = 46886
#: 46886 mod 8 = 6。**不是 2**——8 × 5860 = 46880，余 6。
#: 这个数 Originally 写成 2，是靠参考实现验算才发现的：
#: 心算「46886 = 8 × 5860 + 2」时把余数算错了。
#: 期望值不验算就等于没设——它只是看起来像真的。
HASH_ABC_BUCKET = 6

#: 五个键分裂后的归属。
#: LumenHash 全部取 8 的倍数，保证「这 5 个键在同一个桶（0 号）」这个前提成立。
#: 最初取的是 0/1/8/9/16，其中 1 和 9 落在 1 号桶，
#: 也就是**前提不成立**——和 V0.4.1 的 mem_bucket_01 一模一样的坑，
#: 在实践任务里又踩了一次。前提不成立的任务必须改题，不能靠答案打补丁。
SPLIT_KEYS = [
    ("k0", 0), ("k8", 8), ("k16", 16), ("k24", 24), ("k32", 32),
]
SPLIT_ASSIGNMENT = [
    {"key": name, "from_bucket": 0, "to_bucket": 0} for name, _ in SPLIT_KEYS
]
SPLIT_MIGRATED_COUNT = 0
SPLIT_STILL_OVER_LIMIT = True

#: 压缩判定：5 个桶，条件 A、B 独立成立情况各异
COMPACT_INPUT = [
    {"bucket": 0, "live": 1, "tombstones": 2},   # 触发
    {"bucket": 1, "live": 2, "tombstones": 3},   # 不触发：B 不成立
    {"bucket": 2, "live": 1, "tombstones": 1},   # 不触发：A 不成立
    {"bucket": 3, "live": 0, "tombstones": 4},   # 触发
    {"bucket": 4, "live": 3, "tombstones": 9},   # 不触发：B 不成立
]
COMPACT_TRIGGERED = [0, 3]
COMPACT_EPOCH_BEFORE = 1
#: 两个桶触发压缩 → epoch 1 → 3。最初写成 2（只加了一次），
#: 同样是靠参考实现验算抓出来的。
COMPACT_EPOCH_AFTER = 3

#: epoch 一致性：4 个键，epoch=0 写入，其间发生 2 次压缩
CONSIST_KEYS = [
    {"key": "x1", "lumen_hash": 2500, "written_epoch": 0},
    {"key": "x2", "lumen_hash": 4000, "written_epoch": 0},
    {"key": "x3", "lumen_hash": 6000, "written_epoch": 0},
    {"key": "x4", "lumen_hash": 8200, "written_epoch": 0},
]
CONSIST_COMPRESSIONS = 2
CONSIST_CURRENT_EPOCH = 2
CONSIST_READABLE = []
CONSIST_NEED_RETAG = ["x1", "x2", "x3", "x4"]
CONSIST_EXPECTED_TAGS = {"x1": 2502, "x2": 4002, "x3": 6002, "x4": 8202}


# ==========================================================================
# 任务定义
# ==========================================================================

OBJECTIVE_SYSTEM = (
    "你是练习者。题目要求你提交一份**结构化产出**（JSON）。"
    "规则以学习材料为准，不要用现实世界里的同名概念替代。"
    "只输出 JSON，不要输出解释文字，也不要用``` 包裹。"
)

KINDS = ("algorithm_output", "json_schema", "sql_exec", "python_unit_test")


@dataclass(frozen=True)
class ObjectiveTask:
    """一个机器可验证的实践任务。

    **`expected` 与 `submit_fields` 是两件事，不能混用。**
    `expected` 存**答案**（单测用例、SQL 期望结果集），
    `submit_fields` 存**模型要提交的字段名**。
    两者结构往往不同：单测任务提交 `{"code": ...}`，
    而 expected 是 `{"cases": [[键, 值], ...]}`。
    早先没区分这两者时，泄题/字段门禁把4 个任务全判成结构错误——
    门禁本身没错，错的是把「答案结构」当成了「提交格式」。

    `expected` 绝不进 prompt——参考答案进 prompt 就是泄题，
    泄了题机器判分就没有意义了（`task_diagnostics` 会拦）。
    """

    task_id: str
    kp: str
    title: str
    kind: str
    prompt: str
    checker: str            # CHECKERS 里的函数名
    expected: Any
    #: 模型提交 JSON 的顶层字段名，必须与 submit_hint 一致
    submit_fields: tuple[str, ...]
    max_score: float = 100.0
    #: 给考生看的提交格式说明（同样不许含答案）
    submit_hint: str = ""


OBJECTIVE_TASKS: tuple[ObjectiveTask, ...] = (
    # ---------------- kp_lumen_hash ----------------
    ObjectiveTask(
        task_id="obj_hash_01",
        kp="kp_lumen_hash",
        title="按材料算法算出指定键的哈希与桶号",
        kind="algorithm_output",
        prompt=(
            "请严格按学习材料里 LumenHash 的定义，逐步计算键 \"abc\" 的 "
            "LumenHash 值，并给出它所属的桶号。\n"
            "必须提交 JSON，包含四个字段：\n"
            "  key       —— 固定为字符串 \"abc\"\n"
            "  steps     —— 整数数组，h 的逐步取值序列，"
            "第一个元素是初值，最后一个元素是最终哈希值\n"
            "  lumen_hash —— 最终哈希值（整数）\n"
            "  bucket     —— 所属桶号（整数）"
        ),
        # 只给字段结构，不给数值。示例里带答案等于泄题，
        # 泄题门禁（task_diagnostics）会直接拒绝跑实验。
        # 必须是**合法 JSON**——否则它在教模型输出非法格式。
        submit_hint=(
            '{"key": "填入键名", "steps": [0, 0, 0, 0],'
            ' "lumen_hash": 0, "bucket": 0}'
        ),
        checker="check_hash_algorithm",
        submit_fields=("key", "steps", "lumen_hash", "bucket"),
        expected={
            "key": "abc",
            "steps": HASH_ABC_STEPS,
            "lumen_hash": HASH_ABC_VALUE,
            "bucket": HASH_ABC_BUCKET,
        },
    ),
    ObjectiveTask(
        task_id="obj_hash_02",
        kp="kp_lumen_hash",
        title="实现 LumenHash 函数并通过用例测试",
        kind="python_unit_test",
        prompt=(
            "请用 Python 实现学习材料里定义的 LumenHash 函数，"
            "函数签名必须是 `def lumen_hash(key):`，返回一个整数。\n"
            "提交 JSON，只含一个字段："
            '  code —— 完整的 Python 源码字符串\n'
            "评测方式：把你的代码放进测试环境，用多组键实际运行，"
            "比对返回值。注意空串也要能处理。"
        ),
        submit_hint='{"code":"def lumen_hash(key):\\n    h = 7\\n    for c in key:\\n        h = (h * 131 + ord(c)) % 65537\\n    return h"}',
        checker="check_hash_unit_test",
        submit_fields=("code",),
        expected={"cases": [list(c) for c in UNIT_CASES_HASH]},
    ),
    # ---------------- kp_lumen_bucket ----------------
    ObjectiveTask(
        task_id="obj_bucket_01",
        kp="kp_lumen_bucket",
        title="算出一级分桶分裂后每个键的桶号",
        kind="algorithm_output",
        prompt=(
            "某个桶里有 5 个键，它们的 LumenHash 值分别是 0、8、16、24、32"
            "（所以它们本来就在同一个桶里）。"
            "现在对0 号桶触发一次分裂。请按学习材料的分裂规则，"
            "算出这 5 个键分裂后各自的桶号，并判断这次分裂有没有减少负载。\n"
            "提交 JSON，包含三个字段：\n"
            "  assignments    —— 数组，每项 {\"key\":\"k0\",\"from_bucket\":0,\"to_bucket\":0}，"
            "五个键都要列出，key 名依次用 k0、k8、k16、k24、k32\n"
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
            "assignments": SPLIT_ASSIGNMENT,
            "migrated_count": SPLIT_MIGRATED_COUNT,
            "still_over_limit": SPLIT_STILL_OVER_LIMIT,
        },
    ),
    ObjectiveTask(
        task_id="obj_bucket_02",
        kp="kp_lumen_bucket",
        title="用 SQL 查出所有超载桶",
        kind="sql_exec",
        prompt=(
            "数据库里有两张表（只读，不要建表不要改数据）：\n"
            "  zephyra_buckets(bucket, live_count, tombstone_count)\n"
            "  zephyra_keys(key, lumen_hash, bucket, ztag, epoch_at_write)\n"
            "每个桶最多容纳 4 个活跃键。请写一条**只读** SQL，"
            "把所有活跃键数超过 4 的桶号查出来，按桶号升序。\n"
            "提交 JSON，只含一个字段："
            "  sql —— SQL 语句字符串"
        ),
        submit_hint='{"sql":"SELECT bucket FROM zephyra_buckets WHERE live_count > 4 ORDER BY bucket"}',
        checker="check_overload_sql",
        submit_fields=("sql",),
        expected={"rows": [[b] for b in EXPECTED_OVERLOADED_BUCKETS]},
    ),
    # ---------------- kp_lumen_compact ----------------
    ObjectiveTask(
        task_id="obj_compact_01",
        kp="kp_lumen_compact",
        title="逐桶判定压缩是否触发并给出压缩后的 epoch",
        kind="json_schema",
        prompt=(
            "下面是 5 个桶的当前状态（live = 活跃键数，tombstones = 墓碑数）：\n"
            "  桶 0：live=1, tombstones=2\n"
            "  桶 1：live=2, tombstones=3\n"
            "  桶 2：live=1, tombstones=1\n"
            "  桶 3：live=0, tombstones=4\n"
            "  桶 4：live=3, tombstones=9\n"
            "当前全局 epoch 是 1。请按学习材料的压缩触发条件，"
            "判断哪些桶会触发压缩，并给出执行完这些压缩之后的全局 epoch。\n"
            "提交 JSON，包含两个字段：\n"
            "  compress   —— 整数数组，列出所有触发压缩的桶号，按升序\n"
            "  epoch_after —— 整数，全部压缩完成后的 epoch"
        ),
        submit_hint='{"compress":[0,3],"epoch_after":2}',
        checker="check_compact_schema",
        submit_fields=("compress", "epoch_after"),
        expected={"compress": COMPACT_TRIGGERED, "epoch_after": COMPACT_EPOCH_AFTER},
    ),
    ObjectiveTask(
        task_id="obj_compact_02",
        kp="kp_lumen_compact",
        title="实现压缩触发判定函数并通过用例测试",
        kind="python_unit_test",
        prompt=(
            "请用 Python 实现一个函数判断某个桶是否会触发压缩，"
            "签名必须是 `def should_compact(tombstones, live):`，"
            "返回布尔值。判定条件以学习材料为准（注意是「且」还是「或」）。\n"
            "提交 JSON，只含一个字段："
            "  code —— 完整的 Python 源码字符串"
        ),
        submit_hint='{"code":"def should_compact(tombstones, live):\\n    return tombstones >= 2 and live <= 1"}',
        checker="check_compact_unit_test",
        submit_fields=("code",),
        expected={"cases": [list(c) for c in UNIT_CASES_COMPACT]},
    ),
    # ---------------- kp_lumen_consist ----------------
    ObjectiveTask(
        task_id="obj_consist_01",
        kp="kp_lumen_consist",
        title="算出 epoch 变化后哪些键还能读、标签该改成多少",
        kind="algorithm_output",
        prompt=(
            "4 个键 x1..x4 的 LumenHash 分别是 2500、4000、6000、8200，"
            "它们全部在 epoch=0 时写入（所以存下来的 Z-Tag 就等于各自的 LumenHash）。"
            "此后数据库发生了 2 次压缩，当前 epoch = 2。\n"
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
            "readable": CONSIST_READABLE,
            "need_retag": CONSIST_NEED_RETAG,
            "new_tags": CONSIST_EXPECTED_TAGS,
        },
    ),
    ObjectiveTask(
        task_id="obj_consist_02",
        kp="kp_lumen_consist",
        title="用 SQL 查出所有读取会被拒的键",
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
            "提交 JSON，只含一个字段："
            "  sql —— SQL 语句字符串"
        ),
        submit_hint='{"sql":"SELECT key FROM zephyra_keys WHERE ztag != (lumen_hash + (SELECT value FROM zephyra_meta WHERE name=\'epoch\')) % 65537 ORDER BY key"}',
        checker="check_unreadable_sql",
        submit_fields=("sql",),
        expected={"rows": [[k] for k in EXPECTED_UNREADABLE_KEYS]},
    ),
)


def get_task(task_id: str) -> ObjectiveTask:
    for t in OBJECTIVE_TASKS:
        if t.task_id == task_id:
            return t
    raise KeyError(f"未知实践任务 {task_id!r}")


def tasks_for_kp(kp: str) -> tuple[ObjectiveTask, ...]:
    return tuple(t for t in OBJECTIVE_TASKS if t.kp == kp)


# ==========================================================================
# 候选产出解析
# ==========================================================================

_FENCE = re.compile(r"```(?:json|python)?\s*(.*?)\s*```", re.DOTALL)


def parse_candidate(raw: str) -> tuple[dict[str, Any] | None, str]:
    """把模型原文解析成 dict。失败返回 (None, 原因)。"""
    text = (raw or "").strip()
    if not text:
        return None, "产出为空"
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j > i:
            try:
                data = json.loads(text[i : j + 1])
            except json.JSONDecodeError as exc:
                return None, f"JSON 解析失败：{exc}"
        else:
            return None, "找不到 JSON 对象"
    if not isinstance(data, dict):
        return None, f"顶层不是 JSON 对象，实际是 {type(data).__name__}"
    return data, ""


# ==========================================================================
# 受限执行
# ==========================================================================

_RESULT_MARK = "__OBJECTIVE_RESULT__"

#: 候选 SQL 里禁止出现的写操作关键字
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
    r"vacuum|reindex|pragma\s*(?!table_info)|begin|commit|rollback)\b",
    re.IGNORECASE,
)


#: 候选代码里禁止 import 的模块。
#:
#: `python -I` 只挡包搜索路径，**挡不住 `import os; os.system(...)`**——
#: 实测真的执行了目录列举。所以必须在**进子进程之前**静态拒绝。
#:
#: 判分任务只需要纯计算，这些模块没有任何正当用途，
#: 出现即视为违规。这不是「防御性编程」，
#: 是承认「执行模型产出的代码本身就是这个设计里最大的风险面」。
_FORBIDDEN_IMPORTS = frozenset(
    {
        "os", "sys", "subprocess", "shutil", "socket", "pathlib",
        "importlib", "ctypes", "multiprocessing", "threading",
        "http", "urllib", "requests", "ftplib", "smtplib",
        "pickle", "tempfile", "glob", "signal", "resource",
    }
)

_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+(?P<m1>[A-Za-z_][\w.]*)"
    r"|from\s+(?P<m2>[A-Za-z_][\w.]*)\s+import\b)",
    re.MULTILINE,
)

#: 用了但拿不到模块的动态导入写法
_DYNAMIC_IMPORT_RE = re.compile(r"__import__|importlib|exec\s*\(|eval\s*\(|compile\s*\(")


def scan_code_safety(code: str) -> list[str]:
    """静态扫描候选代码里的危险构造。返回违规描述列表（空=安全）。"""
    problems: list[str] = []
    for m in _IMPORT_RE.finditer(code):
        mod = (m.group("m1") or m.group("m2") or "").split(".")[0]
        if mod in _FORBIDDEN_IMPORTS:
            problems.append(f"禁止 import {mod}")
    if _DYNAMIC_IMPORT_RE.search(code):
        problems.append("使用了 __import__ / exec / eval / compile 动态执行")
    return problems


def _py_literal(value: Any) -> str:
    """把 Python 值渲染成**可被 Python 解释器解析**的字面量。

    踩过的坑：原本用 `json.dumps` 注入用例表，但 `json.dumps(True)`
    产出的是 JS 风格的 `true`，粘进 driver 就是 SyntaxError。
    症状是「明明实现正确却判 0 分」——判分器坏了而不是答案错了。
    用 `repr` 才对：它输出的是 Python 自己的字面量。
    """
    return repr(value)


def _run_python_isolated(
    code: str, driver: str, timeout: float = 15.0
) -> dict[str, Any]:
    """在隔离模式 + 临时目录里执行候选代码。

    `python -I` 忽略 PYTHONPATH 与用户 site 包，cwd 指向一次性临时目录，
    环境变量只留 SYSTEMROOT（Windows 上某些运行时要），15 秒超时。
    """
    env = {
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="wb_obj_") as td:
            path = os.path.join(td, "candidate.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
                f.write("\n\n")
                f.write(driver)
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-B", path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=td,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"执行超时（>{timeout:g}s）"}
    except OSError as exc:  # pragma: no cover — 临时目录创建失败
        return {"ok": False, "error": f"无法启动子进程：{exc}"}

    out = proc.stdout or ""
    if _RESULT_MARK in out:
        payload = out.split(_RESULT_MARK, 1)[1].strip().splitlines()
        if payload:
            try:
                return {"ok": True, "data": json.loads(payload[0])}
            except json.JSONDecodeError:
                return {"ok": False, "error": "子进程结果不是合法 JSON"}
    detail = (proc.stderr or out or "").strip().splitlines()
    return {
        "ok": False,
        "error": (detail[-1] if detail else "子进程无输出")[:300],
    }


def _build_fixture_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(FIXTURE_SCHEMA)
    conn.executemany("INSERT INTO zephyra_meta(name, value) VALUES (?,?)", FIXTURE_META)
    conn.executemany(
        "INSERT INTO zephyra_buckets(bucket, live_count, tombstone_count) VALUES (?,?,?)",
        FIXTURE_BUCKETS,
    )
    conn.executemany(
        "INSERT INTO zephyra_keys(key, lumen_hash, bucket, ztag, epoch_at_write)"
        " VALUES (?,?,?,?,?)",
        FIXTURE_KEYS,
    )
    conn.commit()
    return conn


def _run_sql(sql: str) -> dict[str, Any]:
    """在预置只读 fixture 上跑候选 SQL。"""
    if _FORBIDDEN_SQL.search(sql):
        return {"ok": False, "error": "SQL 含写操作，本任务只接受只读查询"}
    if ";" in sql.rstrip(";"):
        return {"ok": False, "error": "只允许一条 SQL 语句"}
    conn = _build_fixture_db()
    try:
        cur = conn.execute(sql)
        rows = [list(r) for r in cur.fetchall()]
        return {"ok": True, "rows": rows}
    except sqlite3.Error as exc:
        return {"ok": False, "error": f"SQL 执行失败：{exc}"}
    finally:
        conn.close()


# ==========================================================================
# 判分器
#
# 每个 checker 返回 list[check]，check = {"name","passed","detail"}。
# objective_score = 100 × 通过数 / 总check 数。
# ==========================================================================


def _ck(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _as_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for v in value:
        n = _as_int(v)
        if n is None:
            return None
        out.append(n)
    return out


def _as_str_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(v, str) for v in value):
        return None
    return list(value)


def check_hash_algorithm(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    steps = data.get("steps")
    steps_i = _as_int_list(steps)
    checks.append(_ck(
        "steps 是整数数组且逐步递减式完整",
        steps_i is not None and len(steps_i) == len(expected["steps"]),
        f"实际 {steps!r}",
    ))
    final = _as_int(data.get("lumen_hash"))
    checks.append(_ck(
        "lumen_hash 正确",
        final == expected["lumen_hash"],
        f"期望 {expected['lumen_hash']}，实际 {data.get('lumen_hash')!r}",
    ))
    if steps_i and len(steps_i) == len(expected["steps"]):
        checks.append(_ck(
            "steps 每一步中间值都正确",
            steps_i == expected["steps"],
            f"期望 {expected['steps']}，实际 {steps_i}",
        ))
        checks.append(_ck(
            "steps 初值正确（材料规定的初值不是 0）",
            steps_i[0] == expected["steps"][0],
            f"期望初值 {expected['steps'][0]}，实际 {steps_i[0]}",
        ))
    else:
        checks.append(_ck("steps 每一步中间值都正确", False, "steps 缺失或长度不对"))
        checks.append(_ck(
            "steps 初值正确（材料规定的初值不是 0）", False, "steps 缺失"
        ))
    bucket = _as_int(data.get("bucket"))
    checks.append(_ck(
        "bucket 正确（哈希值对 8 取模）",
        bucket == expected["bucket"],
        f"期望 {expected['bucket']}，实际 {data.get('bucket')!r}",
    ))
    return checks


_HASH_DRIVER = """
_cases = __CASES__
try:
    fn = lumen_hash
except NameError:
    print("__OBJECTIVE_RESULT__" + json.dumps({"error": "未定义函数 lumen_hash"}))
else:
    fails = []
    for key, want in _cases:
        try:
            got = fn(key)
        except Exception as exc:
            fails.append({"key": key, "error": type(exc).__name__})
            continue
        if got != want:
            fails.append({"key": key, "want": want, "got": got})
    print("__OBJECTIVE_RESULT__" + json.dumps({"fails": fails, "n": len(_cases)},
          ensure_ascii=False))
"""


def check_hash_unit_test(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    code = data.get("code")
    if not isinstance(code, str) or "def lumen_hash" not in code:
        return [_ck("提交了 lumen_hash 函数实现", False, "code 字段缺失或没有 def lumen_hash")]
    unsafe = scan_code_safety(code)
    if unsafe:
        return [_ck("代码只做纯计算（无危险 import）", False, "、".join(unsafe))]
    driver = "import json\n" + _HASH_DRIVER.replace(
        "__CASES__", _py_literal(expected["cases"])
    )
    res = _run_python_isolated(code, driver)
    if not res["ok"]:
        return [_ck("代码可执行", False, res.get("error", "执行失败"))]
    fails = res["data"].get("fails") or []
    total = res["data"].get("n") or len(expected["cases"])
    return [
        _ck("代码只做纯计算（无危险 import）", True),
        _ck("代码可执行", True),
        _ck(
            f"{total} 组用例全部通过（含模型没见过的键）",
            not fails,
            f"失败 {len(fails)} 组：" + json.dumps(fails[:3], ensure_ascii=False),
        ),
    ]


def check_split_algorithm(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    assigns = data.get("assignments")
    ok_assign = isinstance(assigns, list) and len(assigns) == len(expected["assignments"])
    checks.append(_ck("assignments 覆盖全部 5 个键", ok_assign, f"实际 {assigns!r}"))
    if ok_assign:
        want_map = {e["key"]: e["to_bucket"] for e in expected["assignments"]}
        got_map: dict[str, Any] = {}
        for row in assigns:
            if isinstance(row, dict) and isinstance(row.get("key"), str):
                got_map[row["key"]] = _as_int(row.get("to_bucket"))
        checks.append(_ck(
            "每个键的分裂后桶号都正确",
            got_map == want_map,
            f"期望 {want_map}，实际 {got_map}",
        ))
        non_zero = [k for k, v in got_map.items() if v not in (0, None)]
        checks.append(_ck(
            "没有键被迁移到非 0 号桶（(0+8) mod 8 = 0）",
            not non_zero,
            f"以下键被算到了别的桶：{non_zero}",
        ))
    else:
        checks.append(_ck("每个键的分裂后桶号都正确", False, "assignments 结构不对"))
        checks.append(_ck(
            "没有键被迁移到非 0 号桶（(0+8) mod 8 = 0）", False, "assignments 结构不对"
        ))
    migrated = _as_int(data.get("migrated_count"))
    checks.append(_ck(
        "migrated_count 正确（分裂是恒等变换，实际迁移 0 个）",
        migrated == expected["migrated_count"],
        f"期望 {expected['migrated_count']}，实际 {data.get('migrated_count')!r}",
    ))
    over = data.get("still_over_limit")
    checks.append(_ck(
        "still_over_limit 正确（分裂不减负载，桶仍然超限）",
        over is expected["still_over_limit"],
        f"期望 {expected['still_over_limit']}，实际 {over!r}",
    ))
    return checks


def _sql_checks(data: dict[str, Any], expected: dict[str, Any], label: str) -> list[dict[str, Any]]:
    sql = data.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        return [_ck("提交了 SQL 语句", False, "sql 字段缺失")]
    res = _run_sql(sql)
    if not res["ok"]:
        return [_ck("SQL 可执行且只读", False, res.get("error", "执行失败"))]
    # **按原始顺序比较，不排序。**
    # 排序等于把「按桶号升序 / 按键名升序」这个明确要求整个抹掉——
    # 实测写 ORDER BY bucket DESC 也能拿满分，
    # 于是题目里那句排序要求变成了一句没人检查的废话。
    rows = res["rows"]
    want = expected["rows"]
    flat_ok = all(len(r) == 1 for r in rows)
    return [
        _ck("SQL 可执行且只读", True),
        _ck("结果只有一列（没有多查无关字段）", flat_ok, f"实际每行列数 {[len(r) for r in rows]}"),
        _ck(
            f"{label} 与期望完全一致（含顺序）",
            rows == want,
            f"期望 {want}，实际 {rows}",
        ),
    ]


def check_overload_sql(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    return _sql_checks(data, expected, "超载桶号集合")


def check_unreadable_sql(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    return _sql_checks(data, expected, "读取被拒的键集合")


def check_compact_schema(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    compress = _as_int_list(data.get("compress"))
    checks.append(_ck(
        "compress 是整数数组",
        compress is not None,
        f"实际 {data.get('compress')!r}",
    ))
    if compress is not None:
        checks.append(_ck(
            "触发压缩的桶号完全正确（条件 A 与 B 必须同时成立）",
            compress == expected["compress"],
            f"期望 {expected['compress']}，实际 {compress}",
        ))
        checks.append(_ck(
            "没有把「墓碑多但活跃键也多」的桶误判为触发",
            1 not in compress and 4 not in compress,
            f"实际 {compress}",
        ))
    else:
        checks.append(_ck("触发压缩的桶号完全正确（条件 A 与 B 必须同时成立）", False, "类型不对"))
        checks.append(_ck(
            "没有把「墓碑多但活跃键也多」的桶误判为触发", False, "类型不对"
        ))
    after = _as_int(data.get("epoch_after"))
    checks.append(_ck(
        "epoch_after 正确（每触发一次压缩全局 epoch 加 1）",
        after == expected["epoch_after"],
        f"期望 {expected['epoch_after']}，实际 {data.get('epoch_after')!r}",
    ))
    return checks


_COMPACT_DRIVER = """
_cases = __CASES__
try:
    fn = should_compact
except NameError:
    print("__OBJECTIVE_RESULT__" + json.dumps({"fails": [{"error": "未定义函数"}], "n": len(_cases)}))
else:
    fails = []
    for t, l, want in _cases:
        try:
            got = fn(t, l)
        except Exception as exc:
            fails.append({"t": t, "l": l, "error": type(exc).__name__})
            continue
        if bool(got) != want:
            fails.append({"t": t, "l": l, "want": want, "got": bool(got)})
    print("__OBJECTIVE_RESULT__" + json.dumps({"fails": fails, "n": len(_cases)},
          ensure_ascii=False))
"""


def check_compact_unit_test(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    code = data.get("code")
    if not isinstance(code, str) or "def should_compact" not in code:
        return [_ck("提交了 should_compact 函数实现", False, "code 字段缺失或函数名不对")]
    unsafe = scan_code_safety(code)
    if unsafe:
        return [_ck("代码只做纯计算（无危险 import）", False, "、".join(unsafe))]
    driver = "import json\n" + _COMPACT_DRIVER.replace(
        "__CASES__", _py_literal(expected["cases"])
    )
    res = _run_python_isolated(code, driver)
    if not res["ok"]:
        return [_ck("代码可执行", False, res.get("error", "执行失败"))]
    fails = res["data"].get("fails") or []
    total = res["data"].get("n") or len(expected["cases"])
    return [
        _ck("代码只做纯计算（无危险 import）", True),
        _ck("代码可执行", True),
        _ck(
            f"{total} 组用例全部通过（'且' 与 '或' 混用会在其中一组挂掉）",
            not fails,
            f"失败 {len(fails)} 组：" + json.dumps(fails[:3], ensure_ascii=False),
        ),
    ]


def check_consistency_algorithm(data: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    readable = _as_str_list(data.get("readable"))
    checks.append(_ck(
        "readable 是字符串数组",
        readable is not None,
        f"实际 {data.get('readable')!r}",
    ))
    checks.append(_ck(
        "readable 正确（epoch 变了，存量键全部失配）",
        readable == expected["readable"],
        f"期望 {expected['readable']}，实际 {readable!r}",
    ))
    retag = _as_str_list(data.get("need_retag"))
    checks.append(_ck(
        "need_retag 正确（4 个键全部需要重算）",
        retag == expected["need_retag"],
        f"期望 {expected['need_retag']}，实际 {retag!r}",
    ))
    tags = data.get("new_tags")
    got_tags: dict[str, Any] = {}
    if isinstance(tags, dict):
        for k, v in tags.items():
            if isinstance(k, str):
                got_tags[k] = _as_int(v)
    checks.append(_ck(
        "new_tags 是「键名 → 整数标签」的对象",
        isinstance(tags, dict) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in tags.values()
        ),
        f"实际 {tags!r}",
    ))
    checks.append(_ck(
        "每个新标签 = (LumenHash + 当前 epoch) mod 65537",
        got_tags == expected["new_tags"],
        f"期望 {expected['new_tags']}，实际 {got_tags}",
    ))
    return checks


CHECKERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]]]] = {
    "check_hash_algorithm": check_hash_algorithm,
    "check_hash_unit_test": check_hash_unit_test,
    "check_split_algorithm": check_split_algorithm,
    "check_overload_sql": check_overload_sql,
    "check_compact_schema": check_compact_schema,
    "check_compact_unit_test": check_compact_unit_test,
    "check_unreadable_sql": check_unreadable_sql,
    "check_consistency_algorithm": check_consistency_algorithm,
}


@dataclass
class ObjectiveResult:
    """一次客观判分的结果。

    **objective_score 与 evaluator_score 是两回事。**
    本对象只回答「机器核对的结果对不对」；
    「这个产出作为工程实践好不好用」由 LLM 评审单独给分，
    两者分开存（practice_runs / practice_evaluations两张表），
    混在一起就看不出到底是哪一环出了问题。
    """

    task_id: str
    kp: str
    kind: str
    checker: str
    passed: bool
    objective_score: float
    checks: list[dict[str, Any]] = field(default_factory=list)
    detail: str = ""
    evaluator: str = "objective_checker"
    parse_error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kp": self.kp,
            "kind": self.kind,
            "checker": self.checker,
            "passed": self.passed,
            "objective_score": self.objective_score,
            "checks": self.checks,
            "detail": self.detail,
            "evaluator": self.evaluator,
            "parse_error": self.parse_error,
        }


def evaluate_objective(task: ObjectiveTask, candidate_output: str) -> ObjectiveResult:
    """对一份候选产出做客观判分。**不会抛异常**——

    判分失败本身就是实验数据（模型提交了不可解析的东西，
    这在真实能力评估里是有意义的信息），不该把整次实验炸掉。
    """
    checker = CHECKERS.get(task.checker)
    if checker is None:  # pragma: no cover — 门禁保证不会走到
        return ObjectiveResult(
            task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
            passed=False, objective_score=0.0,
            detail=f"未注册的判分器 {task.checker!r}",
        )

    data, err = parse_candidate(candidate_output)
    if data is None:
        return ObjectiveResult(
            task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
            passed=False, objective_score=0.0,
            detail="产出无法解析为 JSON 对象",
            parse_error=err,
        )

    try:
        checks = checker(data, task.expected)
    except Exception as exc:  # noqa: BLE001 — 判分器自身出错也不能炸实验
        return ObjectiveResult(
            task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
            passed=False, objective_score=0.0,
            detail=f"判分器异常：{type(exc).__name__}: {exc}",
        )

    if not checks:  # pragma: no cover
        return ObjectiveResult(
            task_id=task.task_id, kp=task.kp, kind=task.kind, checker=task.checker,
            passed=False, objective_score=0.0, detail="判分器没有产出任何 check",
        )

    hit = sum(1 for c in checks if c["passed"])
    score = round(100.0 * hit / len(checks), 2)
    failed = [c["name"] for c in checks if not c["passed"]]
    return ObjectiveResult(
        task_id=task.task_id,
        kp=task.kp,
        kind=task.kind,
        checker=task.checker,
        passed=hit == len(checks),
        objective_score=score,
        checks=checks,
        detail=("全部通过" if not failed else "未通过：" + "；".join(failed)),
    )


# ==========================================================================
# 指纹与门禁
# ==========================================================================


def _checker_source(name: str) -> str:
    fn = CHECKERS[name]
    try:
        return inspect.getsource(fn)
    except OSError:  # pragma: no cover
        return name


def _sha256_of(payload: Any) -> str:
    import hashlib

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fingerprint() -> str:
    payload = [
        {
            "task_id": t.task_id,
            "kp": t.kp,
            "title": t.title,
            "kind": t.kind,
            "prompt": t.prompt,
            "submit_hint": t.submit_hint,
            "checker": t.checker,
            "checker_source": _checker_source(t.checker),
            "expected": t.expected,
        }
        for t in OBJECTIVE_TASKS
    ]
    return _sha256_of(payload)


TASKS_FINGERPRINT = _fingerprint()

#: fixture 也进指纹：数据集变了，超载桶 / 拒读键的期望值就跟着变了，
#: 不算进指纹就等于给「悄悄改数据集」留了个后门。
FIXTURE_FINGERPRINT = _sha256_of(
    {
        "schema": FIXTURE_SCHEMA,
        "meta": FIXTURE_META,
        "buckets": FIXTURE_BUCKETS,
        "keys": FIXTURE_KEYS,
    }
)


def task_diagnostics() -> dict[str, Any]:
    """实践任务集诊断。"""
    problems: list[str] = []
    warnings: list[str] = []

    if len(OBJECTIVE_TASKS) < 6:
        problems.append(f"实践任务至少 6 个，当前 {len(OBJECTIVE_TASKS)} 个")

    ids = [t.task_id for t in OBJECTIVE_TASKS]
    if len(ids) != len(set(ids)):
        problems.append("实践任务 id 重复")

    for t in OBJECTIVE_TASKS:
        if t.kind not in KINDS:
            problems.append(f"{t.task_id}：kind 取值非法 {t.kind!r}")
        if t.checker not in CHECKERS:
            problems.append(f"{t.task_id}：引用了未注册的判分器 {t.checker!r}")
        if not t.prompt.strip():
            problems.append(f"{t.task_id}：prompt 为空")
        # submit_hint 是给模型看的格式示例，必须本身是合法 JSON——
        # 非法示例在教模型输出非法格式，等于自己给自己制造解析失败
        hint, hint_err = parse_candidate(t.submit_hint)
        if hint is None:
            problems.append(
                f"{t.task_id}：submit_hint 不是合法 JSON（{hint_err}），"
                f"格式示例本身非法会教模型输出非法格式"
            )
        elif set(hint.keys()) != set(t.submit_fields):
            problems.append(
                f"{t.task_id}：submit_hint 的字段 {sorted(hint)} "
                f"与 submit_fields {sorted(t.submit_fields)} 不一致，"
                f"模型按示例提交会漏字段"
            )
        # 期望值绝不能出现在题面里——泄题则机器判分失去意义
        blob = t.prompt + "\n" + t.submit_hint
        for token in _expected_tokens(t.expected):
            if token in blob:
                problems.append(
                    f"{t.task_id}：题面/示例里出现了期望值 {token!r}，这是泄题"
                )

    kinds = {t.kind for t in OBJECTIVE_TASKS}
    if "python_unit_test" not in kinds:
        warnings.append("没有 python_unit_test 类任务，模型可以写死答案蒙混过关")
    if "sql_exec" not in kinds:
        warnings.append("没有 sql_exec 类任务，缺少真实执行结果的校验")

    per_kp: dict[str, int] = {}
    for t in OBJECTIVE_TASKS:
        per_kp[t.kp] = per_kp.get(t.kp, 0) + 1
    for kp, n in per_kp.items():
        if n < 2:
            warnings.append(f"{kp} 只有 {n} 个实践任务，逐 kp 结论样本太少")

    return {
        "problems": problems,
        "warnings": warnings,
        "per_kp": per_kp,
        "total": len(OBJECTIVE_TASKS),
        "kinds": sorted(kinds),
    }


def _expected_tokens(expected: Any) -> list[str]:
    """从期望值里抽出「足够独特、可用于泄题检测」的字符串/数字。"""
    tokens: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, int):
            # 只查 3 位以上的数字：桶号、单个 0/1 撞词太常见会误报
            if abs(node) >= 100:
                tokens.append(str(node))
        elif isinstance(node, str) and len(node) >= 4:
            tokens.append(node)

    walk(expected)
    return tokens


def assert_tasks_intact() -> None:
    """硬门禁：任务集有结构性问题就拒绝跑实验。"""
    diag = task_diagnostics()
    if diag["problems"]:
        raise ValueError(
            "实践任务诊断发现问题，拒绝跑实验：\n  - " + "\n  - ".join(diag["problems"])
        )


def tasks_meta() -> dict[str, Any]:
    assert_tasks_intact()
    return {
        "fingerprint": TASKS_FINGERPRINT,
        "fixture_fingerprint": FIXTURE_FINGERPRINT,
        "total": len(OBJECTIVE_TASKS),
        "kinds": sorted({t.kind for t in OBJECTIVE_TASKS}),
        "per_kp": task_diagnostics()["per_kp"],
    }


__all__ = [
    "OBJECTIVE_TASKS",
    "ObjectiveTask",
    "ObjectiveResult",
    "KINDS",
    "TASKS_FINGERPRINT",
    "FIXTURE_FINGERPRINT",
    "OBJECTIVE_SYSTEM",
    "assert_tasks_intact",
    "evaluate_objective",
    "get_task",
    "parse_candidate",
    "task_diagnostics",
    "tasks_for_kp",
    "tasks_meta",
]
