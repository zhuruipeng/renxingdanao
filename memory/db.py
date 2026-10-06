"""SQLite 存储层：连接管理与表结构定义。

设计原则：
- 只用标准库 sqlite3，不引入 ORM。
- 单文件数据库 agent.db，所有子系统共用。
- 所有写入走 execute/transaction，保证 exam 模式下也有一致的提交语义。
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence


def checksum(text: str) -> str:
    """内容 SHA256。用于冻结学习材料版本、防事后篡改。"""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()

SCHEMA = """
PRAGMA foreign_keys = ON;

-- 已学习的知识条目
CREATE TABLE IF NOT EXISTS knowledge (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    keywords    TEXT    NOT NULL DEFAULT '',
    source      TEXT    NOT NULL DEFAULT 'llm',
    confidence  REAL    NOT NULL DEFAULT 1.0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);

-- 沉淀的经验（成功/失败都要记）
CREATE TABLE IF NOT EXISTS experiences (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    stage       TEXT    NOT NULL DEFAULT '',
    scenario    TEXT    NOT NULL,
    action      TEXT    NOT NULL DEFAULT '',
    result      TEXT    NOT NULL DEFAULT '',
    score       REAL,
    lesson      TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiences_stage ON experiences(stage);

-- 技能项（能力维度），随表现升级
CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT    NOT NULL DEFAULT '',
    level       INTEGER NOT NULL DEFAULT 1,
    xp          INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

-- 任务：学习/实践/考试统一建模
CREATE TABLE IF NOT EXISTS tasks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type         TEXT    NOT NULL,          -- learn | practice | exam
    stage             TEXT    NOT NULL DEFAULT '',  -- pre_test | post_test | practice | learn
    topic             TEXT    NOT NULL,
    question          TEXT    NOT NULL,
    reference_answer  TEXT,
    candidate_answer  TEXT,
    difficulty        TEXT    NOT NULL DEFAULT 'medium',  -- easy | medium | hard
    blind             INTEGER NOT NULL DEFAULT 0,-- 1=严格考试模式（禁读 knowledge/training_samples）
    status            TEXT    NOT NULL DEFAULT 'pending',  -- pending | answered | evaluated
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_stage ON tasks(stage);
CREATE INDEX IF NOT EXISTS idx_tasks_topic ON tasks(topic);

-- 独立评估结果
CREATE TABLE IF NOT EXISTS evaluations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER REFERENCES tasks(id) ON DELETE CASCADE,
    evaluator   TEXT    NOT NULL DEFAULT 'llm_judge',
    score       REAL    NOT NULL,
    max_score   REAL    NOT NULL DEFAULT 100.0,
    verdict     TEXT    NOT NULL DEFAULT '',
    feedback    TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluations_task ON evaluations(task_id);

-- 训练样本：由 knowledge 生成的问答对，考试模式禁止读取
CREATE TABLE IF NOT EXISTS training_samples (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    topic                TEXT    NOT NULL,
    question             TEXT    NOT NULL,
    answer               TEXT    NOT NULL,
    source_knowledge_id  INTEGER REFERENCES knowledge(id) ON DELETE SET NULL,
    difficulty           TEXT    NOT NULL DEFAULT 'medium',
    created_at           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_topic ON training_samples(topic);
"""

# ==========================================================================
# V0.2：实验记录相关表
#
# 为什么要单独建表而不是复用 V0.1 的表：
# 一次「学习是否带来提升」的实验必须冻结输入——题目快照、学习材料版本、
# 每道题的前后得分。复用 tasks/evaluations 会导致无法证明「前后比的是同一件事」。
# ==========================================================================

SCHEMA_V2 = """
-- 学习材料版本快照：实验必须记录当时学的是哪一版材料
CREATE TABLE IF NOT EXISTS materials (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT    NOT NULL,
    version     TEXT    NOT NULL,
    title       TEXT    NOT NULL DEFAULT '',
    content     TEXT    NOT NULL,
    checksum    TEXT    NOT NULL,          -- 内容 SHA256，防止事后篡改
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_materials_topic ON materials(topic);

-- 固定题库指纹：证明 PRE/POST 卷在实验期间没被改过
CREATE TABLE IF NOT EXISTS question_bank (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    version        TEXT    NOT NULL,
    fingerprint    TEXT    NOT NULL UNIQUE,  -- 全卷 SHA256
    pre_fingerprint TEXT   NOT NULL,
    post_fingerprint TEXT   NOT NULL,
    pre_count      INTEGER NOT NULL,
    post_count     INTEGER NOT NULL,
    payload_json   TEXT    NOT NULL,        -- 完整题目快照，事后可复核
    created_at     TEXT    NOT NULL
);

-- 实验主记录
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT    NOT NULL,
    material_id     INTEGER REFERENCES materials(id) ON DELETE SET NULL,
    bank_id         INTEGER REFERENCES question_bank(id) ON DELETE SET NULL,
    pre_score       REAL,
    post_score      REAL,
    delta           REAL,
    delta_pct       REAL,
    status          TEXT    NOT NULL DEFAULT 'running',  -- running | done | aborted
    detail_json     TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL,
    finished_at     TEXT
);

-- 每道题的前后作答与得分（逐知识点对比的原始数据）
CREATE TABLE IF NOT EXISTS experiment_items (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id     INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    phase             TEXT    NOT NULL,        -- pre | post
    item_id           TEXT    NOT NULL,        -- 题库里的稳定 id
    kp                TEXT    NOT NULL,        -- 知识点标签
    question          TEXT    NOT NULL,        -- 题目快照
    reference_answer  TEXT    NOT NULL,        -- 参考答案快照（仅评估器可见）
    candidate_answer  TEXT    NOT NULL DEFAULT '',
    score             REAL,
    verdict           TEXT    NOT NULL DEFAULT '',
    feedback          TEXT    NOT NULL DEFAULT '',
    missing_points    TEXT    NOT NULL DEFAULT '[]',
    task_id           INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
    evaluation_id     INTEGER REFERENCES evaluations(id) ON DELETE SET NULL,
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_items ON experiment_items(experiment_id, phase);

-- 实践过程记录：成功 / 失败 / 经验
CREATE TABLE IF NOT EXISTS practice_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    stage         TEXT    NOT NULL DEFAULT 'practice',
    outcome       TEXT    NOT NULL,        -- success | failure
    kp            TEXT    NOT NULL DEFAULT '',
    scenario      TEXT    NOT NULL,
    action        TEXT    NOT NULL DEFAULT '',
    result        TEXT    NOT NULL DEFAULT '',
    lesson        TEXT    NOT NULL DEFAULT '',
    score         REAL,
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_practice_exp ON practice_records(experiment_id);
"""

# ==========================================================================
# V0.3：结构化学习笔记
#
# V0.2 暴露的核心矛盾：盲考禁读知识库 → 学到的知识在考试时用不上；
# 开卷放行知识库 → 但如果放行的是原始材料全文，等于把答案抄给模型。
#
# 解法：学习阶段强制产出「Agent 自己写的结构化笔记」，
#考试时只放行笔记，不放行材料原文。笔记是二次加工的产物，
# 不是原材料的搬运——这正是它能测「学习是否真的发生」的原因。
# ==========================================================================

SCHEMA_V3 = """
-- 结构化学习笔记：concept/principle/example/common_mistake/application_rule
CREATE TABLE IF NOT EXISTS learning_notes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id     INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    topic             TEXT    NOT NULL,
    kp                TEXT    NOT NULL DEFAULT '',
    concept           TEXT    NOT NULL,   -- 这是什么
    principle         TEXT    NOT NULL,   -- 为什么是这样（底层机制）
    example           TEXT    NOT NULL DEFAULT '',  -- 具体例子
    common_mistake    TEXT    NOT NULL DEFAULT '',  -- 容易错在哪
    application_rule  TEXT    NOT NULL DEFAULT '',  -- 什么时候用/怎么用
    checksum          TEXT    NOT NULL,   -- 五要素拼接后的 SHA256，防篡改+可校验
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_exp ON learning_notes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_notes_kp ON learning_notes(experiment_id, kp);

-- V0.3：experiment_items 增加 mode 列，区分三种考试
-- （SQLite 不支持 ADD COLUMN IF NOT EXISTS，走 _migrate）
"""

# ==========================================================================
# V0.4：笔记质量评估 + 重复实验
#
# V0.4 要回答的问题比V0.3 更难：
#   「模型是真的形成了概念，还是只是记住了笔记里的措辞？」
#
# 光看开卷分数不够 —— 开卷高分可能来自笔记写得对，也可能来自题目简单。
# 所以笔记质量必须**独立测量**：写笔记的是模型，评笔记的也必须是模型，
# 但是两个独立调用、两个独立人格（同一次上下文里自评必然虚高）。
# ==========================================================================

SCHEMA_V4 = """
-- 笔记质量评估：四维度（准确/完整/可迁移/无幻觉）
-- evaluator 必须与写笔记的 provider 是不同实例
CREATE TABLE IF NOT EXISTS note_quality (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id     INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    note_id           INTEGER NOT NULL REFERENCES learning_notes(id) ON DELETE CASCADE,
    accuracy          REAL NOT NULL DEFAULT 0,   -- 机制描述是否与材料一致
    completeness      REAL NOT NULL DEFAULT 0,   -- 五要素是否有实质内容
    transferable      REAL NOT NULL DEFAULT 0,   -- 讲清「为什么」还是只背结论
    no_hallucination  REAL NOT NULL DEFAULT 0,   -- 有没有编造材料里没有的概念
    total             REAL NOT NULL DEFAULT 0,   -- 加权总分（transferable 权重最高）
    verdict           TEXT    NOT NULL DEFAULT '',  -- pass | fail
    feedback          TEXT    NOT NULL DEFAULT '',
    problems          TEXT    NOT NULL DEFAULT '[]',
    evaluator         TEXT    NOT NULL DEFAULT 'llm_judge',
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_note_quality_exp ON note_quality(experiment_id);
CREATE INDEX IF NOT EXISTS idx_note_quality_note ON note_quality(note_id);

-- 实践任务明细：V0.4 要把「学习能否转化为实际任务能力」单独量化，
-- 所以不能再只看 outcome + 结果长度这种粗粒度数值。
CREATE TABLE IF NOT EXISTS practice_evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    kp              TEXT    NOT NULL DEFAULT '',
    outcome         TEXT    NOT NULL DEFAULT '',  -- success | failure
    task_prompt     TEXT    NOT NULL DEFAULT '',
    candidate_answer TEXT   NOT NULL DEFAULT '',
    score           REAL    NOT NULL DEFAULT 0,
    usable          INTEGER NOT NULL DEFAULT 0,  -- 产出是否真的可执行
    verdict         TEXT    NOT NULL DEFAULT '',
    feedback        TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_practice_eval_exp
    ON practice_evaluations(experiment_id);

-- 重复实验批次：单次结果不作数，必须能看跨次分布。
-- 每次跑一个trial，同一 batch_id 的多次跑共享 batch_id。
CREATE TABLE IF NOT EXISTS experiment_batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT    NOT NULL UNIQUE,
    topic           TEXT    NOT NULL,
    planned_trials  INTEGER NOT NULL DEFAULT 1,
    bank_fingerprint TEXT   NOT NULL,
    model           TEXT    NOT NULL DEFAULT '',
    material_checksum TEXT  NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'running',  -- running | done | aborted
    summary_json    TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL,
    finished_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_batch ON experiment_batches(batch_id);

-- V0.4 列迁移见 MIGRATIONS_V4
"""

# ==========================================================================
# V0.5：Benchmark V2
#
# V0.4.1 暴露的三个统计缺陷，V0.5 用**新表**解决，不改老表：
#
# 1. **题量不足** → V2 每 kp 6 题（V0.4 是每 kp 1 题）。
#    老表 `experiment_items` 一行 = 一道题一次作答，
#    结构上能装 24 题，**但它没有 question_type 字段**——
#    题型是 V2 的核心维度（basic/application/transfer 决定难度），
#    不另起表就只能塞进 difficulty 字符串，而那正是 V0.4.1 判定
#    「难度不可控」的原因。所以 V2 单独建表。
#
# 2. **实践评分主观** → `practice_runs` 表只存**机器判分**结果。
#    它与 `practice_evaluations`（LLM 评审）**分表存放**：
#    两个分数回答的是不同问题，混在一张表就看不出
#    到底是产出真错、还是评审器主观。
#
# 3. **天花板无法测量** → `v2_batches` 存题库/材料/任务三份指纹，
#    任何一份变了整个批次作废。
# ==========================================================================

SCHEMA_V5 = """
-- Benchmark V2 的一次完整实验（一个 trial）
CREATE TABLE IF NOT EXISTS v2_experiments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    topic             TEXT    NOT NULL,
    batch_id          TEXT    NOT NULL DEFAULT '',
    trial_no          INTEGER NOT NULL DEFAULT 1,
    bank_fingerprint  TEXT    NOT NULL,      -- 24 题指纹
    material_checksum TEXT    NOT NULL,      -- Zephyra 材料 checksum
    tasks_fingerprint TEXT    NOT NULL,      -- 实践任务集指纹
    -- 四场考试 + 一个客观实践
    pre_score         REAL,
    open_book_score   REAL,
    memory_blind_score   REAL,
    transfer_blind_score REAL,
    objective_practice_score REAL,           -- 机器判分均分
    objective_practice_pass_rate REAL,       -- 机器判分全过率
    evaluator_practice_score REAL,           -- LLM 评审均分（与上面分开）
    note_quality_avg  REAL,
    status            TEXT    NOT NULL DEFAULT 'running',
    detail_json       TEXT    NOT NULL DEFAULT '{}',
    created_at        TEXT    NOT NULL,
    finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_v2_exp_batch ON v2_experiments(batch_id, trial_no);

-- V2 的逐题作答：每阶段每题一行
-- **每个阶段都必须落齐 8 列**：question / answer / score / evaluator /
-- context_used / experiment_id / knowledge_point / question_type。
-- 少任何一列，「这个阶段到底测了什么」就变成事后猜。
CREATE TABLE IF NOT EXISTS v2_stage_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   INTEGER NOT NULL REFERENCES v2_experiments(id) ON DELETE CASCADE,
    stage           TEXT    NOT NULL,   -- pre_blind|study|practice|open_book|memory_blind|transfer_blind
    item_id         TEXT    NOT NULL,   -- 题库/任务里的稳定 id
    knowledge_point TEXT    NOT NULL,   -- kp id
    question_type   TEXT    NOT NULL,   -- basic|application|transfer|practice
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL DEFAULT '',   -- 考生产出
    reference_answer TEXT   NOT NULL DEFAULT '',   -- 快照，评估器才可见
    score           REAL,
    evaluator       TEXT    NOT NULL DEFAULT '',
    context_used    TEXT    NOT NULL DEFAULT '',   -- 考生实际看到的上下文
    mode            TEXT    NOT NULL DEFAULT 'blind_model',
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_v2_items
    ON v2_stage_items(experiment_id, stage);

-- 客观实践任务：一条 = 一个任务的一次作答 + 机器判分
--
-- **candidate_output 与 objective_result 分开存**：
--   candidate_output  模型原始产出（可能根本不是 JSON）
--   objective_result  逐 check 的机器核对结果
-- 合成一列的话，判分逻辑改了以后就分不清
-- 「产出变了」还是「判分变了」——而这两者含义完全不同。
CREATE TABLE IF NOT EXISTS practice_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id     INTEGER NOT NULL REFERENCES v2_experiments(id) ON DELETE CASCADE,
    task_id           TEXT    NOT NULL,     -- obj_hash_01 等
    kp                TEXT    NOT NULL,
    kind              TEXT    NOT NULL,     -- algorithm_output|json_schema|sql_exec|python_unit_test
    title             TEXT    NOT NULL DEFAULT '',
    candidate_output  TEXT    NOT NULL DEFAULT '',
    objective_result  TEXT    NOT NULL DEFAULT '{}',  -- 逐 check JSON
    objective_score   REAL    NOT NULL DEFAULT 0,
    objective_passed  INTEGER NOT NULL DEFAULT 0,
    parse_error       TEXT    NOT NULL DEFAULT '',
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_practice_runs_exp ON practice_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_practice_runs_task ON practice_runs(task_id);

-- V0.5 批次：冻结三份指纹 + 跨 trial 统计结果
CREATE TABLE IF NOT EXISTS v2_batches (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id          TEXT    NOT NULL UNIQUE,
    topic             TEXT    NOT NULL,
    planned_trials    INTEGER NOT NULL DEFAULT 1,
    bank_fingerprint  TEXT    NOT NULL,
    material_checksum TEXT    NOT NULL,
    tasks_fingerprint TEXT    NOT NULL,
    model             TEXT    NOT NULL DEFAULT '',
    status            TEXT    NOT NULL DEFAULT 'running',
    summary_json      TEXT    NOT NULL DEFAULT '{}',
    created_at        TEXT    NOT NULL,
    finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_v2_batch ON v2_batches(batch_id);

-- V0.5 列迁移见 MIGRATIONS_V5
"""

# ==========================================================================
# V0.5：V2 复用老表时的 id 归属
#
# V0.5 要复用 `learning_notes` / `note_quality` / `practice_evaluations`
# 三张老表（重写它们会破坏 V0.4 那条已验证的链路），
# 但这三张表的 `experiment_id` 都是 `NOT NULL REFERENCES experiments(id)`。
#
# V2 的实验 id 来自 `v2_experiments` 自增，是**另一套 id 空间**。
# 早先没处理归属时V2 直接往 `experiment_id` 写自己的 id，
# 于是「第 3 号 V2 实验的笔记」会挂在「第 3 号 V0.4 实验」名下——
# 数字撞上了，读出来就是别人的数据。**实验间污染，且不报错。**
#
# 两种方案都试过：
#   1. 留空（NULL）→ **不可行**。三列都是 NOT NULL，
#      SQLite 的 ALTER TABLE 不能去掉 NOT NULL 约束，
#      重建表又会动到用户已有的 .db 文件。
#   2. 负数哨兵（采用）→ V2 写`-experiment_id`。
#      V0.4 的 experiments.id 是 AUTOINCREMENT 正整数，物理上不可能为负，
#      所以**负数就是 V2 的专属标记**，不需要额外列，也不会撞号。
#
# 所以 `v2_experiment_id` 列保留为**冗余的正数 id**（便于人读与join），
# 但**权威归属靠 experiment_id 的正负号判断**，不靠列存不存在。
# ==========================================================================
MIGRATIONS_V5: tuple[tuple[str, str, str], ...] = (
    ("v2_experiments", "trial_key", "TEXT NOT NULL DEFAULT ''"),
    ("v2_experiments", "diagnostics_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("learning_notes", "v2_experiment_id", "INTEGER"),
    ("note_quality", "v2_experiment_id", "INTEGER"),
    ("practice_evaluations", "v2_experiment_id", "INTEGER"),
    # 机器判分在 practice_runs 里有 task_id，评审侧没有的话两侧无法对齐：
    # 看到「一个 55 分一个 70 分」却不知道分别是哪个任务，
    # 分歧就成了无法定位的噪声，而不是可分析的信号。
    ("practice_evaluations", "task_id", "TEXT NOT NULL DEFAULT ''"),
)


def v2_experiment_key(v2_experiment_id: int) -> int:
    """V2 实验 id → 老表 experiment_id 列里该写的值（负数）。"""
    return -abs(int(v2_experiment_id))


#: V2 哨兵行在老experiments 表里的 topic 标记
V2_SENTINEL_PREFIX = "[V2]"

#: V3 哨兵行标记。与 V2 分开，否则 V3 的笔记会被当成 V2 产物
V3_SENTINEL_PREFIX = "[V3]"


def v3_experiment_key(v3_experiment_id: int) -> int:
    """V3 实验 id → 老表 experiment_id 列里该写的值。

    **必须与 V2 的取法错开。**
    两者都用 `-|id|`，若同号，V3 写的笔记会被 V2 的查询读出来——
    那是跨版本污染，而且两边都查不出问题。

    错开办法：V3 占偶数侧（-2*id），V2 占奇数侧由`-id` 取得。
    实测 `-2*id` 永远是偶数、`-id` 的绝对值任意，
    两套只在 `id` 本身是 0 时才可能重合，而 id 从 1 起自增。
    """
    return -2 * abs(int(v3_experiment_id))


def create_v3_sentinel(conn: sqlite3.Connection, v3_experiment_id: int, topic: str) -> None:
    """在老experiments 表里为 V3 实验建一条哨兵行。

    存在的理由与 :func:`create_v2_sentinel` 完全相同：
    `learning_notes.experiment_id` 是 `NOT NULL REFERENCES experiments(id)`，
    不建哨兵行外键就拒绝。
    """
    conn.execute(
        """INSERT OR IGNORE INTO experiments(id, topic, status, detail_json, created_at)
           VALUES (?,?,?,?,?)""",
        (v3_experiment_key(v3_experiment_id), f"{V3_SENTINEL_PREFIX} {topic}",
         "done", "{}", datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def is_v3_row(row: Any) -> bool:
    """判断一行是不是 V3 产生的（负偶数id）。

    只看符号与奇偶，不看列存不存在——
    V2 的 id 也是负数，靠奇偶区分。
    """
    eid = row["experiment_id"] if not isinstance(row, dict) else row.get("experiment_id")
    return eid is not None and int(eid) < 0 and int(eid) % 2 == 0


def create_v2_sentinel(conn: sqlite3.Connection, v2_experiment_id: int, topic: str) -> None:
    """在老 experiments 表里为 V2 实验建一条哨兵行（id = 负数）。

    **为什么需要它**：learning_notes / note_quality / practice_evaluations
    三张表的 experiment_id 都是 `NOT NULL REFERENCES experiments(id)`。
    写负数时外键会拒绝（`experiments` 里没有负 id），
    写NULL 又过不了 NOT NULL —— 这条路两头都堵。

    办法是让那条负 id **真实存在**。于是：

      - 外键成立（指向哨兵行）
      - 隔离也成立：V0.4 的代码全部按正 id 查 `experiments`，
        负 id 永远查不到，读不出来
      - 顺带获得一个副产品：`experiments` 表里能直接看到
        「有几个 V2 实验、各自什么题库指纹」，不需要 join v2_experiments

    哨兵行的 topic 带 `[V2]` 前缀，报告与清理脚本据此识别。
    """
    conn.execute(
        """INSERT OR IGNORE INTO experiments(id, topic, status, detail_json, created_at)
           VALUES (?,?,?,?,?)""",
        (v2_experiment_key(v2_experiment_id), f"{V2_SENTINEL_PREFIX} {topic}",
         "done", "{}", datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def is_v2_row(row: Any) -> bool:
    """判断一行是不是 V2 产生的。**只看符号与奇偶，不看列存不存在。**

    V3 用负偶数 id（见 :func:`v3_experiment_key`），
    所以这里必须排除偶数——否则 V3 的行会被 V2 的查询读出来。
    """
    eid = row["experiment_id"] if not isinstance(row, dict) else row.get("experiment_id")
    return eid is not None and int(eid) < 0 and int(eid) % 2 != 0


# ==========================================================================
# V0.6：对照组实验（Difference-in-Gains）
#
# 为什么另起一套表，而不是给 V0.5 的表加 group 列
# --------------------------------------------------
# V0.5 的结论要能拿来做基线对比，**不能被 V0.6 的字段改动影响**。
# 而且 V0.5 的 `v2_stage_items` 有 `experiment_id → v2_experiments` 外键，
# 一旦两组共用它，就必须给每行加 group，而 V0.5 的所有查询
# （报告、跨次统计）都得跟着改，且改漏一处就是静默串组。
#
# 所以 V0.6 全部用新表。代价是V0.5 与 V0.6 的实验记录不共表，
# 换来的是：**V0.5 的历史结论一个字节都不受影响**，
# 且 V0.6 的隔离性可以在数据库层面一眼查清。
#
# 三张表分工
# ----------
# v3_experiments  一次 run = 一对实验（learning + control），
#                 带 candidate_provider / evaluator_provider 两个独立配置
# v3_stage_items  逐题作答，**强制带 group 列**——
#                 没有 group 就无法回答「这一行是哪个组的」
# v3_objective    客观 held-out 判分（机器判定，无LLM）
# ==========================================================================
SCHEMA_V6 = """
-- 一次双组实验 run
CREATE TABLE IF NOT EXISTS v3_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL UNIQUE,
    topic             TEXT    NOT NULL,
    trial_no          INTEGER NOT NULL DEFAULT 1,
    -- **两个 provider 必须分别记录，不允许只存一个。**
    -- V0.5 只有一个 provider 字段，于是「考生和评委是同一族模型」
    -- 这件事在数据里根本查不出来，只能靠回忆。V0.6 把它变成数据。
    candidate_provider TEXT   NOT NULL DEFAULT '',
    candidate_model   TEXT    NOT NULL DEFAULT '',
    evaluator_provider TEXT   NOT NULL DEFAULT '',
    evaluator_model   TEXT    NOT NULL DEFAULT '',
    -- 五份冻结指纹：题库/ 学习材料 / 对照材料 / practice 任务 / held-out 任务
    pre_fingerprint   TEXT    NOT NULL DEFAULT '',
    post_fingerprint  TEXT    NOT NULL DEFAULT '',
    transfer_fingerprint TEXT NOT NULL DEFAULT '',
    learning_material_checksum  TEXT NOT NULL DEFAULT '',
    control_material_checksum   TEXT NOT NULL DEFAULT '',
    practice_tasks_fingerprint  TEXT NOT NULL DEFAULT '',
    heldout_tasks_fingerprint   TEXT NOT NULL DEFAULT '',
    status            TEXT    NOT NULL DEFAULT 'running',
    summary_json      TEXT    NOT NULL DEFAULT '{}',
    diagnostics_json  TEXT    NOT NULL DEFAULT '{}',
    created_at        TEXT    NOT NULL,
    finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_v3_runs ON v3_runs(run_id);

-- 一个组的一次实验（learning / control 各一行，共用 run_id）
CREATE TABLE IF NOT EXISTS v3_experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL REFERENCES v3_runs(run_id) ON DELETE CASCADE,
    exp_group       TEXT    NOT NULL,     -- 'learning' | 'control'
    material_kind   TEXT    NOT NULL,     -- 'novel_zephyra' | 'control_veyra'
    material_checksum TEXT  NOT NULL,
    -- 本组自己独立的 experiment_id 空间。
    -- knowledge / learning_notes / experiences / training_samples /
    -- evaluator context / answer history 全部按这个 id 隔离。
    pre_score            REAL,
    post_score           REAL,
    transfer_score       REAL,
    pre_objective_score  REAL,
    post_objective_score REAL,
    pre_objective_pass_rate   REAL,
    post_objective_pass_rate  REAL,
    status         TEXT    NOT NULL DEFAULT 'running',
    detail_json    TEXT    NOT NULL DEFAULT '{}',
    created_at     TEXT    NOT NULL,
    finished_at    TEXT,
    -- 同一 run 内 group 必须唯一：防止同一个组被写两遍而指标被平均掉
    UNIQUE (run_id, exp_group)
);
CREATE INDEX IF NOT EXISTS idx_v3_exp_run ON v3_experiments(run_id, exp_group);

-- 逐阶段逐题作答。
-- **每一行都必须能回答「哪个组、哪个阶段、哪道题、
-- 考生看到了什么、谁在评、什么时候」。**
-- 少任何一列，污染检测就少一条可查的证据。
CREATE TABLE IF NOT EXISTS v3_stage_items (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    experiment_id   INTEGER NOT NULL REFERENCES v3_experiments(id) ON DELETE CASCADE,
    exp_group       TEXT    NOT NULL,   -- 'learning' | 'control'
    stage           TEXT    NOT NULL,   -- 见 STAGES_V3 十阶段
    seq             INTEGER NOT NULL,   -- 该阶段在固定顺序里的第几步
    item_id         TEXT    NOT NULL,
    knowledge_point TEXT    NOT NULL,
    question_type   TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL DEFAULT '',
    score           REAL,
    --考生实际看到的上下文。盲考时必须为空串，
    -- 非空就是污染，contamination_check 会直接拦。
    context_used    TEXT    NOT NULL DEFAULT '',
    model           TEXT    NOT NULL DEFAULT '',
    provider        TEXT    NOT NULL DEFAULT '',
    evaluator       TEXT    NOT NULL DEFAULT '',
    reference_answer TEXT   NOT NULL DEFAULT '',
    mode            TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_v3_items
    ON v3_stage_items(run_id, exp_group, stage);
CREATE INDEX IF NOT EXISTS idx_v3_items_exp
    ON v3_stage_items(experiment_id);

-- 客观 held-out 任务判分（**机器判定，全程不过 LLM**）
CREATE TABLE IF NOT EXISTS v3_objective_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    experiment_id     INTEGER NOT NULL REFERENCES v3_experiments(id) ON DELETE CASCADE,
    exp_group         TEXT    NOT NULL,
    stage             TEXT    NOT NULL,   -- 'pre_objective' | 'post_objective'
    seq               INTEGER NOT NULL DEFAULT 0,
    task_id           TEXT    NOT NULL,
    kp                TEXT    NOT NULL,
    kind              TEXT    NOT NULL,
    candidate_output  TEXT    NOT NULL DEFAULT '',
    checks_json       TEXT    NOT NULL DEFAULT '[]',
    objective_score   REAL    NOT NULL DEFAULT 0,
    objective_passed  INTEGER NOT NULL DEFAULT 0,
    parse_error       TEXT    NOT NULL DEFAULT '',
    model             TEXT    NOT NULL DEFAULT '',
    provider          TEXT    NOT NULL DEFAULT '',
    context_used      TEXT    NOT NULL DEFAULT '',
    created_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_v3_obj
    ON v3_objective_runs(run_id, exp_group, stage);

-- 调用失败单独一张表。
--
-- 早先考生崩溃被记成一句「（考生调用失败：XXX）」的文本，
-- 照样送进评估器打分——于是「调用炸了」和「模型说不会」
-- 在数据里长得一模一样，两组都拿到同一个中间分，
-- difference-in-gains 凭空归零而报告里看不出任何异常。
-- 失败必须**结构化落库并中止 run**，不能抹平成一个分数。
CREATE TABLE IF NOT EXISTS v3_call_failures (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    exp_group     TEXT    NOT NULL,
    stage         TEXT    NOT NULL,
    item_id       TEXT    NOT NULL,
    error         TEXT    NOT NULL,
    provider      TEXT    NOT NULL DEFAULT '',
    model         TEXT    NOT NULL DEFAULT '',
    created_at    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_v3_fail
    ON v3_call_failures(run_id, exp_group, stage);

-- V0.6 列迁移见 MIGRATIONS_V6
"""

#: V0.6 的实验组名
GROUP_LEARNING = "learning"
GROUP_CONTROL = "control"
GROUPS_V3: tuple[str, str] = (GROUP_LEARNING, GROUP_CONTROL)

#: 固定十阶段顺序。**顺序本身就是实验设计的一部分**：
#: 两组的 PRE 都在任何材料介入之前跑完，之后才交替给材料。
#: 若Learning 组先跑完 POST 再给Control 组跑 PRE，
#: Control 的 PRE 就发生在 Learning 的学习之后——虽然两个组用的是
#: 不同 context，但同一模型实例可能已经从上一轮的 prompt 里
#: 残留了目标知识，Control 的 PRE 会被抬高，difference 被压低。
STAGES_V3: tuple[str, ...] = (
    "pre_learning",           # 1
    "pre_control",            # 2
    "study_learning",         # 3
    "control_material",       # 4
    "practice_learning",      # 5
    "practice_control",       # 6
    "post_learning",          # 7
    "post_control",           # 8
    "transfer_learning",      # 9
    "transfer_control",       # 10
)

#: 盲考阶段：context_used 必须为空
BLIND_STAGES_V3: frozenset[str] = frozenset(
    {
        "pre_learning", "pre_control",
        "post_learning", "post_control",
        "transfer_learning", "transfer_control",
        "pre_objective", "post_objective",
    }
)

MIGRATIONS_V6: tuple[tuple[str, str, str], ...] = (
    # V2 复用了 learning_notes 并加了 v2_experiment_id，
    # V3 同样复用它，所以也要一份 v3_experiment_id。
    # 两者都留着：老代码读 v2_experiment_id，新代码读 v3_experiment_id，
    # 谁也不破坏谁。
    ("learning_notes", "v3_experiment_id", "INTEGER"),
    # 这五张表 V0.5 就存在，但都**没有实验维度**，
    # 于是 V0.5 阶段模型写进去的 experiences / training_samples /
    # tasks / evaluations / materials 无法区分属于哪次实验——
    # 跨实验污染既查不出来也防不住。
    # V0.6 给它们补 `v3_experiment_id`：
    #   两组各写各的 id，污染检测靠「这个 id 被几组用过」来判定。
    ("experiences", "v3_experiment_id", "INTEGER"),
    ("training_samples", "v3_experiment_id", "INTEGER"),
    ("tasks", "v3_experiment_id", "INTEGER"),
    ("evaluations", "v3_experiment_id", "INTEGER"),
    ("materials", "v3_experiment_id", "INTEGER"),
    # materials 还要记来源：对照组材料必须能一眼认出是 control，
    # 否则报告里两份材料混在一起看不出谁是谁
    ("materials", "exp_group", "TEXT NOT NULL DEFAULT ''"),
    # 客观任务也要记seq。
    # 没有它，practice_learning / practice_control /
    # pre_objective / post_objective 四个阶段在顺序审计里
    # **完全不存在**——声称「十阶段顺序」的门禁其实只覆盖八个，
    # 而且报告里还写着 clean=True。这类缺口比没有门禁更难发现。
    ("v3_objective_runs", "seq", "INTEGER NOT NULL DEFAULT 0"),
)

# V0.3 需要对既有表做的迁移：(表名, 列名, 列定义)
# _migrate() 会先查 pragma table_info，已存在的列直接跳过——幂等，
# 因此可以安全地与 V0.2 的表结构共存（pre_score/post_score 在 V0.2 已有）。
MIGRATIONS_V3: tuple[tuple[str, str, str], ...] = (
    ("knowledge", "experiment_id", "INTEGER"),
    ("learning_notes", "verified", "INTEGER NOT NULL DEFAULT 0"),
    ("experiment_items", "mode", "TEXT NOT NULL DEFAULT 'blind_model'"),
    ("experiment_items", "context_used", "TEXT NOT NULL DEFAULT ''"),
    ("experiments", "open_book_score", "REAL"),
)

# V0.4 列迁移：把 V0.3 的「三场考试」扩成
#   PRE / MEMORY_OPEN / MEMORY_BLIND / TRANSFER_BLIND
# 加上批次归属与跨次统计所需的列。
MIGRATIONS_V4: tuple[tuple[str, str, str], ...] = (
    # 批次归属：把同一配置的多次跑串起来
    ("experiments", "batch_id", "TEXT NOT NULL DEFAULT ''"),
    ("experiments", "trial_no", "INTEGER NOT NULL DEFAULT 1"),
    # V0.4 的四个分数
    ("experiments", "memory_open_score", "REAL"),
    ("experiments", "memory_blind_score", "REAL"),
    ("experiments", "transfer_blind_score", "REAL"),
    # 两个新指标：实践可用率、实践均分、笔记质量
    ("experiments", "practice_usable_rate", "REAL"),
    ("experiments", "practice_avg_score", "REAL"),
    ("experiments", "note_quality_avg", "REAL"),
    # exam_kind 区分这4 项属于 memory 卷还是 transfer 卷
    ("experiment_items", "exam_kind", "TEXT NOT NULL DEFAULT 'memory'"),
    # 笔记质量结果直接挂在 note 上，便于单份复核
    ("learning_notes", "quality_total", "REAL"),
    ("learning_notes", "quality_verdict", "TEXT NOT NULL DEFAULT ''"),
    # 实践评审同样要留痕：规则兜底的分和真判分的分不是一回事
    ("practice_evaluations", "evaluator", "TEXT NOT NULL DEFAULT 'llm_judge'"),
)


class Database:
    """薄封装的 SQLite 访问对象。"""

    def __init__(self, path: str = "agent.db") -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")

    # ---------- 生命周期 ----------
    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.executescript(SCHEMA_V2)
        self.conn.executescript(SCHEMA_V3)
        self.conn.executescript(SCHEMA_V4)
        self.conn.executescript(SCHEMA_V5)
        self.conn.executescript(SCHEMA_V6)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """幂等列迁移：已存在的列跳过，不存在则 ADD COLUMN。

        V0.3 / V0.4 / V0.5 / V0.6 的迁移叠在一起跑 ——
        先 V0.3 再 V0.4 再 V0.5 再 V0.6，
        因为后一版的部分列依赖前一版建出来的表。
        """
        for migration in (
            MIGRATIONS_V3, MIGRATIONS_V4, MIGRATIONS_V5, MIGRATIONS_V6
        ):
            for table, column, definition in migration:
                existing = {
                    r["name"] for r in self.query_all(f"PRAGMA table_info({table})")
                }
                if not existing:
                    continue  # 表不存在（不该发生，init_schema 已保证）
                if column in existing:
                    continue
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- 基础操作 ----------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        self.conn.executemany(sql, rows)
        self.conn.commit()

    def query_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        cur = self.conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------- 便捷统计 ----------
    def table_names(self) -> list[str]:
        rows = self.query_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        return [r["name"] for r in rows]

    def count(self, table: str) -> int:
        return int(self.scalar(f"SELECT COUNT(*) FROM {table}") or 0)
