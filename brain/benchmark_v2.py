"""Benchmark V2 —— V0.5 固定题库：4 知识点 × 6 题 = 24 题。

V0.4.1 留下的三个统计缺陷，本题库逐一对应解决
------------------------------------------------
1. **题量不足** → 每 kp 1 题 = 单题判分误差直接等于整个 kp 的分数。
   V2 每 kp 6 题，逐 kp 结论有了统计基础。

2. **天花板效应** → 换材料（`brain/novel_material.py` 的 Zephyra 虚构系统）。
   V2 的题全部基于虚构规则，**模型不可能靠先验知识答对**：
   LumenHash、Z-8 分桶、Tombstone 压缩、Z-Tag 在现实世界里不存在。
   这不是「题目变难了」，而是**先验知识被切断了**——
   V0.4.1 的 PRE 是 90 分，根因不是题难，而是题问的是模型已经会的东西。

3. **难度不可控** → 每 kp 固定 3 种题型，题型本身就是难度刻度：
   basic（直接套定义）< application（多步组合）< transfer（换场景用同一规则）。
   报告可按题型拆开看，不再依赖主观的difficulty 标签。

24 题如何切分成三场考试
------------------------
每 kp 6 题按题型均分给三场，每场 8 题（每 kp 2 题）：

    PRE_BLIND          basic        8 题   学习前裸能力，基线
    MEMORY OPEN_BOOK  ┐application  8 题   同一批题，先开卷再盲考
    MEMORY BLIND      ┘                测：拿掉资料还记不记得住
    TRANSFER BLINDtransfer        8 题   全新场景同一知识点，测是否形成概念

`question_type` 与 `exam_kind` 是**两个字段**，别混：
    question_type = basic | application | transfer  —— 题目形态（决定难度）
    exam_kind     = pre | memory | transfer       —— 哪一场考试

**为什么每场只考 8 题而不是 24 题全考**
--------------------------------------
一次会话考 24 题会产生两个问题：模型在同一次会话里被同一知识点问 6 次，
后面的题会受前面题的影响（上下文效应）；而且耗时是 3 倍。
拆成 3 × 8 让每场独立、可单独复跑、彼此不污染。

**题型与考试卷的绑定是刻意的**
------------------------------
basic 只在 PRE 考、transfer 只在 TRANSFER 考。这意味着
「题型难度」在三场之间**系统性变化**——这是设计选择而非缺陷，
因为三场本来就在回答不同问题（会不会 / 记不记得 / 会不会用）。
真正需要配平的是**同一场内部**四个 kp 的难度，
这一点由「每 kp 题型数量相同」保证。

**TRANSFER 的 premise_in_question 强制为 "asserted"**
-----------------------------------------------------
V0.4.1 最痛的一次教训：`tr_covering_01` 问「要让它是覆盖索引还缺什么」，
但按题面字面推它已经是覆盖索引了，模型答「什么都不缺」**在逻辑上是对的**，
却被判 45分。**题目歧义会伪装成模型能力不足。**

V2 的铁律：**每道 transfer 题都必须把需要判断的前提直接写进题面**，
不能靠考生自己猜题目想问什么。`assert_bank_intact()` 是硬门禁。

**题库不可被模型修改**：题目只存在于本模块的元组常量里，
LLM 出题接口在实验流程中零调用；入库记 SHA256 指纹，
实验开始与结束各校验一次（`BankStore.verify`）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, NamedTuple

BANK_VERSION = "v2.0.0"

#: 四个知识点（与 novel_material 的五节规则对应）
KP_LABELS: dict[str, str] = {
    "kp_lumen_hash": "LumenHash 函数与桶号计算",
    "kp_lumen_bucket": "Z-8 分桶与分裂规则",
    "kp_lumen_compact": "Tombstone 与压缩触发",
    "kp_lumen_consist": "Epoch 标签与读校验",
}

QUESTION_TYPES = ("basic", "application", "transfer")

#: 每个知识点总题量
QUESTIONS_PER_KP = 6

#: 每种题型每个知识点的题量（2 + 2 + 2 = 6）
QUESTIONS_PER_TYPE = 2


class ExamItem(NamedTuple):
    """一道固定题目。

    `premise_in_question` 沿用V0.4.1 的显式声明（不靠关键词猜）：
        "yes"      前提在题面里，无需额外假设
        "asserted" 题面直接给出一个（可能错误的）说法让考生判断
        "no"       题面不含判断依据 → 题目不可解，**门禁直接拒绝跑**
    """

    item_id: str
    kp: str
    question: str
    answer: str
    question_type: str
    difficulty: str = "medium"
    premise_in_question: str = "yes"


# ==========================================================================
# PRE 卷：basic，每 kp 2 题，共 8 题
# ==========================================================================

PRE_ITEMS: tuple[ExamItem, ...] = (
    # ---------------- kp_lumen_hash (2) ----------------
    ExamItem(
        item_id="v2_pre_hash_01",
        kp="kp_lumen_hash",
        question_type="basic",
        question=(
            "Zephyra 的 LumenHash 函数对非空字符串 key 的计算过程是："
            "先令 h = 7，然后从左到右遍历 key 的每个字符 c，"
            "执行 h = (h * 131 + ord(c)) mod 65537，最后返回 h。"
            "请计算 LumenHash(\"ab\") 的值，并写出计算过程中每一步的中间值。"
        ),
        answer=(
            "初始 h = 7。\n"
            "第一个字符 'a'：ord('a') = 97，"
            "h = (7 * 131 + 97) mod 65537 = (917 + 97) mod 65537 = 1014。\n"
            "第二个字符 'b'：ord('b') = 98，"
            "h = (1014 * 131 + 98) mod 65537 = (132834 + 98) mod 65537 "
            "= 132932 mod 65537。\n"
            "132932 大于 65537，需减两次：132932 - 65537 = 67395，"
            "67395 - 65537 = 1858。\n"
            "所以 LumenHash(\"ab\") = 1858。\n"
            "易错点：模 65537 要反复相减直到结果小于 65537，"
            "只减一次得到 67395 是错的。"
        ),
        difficulty="medium",
    ),
    ExamItem(
        item_id="v2_pre_hash_02",
        kp="kp_lumen_hash",
        question_type="basic",
        question=(
            "已知 Zephyra 的分桶规则是 bucket(key) = LumenHash(key) mod 8，"
            "且 LumenHash(\"ab\") = 1858。"
            "请计算键 \"ab\" 属于哪个桶，"
            "并说明「LumenHash 值相同的两个键」是否一定属于同一个桶。"
        ),
        answer=(
            "bucket(\"ab\") = 1858 mod 8。\n"
            "1858 = 8 × 232 + 2，所以 1858 mod 8 = 2。\n"
            "因此键 \"ab\" 属于 2 号桶。\n"
            "LumenHash 相同的两个键**一定同桶**："
            "因为桶号只由 LumenHash(key) mod 8 决定，"
            "LumenHash 一旦相同，取模结果必然相同，"
            "与键的长度和内容无关。\n"
            "反方向不成立：LumenHash 不同的两个键**可能**同桶"
            "（因为要对 8 取模，不同的哈希值会撞进同一个桶）。"
            "所以「LumenHash 相同」是同桶的充分条件，但不是必要条件。"
        ),
        difficulty="medium",
    ),
    # ---------------- kp_lumen_bucket (2) ----------------
    ExamItem(
        item_id="v2_pre_bucket_01",
        kp="kp_lumen_bucket",
        question_type="basic",
        question=(
            "Zephyra 有 8 个桶（编号 0-7），每桶最多容纳 4 个活跃键，"
            "分桶规则是 bucket(key) = LumenHash(key) mod 8。"
            "请计算 LumenHash = 132 的键属于哪个桶，"
            "并说明该桶的活跃键数会不会因为写入这个键而达到 5 个。"
        ),
        answer=(
            "bucket = 132 mod 8 = 4（132 = 8 × 16 + 4）。\n"
            "所以该键属于 4 号桶。\n"
            "**不必然达到 5 个。**每桶上限是 4 个活跃键，"
            "触发分裂的条件是「桶里已有 4 个活跃键」。"
            "如果该桶当前只有 0-3 个活跃键，写入后最多变成 4 个，不会超。\n"
            "只有当该桶当前恰好已有 4 个活跃键时，写入才会让它变成 5 个；"
            "而按材料的一级分桶规则，分裂并不能减少负载，"
            "第 5 个键会留在原桶，导致超限。"
        ),
        difficulty="medium",
    ),
    ExamItem(
        item_id="v2_pre_bucket_02",
        kp="kp_lumen_bucket",
        question_type="basic",
        question=(
            "Zephyra 的分裂规则是：当桶已满（4 个活跃键）时触发分裂，"
            "桶内键若满足 LumenHash mod 16 >= 8 则迁移到 (桶号 + 8) mod 8。"
            "请回答：桶号 3 在分裂后会变成几号桶？请写出计算过程并说明原因。"
        ),
        answer=(
            "迁移目标桶号 = (3 + 8) mod 8 = 11 mod 8 = 3。\n"
            "所以仍然是 3 号桶，桶号没有变化。\n"
            "原因：对任何 0-7 的桶号 b，都有 (b + 8) mod 8 = b，"
            "因为 +8 恰好是一个完整周期。\n"
            "这意味着一级分桶的分裂是**恒等变换**："
            "无论 LumenHash mod 16 的判断结果如何，键都不离开原来的桶，"
            "分裂做了一次搬移工作但负载一点没变。这是 Zephyra 的设计缺陷。"
        ),
        difficulty="hard",
    ),
    # ---------------- kp_lumen_compact (2) ----------------
    ExamItem(
        item_id="v2_pre_compact_01",
        kp="kp_lumen_compact",
        question_type="basic",
        question=(
            "Zephyra 删除键时留下 Tombstone（占用槽位但不算活跃键）。"
            "压缩的触发条件是：Tombstone 数量 >= 2 **且** 活跃键数量 <= 1。"
            "现桶内有 2 个活跃键、3 个 Tombstone。请回答：这个桶会不会触发压缩？"
        ),
        answer=(
            "**不会触发。**两个条件是「且」的关系，必须同时成立：\n"
            "  条件 A：Tombstone = 3 >= 2，**成立**。\n"
            "  条件 B：活跃键 = 2 <= 1，**不成立**。\n"
            "只要有一个条件不成立就不压缩，所以本桶维持原状。\n"
            "这是最典型的陷阱：Tombstone 数量看起来「已经超标 3 个了」，"
            "但压缩还要求活跃键很少，而这里活跃键有 2 个，不满足。"
        ),
        difficulty="medium",
    ),
    ExamItem(
        item_id="v2_pre_compact_02",
        kp="kp_lumen_compact",
        question_type="basic",
        question=(
            "Zephyra 的压缩执行三个动作：清除所有 Tombstone、"
            "按 LumenHash 从小到大重排活跃键、epoch 加 1。"
            "已知 epoch 初始为 0，某个键 K 的 LumenHash = 4000。"
            "请回答：K 所在的桶触发压缩后，K 的 Z-Tag 还能通过读取校验吗？"
        ),
        answer=(
            "**不能通过，读取会被拒绝。**\n"
            "K 在 epoch = 0 时写入的 Z-Tag = (4000 + 0) mod 65537 = 4000。\n"
            "压缩使 epoch 加 1 变为 1，读取时期望的标签是 "
            "(4000 + 1) mod 65537 = 4001。\n"
            "实际存储的标签是 4000，期望是 4001，不相等 → 校验失败，读取被拒。\n"
            "根因：Z-Tag 只在写入那一刻按当时的 epoch 算好并存储，"
            "之后不会自动跟着 epoch 更新。"
        ),
        difficulty="hard",
    ),
    # ---------------- kp_lumen_consist (2) ----------------
    ExamItem(
        item_id="v2_pre_consist_01",
        kp="kp_lumen_consist",
        question_type="basic",
        question=(
            "Zephyra 中 ZTag(key) = (LumenHash(key) + epoch) mod 65537，"
            "epoch 初始为 0。"
            "请回答：epoch = 0 时，一个键的 Z-Tag 与它的 LumenHash 有什么关系？"
        ),
        answer=(
            "epoch = 0 时，ZTag(key) = (LumenHash(key) + 0) mod 65537 "
            "= LumenHash(key) mod 65537。\n"
            "而 LumenHash 的返回值范围是 0 到 65536，本身就小于 65537，"
            "再取模不会改变它。\n"
            "所以 **epoch = 0 时 Z-Tag 就等于 LumenHash 的值**。\n"
            "这个结论说明：初始化状态下标签校验是恒等的，"
            "不会出现「标签算错」这类问题；"
            "也说明 Z-Tag 的所有变化都来自 epoch 的累加。"
        ),
        difficulty="medium",
    ),
    ExamItem(
        item_id="v2_pre_consist_02",
        kp="kp_lumen_consist",
        question_type="basic",
        question=(
            "Zephyra 中 ZTag(key) = (LumenHash(key) + epoch) mod 65537。"
            "已知某键的 LumenHash = 4000，当前 epoch = 2。"
            "请计算该键当前的 Z-Tag 应该是多少，"
            "并说明如果它是在 epoch = 0 时写入的，现在能否被正常读取。"
        ),
        answer=(
            "ZTag = (4000 + 2) mod 65537 = **4002**（4002 < 65537，无需再取模）。\n"
            "如果该键是在 epoch = 0 时写入的，"
            "它存储的标签是 (4000 + 0) mod 65537 = 4000。\n"
            "现在读取时期望的标签是 4002，实际存的是 4000，"
            "两者不等 → **校验失败，读不出来**。\n"
            "要恢复可读，必须把它的 Z-Tag 重算为 4002 并更新。"
        ),
        difficulty="medium",
    ),
)


# ==========================================================================
# MEMORY 卷：application，每 kp 2 题，共 8 题
#
# 与 PRE 的差异必须**只在表面**：换键名、换哈希值、换桶号，
# 但解题要用到的规则完全相同。
# 如果 MEMORY 卷换了规则，它就变成 TRANSFER 卷了，
# 「记住 vs 学会」这个对比也就失效了。
# ==========================================================================

MEMORY_ITEMS: tuple[ExamItem, ...] = (
    # ---------------- kp_lumen_hash (2) ----------------
    ExamItem(
        item_id="v2_mem_hash_01",
        kp="kp_lumen_hash",
        question_type="application",
        question=(
            "Zephyra 的 LumenHash 定义为 h = (h * 131 + ord(c)) mod 65537，初值 7。"
            "已知 LumenHash(\"ab\") = 1858。"
            "请计算键 \"abc\" 的 LumenHash 值，"
            "并回答：\"abc\" 和 \"ab\" 的桶号是否必然不同？"
        ),
        answer=(
            "第三步：ord('c') = 99。\n"
            "h = (1858 * 131 + 99) mod 65537 = (243398 + 99) mod 65537 "
            "= 243497 mod 65537。\n"
            "反复减65537：243497 - 65537 = 177960；177960 - 65537 = 112423；"
            "112423 - 65537 = 46886。\n"
            "所以 LumenHash(\"abc\") = 46886。\n"
            "桶号：bucket(\"ab\") = 1858 mod 8 = 2（1858 = 8 × 232 + 2）；"
            "bucket(\"abc\") = 46886 mod 8 = 6（46886 = 8 × 5860 + **6**）。"
            "本题里两者**不同**：一个是 2，一个是 6。\n"
            "但**不必然不同**：追加字符后哈希值从 1858 变到 46886，"
            "变化很大，但对 8 取模的结果仍可能撞回同一个桶。"
            "所以无法从「追加了一个字符」推出桶号必然改变，"
            "只能把哈希值算出来再逐个取模。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="v2_mem_hash_02",
        kp="kp_lumen_hash",
        question_type="application",
        question=(
            "Zephyra 的 LumenHash 初值为 7，每字符执行 "
            "h = (h * 131 + ord(c)) mod 65537。"
            "工程师想优化：把「遍历全部字符」改成「只遍历前 3 个字符」，"
            "理由是「后缀字符不影响分桶，因为分桶只看 mod 8」。"
            "请判断这个说法是否成立，并说明理由。"
        ),
        answer=(
            "**不成立。**分桶只看 mod 8 没错，"
            "但「后缀不影响分桶」和「后缀不影响 LumenHash」是两件事，"
            "而材料里好几个机制都依赖**完整的 LumenHash 值**，不只是 mod 8 的结果。\n"
            "受影响的机制至少三处：\n"
            "1. **Z-Tag**：ZTag = (LumenHash + epoch) mod 65537，用的是完整哈希值。"
            "LumenHash 一变，全库存量键的标签全部失配，读不出来。\n"
            "2. **压缩时的重排**：压缩会把活跃键按 LumenHash 从小到大重排，"
            "截断后排序结果会变，槽位位置跟着变。\n"
            "3. **同桶判定**：材料规定「按 LumenHash 相等判定是同一个键」，"
            "截断后两个不同键可能变得 LumenHash 相等，被误判为更新而非新增。\n"
            "所以这是「看似只影响一处、实际破坏全局」的典型错误。"
        ),
        difficulty="hard",
    ),
    # ---------------- kp_lumen_bucket (2) ----------------
    ExamItem(
        item_id="v2_mem_bucket_01",
        kp="kp_lumen_bucket",
        question_type="application",
        question=(
            "Zephyra 的桶 b 有 4 个活跃键，它们的 LumenHash 值分别是 "
            "24、25、32、33。现在要写入第 5 个键，它的 LumenHash = 40。"
            "请写出触发分裂后这 5 个键各自的桶号，并指出这个结果说明了什么。"
        ),
        answer=(
            "先算各键的桶号（LumenHash mod 8）：\n"
            "  24 mod 8 = 0，25 mod 8 = 1，32 mod 8 = 0，33 mod 8 = 1，40 mod 8 = 0。\n"
            "**这里必须先指出题面的一处不一致**："
            "题设说「桶 b 有这 4 个键」，但 LumenHash = 24 的键在 0 号桶、"
            "= 33 的键在 1 号桶，一个桶不可能同时装下桶号不同的键。"
            "所以严格按题面算，这五个键分布在 0、1 两个桶里，"
            "「桶 b 有 4 个活跃键」这个前提不成立。\n"
            "若按「四个键桶号都是 0」理解（取 LumenHash = 0、1、8、9，第 5 个取 16）：\n"
            "  0 的 mod 16 = 0 < 8 → 留 0 号桶；1 的 mod 16 = 1 < 8 → 留 0 号桶；\n"
            "  8 的 mod 16 = 8 >= 8 → 迁移到 (0+8) mod 8 = 0；\n"
            "  9 的 mod 16 = 9 >= 8 → 迁移到 (0+8) mod 8 = 0；\n"
            "  16 的 mod 16 = 0 < 8 → 留 0 号桶。\n"
            "五个键桶号全是 0，一个都没离开，桶从 4 个变成 5 个、**超限**。\n"
            "结论：(b+8) mod 8 = b 是恒等变换，一级分桶的分裂无效。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="v2_mem_bucket_02",
        kp="kp_lumen_bucket",
        question_type="application",
        question=(
            "Zephyra 的分裂无效已被确认。工程师提出第二个方案："
            "分裂时把「一半的键」按 LumenHash 排序后，"
            "把排序靠后的那部分键的 **Z-Tag** 重算一遍，"
            "理由是「这样至少能让标签保持新鲜」。"
            "请判断这个方案能否解决桶超载问题，并说明理由。"
        ),
        answer=(
            "**不能。**因为 Z-Tag 与负载分布是**两个完全正交的维度**。\n"
            "Z-Tag = (LumenHash + epoch) mod 65537，"
            "它只决定「这个键能不能被读出来」，完全不决定「这个键在哪个桶」。\n"
            "重算 Z-Tag 不会让任何键离开这个桶，"
            "桶的活跃键数还是 4（第 5 个键写入后还是 5），超载问题原封不动。\n"
            "而且这个方案与压缩的职责重叠："
            "epoch 没变时重算出来的 Z-Tag 与原来的一模一样，纯属白做；"
            "epoch 变了才需要重算，而材料明确指出压缩"
            "（唯一会改 epoch 的操作）已经让所有键失效、重算是必需的。\n"
            "所以「分裂时重算标签」是拿分裂去做压缩的活，两件事都没做对。"
        ),
        difficulty="hard",
    ),
    # ---------------- kp_lumen_compact (2) ----------------
    ExamItem(
        item_id="v2_mem_compact_01",
        kp="kp_lumen_compact",
        question_type="application",
        question=(
            "Zephyra 某桶有 1 个活跃键（键 K，LumenHash = 8888）、2 个 Tombstone，"
            "压缩触发。请回答：(1) 压缩后该桶占用几个槽位；"
            "(2) K 的桶号会不会变，为什么。"
        ),
        answer=(
            "(1) 压缩后该桶占用 **1 个槽位**。\n"
            "原先是 3 个槽位（1 活跃 + 2 墓碑）。压缩会清除所有 Tombstone "
            "并回收槽位，剩下的只有 K 占用的 1 个。"
            "依据：压缩的三个动作（清墓碑 / 重排 / epoch 加 1）里，"
            "没有任何一个会增加槽位，所以槽位只会减少。\n"
            "(2) K 的桶号**不会变**。\n"
            "桶号只由 bucket = LumenHash mod 8 决定，"
            "而重排只改变槽位顺序，不改变 LumenHash 的值，"
            "所以 K 仍在 8888 mod 8 = 0 号桶。\n"
            "一句话概括：压缩改变的是**槽位位置**，不改变**桶归属**。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="v2_mem_compact_02",
        kp="kp_lumen_compact",
        question_type="application",
        question=(
            "Zephyra 一个桶里有 1 个活跃键、2 个 Tombstone，压缩触发。"
            "有人认为「压缩后 epoch 变了，所以这个桶里的键需要被删除再重新插入，"
            "否则标签永远失效」。请判断这个说法是否成立，并给出更准确的处置方式。"
        ),
        answer=(
            "**前半部分对，后半部分错。**\n"
            "对的部分：epoch 变了，键原本的 Z-Tag = (LumenHash + 旧 epoch) mod 65537 "
            "确实失配了，读不出来。\n"
            "错的部分：**不需要删除再重新插入**。删除再插入是多余的，"
            "而且有副作用：删除会在原位留下一个 Tombstone。"
            "删除后这个桶变成 0 个活跃键 + 3 个 Tombstone，"
            "满足条件 A（3 >= 2）和条件 B（0 <= 1），**又触发一次压缩**，"
            "epoch 再加 1，于是刚重算的标签又失效了。\n"
            "正确的处置方式是：直接**重算该键的 Z-Tag** = "
            "(LumenHash + 当前 epoch) mod 65537 并更新，不走删除流程。\n"
            "这印证了材料里的结论：压缩的真正代价不是压缩本身，"
            "而是它触发的**全量标签重算**。"
        ),
        difficulty="hard",
    ),
    # ---------------- kp_lumen_consist (2) ----------------
    ExamItem(
        item_id="v2_mem_consist_01",
        kp="kp_lumen_consist",
        question_type="application",
        question=(
            "Zephyra 某键 K 的 LumenHash = 2500，它在 epoch = 0 时写入且标签正确。"
            "此后发生了 1 次压缩，压缩后又发生了 1 次压缩。"
            "请回答：(1) 现在 K 读得出来吗；(2) K 的槽位位置会不会因为压缩而改变。"
        ),
        answer=(
            "(1) **读不出来。**逐步推：\n"
            "  存入标签 = (2500 + 0) mod 65537 = 2500。\n"
            "  第 1 次压缩后 epoch = 1，期望 (2500 + 1) = 2501，实际存 2500 → 失配。\n"
            "  第 2 次压缩后 epoch = 2，期望 2502，实际存 2500 → 仍失配。\n"
            "所以 K 的标签自 epoch 离开 0 那一刻起就永久失效了，除非有人重算它。\n"
            "(2) **槽位位置会变。**压缩的三个动作之一是"
            "「把桶内所有活跃键按 LumenHash 从小到大重新排序」，"
            "只要 K 所在的桶触发了压缩，K 的槽位就可能被重排到别的位置。\n"
            "注意耦合关系：**epoch 的变化是全局的**（任何桶压缩都会改 epoch），"
            "**槽位变化是局部的**（只有 K 自己那个桶会重排）。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="v2_mem_consist_02",
        kp="kp_lumen_consist",
        question_type="application",
        question=(
            "Zephyra 的 epoch 初始为 0。运维观察到：一次压缩之后，"
            "全库「读取被拒」的键数量与「活跃键总数」几乎相等。"
            "请解释这个现象，并指出它对 Zephyra 可用性的影响。"
        ),
        answer=(
            "**现象的原因**：一次压缩让全局 epoch 加 1，"
            "而 ZTag = (LumenHash + epoch) mod 65537，"
            "所有存量键的标签都是按**旧 epoch** 算出来并存储的。"
            "epoch 一变，全库每一个键的期望标签都变了，"
            "于是**每一个键的校验都失配**，读取全部被拒。\n"
            "所以「读取被拒的键数 ≈ 活跃键总数」不是巧合，"
            "而是 epoch 作为全局变量的必然结果。\n"
            "对可用性的影响：这是**全局性的短暂不可用**。"
            "在「全量重算标签」完成之前，Zephyra 实际上读不出任何数据。"
            "如果重算需要扫描全库、耗时较长，"
            "那么压缩的代价就从材料描述的「压缩很廉价」"
            "变成了「一次可能持续很久的全库停机」。\n"
            "这正是「一级分桶」与「全局 epoch」两个设计叠加出来的后果："
            "**局部触发的操作造成了全局影响**。"
        ),
        difficulty="hard",
    ),
)


# ==========================================================================
# TRANSFER 卷：transfer，每 kp 2 题，共 8 题
#
# 判卷标准（照 V0.4.1 的血泪教训定下来）：
#
# 1. **premise_in_question 必须为 "asserted"**
#    题面里必须**显式给出需要判断的前提或一个错误说法**，
#    让考生去驳。不允许「题意暗示型」出题——
#    那种题测的是「能不能猜到出题人在想什么」，不是「会不会用规则」。
#    V0.4.1 的 tr_covering_01 就是反面教材：模型答「什么都不缺」是对的，
#    却被判 45 分。**题目歧义会伪装成模型能力不足。**
#
# 2. **场景必须是新的，但不能引入材料没讲的概念**
#    学的是「(b+8) mod 8 = b 所以分裂无效」，
#    transfer 不能重问分裂怎么算，要问「容量规划/键空间设计」这类
#    需要用到「分裂无效」这个结论的新场景。
#    答案只能由材料已给规则推出，不能搬现实世界的扩容哈希表知识。
#
# 3. **答案必须唯一**
#    凡是「可以有多种合理答案」的题一律不出一道，
#    因为判分器无法公平判多个正确答案。
# ==========================================================================

TRANSFER_ITEMS: tuple[ExamItem, ...] = (
    # ---------------- kp_lumen_hash (2) ----------------
    ExamItem(
        item_id="v2_tr_hash_01",
        kp="kp_lumen_hash",
        question_type="transfer",
        premise_in_question="asserted",
        question=(
            "Zephyra 的容量规划团队要预估「某个桶会因为超载而无法写入」的概率，"
            "他们因此提出一个前提：「只要把 LumenHash 的模数从 65537 改成 65536，"
            "就能让键在 8 个桶里分布得更均匀，从而减少超载」。"
            "请判断这个前提是否成立，并说明理由。"
        ),
        answer=(
            "**不成立。**理由是这个前提混淆了两个不同的模数：\n"
            "  - 65537 是 LumenHash **内部**的模数："
            "h = (h * 131 + ord(c)) mod 65537。改它会改变所有键的哈希值本身。\n"
            "  - 8 是**分桶**的模数：bucket = LumenHash mod 8。\n"
            "均匀性由分桶模数 8 决定，**与哈希内部模数 65537 无关**。"
            "把 65537 改成 65536（甚至改成任意别的数）"
            "并不改变「对 8 取模」这回事，"
            "所以不可能让 8 个桶分布得更均匀。\n"
            "更严重的是副作用：改哈希内部模数会让全库存量键的 LumenHash 变，"
            "于是 (1) 所有键的 Z-Tag 失配，读不出来；"
            "(2) 键的桶号可能改变，等于全库重新分桶。\n"
            "所以这个前提不仅无效，而且会破坏 Zephyra 的可用性。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="v2_tr_hash_02",
        kp="kp_lumen_hash",
        question_type="transfer",
        premise_in_question="asserted",
        question=(
            "Zephyra 的运维发现：某个键 K 读不出来，"
            "排查后确认它的 LumenHash 本身没算错，"
            "Z-Tag 也没被改过——它就是当初写入时算好存下来的那个值。"
            "运维团队因此提出一个前提：「既然 LumenHash 和 Z-Tag 都没错，"
            "那一定是这个键的 LumenHash 发生了碰撞，被别的键覆盖了」。"
            "请判断这个前提是否成立，并说明真正可能的原因。"
        ),
        answer=(
            "**不成立。**先看前提里的矛盾："
            "「LumenHash 没算错」和「发生了碰撞」是互斥的——"
            "碰撞的定义就是两个不同键算出了相同的 LumenHash，"
            "而如果 K 的 LumenHash 算错了，那就是另一个问题了。\n"
            "**真正可能的原因只有一个：epoch 变了。**\n"
            "读取校验的条件是 ZTag(key) == (LumenHash(key) + 当前 epoch) mod 65537。"
            "K 的 LumenHash 和存储的 Z-Tag 都没错，"
            "说明这两项都自洽，唯一的变量是**右边的当前 epoch**。\n"
            "只要发生过任何一次压缩，epoch 就加 1，"
            "K 存储的标签（按旧 epoch 算）就不再等于期望值 → 读取被拒。\n"
            "所以正确诊断是：键没被碰撞，是**全局 epoch 变化导致标签失配**，"
            "处置办法是重算它的 Z-Tag。"
        ),
        difficulty="hard",
    ),
    # ---------------- kp_lumen_bucket (2) ----------------
    ExamItem(
        item_id="v2_tr_bucket_01",
        kp="kp_lumen_bucket",
        question_type="transfer",
        premise_in_question="asserted",
        question=(
            "Zephyra 的架构评审会上，有人基于分裂规则提出一个前提："
            "「由于分裂时键会按 LumenHash mod 16 被分到两个不同的桶，"
            "所以 Zephyra 的有效桶数是 16 而不是 8，"
            "容量应当按 16 个桶来规划」。"
            "请判断这个前提是否成立，并说明理由。"
        ),
        answer=(
            "**不成立。**这个前提把「分裂规则里出现了 mod 16」"
            "误读成「系统有 16 个桶」。\n"
            "两点：\n"
            "1. Zephyra 定义的桶数就是 8 个（编号 0-7），"
            "材料明确说「迁移后的桶号 = (bucket + 8) mod 8」——"
            "这个对 8 取模本身就说明桶号空间是 8，不是 16。\n"
            "2. 更根本的是：分裂是**恒等变换**。"
            "对任何桶号 b，(b + 8) mod 8 = b，"
            "所以无论 mod 16 的判断结果如何，键都不离开原桶。"
            "既然「迁移到另一个桶」这件事实际上从未发生，"
            "就不存在「有效桶数变成 16」这回事。\n"
            "所以容量必须按 **8 个桶**规划，"
            "而且由于分裂无效，**每个桶的实际负载可能超过 4 个活跃键的上限**，"
            "规划时不能假设「满了会自动分流」。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="v2_tr_bucket_02",
        kp="kp_lumen_bucket",
        question_type="transfer",
        premise_in_question="asserted",
        question=(
            "Zephyra 要做一次键空间重构，团队提出一个前提："
            "「只要把每个键的 LumenHash 值都减去一个常数 C，"
            "就能让原本挤在同一个桶里的键分散到不同的桶，"
            "从而解决超载」。请判断这个前提是否成立，并说明理由。"
        ),
        answer=(
            "**不成立。**理由在于「减去常数」和「对 8 取模」不能交换顺序。\n"
            "设某键的 LumenHash = h，它当前的桶号是 h mod 8。"
            "改后的哈希是 h - C，桶号变成 (h - C) mod 8。\n"
            "要让「原桶号 → 新桶号」是一个有规律的平移，需要 (h - C) mod 8 "
            "与 h mod 8 之间是固定偏移，这在数学上成立时，"
            "新桶号 = (旧桶号 - C mod 8) mod 8，"
            "本质上是**所有桶整体平移相同的量**——"
            "两个原本同桶的键，平移后仍然同桶。\n"
            "所以这个操作只能整体挪动桶的编号，"
            "**无法把同桶的键拆开**。\n"
            "而按 Zephyra 一级分桶的实际情况，"
            "本来就没有「拆开」的手段可用："
            "分裂是恒等变换，(b+8) mod 8 = b，"
            "所以同桶的键无论用什么偏移量都还在一起。"
            "唯一有效的办法是**增加桶数**（材料提到的二级分桶方向）。"
        ),
        difficulty="hard",
    ),
    # ---------------- kp_lumen_compact (2) ----------------
    ExamItem(
        item_id="v2_tr_compact_01",
        kp="kp_lumen_compact",
        question_type="transfer",
        premise_in_question="asserted",
        question=(
            "Zephyra 的运维手册里写着一条规定："
            "「每当观测到某个桶出现读取被拒，就立刻对该桶执行一次压缩，"
            "以恢复该桶内的键」。团队据此提出一个前提："
            "「这个规定是安全的，因为它只影响出问题的那个桶，"
            "不会波及其他桶」。请判断这个前提是否成立，并说明理由。"
        ),
        answer=(
            "**前半部分（规定不安全）成立，后半部分（只影响那个桶）不成立。**\n"
            "1. **规定不安全**：压缩会让全局epoch 加 1，"
            "而 ZTag = (LumenHash + epoch) mod 65537。"
            "epoch 一变，**全库所有键**的期望标签都变了，"
            "于是所有键的校验都会失配。"
            "也就是说「恢复一个桶」的动作会让**其余本来正常的桶全部读不出来**。"
            "所以每观测到一次读取被拒就压缩一次，"
            "会让故障范围从1 个桶扩大到全库。\n"
            "2. **影响是全局的，不是局部的**："
            "触发压缩的条件只看单个桶（条件 A、B），"
            "但 epoch 是**全局计数器**，"
            "所以任何一次压缩都会改变全库的 Z-Tag 期望值。\n"
            "这正是 Zephyra 里「局部触发 → 全局影响」的典型案例，"
            "和材料里那条推论完全一致。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="v2_tr_compact_02",
        kp="kp_lumen_compact",
        question_type="transfer",
        premise_in_question="asserted",
        question=(
            "Zephyra 的一个桶长期只有 1 个活跃键，但因为删除操作频繁，"
            "积累了大量 Tombstone。团队提出一个前提："
            "「既然这个桶的活跃键一直不超过 1，"
            "压缩条件 A（墓碑 >= 2）迟早会满足，"
            "所以可以放心地不断删除，直到压缩自动兜底清理空间」。"
            "请判断这个前提是否成立，并指出它忽略的两个代价。"
        ),
        answer=(
            "**不成立。**这个前提把「压缩会兜底」当成了「删除是免费的」，"
            "忽略了两个代价：\n"
            "1. **每次压缩都要全库重算 Z-Tag。**"
            "压缩使全局 epoch 加 1，全库所有键的期望标签都变了，"
            "都必须重算才能恢复读取。"
            "所以「频繁删除 → 频繁压缩 → 频繁全库重算」，"
            "一次比一次贵，最后是全库范围的反复停机。\n"
            "2. **墓碑会挤占桶容量。**墓碑占用槽位但不算活跃键，"
            "而每桶最多 4 个活跃键、槽位总数也受桶大小限制。"
            "在压缩触发之前，墓碑是一直占着位置的——"
            "而按一级分桶的实际情况（分裂恒等变换、无法真正分流），"
            "桶满了并不会自动把键挪走。\n"
            "所以正确做法不是「放心删」，而是控制删除节奏，"
            "或在墓碑数接近阈值前主动压缩——"
            "但要知道每一次压缩都要付全库重算的代价。"
        ),
        difficulty="hard",
    ),
    # ---------------- kp_lumen_consist (2) ----------------
    ExamItem(
        item_id="v2_tr_consist_01",
        kp="kp_lumen_consist",
        question_type="transfer",
        premise_in_question="asserted",
        question=(
            "Zephyra 要做一次「标签自愈」改造："
            "在每次读取失败时，顺手把该键的 Z-Tag 重算成当前 epoch 下的正确值。"
            "方案提出一个前提：「这样就彻底解决了标签问题，"
            "以后再也不会出现读取被拒」。请判断这个前提是否成立，"
            "并指出这个方案的固有局限。"
        ),
        answer=(
            "**不成立。**读取时自愈只能治**已经读到**的键，"
            "有一个固有局限：**它不是并发安全的**。\n"
            "具体问题在于自愈有先后顺序：\n"
            "  假设读取键 K 的过程中，"
            "恰好又发生了一次压缩（epoch 从 1 变成 2）。\n"
            "K 的标签按 epoch = 1 重算成 (h + 1)，"
            "但写回时当前 epoch 已经是 2 了，"
            "于是写回去的标签**一存下就又失配**。\n"
            "也就是说，自愈动作和 epoch 变化如果交错，"
            "就会写入一个「算的时候对、存的时候错」的标签。\n"
            "而 Zephyra 里epoch 变化是**全局的、由任意桶的任意压缩触发**的，"
            "所以这种交错不是罕见情况而是必然存在。\n"
            "因此正确做法是**批量重算**（停写、全量扫一遍再统一写回），"
            "而不是逐个读取时顺手改。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="v2_tr_consist_02",
        kp="kp_lumen_consist",
        question_type="transfer",
        premise_in_question="asserted",
        question=(
            "Zephyra 的测试环境里只有一个桶，测试人员发现"
            "「只有这一个桶会触发压缩，所以 epoch 变化只影响这一个桶的键」，"
            "并据此提出一个前提：「测试环境下可以忽略标签全局失效的问题，"
            "只测这一个桶就行」。请判断这个前提在生产环境是否也成立，"
            "并说明理由。"
        ),
        answer=(
            "**在生产环境不成立。**这个前提错在把「epoch 是全局变量」"
            "当成了「epoch 的影响范围等于触发范围」。\n"
            "拆开看两件事：\n"
            "  - **触发范围是局部的**：压缩条件（墓碑 >= 2 且活跃键 <= 1）"
            "只看一个桶的内容。\n"
            "  - **影响范围是全局的**：epoch 是全局计数器，"
            "ZTag = (LumenHash + epoch) mod 65537 对**每一个键**成立。\n"
            "所以只要有任何一次压缩发生，"
            "**全库所有键**的期望标签都变了，与那些键所在的桶有没有压缩无关。\n"
            "在测试环境里「只有一个桶」，"
            "意味着触发范围和影响范围恰好重合，"
            "看起来像是局部的——这是测试环境的特殊条件造成的假象，"
            "掩盖了真实行为。\n"
            "生产环境里只要有多个桶，"
            "「A 桶压缩 → B 桶的键也读不出来」就会立刻暴露。\n"
            "所以测试必须显式构造「压缩在A 桶、验证 B 桶失效」这个场景，"
            "否则等于没测标签机制。"
        ),
        difficulty="hard",
    ),
)


# ==========================================================================
# 集合、指纹、门禁
# ==========================================================================

PRE_SET = PRE_ITEMS
MEMORY_SET = MEMORY_ITEMS
TRANSFER_SET = TRANSFER_ITEMS

#: 三场考试各用哪一卷
EXAM_SETS: dict[str, tuple[ExamItem, ...]] = {
    "pre": PRE_SET,
    "memory": MEMORY_SET,
    "transfer": TRANSFER_SET,
}

ALL_ITEMS: tuple[ExamItem, ...] = PRE_ITEMS + MEMORY_ITEMS + TRANSFER_ITEMS


def _fingerprint(items: tuple[ExamItem, ...]) -> str:
    """题目 + 答案 + 题型 + premise 的 SHA256。

    **必须把 question_type 与 premise_in_question 一起算进指纹。**
    V0.4.1 踩过：改题面不改指纹，等于指纹没守住。
    只算 question+answer 的话，把一道 transfer 题偷偷改成 basic 题，
    指纹对不上会被发现；但如果连答案一起改成「更像basic 的答案」，
    就需要 type 参与才拦得住。
    """
    payload = [
        {
            "item_id": i.item_id,
            "kp": i.kp,
            "question": i.question,
            "answer": i.answer,
            "question_type": i.question_type,
            "premise_in_question": i.premise_in_question,
        }
        for i in items
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


BANK_FINGERPRINT = _fingerprint(ALL_ITEMS)
PRE_FINGERPRINT = _fingerprint(PRE_ITEMS)
MEMORY_FINGERPRINT = _fingerprint(MEMORY_ITEMS)
TRANSFER_FINGERPRINT = _fingerprint(TRANSFER_ITEMS)

#: TRANSFER 卷的出题说明（不参与指纹，仅供人工复核）
TRANSFER_NOTES: dict[str, dict[str, str]] = {
    "v2_tr_hash_01": {
        "transfers_from": "分桶模数 8 与哈希内部模数 65537 是两件事",
        "prior_intuition": "以为「模数影响分布」，混淆了哈希值本身和取模分桶",
    },
    "v2_tr_hash_02": {
        "transfers_from": "读取校验的变量在当前 epoch，不在 LumenHash",
        "prior_intuition": "以为读不出来就是哈希碰撞，先怀疑哈希算错",
    },
    "v2_tr_bucket_01": {
        "transfers_from": "分裂恒等变换 → 有效桶数仍是 8",
        "prior_intuition": "看到规则里出现 mod 16 就以为有 16 个桶",
    },
    "v2_tr_bucket_02": {
        "transfers_from": "整体平移哈希值无法拆开同桶的键",
        "prior_intuition": "以为给所有键减同一个常数就能重新洗牌",
    },
    "v2_tr_compact_01": {
        "transfers_from": "压缩触发局部、影响全局",
        "prior_intuition": "以为修一个桶不会连累其他桶",
    },
    "v2_tr_compact_02": {
        "transfers_from": "压缩的代价是全库重算，不是压缩本身",
        "prior_intuition": "以为压缩会自动兜底、删除是免费的",
    },
    "v2_tr_consist_01": {
        "transfers_from": "逐个自愈不是并发安全的，批量重算才可靠",
        "prior_intuition": "以为读取时顺手改标签就彻底解决了",
    },
    "v2_tr_consist_02": {
        "transfers_from": "测试环境单桶会掩盖 epoch 的全局性",
        "prior_intuition": "以为触发范围就是影响范围",
    },
}


def get_set(which: str) -> tuple[ExamItem, ...]:
    """按名字取卷：'pre' | 'memory' | 'transfer'。"""
    if which in EXAM_SETS:
        return EXAM_SETS[which]
    raise ValueError(f"未知卷名 {which!r}，只支持 {sorted(EXAM_SETS)}")


def kp_coverage(items: tuple[ExamItem, ...]) -> set[str]:
    return {i.kp for i in items}


def bank_diagnostics() -> dict[str, Any]:
    """题库诊断：把「跑完实验才发现」的问题提前暴露。

    继承 V0.4.1 的教训：**题目歧义会伪装成模型能力不足**。
    报告里看到「这个知识点没迁移」，真相可能是「这道题问错了」。
    所以歧义必须机器可检，不能靠跑完实验靠人眼发现。

    V0.4.1 试过用关键词匹配（「题面含『还缺』且含『已经有』」）抓歧义，
    拿旧题面一测发现**根本抓不到**——坑在「按字面推」这个推理链上。
    关键词方案给的是虚假的安全感，所以 V2 改成显式字段 + 硬门禁。
    """
    problems: list[str] = []
    warnings: list[str] = []

    # --- 1. premise 声明 ---
    for item in ALL_ITEMS:
        flag = item.premise_in_question
        if flag == "no":
            problems.append(
                f"{item.item_id}：显式标记 premise_in_question='no'，"
                f"即题面没有给出判断所需的依据 → 这道题本身不可解，必须补前提"
            )
        elif flag not in ("yes", "asserted"):
            problems.append(
                f"{item.item_id}：premise_in_question 取值非法 {flag!r}，"
                f"只允许 'yes' / 'no' / 'asserted'"
            )

    # --- 2. TRANSFER 必须 asserted ---
    for item in TRANSFER_ITEMS:
        if item.premise_in_question != "asserted":
            problems.append(
                f"{item.item_id}：TRANSFER 题必须显式给出待判断的前提"
                f"（premise_in_question='asserted'），当前是 "
                f"{item.premise_in_question!r}。"
                f"V0.4.1 的 tr_covering_01 就是因为题意靠暗示而被判错"
            )
        if item.item_id not in TRANSFER_NOTES:
            problems.append(
                f"{item.item_id}：TRANSFER 题缺少出题说明，无法复核是否真在测迁移"
            )
        if len(item.answer) < 120:
            problems.append(
                f"{item.item_id}：TRANSFER 参考答案过短（{len(item.answer)} 字），"
                f"判分会失真"
            )

    # --- 3. 题量与配平诊断 ---
    per_kp: dict[str, dict[str, int]] = {}
    for name, items in EXAM_SETS.items():
        counts: dict[str, int] = {}
        for it in items:
            counts[it.kp] = counts.get(it.kp, 0) + 1
        per_kp[name] = counts

    for name, items in EXAM_SETS.items():
        thin = [kp for kp, n in per_kp[name].items() if n < 2]
        if thin:
            problems.append(
                f"{name} 卷以下知识点只有 1 道题：{sorted(thin)}；"
                f"单题判分误差会直接等于该知识点的分数"
            )

    # 题型与卷的绑定检查：basic 只在 pre、application 只在 memory、
    # transfer 只在 transfer。这保证「题型难度」在三场之间是系统性变化的。
    expected_type = {"pre": "basic", "memory": "application", "transfer": "transfer"}
    for name, items in EXAM_SETS.items():
        wrong = [i.item_id for i in items if i.question_type != expected_type[name]]
        if wrong:
            problems.append(
                f"{name} 卷混入了不属于 {expected_type[name]} 的题目：{wrong}"
            )

    return {
        "version": BANK_VERSION,
        "problems": problems,
        "warnings": warnings,
        "items_per_kp": per_kp,
        "total_items": len(ALL_ITEMS),
    }


def assert_bank_intact() -> None:
    """硬门禁：题库有任何结构性问题就拒绝跑实验。

    这些不是「建议」而是「必须」——
    V0.4.1 的一道歧义题让逐知识点迁移出现 -32.9的假象，
    报告上看起来像「模型没学会」，真相是「题目问错了」。
    """
    if len(ALL_ITEMS) != 24:
        raise ValueError(f"题库必须是 24 题，当前 {len(ALL_ITEMS)} 题")

    ids = [i.item_id for i in ALL_ITEMS]
    if len(ids) != len(set(ids)):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        raise ValueError(f"题库存在重复题目 id：{dupes}")

    base_kp = kp_coverage(PRE_SET)
    if base_kp != set(KP_LABELS):
        raise ValueError(
            f"知识点集合与 KP_LABELS 不一致：{sorted(base_kp)} vs {sorted(KP_LABELS)}"
        )
    for name, items in EXAM_SETS.items():
        missing = base_kp.symmetric_difference(kp_coverage(items))
        if missing:
            raise ValueError(f"{name} 卷知识点不配平，差异：{sorted(missing)}")

    sizes = {name: len(items) for name, items in EXAM_SETS.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"各卷题量不等，无法逐知识点对比：{sizes}")

    # 三卷题目必须互不相同，否则「记忆 vs 迁移」的对比失效
    questions = [i.question for i in ALL_ITEMS]
    if len(questions) != len(set(questions)):
        raise ValueError("题库存在重复题面")

    # TRANSFER 不得复用 PRE/MEMORY 的题面
    known = {i.question for i in PRE_ITEMS} | {i.question for i in MEMORY_ITEMS}
    for item in TRANSFER_ITEMS:
        if item.question in known:
            raise ValueError(
                f"TRANSFER 题 {item.item_id} 与 PRE/MEMORY 重复，退化成记忆题"
            )

    diag = bank_diagnostics()
    if diag["problems"]:
        raise ValueError(
            "题库诊断发现问题，拒绝跑实验：\n  - " + "\n  - ".join(diag["problems"])
        )


def bank_meta() -> dict[str, Any]:
    """入库用的题库元信息。"""
    assert_bank_intact()
    return {
        "version": BANK_VERSION,
        "fingerprint": BANK_FINGERPRINT,
        "pre_fingerprint": PRE_FINGERPRINT,
        "memory_fingerprint": MEMORY_FINGERPRINT,
        "transfer_fingerprint": TRANSFER_FINGERPRINT,
        "total_count": len(ALL_ITEMS),
        "per_set_count": {k: len(v) for k, v in EXAM_SETS.items()},
        "kps": sorted(kp_coverage(PRE_SET)),
        "per_kp": QUESTIONS_PER_KP,
    }
