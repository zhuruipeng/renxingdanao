"""固定题库 —— PRE / POST / TRANSFER 三卷。

V0.2 的全部价值取决于一件事：**PRE 和 POST 必须是可比的**。
V0.1 的题目是 LLM 现场出的，pre/post 两次抽到的是不同题、比的是不同东西，
分数差可能纯出题随机性 —— 那种实验证明不了任何事。

所以 V0.2 把题目硬编码成模块级常量：
- 题目只在代码里，LLM 碰不到，也就没法"自己给自己改题"
- 每道题带 kp（knowledge point）标签，用于逐知识点对比
- PRE / POST 覆盖**同一组 kp**，但表述不同、难度配平
- BANK_FINGERPRINT = SHA256(全部题目 + 答案)。入库时记录指纹，
  一旦有人事后改了题，指纹对不上，实验作废（见 assert_bank_intact）

**改题库的正确做法**：同时改三卷、保持 kp 对齐、改完更新
BANK_VERSION 并在 README 记录原因。单独改一侧等于毁掉整个实验。

V0.4 新增第三卷：TRANSFER
--------------------------------
为什么 POST 不够
----------------
V0.3 用「同一批题先开卷再盲考」来区分「会用外部记忆」和「模型自身变了」。
这对测**记忆**是有效的——但它测不出**概念是否形成**。

原因很直接：盲考时模型可能只是模糊地记得「我刚才答过一道讲回表的题」，
靠的是题面里的线索，而不是对原理的理解。题目字面越具体，越容易蒙对。

所以 V0.4 加 TRANSFER 卷，规则是**同一个原理、换一个新场景**：

    学的     ：B-Tree 为什么需要分裂
    TRANSFER ：节点到容量上限后，为什么不能简单继续增加子节点

V0.4.1 修订：修 tr_covering_01 的题目歧义
------------------------------------------
真实模型跑10 次后，这题被判45 分，原因是**题目本身写错了**：

    原题面：「...并说明要让它是覆盖索引还缺什么」
    但查询只 SELECT month，而索引 (month) 里就有month。
    按字面推，索引 (month) 确实**已经**是覆盖索引，「什么都不缺」。

模型答「按题面字面，缺的不是额外列」在技术上是对的，
但参考答案要求它答「缺数据列」→ 判分器按参考答案判它错。

这一题同时违背TRANSFER 卷的立题本意：
它在测「模型能不能发现题目自己有问题」，而不是「能不能用原理」。

改成让错误说法直接出现在题面里、由考生去驳：

    新题面：「...并说『索引里有 month，已经是覆盖索引了，没有回表开销』」

这样考点唯一且无歧义——把「列都在索引里」和「不用读行」拆开，
考的是「覆盖索引省的是每行少读列，省不掉必须读所有行」。

教训记下来：**题目歧义会伪装成模型能力不足。**
判分扣分之前必须先回读题面，确认扣分扣在模型身上而不是题自己身上。

原则不变、应用场景变。能答对说明形成的是概念，不是记住了题面。

**出题纪律**：TRANSFER 的参考答案只能由 DEFAULT_MATERIAL 里已出现的原理推出，
不能引入材料没讲的新概念（如复合索引最左前缀、分区表）——
否则测的是「模型本来就知道多少」，不是「学完之后会了多少」。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, NamedTuple

BANK_VERSION = "v0.4.1"

# --------------------------------------------------------------------------
# 主题：SQLite 索引（与 V0.1 一致，便于对比）
# --------------------------------------------------------------------------
# kp 列表（4 个知识点，PRE/POST 两侧必须一一对应）：
#   kp_btree     —— B-Tree 结构与为什么快
#   kp_covering  —— 覆盖索引与回表代价
#   kp_clustered —— 聚簇索引 / 行id
#   kp_write     —— 写放大与索引维护代价


class ExamItem(NamedTuple):
    """一道固定题目。answer 是参考答案（考生永远看不到）。

    `premise_in_question`：题面是否**显式包含**判断所需的前提。
    这是 v0.4.1 加的字段，起因是 tr_covering_01 的歧义：

        题面问「要让它是覆盖索引还缺什么」，但题面里没写「COUNT(*) 需要读哪些列」，
        考生按自己的理解答「索引 (month) 已经是覆盖索引了，什么都不缺」——
        这个回答在逻辑上是对的，参考答案却要求答「缺数据列」，于是被判错。

    教训：**题目歧义会伪装成模型能力不足。**
    报告里看到的是「这个知识点没迁移」，真相是「这道题问错了」。

    三种取值：
      "yes"     —— 前提写进题面了，考的是原理
      "no"      —— 题面不含前提，题目本身不可解 → 硬错误，必须修
      "asserted"—— 题面里直接给出一个**错误说法**让考生去驳（如 v0.4.1 的改法）
    """

    item_id: str
    kp: str
    question: str
    answer: str
    difficulty: str = "medium"
    premise_in_question: str = "yes"


# ----------------------------- PRE 卷 -------------------------------------
PRE_ITEMS: tuple[ExamItem, ...] = (
    ExamItem(
        item_id="pre_btree_01",
        kp="kp_btree",
        question=(
            "SQLite 在底层用什么数据结构组织索引？"
            "请说明这个数据结构为什么能让查找次数保持稳定。"
        ),
        answer=(
            "SQLite 使用 B-Tree 组织索引。B-Tree 的所有叶子节点都位于同一层，"
            "且每个内部节点的分叉数固定，因此从根到叶的路径长度基本恒定；"
            "再配合节点内有序数组做二分，查找一个键所需的比较次数不会随数据量增长而变大。"
        ),
        difficulty="medium",
    ),
    ExamItem(
        item_id="pre_covering_01",
        kp="kp_covering",
        question=(
            "什么是覆盖索引？它在什么情况下能带来性能收益，"
            "判断一个查询能不能走覆盖索引要看什么？"
        ),
        answer=(
            "当查询用到的所有列都包含在某个索引中时，无需再回到主表读取数据行，"
            "称为覆盖索引。判断依据是：把 SELECT 列表里的每一列逐个对照索引的列集合，"
            "全部命中才算覆盖。收益是省掉随机 IO；缺一列就会退化成回表。"
        ),
        difficulty="medium",
    ),
    ExamItem(
        item_id="pre_clustered_01",
        kp="kp_clustered",
        question=(
            "在 SQLite 里对一个普通表建索引时，被索引的记录和存放这些记录的数据页"
            "是什么关系？这一点和MySQL 的聚簇索引有什么本质不同？"
        ),
        answer=(
            "SQLite 的表数据存储在表 B-Tree 中，索引 B-Tree 存放的是键值加行号(rowid)，"
            "两者是分离的；普通表本身就是按 rowid 组织的聚簇结构。"
            "MySQL 的 InnoDB 聚簇索引则是索引叶子节点直接存放整行数据，"
            "二级索引回表要回到聚簇索引。本质区别是行数据与索引是否同树。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="pre_write_01",
        kp="kp_write",
        question=(
            "给一个有两万行的表每列都建索引，会付出什么代价？"
            "如果这张表以批量写入为主，索引策略应该怎么调整？"
        ),
        answer=(
            "代价包括：每次 INSERT/DELETE 都要同步维护所有索引（写放大），"
            "索引占用额外存储，且每次维护涉及随机页写入。批量写入场景下应"
            "减少索引数量、推迟建索引（先灌数据再建）或直接使用无索引的堆表，"
            "用少量索引支撑少数关键查询。"
        ),
        difficulty="hard",
    ),
)

# ----------------------------- POST 卷 ------------------------------------
# 与 PRE 覆盖同一组 kp，但换一种提问角度与措辞，难度配平。
POST_ITEMS: tuple[ExamItem, ...] = (
    ExamItem(
        item_id="post_btree_01",
        kp="kp_btree",
        question=(
            "有人问：为什么 SQLite 查一条数据不会因为表变大而越来越慢？"
            "请从索引的数据结构角度解释原因。"
        ),
        answer=(
            "因为索引用 B-Tree 存储。B-Tree 是平衡多路树，所有叶子在同一深度，"
            "内部节点分叉数固定，所以从根节点到目标叶子的层数几乎不随数据量增加而增加；"
            "节点内按键有序，可二分定位。因此单次查找的 I/O 次数被树高（对数级）限制住，"
            "而不是线性扫描。"
        ),
        difficulty="medium",
    ),
    ExamItem(
        item_id="post_covering_01",
        kp="kp_covering",
        question=(
            "一个查询写了 SELECT name, price FROM items WHERE sku = ?，"
            "而sku 上恰好建了索引 (sku, name, price)。这个查询会额外读数据页吗？"
            "如果把 price 从索引里去掉，结论会怎么变？"
        ),
        answer=(
            "不会额外读数据页。索引 (sku, name, price) 已包含查询所需的全部列，"
            "是覆盖索引，只需在索引树内定位并返回，无需回表。"
            "若去掉 price，索引只剩 (sku, name)，price 不在索引中，"
            "必须拿到 rowid 再回表读取完整数据行，产生一次随机 IO。"
        ),
        difficulty="medium",
    ),
    ExamItem(
        item_id="post_clustered_01",
        kp="kp_clustered",
        question=(
            "SQLite 的索引里存的是什么？它是怎么定位到实际数据行的？"
            "对比一下 InnoDB 的一句话说清两者的差别。"
        ),
        answer=(
            "SQLite 的索引项存的是「索引列值 + rowid（行号）」，定位数据行时要拿rowid "
            "去表 B-Tree 里再查一次。InnoDB 的聚簇索引叶子节点直接存整行数据，"
            "所以二级索引拿到的就是主键值，回表是回到聚簇索引本身取整行。"
            "差别在于：SQLite 索引与数据分属两棵树，InnoDB 的数据就长在聚簇索引里。"
        ),
        difficulty="hard",
    ),
    ExamItem(
        item_id="post_write_01",
        kp="kp_write",
        question=(
            "一张日志表每天追加两百万行，其中只有时间戳一个字段会用于查询。"
            "你会给它建几个索引？说明取舍。"
        ),
        answer=(
            "至多一个：只给时间戳建单列索引。理由是索引越多，每次追加都要同步维护，"
            "写放大和额外存储都随索引数量线性上升，而未被查询使用的索引是纯成本。"
            "如果时间戳查询还常带范围条件，可以评估是否改用分区/按时间分表，"
            "用物理裁剪替代索引过滤。"
        ),
        difficulty="hard",
    ),
)

PRE_SET = PRE_ITEMS
POST_SET = POST_ITEMS

# ----------------------------- TRANSFER 卷 --------------------------------
# 判卷标准与 PRE/POST 不同，请照这个标准写出题：
#
#   好题 = 材料里有这个原理 + 但材料从没这么用过
#        + 且模型的先验直觉多半是错的
#
# 反例（不要这样出）：
#   - 「B-Tree 为什么平衡」→ 和 PRE 几乎同题，退化成记忆测试
#   - 「MySQL 聚簇索引怎么建」→ 材料没讲，测的是先验知识不是学习效果
#
# 每题都在 `notes` 里注明「迁移自哪条原理」和「先验直觉错在哪」，
# 方便日后复核题目质量，也提醒改题的人别把题改成记忆题。

TRANSFER_ITEMS: tuple[ExamItem, ...] = (
    ExamItem(
        item_id="tr_btree_01",
        kp="kp_btree",
        question=(
            "有人向B-Tree 提出一个改良方案：既然树这么深是因为节点装不下太多键，"
            "那把节点上限从一页放宽到一整页的一半以外，"
            "是不是就能少分裂、树也就能更矮，查得更快？"
            "请判断这个方案为什么行不通。"
        ),
        answer=(
            "行不通。分叉数上限不是为了省空间，而是被「一个节点必须能放进一页」"
            "这个硬约束决定的：节点内部要按键有序并支持二分，前提是能在一次页读取里"
            "把这个节点完整读进内存。如果节点大到需要多次页读取才能读全，"
            "二分查找的前提就不成立了，查找 I/O 反而从「每层一次」变成「每层多次」。"
            "树变矮省下的层数，会被每层多出来的页读取次数抵消甚至超过——"
            "这就是为什么查找次数能被树高（对数级）而不是节点大小限制住。"
        ),
        difficulty="hard",
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="tr_covering_01",
        kp="kp_covering",
        question=(
            "一条报表查询 SELECT month, COUNT(*) FROM orders "
            "GROUP BY month，数据量很大。有人提议建索引 (month) 来加速，"
            "并说「索引里有 month，这条查询已经是覆盖索引了，所以没有回表开销」。"
            "请指出这个说法错在哪里，并解释：要让这类聚合查询真正变快，"
            "应该从减少「读的行数」还是减少「每行读的列数」入手？"
        ),
        answer=(
            "错在把「查询要 SELECT 的列恰好都在索引里」等同于「查询不读数据行」。"
            "COUNT(*) 需要对分组内的每一条记录各贡献1，"
            "也就是说聚合必须遍历所有行，一条都不能少。"
            "索引 (month) 能让 month 有序、能让分组边界免于排序，"
            "但它内部只存month 和 rowid，不存这些行的业务数据，"
            "所以每处理一条记录仍然要拿 rowid 回表读一次——"
            "回表次数仍然和数据量成正比，索引在这里省不下IO。"
            "即使把索引建成 (month, 业务列) 的覆盖索引、不回表了，"
            "要读的「行数」一条也没少，省下的只是每行从「读索引+读行」变成「只读索引」。"
            "所以正确的方向是：覆盖索引能省的是「每行少读列」，"
            "省不掉「必须读所有行」这件事本身。"
            "要让这类聚合真正变快，得想办法减少要读的行数——"
            "比如按月预聚合出一张汇总表、缩小时间窗口、或用分区裁剪，"
            "而不是继续在原表上叠索引。"
        ),
        difficulty="hard",
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="tr_clustered_01",
        kp="kp_clustered",
        question=(
            "有人发现 SQLite 的表本身就是按 rowid 有序存储的，"
            "于是想：那还需要建索引吗？直接按 rowid 顺序扫表不就行了？"
            "请说明这个推理错在哪里，并给出一个反例。"
        ),
        answer=(
            "错在 rowid 是「插入顺序」，不是「被查询的列」。"
            "按 rowid 有序意味着省掉的是一种特定查询："
            "按 rowid 或 rowid 范围定位，因为这时顺序恰好对齐。"
            "但绝大多数查询是按 name、sku、created_at 这类业务列过滤的，"
            "这些列的值与 rowid 顺序毫无关系，"
            "顺序扫描反而要把每一行都读出来再丢弃不符合条件的，"
            "复杂度是全表 O(n) 而不是定位查找。"
            "反例：一张按时间顺序插入的订单表，"
            "查「某个用户的订单」——如果用户恰好总是最后插入的，"
            "顺序扫描要跳过大半张表；若该用户分散在整张表里，"
            "则必须扫全表。而 (user_id) 索引能直接定位到对应区间。"
        ),
        difficulty="hard",
        premise_in_question="asserted",
    ),
    ExamItem(
        item_id="tr_write_01",
        kp="kp_write",
        question=(
            "一个运维方案是：为了加快查询，先给表建好全部索引，"
            "然后把旧数据批量导进去，导完直接开始提供查询服务。"
            "请指出这个方案在「导入期间」的问题，并给出改造方案。"
        ),
        answer=(
            "问题在于导入期每一行的 INSERT 都要同步维护所有已建好的索引，"
            "写放大成本随着导入行数线性累积，而这段时间索引根本用不上"
            "（没人查询，数据还不存在）——"
            "把最贵的写入成本花在收益为零的阶段，且慢到不可接受。"
            "正确顺序是反过来的：先只灌数据、导入期不建索引（让写入走最快的路径），"
            "灌完之后再一次性建索引。"
            "建索引虽然也要扫描全表，但它是单次顺序操作，"
            "远快于「每插一行都维护一遍所有索引」。"
            "这就是「批量写入为主的表应减少索引数量、推迟建索引」的实际含义。"
        ),
        difficulty="hard",
        premise_in_question="asserted",
    ),
)

#: TRANSFER 卷的出题说明（不参与指纹，仅供复核与改题参考）
TRANSFER_NOTES: dict[str, dict[str, str]] = {
    "tr_btree_01": {
        "transfers_from": "所有叶子同层 + 节点内二分",
        "prior_intuition": "以为树越矮越快，忽略单节点必须能一次读全这个前提",
    },
    "tr_covering_01": {
        "transfers_from": "覆盖索引省的是「每行少读列」，省不掉「必须读所有行」",
        "prior_intuition": "以为查询列都在索引里就等于没有行读取开销，忽略聚合要遍历全部行",
    },
    "tr_clustered_01": {
        "transfers_from": "索引与行分属两棵树 + 覆盖索引省回表",
        "prior_intuition": "以为「表本身有序」等价于「查询快」，混淆 rowid 与业务列顺序",
    },
    "tr_write_01": {
        "transfers_from": "写放大随索引数量与写入次数线性上升",
        "prior_intuition": "以为先建索引再导数据是一步到位的做法",
    },
}

TRANSFER_SET = TRANSFER_ITEMS

# 全部题目拍平成一个稳定顺序，用于计算指纹
ALL_ITEMS: tuple[ExamItem, ...] = PRE_ITEMS + POST_ITEMS + TRANSFER_ITEMS


def _fingerprint(items: tuple[ExamItem, ...]) -> str:
    payload = json.dumps(
        [
            {"id": i.item_id, "kp": i.kp, "q": i.question, "a": i.answer, "d": i.difficulty}
            for i in items
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: 题库指纹。任何对题目的改动都会改变它。
BANK_FINGERPRINT = _fingerprint(ALL_ITEMS)
PRE_FINGERPRINT = _fingerprint(PRE_ITEMS)
POST_FINGERPRINT = _fingerprint(POST_ITEMS)
TRANSFER_FINGERPRINT = _fingerprint(TRANSFER_ITEMS)

PRE_TOPIC = "SQLite 索引"
KP_LABELS = {
    "kp_btree": "B-Tree 结构与查找稳定性",
    "kp_covering": "覆盖索引与回表代价",
    "kp_clustered": "索引与行的存储关系",
    "kp_write": "写放大与索引策略",
}

#: V0.4 的四场考试各自用哪一卷
EXAM_SETS = {
    "pre": PRE_SET,               # 学习前基线
    "memory": POST_SET,           # 记忆测试：先开卷再盲考
    "transfer": TRANSFER_SET,     # 迁移测试：只盲考，测是否形成概念
}


def get_set(which: str) -> tuple[ExamItem, ...]:
    """按名字取卷子：'pre' | 'post' | 'transfer'。未知名字报错，避免静默拿错卷。"""
    if which in EXAM_SETS:
        return EXAM_SETS[which]
    raise ValueError(f"未知卷名 {which!r}，只支持 {sorted(EXAM_SETS)}")


def kp_coverage(items: tuple[ExamItem, ...]) -> set[str]:
    return {i.kp for i in items}


def bank_diagnostics() -> dict[str, Any]:
    """题库诊断：把「跑完实验才发现」的问题提前暴露出来。

    v0.4.1 起因：真实模型跑10 次后才发现 tr_covering_01 题面自身有歧义
    （按字面推索引已是覆盖索引，模型答「不缺额外列」反而是对的，被判45 分）。
    题目歧义会**伪装成模型能力不足** —— 报告里看到的是「这个知识点没迁移」，
    真相是「这道题问错了」。所以歧义不能靠跑完实验再靠人眼发现。

    这里做两件机器能做的检查：

    1. **题面自足性**：读 `ExamItem.premise_in_question`。
       这一项刻意做成**显式字段而不是关键词匹配** —— 试过用
       「题面含『还缺』且含『已经有』」来检测，跑 v0.4.0 的旧题面发现根本抓不到：
       旧题面里没有「已经有」三个字，它的坑在于「按字面推」这个推理链，
       关键词匹配不出来。关键词方案在这里给的是虚假的安全感，
       不如逼改题的人显式声明前提在不在题面里。
    2. **题量诊断**：每 kp 每卷只有 1 道题时，单题判分误差 = 整个 kp 的分数，
       σ 必然被放大，逐 kp 结论没有第二个样本支撑。返回明细供报告引用。
    """
    problems: list[str] = []
    warnings: list[str] = []

    # --- 1. 题面自足性（读显式字段，不猜） ---
    for item in ALL_ITEMS:
        flag = item.premise_in_question
        if flag == "no":
            problems.append(
                f"{item.item_id}：显式标记为 premise_in_question='no'，"
                f"即题面没有给出判断所需的依据 → 这道题本身不可解，必须补前提"
            )
        elif flag not in ("yes", "asserted"):
            problems.append(
                f"{item.item_id}：premise_in_question 取值非法 {flag!r}，"
                f"只允许 'yes' / 'no' / 'asserted'"
            )

    # --- 2. 题量诊断 ---
    per_kp: dict[str, dict[str, int]] = {}
    for name, items in EXAM_SETS.items():
        counts: dict[str, int] = {}
        for it in items:
            counts[it.kp] = counts.get(it.kp, 0) + 1
        per_kp[name] = counts
    thin = [
        f"{name}:{kp}={n} 题"
        for name, counts in per_kp.items()
        for kp, n in counts.items()
        if n < 2
    ]
    if thin:
        per_kp_desc = "、".join(
            f"{name}/{kp}={n} 题"
            for name, counts in per_kp.items()
            for kp, n in counts.items()
            if n < 2
        )
        warnings.append(
            f"每卷每个知识点只 1 道题（{per_kp_desc}），"
            f"单题判分误差会直接放大成该知识点的 σ，"
            f"逐知识点结论目前没有第二个样本支撑"
        )

    return {
        "version": BANK_VERSION,
        "problems": problems,
        "warnings": warnings,
        "items_per_kp": per_kp,
    }


def assert_bank_intact() -> None:
    """校验三卷知识点配平、题目无重复、题量一致。

    这是实验能成立的前提检查：卷子 kp 集合不一致就说明配平被破坏，
    这种实验的分数差没有意义，必须直接拒绝跑。
    """
    base_kp = kp_coverage(PRE_SET)
    for name, items in EXAM_SETS.items():
        missing = base_kp.symmetric_difference(kp_coverage(items))
        if missing:
            raise ValueError(f"{name} 卷知识点不配平，差异：{sorted(missing)}")
    ids = [i.item_id for i in ALL_ITEMS]
    if len(ids) != len(set(ids)):
        dupes = {x for x in ids if ids.count(x) > 1}
        raise ValueError(f"题库存在重复题目 id：{sorted(dupes)}")
    sizes = {name: len(items) for name, items in EXAM_SETS.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"各卷题量不等，无法逐知识点对比：{sizes}")
    # TRANSFER 必须是新题，不能是 PRE/POST 的复制，否则迁移测试退化成记忆测试
    known = {i.question for i in PRE_ITEMS} | {i.question for i in POST_ITEMS}
    for item in TRANSFER_ITEMS:
        if item.question in known:
            raise ValueError(f"TRANSFER 题 {item.item_id} 与 PRE/POST 重复，退化成记忆题")
        if item.item_id not in TRANSFER_NOTES:
            raise ValueError(f"TRANSFER 题 {item.item_id} 缺少出题说明，无法复核质量")
        if len(item.answer) < 60:
            raise ValueError(f"TRANSFER 题 {item.item_id} 参考答案过短，判分会失真")
    # v0.4.1：题面自足性是硬门禁。题目歧义会伪装成模型能力不足 ——
    # 报告里看到「这个知识点没迁移」，真相可能是「这道题问错了」。
    diag = bank_diagnostics()
    if diag["problems"]:
        raise ValueError("题库诊断发现问题，拒绝跑：\n  - " + "\n  - ".join(diag["problems"]))


def bank_meta() -> dict[str, Any]:
    """入库用的题库元信息。"""
    assert_bank_intact()
    return {
        "version": BANK_VERSION,
        "fingerprint": BANK_FINGERPRINT,
        "pre_fingerprint": PRE_FINGERPRINT,
        "post_fingerprint": POST_FINGERPRINT,
        "transfer_fingerprint": TRANSFER_FINGERPRINT,
        "pre_count": len(PRE_SET),
        "post_count": len(POST_SET),
        "transfer_count": len(TRANSFER_SET),
        "kps": sorted(kp_coverage(PRE_SET)),
    }
