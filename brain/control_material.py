"""brain/control_material.py —— V0.6 对照组材料。

为什么必须有对照组
------------------
V0.5 报告里四类gain 全部是「自己的前后差」：

    learning_gain = POST - PRE

这个差值里混着两样东西：

    (1) 真的因学习而获得的��力
    (2) 任何一组都会发生的自然变化 —— 同一套题第二次做本来就可能更高
        （被测效应、格式熟悉、题面措辞被记住、被测当���不是机器）

没有对照组时，(2) 会被算成 (1)。
V0.4.1 的真实模型实验里 PRE 是 90 分OPEN_BOOK 只剩 10 分空间，
「增益小」完全可能只是「没有空间」——这种时候讨论「学习有没有用」
是越权结论。V0.6 把对照组补上，让差值变成**差值的差值**：

    difference_in_gains = (POST_L - PRE_L) - (POST_C - PRE_C)

对照材料的三条硬约束
--------------------
1. **token 长度接近**。差得太多时，「多读了点东西」本身就成了变量。
   `token_estimate()` 用中文字符 + 空白分词做保守估计，
   `length_ratio()` 给出比值，`diagnostics()` 在比值越界时报problem。

2. **与目标知识无关**。它讲的是另一个虚构系统（Veyra 区间树索引），
   不含任何 LumenHash / 桶号 / 压缩 / Z-Tag 相关内容。
   `assert_control_clean()` 用关键词交叉检查——这不是「靠感觉无关」。

3. **不包含目标知识答案**。同上，且额外检查**数字**：
   目标材料里的关键常量（7、131、65537）不得出现在对照材料中。
   这一条比关键词检查更硬：换个说法提到「8 个桶」也算泄漏。
"""

from __future__ import annotations

import re

from memory.db import checksum

#: 对照组材料版本
CONTROL_VERSION = "ctl1.0.0"

#: 对照组主题（与目标主题完全不同的另一个虚构系统）
CONTROL_TOPIC = "Veyra 区间树日志索引"

CONTROL_MATERIAL = """# Veyra 日志索引引擎手册（v1）

## 一、概览

Veyra 是一套面向只追加日志场景的区间树索引引擎。
它把日志切成固定大小的 record，把每条 record 按其时间戳
插入一棵按时间排序的区间树。查询时给定一个时间窗口，
引擎返回该窗口内的全部 record。

本手册定义 Veyra 的定位、区间分裂、区间合并、校验与轮转五组规则。
读完本手册应当能够：判断一条 record 落在哪个区间、
计算一次分裂后哪些区间需要新建、给出一条损坏 record 的判定结果。

## 二、区间定位

每条 record 有一个64 位时间戳 T。Veyra 把时间轴按

    SPAN = 60000

切成等宽区间。record 所属区间号是

    seg = T // SPAN

即按秒对齐、按分钟分段。区间编号从 0 开始连续增长，
不做哈希、不做取模、不做再散列——这是 Veyra 与其他索引引擎
最核心的差别：定位是纯除法，完全可预测。

若 T 恰好等于区间右端点，则归入下一区间。例如 T = 120000 时
seg = 2而不是 1。这一约定必须与写入侧保持一致，
否则查询会漏掉边界上的 record。

## 三、区间分裂

每个区间最多容纳 SEG_LIMIT = 8 条活跃 record。
当某个区间的活跃条数**严格大于** 8 时触发分裂。

分裂后的两个子区间分别是

    left  = (2 * seg)
    right = (2 * seg + 1)

即子区间号由原区间号左移一位再补 0/1。
原区间内的 record 按归属重新落位：seg 落在 left，
其余落在 right。分裂后原区间不再独立存在，
其全部 record 都被分配到两个子区间里。

若某个子区间分裂后仍超出 8 条，则对**该子区间**继续分裂，
不回头再分裂原区间。分裂过程是自顶向下的递归。

## 四、区间合并

与分裂不同，合并**不会自动发生**。合并只在两种情形下被触发：

情形一：两个兄弟区间均为空或均已失效，且各自的 record 数为 0。
情形二：手动执行的收拢操作（operator 名为 merge）。

合并后父区间的编号按

    parent = seg // 2

计算。合并后父区间的 record 数为两个子区间之和。
两个子区间的 record 总数**恰好等于** SEG_LIMIT 时不触发合并——
这个边界情形按「不合并」处理，Veyra 不做边界回收。

需要特别说明的是，Veyra 的合并与「删除无效条目」是两件不同的事。
删除无效条目由校验环节负责（见第五节），它只把损坏 record 从
结果里跳过，不改变区间结构。合并只处理区间编号层面的结构，
不接触 record 内容。把两者混为一谈会导致对空闲率与回收率的
任何统计都失去意义。

## 五、校验与损坏判定

每条 record 在写入时记录一个校验码：

    check = (len(payload) * 31 + seq) mod 8191

其中 len(payload) 是 record载荷的字节长度，seq 是它在所属区间内的
序号（从 0 开始）。读取时重算一次，两次结果不同即判定该 record 损坏。

损坏的 record 不会导致整条查询失败。Veyra 的策略是：
**跳过**损坏 record，返回其余可读 record，并在响应里带上
skipped 计数。因此一次查询的返回条数可能少于窗口内实际条数，
这是正常行为而不是 bug。

判断「能否读到某条 record」只看校验码是否匹配，
与区间是否分裂过、是否合并过**无关**。也就是说，
一条 record 即使经历了任意次数的结构调整，
只要它的校验码仍然匹配，就依然可读。

这一点对容量评估很重要：结构调整（分裂、合并）
只影响区间编号的分布，不影响任何一条既有 record 的可读性。
因此统计「不可读比例」时，**不要**把区间结构调整计入分子。

## 五之二、顺序号的取值范围

seq 从 0 开始，但**在区间分裂后会重新编号**：
record 进入子区间后，按它在子区间内的落位顺序重排序号，
而不是沿用父区间里的原序号。因此同一条 record 在分裂前后
的 seq 可能不同，校验码也随之不同。

推论：分裂之后必须重算受影响 record 的校验码。
若只搬动位置而不重算，这些 record 会被读为损坏并被跳过，
表现为「分裂后突然少了一部分数据」。

重算是幂等的：对同一条 record、在同一个区间内、
以同一个落位顺序重算两次，结果必须完全一致。

## 五之三、负载倾斜的处理

如果某个区间的活跃条数持续高于 8 而分裂条件始终不满足
（例如区间内全部 record 均为损坏且被跳过，导致实际活跃数统计口径不同），
则该区间不再分裂，Veyra 记为 hot segment。

hot segment 的处理方式是允许查询绕过区间树直接顺序扫描该区间，
代价是复杂度退化到 O(n)。Veyra 不为 hot segment 做再平衡——
一旦允许按负载迁移记录，就需要维护 record 的区间归属索引，
而该索引本身会成为新的不一致来源。宁可承认存在退化区间，
也不引入第二套状态。

统计 hot segment 数量时，分母是**区间总数**，
而不是 record 总数。两者混用会得到完全不同的倾斜率。

## 六、轮转

当文件超过 ROTATE_SIZE = 1048576 字节时触发轮转。
轮转会关闭当前文件并开启新文件，时间戳**不重置**——
新文件的第一个区间号接着旧文件继续增长。

这一点与很多日志系统不同：有的实现会在轮转时把时间戳归零，
于是新旧文件的时间戳空间重合，查询必须同时打开两个文件。
Veyra 不需要这样做，因为时间戳全局单调。

轮转不影响任何已写入 record 的校验码，也不改变区间编号的计算方式。
轮转也不重置任何区间的活跃计数——计数属于索引状态，
与文件边界无关。

## 七、本手册不涉及的内容

以下内容**不在** Veyra 的设计范围内，遇到时按「未定义」处理，
不要自行推断：

- 记录级别的更新（Veyra 只支持追加，不支持原地修改）
- 跨文件的分布式放置（Veyra 是单机引擎）
- 索引重建（Veyra 没有 rebuild 流程，损坏只能跳过）
- 时钟回拨（假设时间戳单调，不处理时钟回拨）

把未定义行为当成有定义行为来实现，是接入 Veyra 时最常见的错误来源。
"""

#: 目标材料中不得出现在对照材料里的关键词（大小写不敏感）
_FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "lumenhash", "lumen", "zephyra", "zt ag", "z-tag", "ztag",
    "epoch", "墓碑", "tombstone", "分桶", "桶号", "bucket",
    "65537", "131", "压缩", "compaction",
)

#: 目标材料里的关键数值常量，逐个禁止出现在对照材料中
_FORBIDDEN_NUMBERS: tuple[str, ...] = ("65537", "131")

#: 长度比超出这个区间就报 problem
LENGTH_RATIO_MIN = 0.90
LENGTH_RATIO_MAX = 1.10


def token_estimate(text: str) -> int:
    """保守的 token 估计。

    真正的分词依赖具体模型的 tokenizer，而本项目不引入 tiktoken
    （架构约束：只允许通过 LLMProvider 调模型，tokenizer 属于
    模型内部实现）。这里用「中文字符数 + 空白分词数」近似：

        - 一个汉字 ≈ 1 个 token（多数中文 tokenizer 就是 1 字 1 token）
        - 一段非中文连续文本 ≈ 1 个 token（按词/符号切）

    近似误差不影响用途——**只用于比较两份材料的长度是否接近**，
    两边用同一个函数，误差方向一致。
    """
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    non_cjk = re.sub(r"[\u4e00-\u9fff]", " ", text)
    return cjk + len(non_cjk.split())


def length_ratio() -> float:
    """对照材料 / 学习材料 的 token 长度比。"""
    from brain import novel_material as nm
    learn = token_estimate(nm.NOVEL_MATERIAL)
    ctrl = token_estimate(CONTROL_MATERIAL)
    return round(ctrl / learn, 4) if learn else 0.0


def leakage_hits() -> dict[str, list[str]]:
    """对照材料里的泄漏点。空dict = 干净。"""
    from brain import novel_material as nm
    haystack = CONTROL_MATERIAL.lower()
    keywords = [k for k in _FORBIDDEN_KEYWORDS if k.lower() in haystack]
    numbers = [n for n in _FORBIDDEN_NUMBERS if n in CONTROL_MATERIAL]
    # 逐句检查：对照材料不能出现目标主题名
    topics = [t for t in (nm.NOVEL_TOPIC, "Zephyra") if t.lower() in haystack]
    return {
        "keywords": keywords,
        "numbers": numbers,
        "topics": topics,
    }


def diagnostics() -> dict[str, Any]:  # type: ignore[name-defined]
    """对照材料的自检报告。"""
    from brain import novel_material as nm
    ratio = length_ratio()
    hits = leakage_hits()
    problems: list[str] = []
    warnings: list[str] = []

    if not (LENGTH_RATIO_MIN <= ratio <= LENGTH_RATIO_MAX):
        problems.append(
            f"对照材料长度比{ratio} 超出 [{LENGTH_RATIO_MIN}, {LENGTH_RATIO_MAX}]，"
            f"「多读了点东西」本身会成为变量"
        )
    if hits["keywords"]:
        problems.append(f"对照材料出现目标知识关键词：{hits['keywords']}")
    if hits["numbers"]:
        problems.append(f"对照材料出现目标知识数值常量：{hits['numbers']}")
    if hits["topics"]:
        problems.append(f"对照材料出现目标主题：{hits['topics']}")

    if CONTROL_TOPIC == nm.NOVEL_TOPIC:
        problems.append("对照组主题与目标主题相同，对照失效")

    if not CONTROL_MATERIAL.strip():
        problems.append("对照材料为空")

    # 结构完整性：对照组也必须是「有结构的可学习材料」，
    # 否则学习组与对照组的差异会来自「有没有材料」而非「材料内容」
    sections = CONTROL_MATERIAL.count("\n## ")
    if sections < 4:
        warnings.append(
            f"对照材料只有 {sections} 个章节，"
            f"与目标材料结构差异过大，两组的「学习体验」不对等"
        )

    return {
        "version": CONTROL_VERSION,
        "topic": CONTROL_TOPIC,
        "checksum": checksum(CONTROL_MATERIAL),
        "target_topic": nm.NOVEL_TOPIC,
        "target_checksum": checksum(nm.NOVEL_MATERIAL),
        "learn_tokens": token_estimate(nm.NOVEL_MATERIAL),
        "control_tokens": token_estimate(CONTROL_MATERIAL),
        "length_ratio": ratio,
        "sections": sections,
        "leakage": hits,
        "problems": problems,
        "warnings": warnings,
    }


def assert_control_intact() -> None:
    """硬门禁：对照材料有任何问题就拒绝跑实验。"""
    d = diagnostics()
    if d["problems"]:
        raise RuntimeError(
            "对照材料不合法，双组实验作废：\n  - "
            + "\n  - ".join(d["problems"])
        )


__all__ = [
    "CONTROL_MATERIAL",
    "CONTROL_TOPIC",
    "CONTROL_VERSION",
    "assert_control_intact",
    "diagnostics",
    "leakage_hits",
    "length_ratio",
    "token_estimate",
]
