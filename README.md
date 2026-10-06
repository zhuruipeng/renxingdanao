# Agent-01

> 最小可运行的 AI 学习实验室。
> 闭环：**学习 → 记忆 → 实践 → 考试 → 独立评估 → 保存经验**
>
> - **V0.1**：跑通闭环
> - **V0.2**：固定题库，验证「学习是否真的带来能力提升」
> - **V0.3**：拆开「用外部记忆」与「模型自身能力」
> - **V0.4**：拆开「记住了」与「学会了」，重复 N 次看分布
> - **V0.4.1**：真实模型跑完后发现的两件事——一道题问错了，以及天花板让指标测不出东西
> - **V0.5**：Benchmark V2 —— 24 道固定题、虚构知识材料、机器判分实践 ← 当前阶段

---

## V0.5：Benchmark V2

### 为什么要有这一版

V0.4.1 拿真实模型跑 10 次，暴露三件测不出来的事：

| 问题 | 后果 | V0.5 的处理 |
|---|---|---|
| PRE 已 90 分（模型本来就会） | 只剩 10 分空间，σ=3~5 装不下 → 「增益测不出来」，但「学习没用」和「没有测量空间」分不开 | 换成现实中不存在的 **Zephyra** 规则系统，PRE 理论上得 0 分 |
| 每 kp 每卷只 1 题 | 单题判分误差 = 整个 kp 的分数，却和 8 题均值并排印在报告里 | 每 kp 每卷 **2 题**，逐 kp 样本数不够时打 `样本不足` 标记 |
| 实践分是 LLM 给的 | 「能不能用」是主观的，且和正确性混在一起 | **机器判分**（算法/SQL/单测/schema）与 **LLM 评审**分表存、分开报 |

### 流程

```text
PRE_BLIND → STUDY → PRACTICE → OPEN_BOOK → MEMORY_BLIND → TRANSFER_BLIND
```

- `PRE_BLIND`：裸能力基线，context 恒空
- `STUDY`：读 Zephyra 材料 → 抽知识点 → 写结构化笔记
- `PRACTICE`：8 个客观任务，**机器判分**
- `OPEN_BOOK`：放行本实验笔记，考 application 卷
- `MEMORY_BLIND`：收回一切，考同一批题
- `TRANSFER_BLIND`：全新场景，只盲考

### 虚构知识：Zephyra 分布式哈希存储

现实世界里**不存在**这套规则，所以模型不可能靠先验知识答对：

```text
LumenHash:   h 初值 7，每字符 h = (h * 131 + ord(c)) mod 65537
桶号:        bucket = LumenHash mod 8
分裂:        (bucket + 8) mod 8  ← 恒等变换，一级分桶分裂不减负载
压缩:        Tombstone >= 2 且 活跃键 <= 1（是「且」不是「或」）
Z-Tag:       (LumenHash + epoch) mod 65537
```

关键耦合：**压缩触发是局部的，epoch 影响是全局的**——一次压缩会让所有存量键的 Z-Tag 失配。

材料里出现任何真实系统名（Redis / LevelDB / Cassandra…）都会让「切断先验知识」这个前提作废，所以有测试专门守这条。

### 题库：24 题

- 4 个知识点 × 6 题
- 每 kp含 `basic` / `application` / `transfer`
- 三卷各 8 题（pre / memory / transfer），每 kp 每卷 2 题
- `TRANSFER` 全部 `premise_in_question="asserted"`——题面自带待判断的前提
- 题库指纹覆盖 `item_id / kp / question / answer / question_type / premise_in_question`

### 实践：8 个机器可验证任务

| task_id | 知识点 | 判分方式 |
|---|---|---|
| `obj_hash_01` | lumen_hash | algorithm_output |
| `obj_hash_02` | lumen_hash | python_unit_test |
| `obj_bucket_01` | lumen_bucket | algorithm_output |
| `obj_bucket_02` | lumen_bucket | sql_exec |
| `obj_compact_01` | lumen_compact | json_schema |
| `obj_compact_02` | lumen_compact | python_unit_test |
| `obj_consist_01` | lumen_consist | algorithm_output |
| `obj_consist_02` | lumen_consist | sql_exec |

`python_unit_test` 会**真的执行**模型提交的代码：`python -I` +临时目录 + 15 秒超时 + 静态拒绝危险 import（实测 `python -I` 挡不住 `import os; os.system(...)`）。`sql_exec` 只接受只读单条查询。

### 报告里禁止出现的措辞

`brain/report_v2.py` 有硬门禁：任何指标都不得被称为「训练成功」。

本项目没有训练、没有微调、没有梯度更新，每次实验都是一次前向推理。报告越权措辞**在生成时抛异常**，不是事后靠人读。

报告给每个 gain 绑定了「能说明 / 不能说明」两句话：

| 指标 | 它回答的问题 | 它**不**能说明 |
|---|---|---|
| `external_memory_gain` | 能不能查自己写的笔记答题 | 不说明规则被内化 |
| `memory_retention_gain` | 笔记收走以后还记得多少 | 分不清「理解」与「记住措辞」 |
| `transfer_gain` | 换场景还能不能做对 | 不等于模型被训练过 |
| `objective_practice_gain` | 能不能写出机器可验的产出 | 与考试卷不同题型，属代理指标 |

### 统计上的几条硬规则

- n < 2 **不返回**置信区间（单样本区间退化为点，看起来精确实际什么都没说）
- CI 跨 0 判为 `noisy`，**不判negative**——跨 0 只说明证据不足
- 逐 kp 样本 < 4 时打 `insufficient_sample` 标记
- 全部增益方差为 0 时（确定性 provider），报「重复次数不提供额外信息」
- 天花板线 85 / 地板线 15

### 离线跑一次

```bash
# 1 次完整链路（不联网、不烧 token）
python tools/run_v05_offline.py

# 5 次，看统计层
python tools/run_v05_offline.py --trials 5
```

**注意**：`ScriptedProvider` 是确定性的，跑 5 次会得到完全相同的结果，方差为 0。它的作用是验证**链路能不能跑通**，不构成任何能力结论。

---

## V0.4.1：真实模型跑完之后（这一节全是实跑暴露的问题）

V0.4 换成真实模型（DeepSeek）跑 10 次之后，暴露了两个**不是 bug、但比bug 更危险**的问题。

### 问题一：题目歧义会伪装成「模型能力不足」

`tr_covering_01` 被判45 分（10 次里 7 次都是 45），逐知识点迁移 **-32.9**，
是四个知识点里唯一大幅退步的。看起来像「这个知识点没学会」。

**真相是这道题自己写错了。**原题面问：

> 请指出这个索引为什么救不了这条查询，并说明**要让它是覆盖索引还缺什么**。

但查询是 `SELECT month, COUNT(*)`，索引是 `(month)` ——
**按字面推，这个索引已经是覆盖索引了，什么都不缺。**

模型答「按本题字面，缺的不是额外列」在技术上**完全正确**，
但参考答案要求答「缺数据列」，于是判它错。

这一题同时违背 TRANSFER 卷的立题本意：它在测「模型能不能发现题目自己有问题」，
而不是「能不能用原理」——正是老板要求 TRANSFER 要避免的那种题。

**改法**：把错误说法直接写进题面，让考生去驳。

```
旧：…并说明要让它是覆盖索引还缺什么。
新：…并说「索引里有 month，这条查询已经是覆盖索引了，所以没有回表开销」。
    请指出这个说法错在哪里。
```

改后同一批 10 次实测：

| 知识点 | 迁移(旧题) | 迁移(新题) | 变化 |
|---|---|---|---|
| kp_btree | -0.7 | 0.0 | +0.7 |
| **kp_covering** | **-32.9** | **-8.6** | **+24.3** |
| kp_clustered | -1.6 | -3.3 | -1.7 |
| kp_write | -2.0 | -3.0 | -1.0 |

只改了一题，所以只有 `kp_covering` 变，其他三个 ±2 分内——这反过来验证了测量本身是可靠的。

**防止再犯**：`ExamItem` 加了 `premise_in_question` 字段（`yes` / `no` / `asserted`），
`bank_diagnostics()` 读它做检查，声明为 `no` 的题**直接拒绝跑实验**。

> 这里试过用关键词匹配（「题面含『还缺』且含『已经有』」），
> 拿旧题面一测发现**根本抓不到**——旧题面里没有「已经有」三个字，
> 坑在「按字面推」这个推理链上。关键词方案给的是虚假的安全感，
> 不如逼改题的人显式声明。

### 问题二：天花板让指标测不出东西

DeepSeek 的 PRE 均值 **90.0**。开卷满分 100，所以最多只能提升 10 分。
而采样噪声 σ=3~5 —— **10 分的空间装不下 3~5 分的噪声。**

于是「OPEN_BOOK_GAIN 测不出来」有两种解释，**本实验分不开**：

|解释 | 含义 |
|---|---|
| A. 学习真没起作用 | 结论是负面的 |
| B. 根本没有可测量的空间 | 结论是**无效的**，不是负面的 |

关键：**绝不能把「测不出来」写成「学习没用」。**
所以报告里新增了 `detect_ceiling()` 和【结论的限制】这一段，
强制结论和它的限制一起出现。

**要区分只能改题库**（提高难度 / 换更冷门知识点），不是改判分或统计方法。

### 还有一条：每卷每知识点只有 1 道题

`PRE` / `MEMORY` / `TRANSFER` 各 4 题，对应 4 个知识点 × 每知识点 1 题。
后果：**单题的判分误差 = 整个知识点的分数**，σ 必然被放大。
上面那个-32.9 完全是 `tr_covering_01` 一题造成的，没有第二个样本能证明它稳定。

`bank_diagnostics()` 会对此报警并写进报告的限制段落。

---


## V0.4：本阶段做了什么

V0.3 有一个**方法论漏洞**，不是 bug：

> V0.3 的两场 POST 是**同一批题**（先开卷、再盲考）。
> 这对「有没有记忆」很够用，但对「有没有真正学会」不够。

盲考时模型可能只是模糊地记得「我刚才答过一道讲回表的题」，
靠的是**题面里的线索**，而不是对原理的理解。
**背答案和形成概念，在 V0.3 的实验设计里长得一模一样。**

V0.4 加了第三场考试来拆开这两件事：

```
PRE → 阅读材料 → 自己整理学习笔记 → 实践
    → MEMORY OPEN_BOOK ┐ 同一批新题
    → MEMORY BLIND     ┘ 测：拿掉资料以后是不是还能记住？
    → TRANSFER BLIND     完全不同的题，同一个知识点，不给笔记
                        测：有没有形成概念，而不是背答案？
```

以 B-Tree 为例：

| | 问什么 | 测什么 |
|---|---|---|
| 学习材料 | B-Tree 为什么需要分裂 | — |
| MEMORY | 为什么所有叶子节点必须在同一层？ | 记住了 |
| **TRANSFER** | 一个节点达到容量上限后，为什么不能简单继续增加子节点？ | **会用** |

关键不对称：**TRANSFER 不开卷。**
开卷能查笔记就说明不了迁移能力——TRANSFER 要测的正是
「不依赖笔记能不能用同一个原理」。

### 六个指标

| 指标 | 意义 |
|---|---|
| `PRE` | 学习前能力 |
| `OPEN_BOOK` | 外部记忆带来的提升 |
| `BLIND` | 模型自身是否出现能力迁移（分 MEMORY / TRANSFER 两场） |
| `Practice success` | 学习能否转化为实际任务能力 |
| `Note quality` | 学习笔记本身是否可靠 |

三个 gain 语义严格区分，**不可混用**：

```
OPEN_BOOK_GAIN = OPEN_BOOK     - PRE   外部记忆的利用能力
MEMORY_GAIN    = MEMORY_BLIND  - PRE   记忆保持（背住了）
TRANSFER_GAIN  = TRANSFER_BLIND- PRE   概念形成（学会了）
```

`TRANSFER_GAIN - MEMORY_GAIN` 的**折损**才是关键信号：
折损大 = 背住了同一批题的话术，换个场景就废。

### 重复实验：单次结果不作数

LLM 判分本身有波动（±10 分），样本又只有 4 题/卷。
单次跑出 TRANSFER +15 分，你无法区分「真迁移」「判分噪声」「题偏易」。

V0.4 强制重复 **N 次（默认 5，建议 5–10）**，看**分布**而不是单点。
判定门槛：

```
mean(gain) - 2 × std(gain) > 0    且    n ≥ 5
```

要求提升幅度大于波动幅度的两倍，才敢说「观察到了提升」。
单看均值会把噪声读成信号。

### 两个独立评估器

V0.3 的 `learning_notes` 只校验 `checksum`（防篡改），
**不判断笔记讲得对不对**。笔记写错了，开卷分数照样漂亮——
整条链路已经在骗人了。V0.4 补上：

**Note quality**（`learning_notes/quality.py`）四维度加权：

| 维度 | 权重 | 判什么 |
|---|---|---|
| `accuracy` | 0.35 | 机制描述与材料是否一致 |
| `completeness` | 0.20 | 五要素是否有实质内容 |
| `transferable` | 0.30 | 讲清「为什么」还是只背结论 |
| `no_hallucination` | 0.15 | 有没有编造材料里不存在的概念 |

**Practice Judge**（`practice/judge.py`）判两件事：
`usable`（产出是否可执行：有具体命令 / SQL / 步骤 / 可观测判据）与 `score`。

两个评估器的**共同纪律**：

1. **人格分离**——写笔记的和评笔记的必须是两个独立调用、两个独立人格。
   同一模型在同一次上下文里既写又评，几乎必然给自己打高分。
2. **不给材料、不给笔记**——「产出好不好」和「有没有用上材料」是两件事。
   混在一起评，判断就废了。
3. **不采信模型自报**——实践的 `outcome` 以评审为准。
4. **兜底必须留痕且刻意保守**——LLM 失效时**不猜准确性**，给中性值并标
   `evaluator='rule_fallback'`。虚高的兜底会让指标彻底失去意义。

**跑法**：

```bash
# 真实模型，重复 10 次（要 OPENAI_API_KEY）
python main.py --provider openai experiment --repeat 10

# 离线冒烟，不联网不烧 token
python main.py --provider scripted experiment --repeat 5

python main.py batch-report              # 跨次统计（六指标 + 三个 gain 判定）
python main.py trials                    # 逐次列出每次的分数
python main.py report --format json      # 单次实验的原始数据
python main.py bank                      # 题库版本与三个指纹
```

### 实验结果怎么读

```
PRE=50 OPEN=90 MEM=80 TRANS=52  → 背住了同题，换场景答不出 = 背诵
PRE=50 OPEN=90 MEM=50 TRANS=70  → 观察到可能的能力迁移迹象
```

**⚠️ 这是能力迁移实验，不是训练实验。**
本项目没有任何参数更新、微调或权重修改。`TRANSFER_GAIN` 再大也**不能**解释为
「模型被改造过」。只要盲考侧有提升，报告强制列出替代解释：
上下文残留、评分噪声、判分漂移、题目难度未标定、题目泄漏。

---

## V0.3：拆开「用外部记忆」与「模型自身能力」

V0.2 暴露了一个**结构性矛盾**：

> 盲考禁止读知识库 → 学到的知识在考试时根本用不上 → 分数差只能来自噪声。

于是 V0.2 那句「提升 0%」是**必然**的，换多少模型都一样。

V0.3 的解法不是放宽闸门，而是**加一场考试**：

```
PRE_TEST_BLIND → STUDY → PRACTICE → OPEN_BOOK_TEST → BLIND_POST_TEST
```

并强制 Agent 在学习阶段产出**结构化笔记**（`learning_notes` 表）：

```
concept          这是什么
principle        为什么是这样（底层机制）
example          具体例子
common_mistake   容易错在哪
application_rule 什么时候用、怎么用
```

笔记带 `checksum`（五要素 SHA256，防篡改）与 `experiment_id`（作用域隔离）。
**结构化笔记强制二次加工**：抽概念、讲原理、举例子、指误区、给规则，
不是把原文切一段。

---

## V0.2：固定题库

V0.1 的题目是 LLM 现场出的，所以 pre/post 两次抽到的是**不同题、比的是不同东西**。
V0.2 把题目固定成**模块级常量 + SHA256 指纹**，模型碰不到，也就没法自己改题。
指纹在实验开始和结束各校验一次，中途被改则实验作废。

---

## 快速开始

```bash
cd agent-01
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # 填入 OPENAI_API_KEY

python main.py init
python main.py --provider openai experiment --repeat 5
python main.py batch-report
python main.py stats
```

**没有 API key 也能验证全链路**（走内置离线脚本，不联网）：

```bash
python main.py --provider scripted experiment --repeat 5
```

## 跑测试

```bash
pytest                                        # 全量：263 个用例
pytest tests/test_v04_experiment.py -v        # V0.4 专项（TRANSFER / 评估器 / 跨次统计）
pytest tests/test_v03_isolation.py -v         # V0.3 专项隔离
```

测试全程离线：`ScriptedProvider` 按 prompt 里的 `[[TASK:KEY]]` 标记返回预设响应，
不联网、不烧 token。

`ScriptedProvider` 支持**上下文分叉**——script 的 value 可以是 dict，
按 prompt 里的特征串选不同回复。这样 PRE / MEMORY_OPEN / MEMORY_BLIND /
TRANSFER 四场能返回**不同**答案，而不是四场一模一样。

> ⚠️ **分叉标记的坑**：标记不能取学习阶段的知识点标题，也不能取知识点内部的措辞——
> 因为 `EVALUATE` 的 prompt 里有「参考答案」，标记若同时出现在参考答案里，
> 盲答也会被误判成开卷。曾经用「内部节点分叉数固定」做标记，它正好在
> `post_btree_01` 的参考答案里，导致盲考首题白拿 47 分。
> V0.4 又踩了一次同类的字面匹配坑（`ANSWER` 与 `transfer_answer` 差一个「来」字，
> 导致 TRANSFER 分支没命中、回落成盲答）。
> `test_evaluate_fork_marker_never_appears_in_question_bank` 锁住了前者。

---

## 目录结构

```
agent-01/
├── main.py                       # CLI 入口（--repeat / --model / batch-report / trials）
├── requirements.txt
├── .env.example
├── pytest.ini
├── brain/
│   ├── llm.py                    # LLMProvider 抽象接口 + OpenAIProvider + ScriptedProvider
│   ├── agent.py                  # V0.1 编排：learn / practice / exam / full_cycle
│   ├── question_bank.py          # V0.4.1 题库：premise_in_question 显式字段 + 硬门禁
│   ├── experiment.py             # V0.4 实验编排：四场考试 + 笔记质量 + 实践评审
│   ├── replicate.py              # 重复实验编排 + 跨次统计 + gain_verdict 判定
│   ├── report.py                 # V0.4 六指标报告 + interpret() + batch_verdict()
│   ├── benchmark_v2.py           # **V0.5** 24 题固定题库（4 kp × 6 题）+ 三卷指纹
│   ├── novel_material.py         # **V0.5** Zephyra 虚构规则手册（现实不存在）
│   ├── experiment_v2.py          # **V0.5** 六阶段流程 + V2Store + 负数哨兵
│   ├── stats_v2.py               # **V0.5** mean/std/CI/天花板/地板/样本量诊断
│   ├── report_v2.py              # **V0.5** 四类 gain 分列 + 越权措辞硬门禁
│   └── materials.py              # 默认学习材料
├── memory/
│   ├── db.py                     # SQLite 连接 + 五段 SCHEMA + 幂等列迁移
│   └── gate.py                   # 访问闸门：两种考试模式 + experiment_id 作用域
├── knowledge/store.py            # 知识存储 + 关键词检索（无向量库）
├── learning_notes/
│   ├── store.py                  # V0.3 结构化学习笔记（核心产物）+ V2 负数 id 路径
│   └── quality.py                # V0.4 笔记质量四维度独立评估
├── practice/
│   ├── judge.py                  # V0.4 实践产出独立评审（usable + score）
│   └── objective.py              # **V0.5** 8 个机器可验证任务 + 沙箱执行
├── tools/
│   └── run_v05_offline.py        # **V0.5** 离线跑完整链路（不联网不烧 token）
├── training_data/generator.py    # 知识 → 问答训练样本
├── tasks/builder.py              # 出题器 + 任务存储
├── evaluation/evaluator.py       # 独立评估器（考官人格，与答题人格分离）
├── experience/store.py           # 经验沉淀（读路径也过闸门）
├── skills/store.py               # 技能项与升级
└── tests/                        # 14 个测试文件
```

## 数据库表（agent.db）

### V0.1 六张

| 表 | 作用 | 关键字段 |
|---|---|---|
| `knowledge` | 已学习的知识点 | topic, title, content, keywords, confidence, **experiment_id** |
| `training_samples` | 由知识生成的问答样本 | question, answer, source_knowledge_id |
| `tasks` | 学习/实践/考试统一建模 | task_type, stage, question, reference_answer, candidate_answer, **blind**, status |
| `evaluations` | 独立评估结果 | score, verdict, feedback |
| `experiences` | 沉淀的经验 | scenario, action, result, lesson, score |
| `skills` | 能力项，随分数升级 | name, level, xp |

### V0.2 新增五张

| 表 | 作用 | 关键字段 |
|---|---|---|
| `materials` | 学习材料版本快照 | version, content, **checksum**（防事后篡改） |
| `question_bank` | 题库指纹 + 完整题目快照 | version, **fingerprint**, pre_fingerprint, post_fingerprint, payload_json |
| `experiments` | 实验主记录 | pre_score, post_score, delta, delta_pct, status, detail_json |
| `experiment_items` | 每道题的前后作答与得分 | phase(pre/post), item_id, kp, question, reference_answer, candidate_answer, score, missing_points |
| `practice_records` | 实践过程记录 | outcome(success/failure), kp, action, result, lesson |

### V0.3 新增一张 + 五处列迁移

| 表 | 作用 | 关键字段 |
|---|---|---|
| `learning_notes` | **结构化学习笔记** | experiment_id, kp, **concept, principle, example, common_mistake, application_rule**, **checksum** |

`MIGRATIONS_V3`（幂等）：`knowledge.experiment_id`、`learning_notes.verified`、
`experiment_items.mode` / `context_used`、`experiments.open_book_score`。

### V0.4 新增三张 + 十一处列迁移

| 表 | 作用 | 关键字段 |
|---|---|---|
| `note_quality` | 笔记质量评估明细 | experiment_id, note_id, **accuracy, completeness, transferable, no_hallucination**, total, verdict, **evaluator**（`llm_judge` / `rule_fallback`） |
| `practice_evaluations` | 实践产出评审明细 | experiment_id, kp, **usable**, score, verdict, feedback, **evaluator** |
| `experiment_batches` | 重复实验批次 | batch_id, topic, planned_trials, bank_fingerprint, model, material_checksum, status, **summary_json** |

`MIGRATIONS_V4`（幂等）：

| 表 | 新增列 |
|---|---|
| `experiments` | `batch_id`, `trial_no`, `memory_open_score`, `memory_blind_score`, `transfer_blind_score`, `practice_usable_rate`, `practice_avg_score`, `note_quality_avg` |
| `experiment_items` | `exam_kind`（`pre` / `memory` / `transfer`） |
| `learning_notes` | `quality_total`, `quality_verdict` |
| `practice_evaluations` | `evaluator` |

**为什么要单独建表而不是复用 V0.1 的表**：一次实验必须**冻结输入**——
题目快照、材料版本、每题前后得分。复用 `tasks`/`evaluations`
就无法证明「前后比的是同一件事」。

### V0.5 新增四张 + 六处列迁移

| 表 | 作用 | 关键字段 |
|---|---|---|
| `v2_experiments` | V2 单次实验主记录 | batch_id, trial_no, **三份冻结指纹**, pre_score, open_book_score, memory_blind_score, transfer_blind_score, **objective_practice_score**, **objective_practice_pass_rate**, evaluator_practice_score, note_quality_avg, detail_json, diagnostics_json |
| `v2_stage_items` | **每阶段每题落齐 8 列** | experiment_id, stage, item_id, knowledge_point, question_type, question, answer, reference_answer, score, evaluator, **context_used**, mode |
| `practice_runs` | **机器判分记录** | experiment_id, task_id, kp, kind, **candidate_output, objective_result, objective_score**, objective_passed, parse_error |
| `v2_batches` | V2 重复实验批次 | batch_id, topic, planned_trials, **三份冻结指纹**, model, status, summary_json |

`MIGRATIONS_V5`（幂等）：

| 表 | 新增列 |
|---|---|
| `v2_experiments` | `trial_key`, `diagnostics_json` |
| `learning_notes` / `note_quality` / `practice_evaluations` | `v2_experiment_id` |
| `practice_evaluations` | **`task_id`** |

**为什么 `practice_evaluations` 也要 `task_id`**：机器判分在 `practice_runs` 里有
`task_id`，评审侧没有的话两侧**无法对齐**——看到「一个 55 分一个 70 分」
却不知道分别是哪个任务，分歧就成了无法定位的噪声。

**V2 为什么用负数 experiment_id**：V0.5 要复用 `learning_notes` /
`note_quality` / `practice_evaluations` 三张老表（重写它们会破坏
V0.4 已验证的链路），但这三张表的 `experiment_id` 都是
`NOT NULL REFERENCES experiments(id)`，而 V2 的 id 来自
`v2_experiments` 自增，是另一套 id 空间。

直接写 V2 的 id → 「第 3 号 V2 实验的笔记」会挂在「第 3 号 V0.4 实验」名下，
**数字撞上，不报任何错**。写 NULL → 过不了 NOT NULL（SQLite 改不掉）。

采用**负数哨兵**：`v2_experiment_key(3) == -3`。
V0.4 的 `experiments.id` 是 AUTOINCREMENT 正整数，物理上不可能为负，
所以负数就是 V2 的专属标记；同时在老 `experiments` 表建一条
`id = -3, topic = "[V2] ..."` 的哨兵行让外键成立。

---

## 严格考试模式（代码层强制，不靠 prompt）

`memory/gate.py` 的两套禁读集：

| 资源 | `OPEN_BOOK_MEMORY` | `BLIND_MODEL` |
|---|---|---|
| 本实验 `knowledge` | ✅ 放行 | ❌ |
| 本实验 `learning_notes` | ✅ 放行（读）/ ❌ 写 | ❌ |
| 原始材料全文 | ❌ | ❌ |
| `reference_answer` | ❌ | ❌ |
| `training_samples` | ❌ | ❌ |
| `evaluation` | ❌ | ❌ |
| PRE/POST/TRANSFER 历史答案 | ❌ | ❌ |
| `experiences` | ❌ | ❌ |

- 读受保护资源抛 `ExamAccessViolation`，**不返回空结果**——
  空结果会被当成「没资料」而静默降级，静默降级会让分数失去意义。
- 开卷有**作用域隔离**：`guard_scope()` 强制只能访问本 `experiment_id` 产出的数据。
- **考试期间禁止写笔记**。否则考生边考边改小抄，`OPEN_BOOK_GAIN` 立刻失去意义。
- **TRANSFER 场不开卷**，也不给任何上下文。
- 每次开考前**自检闸门是否生效**，失效直接 `RuntimeError`，不产出无效分数。

### 独立评估

答题和判分是**两套独立人格、两次独立调用**：
判分（考官）只看题目 + 参考答案 + 考生答案，**看不到考生答题时的上下文**。
`ExperimentRunner` 支持传入独立的 `evaluator_provider`，
测试验证了独立 provider 确实承担了全部 24 次判分与评审，考生 provider 一次都没判分。

对应测试在 `tests/test_exam_isolation.py`（V0.1）、
`tests/test_experiment_isolation.py`（V0.2）、
`tests/test_v03_isolation.py`（V0.3，65 个）、
`tests/test_v04_experiment.py`（V0.4 + V0.4.1）。

---

## CLI 命令

| 命令 | 说明 |
|---|---|
| `experiment [--topic T] [--repeat N] [--material-file F] [--batch-id B] [--format text\|json]` | **跑完整实验，默认重复 5 次** |
| `batch-report [--id B] [--format text\|json]` | 最近批次的跨次统计 + 三个 gain 的判定 |
| `trials [--id B]` | 逐次列出每次实验的分数 |
| `report [--id N] [--format text\|json]` | 单次实验报告（默认最近一次） |
| `bank` | 题库版本与三个指纹 |
| `learn --topic T [--max-items N]` | 学习模式，生成知识点 + 训练样本 |
| `practice --topic T [--count N]` | 实践模式，带知识库作答并评估 |
| `exam --topic T [--count N] [--stage pre_test\|post_test\|exam]` | 严格考试 |
| `cycle --topic T [--count N]` | V0.1 闭环 |
| `ask --question Q [--topic T] [--blind]` | 单题问答；`--blind` 触发闸门自检 |
| `stats` | 各表数据量（**不需要 API key**） |

全局参数：`--db`（默认 `agent.db`）、`--provider openai|scripted`、
`--model`（覆盖 `OPENAI_MODEL`）。

退出码：`0` 正常 / `2` 缺 API key / `3` 被闸门拦截 / `4` 无历史实验或批次。

---

## 环境变量

| 变量 | 必填 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | 是（除离线模式） | 从环境变量读取，不硬编码 |
| `OPENAI_MODEL` | 否 | 默认 `gpt-4o-mini`，也可用 `--model` |
| `OPENAI_BASE_URL` | 否 | 换任意 OpenAI 协议端点（DeepSeek / 通义 / vLLM / Ollama） |

---

## 离线基线（scripted 模式，10 次）

| 指标 | 均值 | 标准差 |
|---|---|---|
| `PRE` | 45.0 | 0.0 |
| `OPEN_BOOK` | 92.0 | 0.0 |
| `MEMORY_BLIND` | 45.0 | 0.0 |
| `TRANSFER_BLIND` | 45.0 | 0.0 |
| `PRACTICE` | 100.0% | 0.0 |
| `NOTE_QUALITY` | 82.3 | 0.0 |

| Gain | 均值 | 判定 |
|---|---|---|
| `OPEN_BOOK_GAIN` | +47.0 | ✔ 观察到提升 |
| `MEMORY_GAIN` | +0.0 | ✘ 未观察到提升 |
| `TRANSFER_GAIN` | +0.0 | ✘ 未观察到提升 |

**⚠️ 离线结果的性质**：这些数字是 `ScriptedProvider` **按设计返回的常量**，
标准差为 0 是因为每次返回完全相同的答案。
**它只证明「实验管线能正确区分四个阶段」，不证明真实模型的学习效果。**
真实效果必须 `--provider openai` 实跑。

离线脚本刻意让 TRANSFER 拿低分（模拟「背住了话术但没形成概念」）。
如果离线脚本让 TRANSFER 也拿高分，那说明管线分不清「记住了」和「学会了」——
那才是真问题。

---

## V0.4.1 的限制清单（历史记录，原样保留）

下面十条是 V0.4.1 收口时的原始判断。**其中第 2、3、7 项已在 V0.5 缓解，
其余仍然成立**——完整的当前清单见 [当前限制（V0.5）](#当前限制-v05)。

0. **【最要紧】天花板让核心指标测不出东西**。DeepSeek 的 PRE 已 90 分，
   开卷最多只剩 10 分空间，而噪声 σ=3~5。`OPEN_BOOK_GAIN` 测不出来有两种解释：
   「学习没用」和「没有可测量的空间」——**本实验分不开**。
   报告已强制把这条限制印在结论旁边（`detect_ceiling()`），
   但要真正解决**只能换更难的题库**。
1. **离线结果只是管线验证**。真实模型上的结果可能完全不同。
2. ~~**样本量仍然小**（每卷 4 题，每知识点 1 题）。~~ → V0.5 已改每 kp 每卷 2 题。
3. ~~**难度配平只做到主观**。~~ → V0.5 换成规则完全定义的虚构材料，
   「难度」变成可计算量（算错就是错），不再靠人工判断。
4. **上下文残留只在「每次 trial 换新 provider」这一层缓解**，
   跨进程复验仍未做。
5. **评估器与考生同源**。两者是同一模型家族，存在共同偏差——
   「换个模型评委会不会判得不一样」这个问题本实验回答不了。
6. **没有对照组**。所有 gain 都是「自己的前后差」，
   没有「同样题给没学习的实例」的对照臂。
7. ~~**题库只覆盖 1 个主题**（SQLite 索引）。~~ → V0.5 换成 Zephyra 虚构系统，
   换主题只需换材料 + 重出题。
8. **规则兜底不判正确性**。LLM 挂掉时笔记质量与实践评审会降级到机械检查，
   此时分数只在同一次实验内可比，不能与 LLM 判分的实验横向比。
9. **V0.4 仍然不做**：微调、训练、Web UI、多 Agent。
   **这是能力迁移实验，不是训练实验。**

## 当前限制（V0.5）

V0.4.1 的九项限制里，第 2、3、7 项在 V0.5 有实质进展，其余仍然成立。
**新增的限制往往比解决的更值得看。**

### V0.5 已缓解

| 原限制 | V0.5 的处理 |
|---|---|
| 每知识点每卷 1 题，单题误差 = 整个 kp 分数 | 改为 2 题/卷/kp；不足 4 样本时打 `insufficient_sample` 标记 |
| 天花板让指标测不出东西 | 换成虚构材料，PRE 理论上得 0 分；`detect_ceiling` 仍保留做兜底 |
| 实践分主观、正确性混在里面 | 机器判分（8 任务）与 LLM 评审分表存、分开报，不做加权 |
| 换主题要重写题库 | 换「材料」即可：题库按知识点配平，规则可替换 |

### V0.5 新引入的限制

1. **虚构材料有一个未验证的前提**：我们**假设**模型对 Zephyra 一无所知。
   这个假设没有直接证据，只能靠「PRE 接近 0 分」间接支持。
   如果某个模型的训练数据恰好覆盖了同名系统（可能性极低但非零），
   天花板会重新出现。当前无检测手段。
2. **`objective_practice_gain` 是代理指标，不是同题对比**。
   它拿「实践机器分」减「考试卷分数」，两者题型不同、卷不同、
   判分器也不同。这个差值可以跨 trial 比较（同口径），
   **不能**解释成「实践比考试容易 30 分」。
3. **逐 kp 的 practice 样本只有 2 × trial 数**，
   而四个 kp 的实践任务都是 100 分时（本轮离线结果就是），
   逐 kp 表在实践列上没有任何区分度。
4. **`v2_stage_items.score` 在 `study` 阶段为 NULL**：
   笔记质量是之后单独评的，不在写入时刻。查该阶段时别以为漏了。
5. **负数哨兵 id 是权宜之计**。`v2_experiment_key()` 把 V2 的 id 取负写进
   三张老表，以满足 `NOT NULL REFERENCES experiments(id)`。
   代价是：任何按 `experiment_id` 正序排的查询都会把 V2 数据排在最前，
   跨代际统计必须显式加符号过滤。
6. **执行模型产出的代码本身就是最大的风险面**。
   现在的护栏是：静态拒绝危险 import + `python -I` + 临时目录 + 15 秒超时。
   这挡得住 `import os`，挡不住更刁钻的纯计算型沙箱逃逸。
   V0.6 之前不应把这一层当作安全边界。
7. **SQL 判分按结果集顺序比对**，因此「排序」这个要求是真的被检查了。
   但也意味着等价的不同写法（子查询顺序不同）可能被判错。

### 仍然成立

- **离线结果只是管线验证**。ScriptedProvider 是确定性的，
  跑 5 次结果完全一样、方差为 0，统计层会显式报出这一点。
- **没有对照组**：所有 gain 都是「自己的前后差」。
- **评估器与考生同源**：共用偏差无法量化。
- **跨进程复验未做**。
- **难度配平仍靠人工**。
- **V0.5 仍然不做**：微调、训练、Web UI、多 Agent。

## 下一步建议（V0.6，按优先级）

1. **加对照组**——同一批题给「没学习的实例」再答一遍。
   现在的 gain 混着「题变简单了」和「真的变强了」。
   这是当前**最大**的未解决的方法论问题。
2. **逐 kp 实践样本扩容**——每个 kp 的实践任务从 2 个加到 4 个，
   或对同一任务出多组参数，让逐 kp 表在实践列上有区分度。
3. **给 objective_practice_gain 配同题对照**——
   出一批「机器判分的考试题」，让该指标与四场考试同题型，
   消除「不同题型相减」这个代理偏差。
4. **换评测模型**——用另一个家族的模型做考官，量化共同偏差。
5. **跨进程复验**——把四场考试拆到不同进程跑，排除上下文残留。
6. **第二个虚构主题**——验证这套流程不是只对 Zephyra 有效。
7. **沙箱评估**——在把「执行候选代码」当成安全边界之前，
   需要一次针对该执行器的逃逸测试。

---

## 架构约束

不用 LangChain、不用向量数据库、不用 Redis、不做 Web 界面、不做多 Agent、
**不训练模型、不做模型微调**。模型调用**只能**通过 `LLMProvider`，
任何模块直接 `import openai` 都算破坏架构。

这一条在 V0.5 有具体落点：报告层有「禁止称训练成功」的硬门禁
（`report_v2.assert_no_forbidden`），越权措辞在**生成时**抛异常。
架构约束不写在文档里靠自觉，是靠门禁。
