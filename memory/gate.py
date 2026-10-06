"""访问策略闸门：保证考试隔离在代码层面成立。

这是本项目的核心约束，不是靠 prompt 里的君子协定：
凡是进入考试模式的流程，读取受保护资源一律抛 ExamAccessViolation，
而不是返回空结果——空结果会被误当成「没资料」而静默降级，
静默降级会让考试分数失去意义。

历史
----
V0.1：单一 blind 模式，禁止 knowledge / training_samples
V0.2：experiences 加入禁止清单（经验含具体答法与得分提示）
V0.3：拆成两种考试模式，并引入 experiment_id 作用域隔离

V0.3 两种考试模式
----------------
OPEN_BOOK_MEMORY（开卷记忆）
    允许：本次实验产生的 knowledge、本次实验产生的 learning_notes
    禁止：原始材料全文、reference_answer、training_samples、
          PRE/POST 历史答案、evaluation 结果、experiences

BLIND_MODEL（盲模型）
    禁止：knowledge、learning_notes、training_samples、experiences、
          reference_answer、evaluation、PRE/POST 历史答案
    只允许看到题目本身

关键区别：OPEN_BOOK 放行的是「Agent 自己写的笔记」，
**不是原始材料**。原始材料全文泄露等于把答案抄给它——
V0.2 的实验（0% 提升）恰恰说明那样测不出东西，V0.3 必须禁掉。
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# 资源分类
# ----------------------------------------------------------------------
RESOURCE_KNOWLEDGE = "knowledge"
RESOURCE_NOTES = "learning_notes"
RESOURCE_MATERIAL = "material"
RESOURCE_SAMPLES = "training_samples"
RESOURCE_EXPERIENCES = "experiences"
RESOURCE_REFERENCE = "reference_answer"
RESOURCE_EVALUATION = "evaluation"
RESOURCE_HISTORY = "exam_history"

#: 任何考试模式下都绝对禁读的东西
FORBIDDEN_IN_ALL_EXAMS = frozenset(
    {
        RESOURCE_SAMPLES,
        RESOURCE_REFERENCE,
        RESOURCE_EVALUATION,
        RESOURCE_HISTORY,
    }
)

#: BLIND_MODEL：禁一切学习产物 + 一切答案来源
BLIND_MODEL_FORBIDDEN = FORBIDDEN_IN_ALL_EXAMS | frozenset(
    {
        RESOURCE_KNOWLEDGE,
        RESOURCE_NOTES,
        RESOURCE_EXPERIENCES,
        RESOURCE_MATERIAL,
    }
)

#: OPEN_BOOK_MEMORY：放行 knowledge / learning_notes，但仍禁原始材料
OPEN_BOOK_FORBIDDEN = FORBIDDEN_IN_ALL_EXAMS | frozenset(
    {
        RESOURCE_MATERIAL,
        RESOURCE_EXPERIENCES,
    }
)

# 向后兼容：V0.1/V0.2 的 blind 语义 == BLIND_MODEL
FORBIDDEN_IN_EXAM = BLIND_MODEL_FORBIDDEN


class ExamAccessViolation(RuntimeError):
    """考试模式下尝试读取受保护资源。"""


class AccessPolicy:
    """控制某次会话能读哪些资源。

    mode 取值：'open'（非考试）/ 'open_book_memory' / 'blind_model'
    experiment_id 用于作用域隔离：考试时只允许访问该实验自己产出的数据。
    """

    def __init__(
        self,
        stage: str,
        blind: bool,
        mode: str = "open",
        experiment_id: int | None = None,
        allowed_resources: frozenset[str] | None = None,
    ) -> None:
        self.stage = stage
        self.blind = blind
        # 向后兼容：V0.1/V0.2 的 AccessPolicy(stage, blind=True) 无mode 参数，
        # 语义是「全盲」。若调用方没显式给 mode 且 blind=True，按 blind_model 处理，
        # 否则老的 blind 调用会静默退化成开卷，考试就泄题了。
        if blind and mode == "open" and experiment_id is None and allowed_resources is None:
            mode = "blind_model"
        self.mode = mode
        self.experiment_id = experiment_id
        # 显式白名单优先；否则按 mode 取默认禁集
        self._forbidden = (
            frozenset() if allowed_resources is not None
            else self._default_forbidden(mode)
        )

    @staticmethod
    def _default_forbidden(mode: str) -> frozenset[str]:
        if mode == "blind_model":
            return BLIND_MODEL_FORBIDDEN
        if mode == "open_book_memory":
            return OPEN_BOOK_FORBIDDEN
        return frozenset()  # 'open'：什么都不禁

    # ------------------------------------------------------------------
    # 两种考试模式的构造器
    # ------------------------------------------------------------------
    @classmethod
    def open_book_memory(cls, stage: str, experiment_id: int) -> "AccessPolicy":
        """开卷记忆：可读本实验的 knowledge 与 learning_notes。"""
        return cls(
            stage=stage,
            blind=True,
            mode="open_book_memory",
            experiment_id=experiment_id,
        )

    @classmethod
    def blind_model(cls, stage: str, experiment_id: int | None = None) -> "AccessPolicy":
        """盲模型：只允许看到题目本身。"""
        return cls(
            stage=stage,
            blind=True,
            mode="blind_model",
            experiment_id=experiment_id,
        )

    @classmethod
    def open_mode(cls, stage: str = "open") -> "AccessPolicy":
        """非考试场景（学习、实践）：什么都能读。"""
        return cls(stage=stage, blind=False, mode="open")

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        scope = f"exp{self.experiment_id}" if self.experiment_id else "no-scope"
        return f"{self.stage}[{self.mode}/{scope}]"

    def allows(self, resource: str) -> bool:
        return resource not in self._forbidden

    def guard(self, resource: str) -> None:
        """访问受保护资源前的检查。考试模式下直接抛异常。"""
        if self.blind and resource in self._forbidden:
            hint = ""
            if self.mode == "open_book_memory":
                hint = "开卷模式只放行本实验产出的 knowledge / learning_notes，不放行原始材料。"
            raise ExamAccessViolation(
                f"考试模式 {self.name} 禁止访问 {resource}；{hint}"
            )

    def guard_scope(self, resource: str, owner_experiment_id: int | None) -> None:
        """作用域检查：考试时只能访问本实验产出的数据。

        这层防的是「读了别的实验的笔记」——V0.3 明确要求
        experiment_id 隔离正常。作用域为空（无实验上下文）时直接拒绝。
        """
        if not self.blind:
            return
        self.guard(resource)
        if self.experiment_id is None:
            raise ExamAccessViolation(
                f"考试模式 {self.name} 缺少 experiment_id，拒绝访问 {resource}"
            )
        if owner_experiment_id != self.experiment_id:
            raise ExamAccessViolation(
                f"跨实验访问被拒：{resource} 属于实验 #{owner_experiment_id}，"
                f"当前考试属于 #{self.experiment_id}"
            )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"AccessPolicy({self.name})"


# 常规模式：什么都能读（非考试场景）
OPEN_POLICY = AccessPolicy.open_mode()
