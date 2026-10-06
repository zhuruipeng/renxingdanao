"""访问闸门测试 —— 严格考试模式的核心保证。

这是整个 V0.1 最重要的测试：证明「严格考试模式禁止读取知识和训练样本」
是代码层强制，而不是 prompt 里的君子协定。
"""

from __future__ import annotations

import pytest

from knowledge.store import KnowledgeStore
from memory.gate import FORBIDDEN_IN_EXAM, AccessPolicy, ExamAccessViolation
from training_data.generator import TrainingSampleStore

# V0.2 起 experiences 也进入禁止清单（经验里含具体答法与得分提示）
# V0.3 起改为按模式区分，BLIND_MODEL 禁一切学习产物
RESOURCES = ["knowledge", "learning_notes", "training_samples", "experiences",
             "material", "reference_answer", "evaluation", "exam_history"]


def test_forbidden_set_is_what_we_expect():
    from memory.gate import BLIND_MODEL_FORBIDDEN, OPEN_BOOK_FORBIDDEN

    assert BLIND_MODEL_FORBIDDEN == {
        "knowledge", "learning_notes", "training_samples", "experiences",
        "material", "reference_answer", "evaluation", "exam_history",
    }
    # 开卷模式放行 knowledge / learning_notes，但仍禁原始材料与答案来源
    assert "knowledge" not in OPEN_BOOK_FORBIDDEN
    assert "learning_notes" not in OPEN_BOOK_FORBIDDEN
    assert "material" in OPEN_BOOK_FORBIDDEN
    assert "reference_answer" in OPEN_BOOK_FORBIDDEN
    # 向后兼容：FORBIDDEN_IN_EXAM 仍指向 BLIND_MODEL
    from memory.gate import FORBIDDEN_IN_EXAM

    assert FORBIDDEN_IN_EXAM == BLIND_MODEL_FORBIDDEN


@pytest.mark.parametrize("resource", RESOURCES)
def test_blind_policy_blocks_resource(resource):
    policy = AccessPolicy(stage="pre_test", blind=True)
    with pytest.raises(ExamAccessViolation):
        policy.guard(resource)


@pytest.mark.parametrize("resource", RESOURCES)
def test_open_policy_allows_resource(resource):
    AccessPolicy(stage="open", blind=False).guard(resource)  # 不抛异常即通过


def test_blind_policy_allows_unprotected_resources():
    """tasks / evaluations 属于考生与考官共同的中立区，不禁。"""
    policy = AccessPolicy(stage="exam", blind=True)
    policy.guard("tasks")
    policy.guard("evaluations")


def test_blind_policy_blocks_experiences(db):
    """V0.2 新增约束：经验里含具体答法与得分提示，考试时必须禁读。"""
    from experience.store import ExperienceStore

    open_store = ExperienceStore(db)
    open_store.add("场景", "动作", "结果", "教训", 90.0, stage="practice")
    blind_store = ExperienceStore(db, AccessPolicy(stage="exam", blind=True))
    with pytest.raises(ExamAccessViolation):
        blind_store.recent()
    with pytest.raises(ExamAccessViolation):
        blind_store.by_stage("practice")


def test_knowledge_store_respects_blind_policy(db):
    open_store = KnowledgeStore(db)
    open_store.add("T", "标题", "内容")
    blind_store = KnowledgeStore(db, AccessPolicy(stage="exam", blind=True))
    with pytest.raises(ExamAccessViolation):
        blind_store.list_by_topic("T")
    with pytest.raises(ExamAccessViolation):
        blind_store.add("T", "x", "y")
    with pytest.raises(ExamAccessViolation):
        blind_store.search("T", "内容")


def test_training_store_respects_blind_policy(db):
    open_store = TrainingSampleStore(db)
    open_store.add("T", "q", "a")
    blind_store = TrainingSampleStore(db, AccessPolicy(stage="exam", blind=True))
    with pytest.raises(ExamAccessViolation):
        blind_store.list_by_topic("T")


def test_violation_message_names_the_resource():
    policy = AccessPolicy(stage="post_test", blind=True)
    with pytest.raises(ExamAccessViolation, match="knowledge"):
        policy.guard("knowledge")


def test_exam_mode_flagged_on_task_rows(agent):
    agent.learn("SQLite 索引原理", max_items=2)
    agent.exam("SQLite 索引原理", count=1, stage="pre_test")
    rows = agent.tasks.by_stage("pre_test")
    assert rows and all(r["blind"] == 1 for r in rows)
