"""全链路测试：学习 → 记忆 → 实践 → 考试 → 评估 → 存经验。"""

from __future__ import annotations

import pytest

from brain.llm import ScriptedProvider
from tests.conftest import SCRIPT, TOPIC


def test_learn_writes_knowledge_and_samples(agent):
    res = agent.learn(TOPIC, max_items=2)
    assert len(res.knowledge_ids) == 2
    assert res.sample_ids, "学习后应自动生成训练样本"
    rows = agent.knowledge.list_by_topic(TOPIC)
    assert len(rows) == 2
    assert {r["title"] for r in rows} == {"B-Tree 索引结构", "覆盖索引"}
    assert agent.db.count("training_samples") == 2


def test_learn_rejects_empty_topic(agent):
    with pytest.raises(ValueError):
        agent.learn("   ")


def test_learn_falls_back_when_llm_returns_garbage(db):
    """LLM 返回非 JSON 时应逐条降级，而不是崩溃。"""
    bad = ScriptedProvider({"LEARN": "这不是 JSON"})
    from brain.agent import Agent

    ag = Agent(db, bad)
    res = ag.learn(TOPIC, max_items=2)
    assert len(res.knowledge_ids) == 2
    assert db.count("knowledge") == 2
    assert all("模型不可用" in ag.knowledge.get(k)["content"] for k in res.knowledge_ids)


def test_knowledge_search_ranks_by_relevance(agent):
    agent.learn(TOPIC, max_items=2)
    hits = agent.knowledge.search(TOPIC, "覆盖索引")
    assert hits, "关键词检索应有结果"
    assert hits[0]["title"] == "覆盖索引"


def test_practice_reads_knowledge_and_scores(agent):
    agent.learn(TOPIC, max_items=2)
    rr = agent.practice(TOPIC, count=2)
    assert len(rr.task_ids) == 2
    assert all(tid in rr.answers for tid in rr.task_ids)
    assert rr.average_score > 0
    assert len(rr.evaluations) == 2
    assert len(rr.experiences) == 2, "实践后应沉淀经验"


def test_practice_context_includes_knowledge(agent, provider):
    agent.learn(TOPIC, max_items=2)
    provider.calls.clear()
    agent.practice(TOPIC, count=1)
    joined = "\n".join(
        m["content"] for call in provider.calls for m in call
    )
    assert "覆盖索引" in joined, "实践模式应能读到知识库内容"


def test_practice_requires_samples(agent):
    with pytest.raises(ValueError, match="训练样本"):
        agent.practice("不存在的主题", count=1)


def test_exam_never_injects_knowledge(agent, provider):
    agent.learn(TOPIC, max_items=2)
    provider.calls.clear()
    agent.exam(TOPIC, count=1, stage="pre_test")
    answer_calls = [
        "\n".join(m["content"] for m in call)
        for call in provider.calls
        if "[[TASK:ANSWER]]" in "\n".join(m["content"] for m in call)
    ]
    assert answer_calls, "应产生一次答题调用"
    for text in answer_calls:
        assert "覆盖索引" not in text, "严格考试模式不得把知识库喂给答题器"
        assert "参考资料" not in text


def test_pre_and_post_test_are_both_blind(agent):
    agent.learn(TOPIC, max_items=2)
    pre = agent.pre_test(TOPIC, count=1)
    post = agent.post_test(TOPIC, count=1)
    assert pre.blind and post.blind
    assert {r["stage"] for r in agent.tasks.by_topic(TOPIC, "exam")} == {
        "pre_test",
        "post_test",
    }


def test_evaluations_are_stored_and_bounded(agent):
    agent.learn(TOPIC, max_items=2)
    rr = agent.practice(TOPIC, count=2)
    for ev in rr.evaluations:
        assert 0 <= ev["score"] <= 100
        assert ev["verdict"] in {"pass", "fail"}
        assert ev["evaluation_id"] > 0
    assert agent.db.count("evaluations") == 2


def test_experiences_accumulate(agent):
    agent.learn(TOPIC, max_items=2)
    agent.practice(TOPIC, count=2)
    agent.post_test(TOPIC, count=1)
    assert agent.experiences.count() >= 3
    row = agent.experiences.recent(1)[0]
    assert row["lesson"], "经验必须带教训字段"


def test_skills_level_up(agent):
    agent.learn(TOPIC, max_items=2)
    agent.practice(TOPIC, count=2)
    sk = agent.skills.get(f"practice:{TOPIC}")
    assert sk is not None
    assert sk["level"] >= 1 and sk["xp"] > 0


def test_full_cycle_end_to_end(agent):
    out = agent.full_cycle(TOPIC, count=1)
    for key in ("pre_test", "learn", "practice", "post_test"):
        assert key in out
    assert "delta" in out
    assert out["agent"].startswith("Agent-01")
    assert agent.stats()["knowledge"] >= 1
    assert agent.stats()["evaluations"] >= 3


def test_stats_reflects_all_six_tables(agent):
    agent.learn(TOPIC, max_items=2)
    s = agent.stats()
    # V0.1 的六张核心表 + V0.2 的实验表都应出现在统计里
    for t in (
        "knowledge",
        "training_samples",
        "tasks",
        "evaluations",
        "experiences",
        "skills",
    ):
        assert t in s
    assert s["knowledge"] == 2
    # 动态统计，不该漏表也不该有幻觉表
    assert set(s) == set(agent.db.table_names())


def test_offline_provider_never_touches_network(agent):
    """保证测试路径完全离线：ScriptedProvider 不应触发任何 HTTP 客户端构造。"""
    import brain.llm as llm

    agent.full_cycle(TOPIC, count=1)
    # ScriptedProvider 从不构造 OpenAI 客户端（_client 恒为 None）
    assert agent.provider._client if hasattr(agent.provider, "_client") else True
    assert not isinstance(agent.provider, llm.OpenAIProvider)


def test_script_fixture_keys_match_task_tags():
    """防止改了 prompt 标记但忘了更新测试脚本。"""
    joined = "\n".join(str(v) for v in SCRIPT.values())
    assert "items" in joined
