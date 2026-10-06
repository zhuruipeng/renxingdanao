"""降级路径测试：LLM 返回垃圾数据时系统不能崩，要能兜住。"""

from __future__ import annotations

import json

from brain.agent import Agent
from brain.llm import ScriptedProvider


def _empty_provider() -> ScriptedProvider:
    return ScriptedProvider({}, fallback="模型不可用")


def test_learn_degrades_to_per_item_templates(db):
    """LLM 不可用时，应按 max_items 逐条兜底，保证下游有足够素材。"""
    agent = Agent(db, _empty_provider())
    res = agent.learn("任意主题", max_items=3)
    assert len(res.knowledge_ids) == 3
    assert len(res.sample_ids) == 3
    rows = [agent.knowledge.get(k) for k in res.knowledge_ids]
    assert all("模型不可用" in r["content"] for r in rows)


def test_sample_generator_degrades_to_template(db):
    agent = Agent(db, _empty_provider())
    agent.learn("任意主题", max_items=2)
    # learn 已生成兜底样本；再单独验证 generator 的模板降级
    db.execute("DELETE FROM training_samples")
    out = agent.generator.generate("任意主题", limit=2)
    assert len(out) == 2
    assert all(s["answer"] for s in out)


def test_question_builder_degrades_to_default_question(db):
    agent = Agent(db, _empty_provider())
    items = agent.builder.make_questions("任意主题", count=1)
    assert items and items[0]["question"]
    assert agent.tasks.get(items[0]["id"])["reference_answer"]


def test_evaluator_rule_fallback_when_json_invalid(db):
    agent = Agent(db, _empty_provider())
    agent.learn("任意主题", max_items=1)
    tid = agent.tasks.create("exam", "任意主题", "q", "参考答案A；参考答案B")
    ev = agent.evaluator.evaluate(tid, "q", "我覆盖了 参考答案A", "参考答案A；参考答案B")
    assert 0 <= ev["score"] <= 100
    assert ev["verdict"] in {"pass", "fail"}


def test_evaluator_scores_zero_when_nothing_matches(db):
    agent = Agent(db, _empty_provider())
    tid = agent.tasks.create("exam", "t", "q", "苹果；香蕉；橘子")
    ev = agent.evaluator.evaluate(tid, "q", "完全无关的内容", "苹果；香蕉；橘子")
    assert ev["score"] == 0.0
    assert ev["verdict"] == "fail"


def test_evaluator_without_reference_gives_neutral_score(db):
    agent = Agent(db, _empty_provider())
    tid = agent.tasks.create("exam", "t", "q")
    ev = agent.evaluator.evaluate(tid, "q", "某个答案", None)
    assert ev["score"] == 50.0


def test_reflector_rule_fallback(db):
    agent = Agent(db, _empty_provider())
    low = agent.reflector.reflect("pre_test", "q", "a", 30.0)
    high = agent.reflector.reflect("pre_test", "q", "a", 90.0)
    assert low["lesson"] and high["lesson"]
    assert low["lesson"] != high["lesson"], "不同分数应给出不同教训"


def test_out_of_range_llm_score_is_clamped(db):
    p = ScriptedProvider({"EVALUATE": json.dumps({"score": 9999, "verdict": "pass"})})
    agent = Agent(db, p)
    tid = agent.tasks.create("exam", "t", "q", "ref")
    ev = agent.evaluator.evaluate(tid, "q", "a", "ref")
    assert ev["score"] == 100.0


def test_task_status_transitions(agent):
    agent.learn("任意主题", max_items=1)
    items = agent.builder.make_questions("任意主题", count=1, stage="exam")
    tid = items[0]["id"]
    assert agent.tasks.get(tid)["status"] == "pending"
    agent.tasks.set_answer(tid, "我的答案")
    assert agent.tasks.get(tid)["status"] == "answered"
    agent.evaluator.evaluate(tid, "q", "我的答案", "ref")
    assert agent.tasks.get(tid)["status"] == "evaluated"
