"""brain/agent.py —— Agent-01 大脑编排层。

完整闭环：
    学习(learn) → 记忆(knowledge) → 实践(practice) → 考试(exam) → 独立评估(evaluate) → 存经验(experience)

三种模式的差别只在 policy：
    learn / practice  → AccessPolicy(blind=False)  可读知识库
    exam(pre/post)    → AccessPolicy(blind=True)   代码层禁止读 knowledge / training_samples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brain.llm import LLMProvider
from evaluation.evaluator import EvaluationStore, Evaluator
from experience.store import ExperienceReflector, ExperienceStore
from knowledge.store import KnowledgeStore
from memory.db import Database
from memory.gate import AccessPolicy
from skills.store import SkillStore
from tasks.builder import QuestionBuilder, TaskStore
from training_data.generator import SampleGenerator, TrainingSampleStore

LEARN_SYSTEM = (
    "你是知识工程师。把用户给的学习主题拆解为可记忆的知识点。"
    "只输出 JSON："
    '{"items":[{"title":"知识点标题","content":"详细解释（150-400字）",'
    '"keywords":["关键词"]}]}'
)
LEARN_TASK_TAG = "[[TASK:LEARN]]"


@dataclass
class LearnResult:
    topic: str
    knowledge_ids: list[int] = field(default_factory=list)
    sample_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "knowledge_ids": self.knowledge_ids,
            "sample_ids": self.sample_ids,
        }


@dataclass
class RoundResult:
    stage: str
    topic: str
    task_ids: list[int] = field(default_factory=list)
    answers: dict[int, str] = field(default_factory=dict)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    experiences: list[dict[str, Any]] = field(default_factory=list)
    average_score: float = 0.0
    blind: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "topic": self.topic,
            "blind": self.blind,
            "task_ids": self.task_ids,
            "average_score": self.average_score,
            "evaluations": self.evaluations,
            "experiences": self.experiences,
        }


class Agent:
    """Agent-01：一个最小可运行的 AI 学习实验室。"""

    name = "Agent-01"
    version = "0.1"

    def __init__(self, db: Database, provider: LLMProvider) -> None:
        self.db = db
        self.provider = provider

        open_policy = AccessPolicy(stage="open", blind=False)
        self.knowledge = KnowledgeStore(db, open_policy)
        self.samples = TrainingSampleStore(db, open_policy)
        self.tasks = TaskStore(db, open_policy)
        self.evaluations = EvaluationStore(db)
        self.experiences = ExperienceStore(db)
        self.skills = SkillStore(db)

        self.generator = SampleGenerator(db, self.knowledge, self.samples, provider)
        self.builder = QuestionBuilder(db, self.tasks, provider)
        self.evaluator = Evaluator(db, self.evaluations, provider)
        self.reflector = ExperienceReflector(db, self.experiences, provider)

    # ------------------------------------------------------------------
    # 1. 学习 + 记忆
    # ------------------------------------------------------------------
    def learn(self, topic: str, max_items: int = 3, with_samples: bool = True) -> LearnResult:
        """学习模式：向 LLM 索取知识点 → 落 knowledge 表 → 生成训练样本。"""
        if not topic.strip():
            raise ValueError("topic 不能为空")
        user = f"{LEARN_TASK_TAG}\n主题：{topic}\n请拆解 {max_items} 个核心知识点。"
        data: Any = None
        try:
            data = self.provider.ask_json(LEARN_SYSTEM, user, temperature=0.4)
        except ValueError:
            data = None
        items = []
        if isinstance(data, dict):
            items = [i for i in (data.get("items") or []) if i.get("title") and i.get("content")]
        if not items:  # LLM 不可用 → 按 max_items 逐条兜底，保证下游有足够素材
            templates = ("定义与范围", "核心原理", "实践方式", "常见陷阱", "延伸应用")
            items = [
                {
                    "title": f"{topic} · {templates[i % len(templates)]}",
                    "content": f"{topic} 的{templates[i % len(templates)]}。"
                    f"（模型不可用，此条为本地模板兜底。）",
                    "keywords": [topic],
                }
                for i in range(max_items)
            ]

        result = LearnResult(topic=topic)
        for item in items[:max_items]:
            kid = self.knowledge.add(
                topic=topic,
                title=str(item["title"]),
                content=str(item["content"]),
                keywords=[str(k) for k in (item.get("keywords") or [])],
                source=f"{self.name}:{self.version}",
            )
            result.knowledge_ids.append(kid)
            if with_samples:
                pass  # 样本在下面批量生成，便于一次性拿到上下文
        if with_samples:
            for s in self.generator.generate(topic, limit=max_items):
                result.sample_ids.append(s["id"])
        return result

    # ------------------------------------------------------------------
    # 2. 实践（可读知识库）
    # ------------------------------------------------------------------
    def practice(self, topic: str, count: int = 2) -> RoundResult:
        """实践模式：拿训练样本当题目，带着知识库作答，再评估。"""
        pool = self.samples.list_by_topic(topic, limit=count)
        if not pool:
            raise ValueError(f"主题「{topic}」没有训练样本，请先 learn")
        context = self._build_context(topic)
        rr = RoundResult(stage="practice", topic=topic, blind=False)
        for s in pool:
            tid = self.tasks.create(
                task_type="practice",
                stage="practice",
                topic=topic,
                question=s["question"],
                reference_answer=s["answer"],
                difficulty=s.get("difficulty", "medium"),
                blind=False,
            )
            answer = self.builder.answer(s["question"], context=context)
            self.tasks.set_answer(tid, answer)
            rr.task_ids.append(tid)
            rr.answers[tid] = answer
            ev = self.evaluator.evaluate(tid, s["question"], answer, s["answer"])
            rr.evaluations.append(ev)
            rr.experiences.append(
                self.reflector.reflect("practice", s["question"], answer, ev["score"], tid)
            )
        rr.average_score = self._avg(rr.evaluations)
        self.skills.record_score(f"practice:{topic}", rr.average_score)
        return rr

    # ------------------------------------------------------------------
    # 3. 严格考试（禁读知识与样本）
    # ------------------------------------------------------------------
    def exam(self, topic: str, count: int = 3, stage: str = "exam") -> RoundResult:
        """严格考试模式。

        这里用独立的 blind store 实例，确保即使内部代码误调knowledge 也会抛
        ExamAccessViolation 而不是静默拿到答案。
        """
        from memory.gate import ExamAccessViolation

        blind_policy = AccessPolicy(stage=stage, blind=True)
        blind_knowledge = KnowledgeStore(self.db, blind_policy)
        blind_samples = TrainingSampleStore(self.db, blind_policy)

        # 闸门自检：确认 blind 确实生效。
        # 严格考试的全部意义都建立在这两条上——如果闸门失效，这次考试就是开卷，分数作废。
        for store, resource in ((blind_knowledge, "knowledge"), (blind_samples, "training_samples")):
            try:
                store.list_by_topic(topic, limit=1)
            except ExamAccessViolation:
                pass
            else:  # pragma: no cover - 闸门失效属于致命错误
                raise RuntimeError(
                    f"访问闸门失效：严格考试模式竟能读取 {resource}，本次考试结果不可信"
                )

        questions = self.builder.make_questions(topic, count=count, blind=True, stage=stage)
        rr = RoundResult(stage=stage, topic=topic, blind=True)
        for q in questions:
            answer = self.builder.answer(q["question"], context="")  # 无 context
            self.tasks.set_answer(q["id"], answer)
            rr.task_ids.append(q["id"])
            rr.answers[q["id"]] = answer
            ev = self.evaluator.evaluate(
                q["id"], q["question"], answer, q.get("reference_answer")
            )
            rr.evaluations.append(ev)
            rr.experiences.append(
                self.reflector.reflect(stage, q["question"], answer, ev["score"], q["id"])
            )
        rr.average_score = self._avg(rr.evaluations)
        self.skills.record_score(f"exam:{topic}", rr.average_score)
        return rr

    def pre_test(self, topic: str, count: int = 3) -> RoundResult:
        """前测：学习前的裸能力基线（严格考试模式）。"""
        return self.exam(topic, count=count, stage="pre_test")

    def post_test(self, topic: str, count: int = 3) -> RoundResult:
        """后测：学习后的裸能力检验（严格考试模式）。"""
        return self.exam(topic, count=count, stage="post_test")

    # ------------------------------------------------------------------
    # 4. 一键闭环
    # ------------------------------------------------------------------
    def full_cycle(self, topic: str, count: int = 3) -> dict[str, Any]:
        """完整闭环：前测 → 学习 → 实践 → 后测 → 经验汇总。"""
        pre = self.pre_test(topic, count=count)
        learn = self.learn(topic, max_items=count)
        prac = self.practice(topic, count=count)
        post = self.post_test(topic, count=count)
        return {
            "agent": f"{self.name} v{self.version}",
            "topic": topic,
            "pre_test": pre.as_dict(),
            "learn": learn.as_dict(),
            "practice": prac.as_dict(),
            "post_test": post.as_dict(),
            "delta": round(post.average_score - pre.average_score, 2),
        }

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _build_context(self, topic: str, limit: int = 5) -> str:
        rows = self.knowledge.list_by_topic(topic, limit=limit)
        return "\n".join(f"- {r['title']}: {r['content']}" for r in rows)

    @staticmethod
    def _avg(evals: list[dict[str, Any]]) -> float:
        if not evals:
            return 0.0
        return round(sum(float(e["score"]) for e in evals) / len(evals), 2)

    def stats(self) -> dict[str, Any]:
        """各表数据量。V0.2 起改为按实际存在的表动态统计。"""
        return {t: self.db.count(t) for t in self.db.table_names()}
