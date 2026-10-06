"""LLMProvider 抽象接口。

整个项目**只**通过 LLMProvider 调用模型，不允许直接 import openai。
好处：
1. 换模型（DeepSeek / 通义 / 本地 Ollama）只需新增一个 Provider 子类。
2. 测试可用 ScriptedProvider 离线跑通全流程，不烧 token。
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

Message = dict[str, str]


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def json(self) -> Any:
        """从回复中提取 JSON（容忍 ```json 包裹和前后废话）。"""
        return extract_json(self.text)


class LLMProvider(ABC):
    """模型调用统一接口。"""

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """给定消息列表，返回模型文本回复。"""

    def ask_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Any:
        """要求模型返回 JSON，并解析为Python 对象。失败抛 ValueError。"""
        resp = self.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            return resp.json()
        except ValueError as exc:
            raise ValueError(f"{self.name} 未返回合法 JSON: {exc}") from exc

    def ask_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        resp = self.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.text


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """从模型输出中稳健地提取 JSON。"""
    text = text.strip()
    if not text:
        raise ValueError("空回复")
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退化：截取第一个 { 或 [ 到最后一个配对符号
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法解析 JSON: {text[:200]}")


class OpenAIProvider(LLMProvider):
    """OpenAI 官方 API 实现。API key 从环境变量 OPENAI_API_KEY 读取。

    兼容所有 OpenAI 协议端点（base_url 可覆盖），因此 DeepSeek /
    通义 / vLLM 等只需改环境变量即可复用。
    """

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "缺少 OPENAI_API_KEY。请在 .env 中设置，或用 export OPENAI_API_KEY=sk-xxx"
            )
        self._client = None  # 延迟初始化，避免 import 期就要求依赖

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "未安装 openai 依赖，请执行 pip install -r requirements.txt"
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
        except TypeError:
            # 部分兼容端点不支持 response_format，去掉重试
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=getattr(resp, "model", self.model),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class ScriptedProvider(LLMProvider):
    """离线测试/演示用：按 prompt 里的标记返回预设回复。

    标记格式 [[TASK:KEY]]，测试用它精确控制模型行为，
    从而在不联网、不烧 token 的情况下验证整条链路。

    V0.3：支持「同名标记按上下文分叉」。
    script 的 value 可以是 str，也可以是 dict：
        {"default": "...", "with_notes": "..."}
    当prompt 里出现 with_notes 特征串时用后者。
    这样 PRE / OPEN_BOOK / BLIND_POST 三场考试能返回**不同**答案，
    而不是三场一模一样 —— 否则离线实验永远测不出东西。
    """

    name = "scripted"

    def __init__(self, script: dict[str, Any], fallback: str = "mock") -> None:
        self.script = script
        self.fallback = fallback
        self.calls: list[list[Message]] = []

    @staticmethod
    def _pick(payload: Any, joined: str) -> str | None:
        """从 script 条目里挑一个回复。dict 表示按上下文分叉。"""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            # 优先匹配特征串（顺序即优先级）
            for marker, text in payload.items():
                if marker == "default":
                    continue
                if marker in joined:
                    return text
            return payload.get("default")
        return None

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        msgs = list(messages)
        self.calls.append(msgs)
        joined = "\n".join(m.get("content", "") for m in msgs)
        for key, payload in self.script.items():
            if f"[[TASK:{key}]]" in joined:
                text = self._pick(payload, joined)
                if text is not None:
                    return LLMResponse(text=text, model="scripted")
        return LLMResponse(text=self.fallback, model="scripted")
