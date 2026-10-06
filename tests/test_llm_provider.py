"""LLMProvider 抽象层测试。

重点验证：所有模型调用都走 LLMProvider 接口，且 JSON 解析足够鲁棒。
"""

from __future__ import annotations

import json

import pytest

from brain.llm import LLMProvider, OpenAIProvider, ScriptedProvider, extract_json


def test_llmprovider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore[abstract]


def test_all_providers_subclass_interface():
    assert issubclass(OpenAIProvider, LLMProvider)
    assert issubclass(ScriptedProvider, LLMProvider)


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def test_openai_provider_reads_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    p = OpenAIProvider()
    assert p.api_key == "sk-test-123"
    assert p.model  # 有默认模型名


def test_openai_provider_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    p = OpenAIProvider()
    assert p.model == "gpt-4o"
    assert p.base_url == "https://example.com/v1"


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        '好的，结果如下：{"a": 1} 希望有帮助。',
        '[{"a": 1}]',
    ],
)
def test_extract_json_is_robust(raw):
    assert extract_json(raw) is not None


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("完全没有 JSON 的一段话")
    with pytest.raises(ValueError):
        extract_json("")


def test_scripted_provider_routes_by_tag():
    p = ScriptedProvider({"LEARN": '{"items":[]}'})
    out = p.ask_json("sys [[TASK:LEARN]]", "user")
    assert out == {"items": []}


def test_scripted_provider_records_calls():
    p = ScriptedProvider({"LEARN": "{}"})
    p.ask_text("sys", "hello")
    assert len(p.calls) == 1
    assert p.calls[0][1]["content"] == "hello"


def test_scripted_provider_fallback_when_no_tag():
    p = ScriptedProvider({}, fallback="默认回复")
    assert p.ask_text("sys", "user") == "默认回复"


def test_ask_json_raises_on_invalid_json():
    p = ScriptedProvider({"BAD": "not json at all"})
    with pytest.raises(ValueError, match="未返回合法 JSON"):
        p.ask_json("sys [[TASK:BAD]]", "user")
