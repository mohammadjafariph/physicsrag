"""LLM factory tests (construction only — no network calls)."""

import pytest

from src.llm.anthropic_client import AnthropicClient
from src.llm.base import extract_json_object
from src.llm.factory import build_llm
from src.llm.openai_compat import OpenAICompatibleClient


def test_builds_openai_compatible_client():
    client = build_llm(provider="openrouter", model="test/model", api_key="k")
    assert isinstance(client, OpenAICompatibleClient)


def test_builds_anthropic_client():
    client = build_llm(provider="anthropic", model="claude-3-5-sonnet-latest", api_key="k")
    assert isinstance(client, AnthropicClient)


def test_local_ollama_needs_no_key():
    client = build_llm(provider="ollama", model="llama3.1:8b", api_key="")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.default_model == "llama3.1:8b"


def test_anthropic_requires_key(monkeypatch):
    monkeypatch.setattr("config.settings.llm_api_key", "")
    monkeypatch.setattr("config.settings.groq_key", "")
    with pytest.raises(Exception, match="API key"):
        build_llm(provider="anthropic", model="x", api_key="")


def test_unknown_provider_raises():
    with pytest.raises(Exception, match="unknown provider"):
        build_llm(provider="nope", model="x")


def test_missing_model_raises(monkeypatch):
    monkeypatch.setattr("config.settings.llm_model", "")
    with pytest.raises(Exception, match="model"):
        build_llm(provider="openai", model="", api_key="k")


def test_extract_json_object_strips_fences():
    raw = "```json\n{\"a\": 1}\n```"
    assert extract_json_object(raw) == '{"a": 1}'
    assert extract_json_object('{"a": 1}') == '{"a": 1}'


def test_anthropic_json_mode_appends_instruction():
    client = AnthropicClient(api_key="k", default_model="m")
    kwargs = client._kwargs(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        model=None, temperature=0.1, json_mode=True,
    )
    assert "JSON" in kwargs["system"]
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["temperature"] == 0.1