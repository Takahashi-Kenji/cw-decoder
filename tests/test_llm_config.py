"""プロバイダ生成ファクトリのテスト."""
import pytest

from src.infer.settings import AppSettings
from src.llm.base import LLMError
from src.llm.config import create_provider
from src.llm.providers.claude import ClaudeProvider
from src.llm.providers.ollama import OllamaProvider


def test_create_ollama_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = AppSettings(llm_provider="ollama", llm_model="llama3.1")
    p = create_provider(s)
    assert isinstance(p, OllamaProvider)
    assert p.endpoint == "http://localhost:11434"


def test_create_claude_reads_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-xyz")
    s = AppSettings(llm_provider="claude", llm_model="claude-haiku-4-5-20251001")
    p = create_provider(s)
    assert isinstance(p, ClaudeProvider)


def test_create_openai_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = AppSettings(llm_provider="openai", llm_model="gpt-4o-mini")
    with pytest.raises(LLMError) as exc:
        create_provider(s)
    assert "OPENAI_API_KEY" in str(exc.value)


def test_unknown_provider_raises():
    s = AppSettings(llm_provider="bogus")
    with pytest.raises(LLMError):
        create_provider(s)
