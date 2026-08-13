"""LLM 基底型のテスト."""
from src.llm.base import LLMError, LLMResult


def test_llm_result_is_frozen_dataclass():
    r = LLMResult(text="本日は晴天", provider="ollama", model="llama3.1")
    assert r.text == "本日は晴天"
    assert r.provider == "ollama"
    assert r.model == "llama3.1"


def test_llm_error_is_exception():
    err = LLMError("接続失敗")
    assert isinstance(err, Exception)
    assert str(err) == "接続失敗"
