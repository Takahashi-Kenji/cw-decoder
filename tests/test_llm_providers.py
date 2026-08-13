"""LLM プロバイダ実装のテスト (post_json をモック)."""
from src.llm.base import LLMError
from src.llm.providers.ollama import OllamaProvider
from src.llm.providers.openai import OpenAIProvider


def test_ollama_transform_builds_request_and_extracts_text(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, *, timeout):
        captured["url"] = url
        captured["json"] = json
        return {"message": {"content": "清書結果 ⟦推測⟧"}}

    monkeypatch.setattr("src.llm.providers.ollama.post_json", fake_post)
    p = OllamaProvider(model="llama3.1", endpoint="http://localhost:11434")
    result = p.transform("CQ DE JH0ILL", mode="european", timeout=10.0)

    assert result.text == "清書結果 ⟦推測⟧"
    assert result.provider == "ollama"
    assert result.model == "llama3.1"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "llama3.1"
    assert captured["json"]["stream"] is False
    assert captured["json"]["messages"][0]["role"] == "system"


def test_openai_transform_sets_auth_and_extracts_choice(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, *, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return {"choices": [{"message": {"content": "清書"}}]}

    monkeypatch.setattr("src.llm.providers.openai.post_json", fake_post)
    p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
    result = p.transform("ABC", mode="european", timeout=10.0)

    assert result.text == "清書"
    assert result.provider == "openai"
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "gpt-4o-mini"


from src.llm.providers.claude import ClaudeProvider


def test_claude_transform_splits_system_and_sets_headers(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, *, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return {"content": [{"type": "text", "text": "清書済み"}]}

    monkeypatch.setattr("src.llm.providers.claude.post_json", fake_post)
    p = ClaudeProvider(model="claude-haiku-4-5-20251001", api_key="ak-test")
    result = p.transform("ABC", mode="european", timeout=10.0)

    assert result.text == "清書済み"
    assert result.provider == "claude"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "ak-test"
    assert "anthropic-version" in captured["headers"]
    # system は messages から分離されている
    assert isinstance(captured["json"]["system"], str)
    assert all(m["role"] != "system" for m in captured["json"]["messages"])
    assert captured["json"]["max_tokens"] > 0


def test_claude_extracts_text_skipping_non_text_blocks(monkeypatch):
    """先頭が thinking 等の非 text ブロックでも text を正しく抽出する."""
    def fake_post(url, json, headers, *, timeout):
        return {
            "content": [
                {"type": "thinking", "thinking": "考え中..."},
                {"type": "text", "text": "清書結果"},
            ],
            "stop_reason": "end_turn",
        }

    monkeypatch.setattr("src.llm.providers.claude.post_json", fake_post)
    p = ClaudeProvider(model="claude-sonnet-4-6", api_key="ak")
    assert p.transform("ABC", mode="european", timeout=10.0).text == "清書結果"


def test_claude_joins_multiple_text_blocks(monkeypatch):
    """複数の text ブロックは連結する."""
    def fake_post(url, json, headers, *, timeout):
        return {"content": [
            {"type": "text", "text": "前半"},
            {"type": "text", "text": "後半"},
        ]}

    monkeypatch.setattr("src.llm.providers.claude.post_json", fake_post)
    p = ClaudeProvider(model="claude-sonnet-4-6", api_key="ak")
    assert p.transform("ABC", mode="european", timeout=10.0).text == "前半後半"


def test_claude_refusal_raises_clear_error(monkeypatch):
    """拒否 (content 空 + stop_reason=refusal) は専用の分かりやすいエラー."""
    import pytest as _pytest

    def fake_post(url, json, headers, *, timeout):
        return {"content": [], "stop_reason": "refusal"}

    monkeypatch.setattr("src.llm.providers.claude.post_json", fake_post)
    p = ClaudeProvider(model="claude-sonnet-4-6", api_key="ak")
    with _pytest.raises(LLMError) as exc:
        p.transform("ABC", mode="european", timeout=10.0)
    assert "拒否" in str(exc.value)


def test_claude_empty_text_error_includes_stop_reason(monkeypatch):
    """本文が無い場合は stop_reason を含む診断付きエラー."""
    import pytest as _pytest

    def fake_post(url, json, headers, *, timeout):
        return {"content": [], "stop_reason": "max_tokens"}

    monkeypatch.setattr("src.llm.providers.claude.post_json", fake_post)
    p = ClaudeProvider(model="claude-sonnet-4-6", api_key="ak")
    with _pytest.raises(LLMError) as exc:
        p.transform("ABC", mode="european", timeout=10.0)
    assert "max_tokens" in str(exc.value)
