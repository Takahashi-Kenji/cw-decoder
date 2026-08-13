"""共通 HTTP クライアントのテスト (httpx をモック)."""
import httpx
import pytest

from src.llm.base import LLMError
from src.llm.client import post_json


def _transport(handler):
    return httpx.MockTransport(handler)


def test_post_json_returns_parsed_body(monkeypatch):
    def handler(request):
        assert request.headers["x-test"] == "1"
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        "src.llm.client._build_client",
        lambda timeout: httpx.Client(transport=_transport(handler), timeout=timeout),
    )
    body = post_json("http://x/api", {"a": 1}, {"x-test": "1"}, timeout=5.0)
    assert body == {"ok": True}


def test_http_error_status_becomes_llmerror(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    monkeypatch.setattr(
        "src.llm.client._build_client",
        lambda timeout: httpx.Client(transport=_transport(handler), timeout=timeout),
    )
    with pytest.raises(LLMError) as exc:
        post_json("http://x/api", {}, {}, timeout=5.0)
    assert "401" in str(exc.value)


def test_connection_error_becomes_llmerror(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(
        "src.llm.client._build_client",
        lambda timeout: httpx.Client(transport=_transport(handler), timeout=timeout),
    )
    with pytest.raises(LLMError) as exc:
        post_json("http://x/api", {}, {}, timeout=5.0)
    assert "接続" in str(exc.value)
