"""Ollama のモデル候補取得と、デコード用デバイス選択のテスト."""
from __future__ import annotations

import pytest

from src.llm.config import FALLBACK_OLLAMA_MODELS, list_ollama_models


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url):
        if self._error:
            raise self._error
        return _FakeResponse(self._payload)


def _patch(monkeypatch, payload=None, error=None):
    import httpx
    monkeypatch.setattr(httpx, "Client", lambda **kw: _FakeClient(payload, error))


def test_smaller_models_come_first(monkeypatch) -> None:
    """清書は文字の変換なので小さいモデルで足りる。候補の先頭を小さいものにする."""
    _patch(monkeypatch, {"models": [
        {"name": "big:27b", "size": 20_000_000_000},
        {"name": "small:2b", "size": 2_000_000_000},
        {"name": "mid:8b", "size": 5_000_000_000},
    ]})
    assert list_ollama_models("http://x") == ["small:2b", "mid:8b", "big:27b"]


def test_embedding_models_are_excluded(monkeypatch) -> None:
    """埋め込み専用モデルは会話に使えない."""
    _patch(monkeypatch, {"models": [
        {"name": "nomic-embed-text:latest", "size": 100},
        {"name": "small:2b", "size": 2_000_000_000},
    ]})
    assert list_ollama_models("http://x") == ["small:2b"]


def test_cloud_models_go_last(monkeypatch) -> None:
    """クラウドは実体が無く size が極小なので、放っておくと先頭に来てしまう.

    GPU をローカル LLM に使うのが目的なので、既定はローカルであるべき。
    """
    _patch(monkeypatch, {"models": [
        {"name": "gpt-oss:120b-cloud", "size": 12},
        {"name": "small:2b", "size": 2_000_000_000},
    ]})
    assert list_ollama_models("http://x") == ["small:2b", "gpt-oss:120b-cloud"]


def test_falls_back_when_ollama_is_down(monkeypatch) -> None:
    """Ollama が動いていなくてもアプリは起動できる必要がある."""
    _patch(monkeypatch, error=OSError("connection refused"))
    assert list_ollama_models("http://x") == list(FALLBACK_OLLAMA_MODELS)


def test_falls_back_when_no_models_installed(monkeypatch) -> None:
    _patch(monkeypatch, {"models": []})
    assert list_ollama_models("http://x") == list(FALLBACK_OLLAMA_MODELS)


class TestResolveDevice:
    """既定は cpu。GPU はローカル LLM に空ける (運用者の判断)."""

    def test_cpu_is_returned_verbatim(self) -> None:
        pytest.importorskip("PySide6")
        from src.app.main_window import resolve_device
        assert resolve_device("cpu").type == "cpu"

    def test_cuda_falls_back_when_unavailable(self) -> None:
        pytest.importorskip("PySide6")
        import torch
        from src.app.main_window import resolve_device
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert resolve_device("cuda").type == expected

    def test_auto_uses_cuda_when_available(self) -> None:
        pytest.importorskip("PySide6")
        import torch
        from src.app.main_window import resolve_device
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert resolve_device("auto").type == expected


def test_default_setting_is_cpu() -> None:
    """既定でデコードは CPU。GPU をローカル LLM に譲る."""
    from src.infer.settings import AppSettings
    assert AppSettings().decode_device == "cpu"


class TestProviderModelDefaults:
    """プロバイダ切替時に選ばれる既定モデル (候補の先頭)."""

    def _models(self):
        import pytest
        pytest.importorskip("PySide6")
        from src.app.main_window import CWDecoderWindow
        return CWDecoderWindow._PROVIDER_MODELS

    def test_openai_default_is_luna(self) -> None:
        assert self._models()["openai"][0] == "gpt-5.6-luna"

    def test_openai_offers_lighter_alternatives(self) -> None:
        """推論モデルは思考が既定でオンで遅いので、軽い選択肢も出す."""
        models = self._models()["openai"]
        assert "gpt-5-mini" in models

    def test_claude_default_is_haiku(self) -> None:
        """清書は文字の変換なので軽い方で足りる (運用者の判断)."""
        assert self._models()["claude"][0] == "claude-haiku-4-5"
