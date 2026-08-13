"""LLM ワーカーのテスト (プロバイダをモックし、Signal を直接検証)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from src.app.llm_worker import LLMWorker
from src.llm.base import LLMError, LLMResult


class _FakeProvider:
    name = "fake"
    model = "m"

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def transform(self, raw_text, mode, *, timeout, lead_text=None, compact=False):
        self.last_call = {"raw_text": raw_text, "mode": mode,
                          "lead_text": lead_text, "compact": compact}
        if self._error:
            raise self._error
        return self._result


def _app():
    return QApplication.instance() or QApplication([])


def test_worker_emits_result():
    _app()
    worker = LLMWorker(timeout_s=5.0)
    worker.set_provider(_FakeProvider(result=LLMResult("清書 ⟦x⟧", "fake", "m")))
    received = []
    worker.result_ready.connect(received.append)
    worker.request_transform("ABC", "european")
    assert received == ["清書 ⟦x⟧"]


def test_worker_emits_error_message():
    _app()
    worker = LLMWorker(timeout_s=5.0)
    worker.set_provider(_FakeProvider(error=LLMError("接続に失敗しました")))
    errors = []
    worker.error.connect(errors.append)
    worker.request_transform("ABC", "european")
    assert errors == ["接続に失敗しました"]


def test_worker_without_provider_emits_error():
    _app()
    worker = LLMWorker(timeout_s=5.0)
    errors = []
    worker.error.connect(errors.append)
    worker.request_transform("ABC", "european")
    assert len(errors) == 1
    assert "プロバイダ" in errors[0]


def test_worker_toggles_busy():
    _app()
    worker = LLMWorker(timeout_s=5.0)
    worker.set_provider(_FakeProvider(result=LLMResult("x", "fake", "m")))
    busy = []
    worker.busy_changed.connect(busy.append)
    worker.request_transform("ABC", "european")
    assert busy == [True, False]


def test_worker_passes_lead_text_to_provider():
    """増分清書: 直前のやり取りをプロバイダへ渡すこと."""
    _app()
    worker = LLMWorker(timeout_s=5.0)
    provider = _FakeProvider(result=LLMResult("ok", "fake", "m"))
    worker.set_provider(provider)
    worker.request_transform("NEW", "japanese", "OLD")
    assert provider.last_call["raw_text"] == "NEW"
    assert provider.last_call["lead_text"] == "OLD"


def test_worker_sends_none_when_lead_text_is_empty():
    """空文字は None にする (プロンプトに空の参考欄を作らない)."""
    _app()
    worker = LLMWorker(timeout_s=5.0)
    provider = _FakeProvider(result=LLMResult("ok", "fake", "m"))
    worker.set_provider(provider)
    worker.request_transform("NEW", "japanese", "")
    assert provider.last_call["lead_text"] is None


def test_worker_passes_compact_flag():
    """短いプロンプトの指定がプロバイダまで届くこと."""
    _app()
    worker = LLMWorker(timeout_s=5.0)
    provider = _FakeProvider(result=LLMResult("ok", "fake", "m"))
    worker.set_provider(provider)
    worker.set_compact(True)
    worker.request_transform("ABC", "japanese")
    assert provider.last_call["compact"] is True


def test_worker_defaults_to_full_prompt():
    """set_compact を呼ばなければ従来どおり重い版."""
    _app()
    worker = LLMWorker(timeout_s=5.0)
    provider = _FakeProvider(result=LLMResult("ok", "fake", "m"))
    worker.set_provider(provider)
    worker.request_transform("ABC", "japanese")
    assert provider.last_call["compact"] is False
