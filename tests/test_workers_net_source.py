"""ワーカーが LAN 経由キャプチャへ差し替わることのテスト.

Qt のイベントループも実デバイスも使わない。生成されるキャプチャの型だけを見る。
"""
from __future__ import annotations

import pytest

from src.app.workers import AudioInferenceWorker


class _StubEngine:
    """ワーカーが触る最小限のエンジン代役.

    ワーカーは生成時に ``engine.word_break_bias`` へ代入する
    (``_apply_word_break_bias``)。素の ``object()`` は属性を持てないため、
    2026-08 に本物へこの代入が入った時点でここが全部落ちていた。
    """

    word_break_bias = 0.0


def _make_worker() -> AudioInferenceWorker:
    return AudioInferenceWorker(
        engine=_StubEngine(), sample_rate=8000, mode="european"
    )


def test_set_net_source_stores_endpoint(monkeypatch) -> None:
    worker = _make_worker()
    worker.set_net_source("192.168.1.20:45000")
    assert worker._net_endpoint == ("192.168.1.20", 45000)


def test_set_net_source_none_clears(monkeypatch) -> None:
    worker = _make_worker()
    worker.set_net_source("192.168.1.20")
    worker.set_net_source(None)
    assert worker._net_endpoint is None


def test_set_net_source_rejects_bad_endpoint() -> None:
    worker = _make_worker()
    with pytest.raises(ValueError):
        worker.set_net_source("192.168.1.20:abc")


def test_start_uses_network_capture_when_endpoint_set(monkeypatch) -> None:
    """--net-source 指定時は NetworkAudioCapture を作ること."""
    created: dict[str, object] = {}

    class _FakeNetCapture:
        def __init__(self, host, port, **kwargs):
            created["host"] = host
            created["port"] = port
            self.source_sample_rate = 16000

        def start(self):
            created["started"] = True

        def stop(self):
            pass

    monkeypatch.setattr("src.app.workers.NetworkAudioCapture", _FakeNetCapture)
    monkeypatch.setattr("src.app.workers.QTimer", lambda: _NoopTimer())

    worker = _make_worker()
    worker.set_net_source("192.168.1.20:45000")
    worker.start()

    assert created["host"] == "192.168.1.20"
    assert created["port"] == 45000
    assert created["started"] is True


class _NoopTimer:
    """QTimer の代わり。start/stop を受けるだけ."""

    def __init__(self) -> None:
        self.timeout = _NoopSignal()

    def start(self, _interval) -> None:
        pass

    def stop(self) -> None:
        pass


class _NoopSignal:
    def connect(self, _slot) -> None:
        pass


def test_start_uses_local_capture_when_no_endpoint(monkeypatch) -> None:
    created: dict[str, object] = {}

    class _FakeLocalCapture:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs
            self.source_sample_rate = 48000

        def start(self):
            created["started"] = True

        def stop(self):
            pass

    monkeypatch.setattr("src.app.workers.AudioCapture", _FakeLocalCapture)
    monkeypatch.setattr("src.app.workers.QTimer", lambda: _NoopTimer())

    worker = _make_worker()
    worker.start()

    assert created["started"] is True
