"""UI ウィジェットのテスト (スペクトログラム)."""
from __future__ import annotations

import pytest


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class TestSpectrogramResolution:
    """**短点と長音が見分けられる分解能であること** (運用者の要望、2026-08-12).

    以前は ``fft_size=512`` / ``hop=256`` で、20 WPM の短点 (60 ms) が
    1.9 列しか占めず符号として読めなかった。**根本は窓の長さ** — 512 サンプルは
    64 ms で、短点そのものより長い。送り幅だけ詰めても見分けられない。
    """

    @staticmethod
    def _view():
        from src.app.widgets import SpectrogramView

        return SpectrogramView(sample_rate=8000)

    def test_窓は短点より短い(self, qapp) -> None:
        """**これが本質。** 窓が短点より長いと energy が塗り広げられる."""
        view = self._view()
        window_s = view.fft_size / view.sample_rate
        dot_at_24wpm = 1.2 / 24.0            # 実運用で一番速い側
        assert window_s < dot_at_24wpm

    def test_短点が何列も占める(self, qapp) -> None:
        view = self._view()
        columns_per_dot = (1.2 / 24.0) * view.sample_rate / view.hop
        assert columns_per_dot >= 5.0

    def test_長音と短点の幅が三倍に開く(self, qapp) -> None:
        view = self._view()
        column_s = view.hop / view.sample_rate
        dot, dash = 1.2 / 20.0, 3 * 1.2 / 20.0
        assert (dash / column_s) - (dot / column_s) >= 10.0

    def test_全体の幅は数秒(self, qapp) -> None:
        """**長すぎると 1 文字が数ピクセルになる。** 短すぎると流れが読めない."""
        view = self._view()
        span_s = view.history_columns * view.hop / view.sample_rate
        assert 2.0 <= span_s <= 6.0

    def test_トーンが見える程度の周波数分解能(self, qapp) -> None:
        """粗くてよいが、600 Hz が判別できる程度には要る."""
        view = self._view()
        assert view.sample_rate / view.fft_size <= 100.0

    def test_音を流すと列が進む(self, qapp) -> None:
        import numpy as np

        view = self._view()
        before = view._spec.copy()
        t = np.arange(8000, dtype=np.float32) / 8000.0
        view.add_audio_block(np.sin(2 * np.pi * 600.0 * t).astype(np.float32))
        assert not np.array_equal(before, view._spec)
