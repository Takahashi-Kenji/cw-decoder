"""スペクトル表示の見え方 (コントラストと表示幅) のテスト.

**見ながら合わせられることが要件** (運用者、2026-08-14)。目的は「符号として
それらしく見える」ことなので、設定画面ではなくスペクトルの隣にスライダを置く。
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from src.app.widgets import SpectrogramView, SpectrogramWithControls  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class TestContrast:
    """下限 dB を上げるほど、弱い信号が切り捨てられて濃く見える."""

    def test_floor_is_applied_to_the_image(self, qapp) -> None:
        view = SpectrogramView()
        view.set_floor_db(-40.0)
        assert view.floor_db() == -40.0
        assert view._img.levels[0] == pytest.approx(-40.0)
        assert view._img.levels[1] == pytest.approx(0.0), "上限は 0 dB 固定"

    def test_slider_drives_the_view(self, qapp) -> None:
        view = SpectrogramView()
        panel = SpectrogramWithControls(view, initial_floor_db=-80.0)
        panel.floor_slider.setValue(-35)
        assert view.floor_db() == pytest.approx(-35.0)
        assert panel.floor_db() == pytest.approx(-35.0)


class TestSpan:
    """表示幅 (秒) を変えても**周波数分解能は変わらない**.

    分解能を決めるのは窓長 (``fft_size``) であって送り幅 (``hop``) ではない。
    ここを取り違えると「送り幅を詰めれば読める」と誤解する
    (``SpectrogramView`` の docstring に経緯がある)。
    """

    def test_span_changes_hop_only(self, qapp) -> None:
        view = SpectrogramView(fft_size=128, hop=64, history_columns=400)
        before_fft = view.fft_size
        view.set_span_s(6.4)
        assert view.fft_size == before_fft, "窓長を変えてはいけない"
        assert view.hop == 128
        assert view.span_s() == pytest.approx(6.4)

    def test_narrow_span_zooms_in(self, qapp) -> None:
        view = SpectrogramView(fft_size=128, hop=64, history_columns=400)
        view.set_span_s(1.6)
        assert view.hop == 32
        assert view.span_s() == pytest.approx(1.6)

    def test_span_is_clamped_to_a_sane_minimum(self, qapp) -> None:
        """0 や負を渡されても hop が 0 にならないこと."""
        view = SpectrogramView()
        view.set_span_s(0.0)
        assert view.hop >= 1

    def test_slider_drives_the_view(self, qapp) -> None:
        view = SpectrogramView(fft_size=128, hop=64, history_columns=400)
        panel = SpectrogramWithControls(view, initial_span_s=3.2)
        panel.span_slider.setValue(int(6.4 * panel._SPAN_SCALE))
        assert view.span_s() == pytest.approx(6.4)


class TestDoesNotConstrainTheWindowWidth:
    """**スライダに窓の幅を決めさせない.**

    Qt の最小幅は中身の sizeHint の合計で決まるので、横に並べたぶんだけ
    窓の下限が上がって細くできなくなる。2026-08-12 に運用者が「狭くならない」と
    報告し、入力デバイス欄・パス表示・確信度スライダの 3 つで同じ対処をした。
    ここで 2 本足すので、同じ轍を踏まないようにする。
    """

    def test_sliders_do_not_claim_width(self, qapp) -> None:
        """幅の主張を捨てていること.

        (``minimumSizeHint`` はつまみの大きさ 16px を返すが、水平方針を
        ``Ignored`` にした時点でレイアウトはこれを無視する。見るべきは方針。)
        """
        from PySide6.QtWidgets import QSizePolicy

        panel = SpectrogramWithControls(SpectrogramView())
        for slider in (panel.floor_slider, panel.span_slider):
            assert slider.minimumWidth() == 0
            assert slider.sizePolicy().horizontalPolicy() == (
                QSizePolicy.Policy.Ignored
            ), "スライダが幅を主張している (窓を細くできなくなる)"

    def test_panel_is_narrower_than_the_spectrogram_itself(self, qapp) -> None:
        """スライダ行を足したせいで下限が上がっていないこと."""
        view = SpectrogramView()
        panel = SpectrogramWithControls(view)
        # 見出しと数値表示のぶんだけは要るが、200 px も要らない
        assert panel.minimumSizeHint().width() < 200

    def test_readouts_have_a_fixed_width(self, qapp) -> None:
        """値の桁数で並びが揺れないこと (動かすたびに横がずれると目障り)."""
        panel = SpectrogramWithControls(SpectrogramView())
        before = panel.floor_label.width()
        panel.floor_slider.setValue(-100)
        panel.floor_slider.setValue(-20)
        assert panel.floor_label.width() == before


class TestInitialValues:
    def test_panel_applies_the_initial_settings(self, qapp) -> None:
        view = SpectrogramView(fft_size=128, hop=64, history_columns=400)
        panel = SpectrogramWithControls(
            view, initial_floor_db=-45.0, initial_span_s=6.4
        )
        assert view.floor_db() == pytest.approx(-45.0)
        assert view.span_s() == pytest.approx(6.4)
        assert panel.floor_db() == pytest.approx(-45.0)
        assert panel.span_s() == pytest.approx(6.4)
