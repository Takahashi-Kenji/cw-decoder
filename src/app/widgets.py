"""UI 用カスタムウィジェット (レベルメータ + スペクトログラム)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class LevelMeter(QWidget):
    """縦型入力レベルメータ (dBFS). スキッシュ閾値を表す横線も描画."""

    FLOOR_DB = -60.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level_db = -120.0
        self._threshold_db: float | None = None
        self.setMinimumWidth(36)
        self.setMinimumHeight(80)

    def set_level(self, level_db: float) -> None:
        self._level_db = max(-120.0, min(0.0, level_db))
        self.update()

    def set_threshold(self, threshold_db: float | None) -> None:
        """スキッシュ閾値を設定 (None で非表示)."""
        self._threshold_db = threshold_db
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, QColor(40, 40, 40))
        floor_db = self.FLOOR_DB
        ratio = max(0.0, min(1.0, (self._level_db - floor_db) / (0.0 - floor_db)))
        filled_h = int(rect.height() * ratio)
        bar_rect = rect.adjusted(0, rect.height() - filled_h, 0, 0)
        # グラデーション風: green (-60..-12), yellow (-12..-3), red (-3..0)
        if self._level_db < -12.0:
            color = QColor(60, 200, 60)
        elif self._level_db < -3.0:
            color = QColor(220, 200, 50)
        else:
            color = QColor(220, 60, 60)
        painter.fillRect(bar_rect, color)
        # 目盛り (-60, -30, -12, -3 dB)
        painter.setPen(QPen(QColor(120, 120, 120)))
        for tick_db in (-60.0, -30.0, -12.0, -3.0):
            r = (tick_db - floor_db) / (0.0 - floor_db)
            y = rect.bottom() - int(rect.height() * r)
            painter.drawLine(rect.left(), y, rect.right(), y)
        # スキッシュ閾値線 (橙色破線)
        if self._threshold_db is not None and self._threshold_db > floor_db:
            t = max(floor_db, min(0.0, self._threshold_db))
            tr = (t - floor_db) / (0.0 - floor_db)
            ty = rect.bottom() - int(rect.height() * tr)
            pen = QPen(QColor(255, 140, 30), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(rect.left() - 2, ty, rect.right() + 2, ty)


class SpectrogramView(pg.GraphicsLayoutWidget):
    """ローリングスペクトログラム (時間軸 = 横、周波数軸 = 縦).

    ``add_audio_block`` で音声を流し込む. 内部で短時間 FFT を計算してロール表示.

    **短点と長音が見分けられる分解能にしてある** (運用者の要望、2026-08-12)。

    以前は ``fft_size=512`` / ``hop=256`` で、20 WPM の短点 (60 ms) が
    **1.9 列**しか占めず符号として読めなかった。**根本は窓の長さである** —
    512 サンプルは 64 ms で、**短点そのものより長い**。送り幅だけ詰めても
    energy が窓の幅に塗り広げられるので見分けられない。

    ===========  =========  =========  ==========  ==========
    設定         窓         1 列       全体の幅    20WPM 短点
    ===========  =========  =========  ==========  ==========
    512 / 256    64.0 ms    32.0 ms    12.8 秒     1.9 列
    **128 / 64**  16.0 ms    8.0 ms    3.2 秒      7.5 列
    ===========  =========  =========  ==========  ==========

    周波数分解能は 15.6 Hz から 62.5 Hz に粗くなるが、**見たいのは
    トーンの ON/OFF であって周波数の細かさではない**。600 Hz のトーンは
    引き続きはっきり横線として見える。

    全体の幅が 12.8 秒から 3.2 秒に縮む。**符号を読むには短いほうがよい** —
    12.8 秒ぶんを画面幅に詰め込むと 1 文字が数ピクセルになる。
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        fft_size: int = 128,
        hop: int = 64,
        history_columns: int = 400,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop = hop
        self.history_columns = history_columns

        self.plot = self.addPlot()
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()

        # 表示する周波数ビン数 = fft_size // 2 + 1, 0〜nyquist
        self._n_freq = fft_size // 2 + 1
        self._spec = np.full(
            (self._n_freq, history_columns), -80.0, dtype=np.float32
        )
        self._img = pg.ImageItem(image=self._spec.T)
        self._img.setLookupTable(self._build_lut())
        self._img.setLevels((-80.0, 0.0))
        self.plot.addItem(self._img)
        # 軸スケール
        self._img.setRect(0, 0, history_columns * hop / sample_rate,
                          sample_rate / 2)
        # 内部バッファ
        self._buffer = np.zeros(0, dtype=np.float32)
        self._window = np.hanning(fft_size).astype(np.float32)
        self._floor_db = -80.0

    # ---- 見え方の調整 (符号として読めることが目的) ----
    def set_floor_db(self, floor_db: float) -> None:
        """表示する下限 dB を変える (コントラスト).

        上限は 0 dB 固定で、下限を上げるほど弱い信号が切り捨てられて
        **強い信号だけがはっきり出る**。ノイズに埋もれて符号が読めないときに
        上げる。既定の -80 dB は弱い信号まで拾うぶん全体に薄い。
        """
        self._floor_db = float(floor_db)
        self._img.setLevels((self._floor_db, 0.0))

    def floor_db(self) -> float:
        return self._floor_db

    def set_span_s(self, span_s: float) -> None:
        """画面全体に映す時間 (秒) を変える (走査速度).

        ``hop`` (1 列あたりのサンプル数) を計算し直す。**窓長 (``fft_size``) は
        変えない。** docstring のとおり分解能を決めるのは窓長であって送り幅では
        ないので、ここを変えても短点が潰れることはない。単に広く映るか、
        拡大して映るかが変わる。

        遅い相手ほど 1 文字が長いので、広めに映した方が読みやすい。
        """
        span_s = max(0.4, float(span_s))
        hop = max(1, int(round(span_s * self.sample_rate / self.history_columns)))
        if hop == self.hop:
            return
        self.hop = hop
        self._img.setRect(
            0, 0, self.history_columns * hop / self.sample_rate,
            self.sample_rate / 2,
        )

    def span_s(self) -> float:
        return self.history_columns * self.hop / self.sample_rate

    def _build_lut(self) -> np.ndarray:
        # シンプルな青→黄→赤 LUT
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            r = min(255, max(0, int(i * 2 - 100)))
            g = min(255, max(0, int(i * 2 - 50)))
            b = min(255, max(0, int(255 - i * 1.5)))
            lut[i] = (r, g, b)
        return lut

    def add_audio_block(self, block: np.ndarray) -> None:
        self._buffer = np.concatenate([self._buffer, block.astype(np.float32)])
        cols_added = 0
        while self._buffer.size >= self.fft_size:
            frame = self._buffer[: self.fft_size] * self._window
            spec = np.fft.rfft(frame)
            mag_db = 20.0 * np.log10(np.abs(spec) + 1e-6)
            self._spec[:, :-1] = self._spec[:, 1:]
            self._spec[:, -1] = mag_db
            self._buffer = self._buffer[self.hop:]
            cols_added += 1
        if cols_added > 0:
            self._img.setImage(self._spec.T, autoLevels=False)

    def reset(self) -> None:
        self._spec.fill(-80.0)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._img.setImage(self._spec.T, autoLevels=False)


class SpectrogramWithControls(QWidget):
    """スペクトログラム + 見え方のスライダ 2 本.

    **見ながら合わせられることが要件。** 目的は「符号としてそれらしく見える」
    ことなので、設定画面に隠さずスペクトルの隣に置く (運用者、2026-08-14)。

    * **コントラスト** — 表示する下限 dB。上げるほど強い信号だけが残る
    * **表示幅** — 画面に映す秒数。遅い相手ほど広く映した方が読みやすい
    """

    floor_changed = Signal(float)
    span_changed = Signal(float)

    # スライダは整数しか扱えないので、内部は 10 倍した整数で持つ
    _SPAN_SCALE = 10

    def __init__(
        self,
        spectrogram: SpectrogramView,
        initial_floor_db: float = -80.0,
        initial_span_s: float = 3.2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.spectrogram = spectrogram
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(spectrogram)

        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(4)

        row.addWidget(self._caption("濃さ"))
        self.floor_slider = QSlider(Qt.Orientation.Horizontal)
        # 下限 dB。-100 (薄い・弱い信号も見える) 〜 -20 (濃い・強い信号だけ)
        self.floor_slider.setRange(-100, -20)
        self.floor_slider.setValue(int(initial_floor_db))
        self.floor_slider.setToolTip(
            "表示する下限 dB。右に振るほど弱い信号が切り捨てられ、\n"
            "強い信号だけがはっきり出ます。ノイズに埋もれて符号が\n"
            "読めないときに右へ。"
        )
        self.floor_slider.valueChanged.connect(self._on_floor)
        self._make_shrinkable(self.floor_slider)
        row.addWidget(self.floor_slider, 1)
        self.floor_label = self._readout()
        row.addWidget(self.floor_label)

        row.addSpacing(8)
        row.addWidget(self._caption("幅"))
        self.span_slider = QSlider(Qt.Orientation.Horizontal)
        # 表示幅 (秒)。0.8 〜 12.8 秒
        self.span_slider.setRange(
            int(0.8 * self._SPAN_SCALE), int(12.8 * self._SPAN_SCALE)
        )
        self.span_slider.setValue(int(initial_span_s * self._SPAN_SCALE))
        self.span_slider.setToolTip(
            "画面に映す時間の長さ。左へ振ると拡大、右へ振ると広く映ります。\n"
            "遅い相手ほど 1 文字が長いので、広めの方が読みやすくなります。\n"
            "**周波数分解能は変わりません** (窓の長さは固定)。"
        )
        self.span_slider.valueChanged.connect(self._on_span)
        self._make_shrinkable(self.span_slider)
        row.addWidget(self.span_slider, 1)
        self.span_label = self._readout()
        row.addWidget(self.span_label)

        layout.addLayout(row)
        self._on_floor(self.floor_slider.value())
        self._on_span(self.span_slider.value())

    # ---- 窓の幅を拘束しないための細工 ----
    #
    # **Qt の最小幅は中身の sizeHint の合計で決まる。** 横に並べたぶんだけ
    # 窓の下限が上がり、細くできなくなる (2026-08-12 に運用者が
    # 「狭くならない」と報告し、入力デバイス欄・パス表示・スライダの 3 つで
    # 同じ対処をしている)。ここで 2 本足すので、同じ轍を踏まないようにする。

    @staticmethod
    def _make_shrinkable(slider: QSlider) -> None:
        """スライダに幅を主張させない (潰れてよい)."""
        slider.setMinimumWidth(0)
        slider.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    @staticmethod
    def _caption(text: str) -> QLabel:
        """小さめの見出し。**さりげなく**支えるための控えめな表示."""
        label = QLabel(text)
        label.setStyleSheet("color: #888; font-size: 10px;")
        return label

    @staticmethod
    def _readout() -> QLabel:
        """現在値の表示。**幅を固定して、値が変わっても並びが動かないように**する.

        桁数で幅が変わると、スライダを動かすたびに横のものが揺れて目障りになる。
        """
        label = QLabel()
        label.setStyleSheet("color: #888; font-size: 10px;")
        label.setFixedWidth(44)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return label

    def _on_floor(self, value: int) -> None:
        self.spectrogram.set_floor_db(float(value))
        self.floor_label.setText(f"{value} dB")
        self.floor_changed.emit(float(value))

    def _on_span(self, value: int) -> None:
        span = value / self._SPAN_SCALE
        self.spectrogram.set_span_s(span)
        self.span_label.setText(f"{span:.1f} 秒")
        self.span_changed.emit(span)

    def floor_db(self) -> float:
        return float(self.floor_slider.value())

    def span_s(self) -> float:
        return self.span_slider.value() / self._SPAN_SCALE


class LevelMeterWithSquelch(QWidget):
    """LevelMeter + スキッシュ閾値スライダのコンポジット."""

    threshold_changed = Signal(float)   # dBFS

    def __init__(
        self,
        initial_threshold_db: float = -60.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.label = QLabel("Lv")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.meter = LevelMeter()
        self.meter.set_threshold(initial_threshold_db)
        layout.addWidget(self.meter, 1)

        # 縦スライダで閾値を設定. -60 (下) 〜 0 (上).
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(-60, 0)
        self.slider.setValue(int(initial_threshold_db))
        self.slider.setFixedWidth(20)
        layout.addWidget(self.slider, 0, Qt.AlignmentFlag.AlignHCenter)

        self.threshold_label = QLabel(f"{int(initial_threshold_db)}dB")
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.threshold_label.setStyleSheet("color: #ff8c1e; font-size: 10px;")
        layout.addWidget(self.threshold_label)

        self.slider.valueChanged.connect(self._on_slider_changed)

    def set_level(self, level_db: float) -> None:
        self.meter.set_level(level_db)

    def threshold_db(self) -> float:
        return float(self.slider.value())

    def _on_slider_changed(self, value: int) -> None:
        self.meter.set_threshold(float(value))
        self.threshold_label.setText(f"{value}dB")
        self.threshold_changed.emit(float(value))


__all__ = ["LevelMeter", "LevelMeterWithSquelch", "SpectrogramView"]
