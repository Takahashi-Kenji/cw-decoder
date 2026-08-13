"""UI 用カスタムウィジェット (レベルメータ + スペクトログラム)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSlider, QVBoxLayout, QWidget, QLabel


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
