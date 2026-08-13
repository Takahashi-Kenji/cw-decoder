"""CW デコーダ メインウィンドウ (PySide6)."""
from __future__ import annotations

import html
import time
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCore import Signal

from src.app.llm_worker import LLMWorker
from src.infer.word_correct import CorrectedSpan, correct_text
from src.app.recorder import Recorder
from src.app.widgets import LevelMeterWithSquelch, SpectrogramView
from src.app.workers import AudioInferenceWorker
from src.infer.audio import list_input_devices
from src.infer.engine import InferenceEngine
from src.infer.settings import AppSettings, load_settings, save_settings
from src.tokens.morse_tokens import DisplayMode
from src.llm import markup
from src.llm.auto import (
    AutoRefineState,
    RefineRequest,
    plan_auto_refine,
    plan_refine_all,
)
from src.llm.base import LLMError
from src.llm.config import (
    FALLBACK_OLLAMA_MODELS,
    create_provider,
    list_ollama_models,
)


class CWDecoderWindow(QMainWindow):
    """メインウィンドウ."""

    # ワーカースレッドへの設定変更は Qt QueuedConnection 経由で発行する
    # (直接呼ぶとクロススレッドで未定義動作になる)
    request_set_mode = Signal(str)
    request_set_threshold = Signal(float)
    request_set_squelch = Signal(float)
    request_set_decoding = Signal(bool)
    request_set_bpf_enabled = Signal(bool)
    request_set_bpf_params = Signal(float, float)
    request_clear_decode = Signal()
    request_llm_transform = Signal(str, str, str)   # (本文, モード, 参考)
    request_set_llm_provider = Signal(object)

    # プロバイダ別の選択候補モデル (先頭が既定)。編集可能なので他の名前も入力できる。
    #
    # **ollama の候補は実機から取る** (``_ollama_models``)。ここに書き固めると、
    # 入っていないモデル名が既定になって「押しても動かない」状態になる
    # (実際に llama3.1 でそうなっていた)。
    _PROVIDER_MODELS = {
        "ollama": list(FALLBACK_OLLAMA_MODELS),
        # 先頭が既定 (プロバイダ切替時に models[0] が選ばれる)
        "openai": ["gpt-5.6-luna", "gpt-5", "gpt-5-mini", "gpt-4.1"],
        # Haiku を先頭に (運用者の判断)。清書は文字の変換なので軽い方で足りる
        "claude": ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"],
    }

    def __init__(
        self,
        engine: InferenceEngine,
        settings: AppSettings | None = None,
        net_source: str | None = None,
        config_path: Path | None = None,
    ) -> None:
        super().__init__()
        # 設定の保存先。**テストは必ず一時パスを渡すこと。**
        # 既定のままだと close() のたびに利用者の実設定を上書きする
        # (実際に UI スモークテストが llm_auto_interval_s を 0.0 に、
        #  llm_model を既定値に書き戻していた)。
        from src.infer.settings import DEFAULT_CONFIG_PATH
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.setWindowTitle("CW デコーダ")
        # **横は控えめに。** 送信ダイアログを並べて置けるようにする
        # (送信中もメイン画面を操作できる。`_open_tx_dialog` 参照)
        self.resize(550, 700)

        self._settings = settings or AppSettings()
        self._engine = engine
        # LAN 経由入力 (--net-source)。指定時は入力デバイス選択を使わない。
        self._net_source = net_source
        self._worker: AudioInferenceWorker | None = None
        self._worker_thread: QThread | None = None
        self._recorder = Recorder(out_dir=Path(self._settings.recording_dir))
        # 開いている送信ダイアログ (``_open_tx_dialog``)。閉じたら必ず None に戻す
        self._tx_dialog: QDialog | None = None
        self._init_live_display_state()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- 上部は 3 段 ----
        #
        # **1 段や 2 段に詰めると窓を細くできない。** Qt の最小幅は中身の
        # sizeHint の合計で決まるので、横に並べたぶんだけ下限が上がる
        # (運用者が「狭くならない」と報告、2026-08-12)。
        #
        # 幅を食っていたのは次の 3 つ。**並べ替えだけでは効かないので、
        # それぞれの下限も外す**:
        #
        # * 入力デバイスの一覧 — デバイス名が長い (``…(Realtek(R) Audio)``)
        # * チェックポイントのパス — 絶対パスがそのまま入る
        # * 確信度閾値のスライダ — 固定 140 px
        top = QHBoxLayout()

        top.addWidget(QLabel("入力デバイス:"))
        self.device_combo = QComboBox()
        self._populate_devices()
        if self._net_source:
            self.device_combo.setEnabled(False)
            self.device_combo.setToolTip(
                f"LAN 経由入力を使用中 ({self._net_source})。入力デバイスは無効です。"
            )
        # **長いデバイス名に窓の幅を決めさせない。** 中身に合わせて広がる
        # 既定のままだと、名前がそのまま最小幅になる
        self.device_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.device_combo.setMinimumContentsLength(8)
        self.device_combo.setMinimumWidth(120)
        # **``Ignored`` にしない** (上の LLM モデル欄と同じ理由)。
        # いまは伸縮 2 で場所をもらえているが、並びを変えた誰かが
        # 潰れる側に倒すのを防ぐ
        self.device_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        top.addWidget(self.device_combo, 2)

        top.addWidget(QLabel("モード:"))
        self.mode_combo = QComboBox()
        # **括弧の中の英語は落とす。** 幅を食うわりに情報が増えない
        self.mode_combo.addItems(["欧文", "和文", "自動"])
        _mode_index = {"european": 0, "japanese": 1, "auto": 2}
        self.mode_combo.setCurrentIndex(_mode_index.get(self._settings.mode, 0))
        top.addWidget(self.mode_combo)

        top.addWidget(QLabel("確信度閾値:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self._settings.confidence_threshold * 100))
        self.threshold_slider.setFixedWidth(70)      # 140 の半分 (運用者の指定)
        top.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel(f"{self._settings.confidence_threshold:.2f}")
        top.addWidget(self.threshold_value_label)
        top.addStretch(1)

        root.addLayout(top)

        # ---- 2 段目: 操作ボタン + チェックポイント ----
        second = QHBoxLayout()

        self.start_btn = QPushButton("開始")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.decode_toggle_btn = QPushButton("● デコード開始")
        self.decode_toggle_btn.setCheckable(True)
        self.decode_toggle_btn.setEnabled(False)
        self.decode_toggle_btn.setToolTip("ライブデコードの ON/OFF を切替")
        self.clear_decode_btn = QPushButton("クリア")
        self.clear_decode_btn.setToolTip("デコード本文をクリアする (清書は別の「クリア」)")
        second.addWidget(self.start_btn)
        second.addWidget(self.stop_btn)
        second.addWidget(self.decode_toggle_btn)
        second.addWidget(self.clear_decode_btn)

        self.record_btn = QPushButton("● 録音開始")
        self.record_btn.setCheckable(True)
        second.addWidget(self.record_btn)

        self.tx_btn = QPushButton("送信…")
        self.tx_btn.setToolTip("無線機を繋いだ PC に打鍵させます")
        self.tx_btn.clicked.connect(self._open_tx_dialog)
        second.addWidget(self.tx_btn)

        # **絶対パスに窓の幅を決めさせない。** ファイル名だけ出し、全体は
        # ツールチップで読めるようにする (パスがそのまま最小幅になっていた)
        self.ckpt_label = QLabel()
        self.ckpt_label.setStyleSheet("color: #888;")
        self._set_ckpt_label(self._settings.checkpoint_path)
        second.addWidget(self.ckpt_label)
        self.load_ckpt_btn = QPushButton("読込…")
        second.addWidget(self.load_ckpt_btn)

        root.addLayout(second)

        # ---- 3 段目: 表示の切り替えと BPF ----
        third = QHBoxLayout()

        self.show_spectrogram_check = QCheckBox("スペクトル")
        self.show_spectrogram_check.setToolTip("下のスペクトログラムの表示を切り替えます")
        self.show_spectrogram_check.setChecked(self._settings.show_spectrogram)
        third.addWidget(self.show_spectrogram_check)

        self.show_provisional_check = QCheckBox("未確定")
        self.show_provisional_check.setChecked(self._settings.show_provisional)
        self.show_provisional_check.setToolTip(
            "確定前の文字をグレーで表示する。"
            "文脈が増えると読みが変わるため、通常はオフのほうが読みやすい"
        )
        third.addWidget(self.show_provisional_check)

        self.word_correct_check = QCheckBox("辞書補正")
        self.word_correct_check.setChecked(self._settings.word_correct_enabled)
        self.word_correct_check.setToolTip(
            "CW 定型語彙で語を切り直し・寄せする (LLM 不要・即時)。"
            "補正した箇所は橙色で表示。コールサイン・RST・プロサインは触りません"
        )
        third.addWidget(self.word_correct_check)

        self.bpf_check = QCheckBox("BPF")
        self.bpf_check.setChecked(self._settings.bpf_enabled)
        self.bpf_check.setToolTip(
            "帯域通過フィルタ — CW トーン以外のノイズを除去"
        )
        third.addWidget(self.bpf_check)

        third.addWidget(QLabel("中心"))
        self.bpf_center_spin = QSpinBox()
        self.bpf_center_spin.setRange(300, 1200)
        self.bpf_center_spin.setSuffix(" Hz")
        self.bpf_center_spin.setValue(int(self._settings.bpf_center_hz))
        self.bpf_center_spin.setFixedWidth(78)
        third.addWidget(self.bpf_center_spin)

        third.addWidget(QLabel("帯域"))
        self.bpf_bw_spin = QSpinBox()
        self.bpf_bw_spin.setRange(100, 1000)
        self.bpf_bw_spin.setSuffix(" Hz")
        self.bpf_bw_spin.setValue(int(self._settings.bpf_bandwidth_hz))
        self.bpf_bw_spin.setFixedWidth(78)
        third.addWidget(self.bpf_bw_spin)

        third.addStretch(1)
        root.addLayout(third)

        # ---- メインエリア: 左テキスト + 右レベルメータ ----
        body = QHBoxLayout()

        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        font = QFont("Consolas", 14)
        self.text_view.setFont(font)
        body.addWidget(self.text_view, 5)

        self.level_meter = LevelMeterWithSquelch(
            initial_threshold_db=self._settings.squelch_threshold_db
        )
        body.addWidget(self.level_meter)

        root.addLayout(body, 5)

        # ---- スペクトログラム ----
        self.spectrogram = SpectrogramView(sample_rate=self._settings.sample_rate)
        self.spectrogram.setVisible(self._settings.show_spectrogram)
        root.addWidget(self.spectrogram, 3)

        # ---- LLM 清書パネル ----
        self.llm_text_view = QTextEdit()
        self.llm_text_view.setReadOnly(True)
        self.llm_text_view.setFont(QFont("Yu Gothic UI", 13))
        self.llm_text_view.setPlaceholderText("LLM 清書結果 (確定=黒 / 推測=赤)")
        root.addWidget(self.llm_text_view, 3)

        llm_bar = QHBoxLayout()
        llm_bar.addWidget(QLabel("LLM:"))
        self.llm_provider_combo = QComboBox()
        self.llm_provider_combo.addItems(["ollama", "openai", "claude"])
        _p_index = {"ollama": 0, "openai": 1, "claude": 2}
        self.llm_provider_combo.setCurrentIndex(
            _p_index.get(self._settings.llm_provider, 0)
        )
        llm_bar.addWidget(self.llm_provider_combo)

        llm_bar.addWidget(QLabel("モデル:"))
        self.llm_model_edit = QComboBox()
        self.llm_model_edit.setEditable(True)
        # **モデル名に窓の幅を決めさせない。** 編集可能なコンボは候補の
        # 一番長いもの (`claude-haiku-4-5` 等) がそのまま最小幅になり、
        # 実測で 582 px と**窓全体の下限を決めていた** (2026-08-12)
        self.llm_model_edit.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.llm_model_edit.setMinimumContentsLength(10)
        self.llm_model_edit.setMinimumWidth(120)
        # **``Ignored`` を使ってはいけない。** 一度そうしたところ、伸縮の
        # 指定が無いこの欄は**幅 0 px に潰れて選べなくなった** (運用者が
        # 「モデル選択ができない」と報告)。``Ignored`` は「sizeHint を無視して
        # 余りを取る」であり、余りが無ければ 0 になる。
        self.llm_model_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        # 起動時: 保存プロバイダの候補で初期化し、保存モデルを現在値にする
        self.llm_model_edit.addItems(self._models_for(self._settings.llm_provider))
        self.llm_model_edit.setEditText(self._settings.llm_model)
        llm_bar.addWidget(self.llm_model_edit, 1)

        self.llm_refine_btn = QPushButton("まとめて清書")
        self.llm_refine_btn.setToolTip(
            "まだ清書していない分を 1 回で全部清書する。"
            "最初からやり直すときは「クリア」を押してから押す"
        )
        # **ここで段を割る。** 1 行に並べると LLM 行だけで 700 px あり、
        # 窓全体の下限を決めていた (2026-08-12)
        llm_bar.addStretch(1)
        root.addLayout(llm_bar)

        llm_bar2 = QHBoxLayout()
        llm_bar2.addWidget(self.llm_refine_btn)
        self.llm_clear_btn = QPushButton("クリア")
        self.llm_clear_btn.setToolTip("清書パネルの表示を消去する")
        llm_bar2.addWidget(self.llm_clear_btn)
        self.llm_auto_check = QCheckBox("自動清書")
        self.llm_auto_check.setChecked(self._settings.llm_auto)
        self.llm_auto_check.setToolTip(
            "確定した分だけを順に清書する (増分方式)。清書済みは送り直さないので"
            "ローカル LLM でも待ち時間が短い"
        )
        llm_bar2.addWidget(self.llm_auto_check)

        self.llm_compact_check = QCheckBox("短いプロンプト")
        self.llm_compact_check.setChecked(self._settings.llm_compact_prompt)
        self.llm_compact_check.setToolTip(
            "小さいローカルモデル向けの短い指示を使う (207 文字 対 1403 文字)。"
            "重い指示だと 4B 前後のモデルは例文をそのまま返したり捏造したりする。"
            "クラウドの大きいモデルを使うときはオフにすると細かい指示が効く"
        )
        llm_bar2.addWidget(self.llm_compact_check)

        self.llm_highlight_check = QCheckBox("推測を赤で表示")
        self.llm_highlight_check.setChecked(self._settings.llm_highlight_guesses)
        self.llm_highlight_check.setToolTip(
            "LLM が推測・補正した箇所を赤くする。赤が多くて読みにくいときはオフに"
            "できる (オフでもマーカー記号は表示しない)"
        )
        llm_bar2.addWidget(self.llm_highlight_check)
        llm_bar2.addStretch(1)
        root.addLayout(llm_bar2)

        # ---- ステータスバー ----
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("待機中")
        # **受信 WPM は常設ウィジェットで出す。** ステータスバーの本文は
        # `status` (毎秒) と `_on_stream_diag` (再デコードごと) が奪い合って
        # いるので、そこに混ぜると出たり消えたりする。
        self.wpm_label = QLabel("")
        self.wpm_label.setToolTip("受信信号から測った速度 (だいたいの目安)")
        self.statusBar().addPermanentWidget(self.wpm_label)

        # ---- LLM 清書状態の初期化 ----
        # NOTE: 実装手順では _init_live_display_state() 近傍 (__init__ 冒頭) と
        # 指示されているが、_init_llm_worker → _refresh_llm_provider が
        # llm_provider_combo 等のウィジェットを参照するため、ウィジェット生成後の
        # この位置で初期化する。
        self._auto_refine_state = AutoRefineState()
        self._llm_busy = False
        self._init_llm_worker()

        # ---- イベント接続 ----
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.decode_toggle_btn.toggled.connect(self._on_decode_toggled)
        self.clear_decode_btn.clicked.connect(self._on_clear_decode)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.load_ckpt_btn.clicked.connect(self._on_load_checkpoint)
        self.record_btn.toggled.connect(self._on_record_toggled)
        self.show_spectrogram_check.toggled.connect(self.spectrogram.setVisible)
        self.show_provisional_check.toggled.connect(self._on_show_provisional_toggled)
        self.llm_refine_btn.clicked.connect(self._on_refine_clicked)
        self.llm_clear_btn.clicked.connect(self._on_llm_clear)
        self.llm_provider_combo.currentTextChanged.connect(self._on_llm_provider_changed)
        self.llm_model_edit.lineEdit().editingFinished.connect(self._on_llm_model_changed)
        self.llm_model_edit.activated.connect(self._on_llm_model_changed)
        # 赤表示の切替は再清書せず、積み上げ済みの結果を描き直すだけ
        self.llm_highlight_check.toggled.connect(lambda _: self._refresh_llm_display())
        self.llm_compact_check.toggled.connect(self._on_compact_toggled)

        self._restore_geometry()

    def _models_for(self, provider: str) -> list[str]:
        """プロバイダのモデル候補。ollama だけは実機に入っているものを返す."""
        if provider == "ollama":
            return list_ollama_models(self._settings.ollama_endpoint)
        return list(self._PROVIDER_MODELS.get(provider, []))

    # ---- デバイス列挙 ----
    def _set_ckpt_label(self, path: str | None) -> None:
        """チェックポイントの表示を更新する. **ファイル名だけ出す。**

        絶対パスをそのまま出すと、その幅が窓の下限になる。全体はツールチップ
        で読める (2026-08-12 に運用者が「窓が狭くならない」と報告)。
        """
        if not path:
            self.ckpt_label.setText("(未読込)")
            self.ckpt_label.setToolTip("未学習モデルを使用中です")
            return
        self.ckpt_label.setText(Path(path).name)
        self.ckpt_label.setToolTip(path)

    def _populate_devices(self) -> None:
        self.device_combo.clear()
        self.device_combo.addItem("(システムデフォルト)", userData=None)
        for info in list_input_devices():
            label = f"[{info.index}] {info.name}"
            self.device_combo.addItem(label, userData=info.index)
        # 設定の input_device を復元
        if self._settings.input_device is not None:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == self._settings.input_device:
                    self.device_combo.setCurrentIndex(i)
                    break

    # ---- ワーカー制御 ----
    def _on_start(self) -> None:
        if self._worker is not None:
            return
        mode = self._current_mode()
        threshold = self.threshold_slider.value() / 100.0
        device_index = self.device_combo.currentData()
        self._worker = AudioInferenceWorker(
            engine=self._engine,
            sample_rate=self._settings.sample_rate,
            mode=mode,
            confidence_threshold=threshold,
            prosign_threshold=self._settings.prosign_threshold,
            switch_on_japanese_only=self._settings.switch_on_japanese_only,
            squelch_threshold_db=self.level_meter.threshold_db(),
            squelch_hold_sec=self._settings.squelch_hold_sec,
            bpf_enabled=self.bpf_check.isChecked(),
            bpf_center_hz=float(self.bpf_center_spin.value()),
            bpf_bandwidth_hz=float(self.bpf_bw_spin.value()),
            window_s=self._settings.window_s,
            hop_s=self._settings.hop_s,
            line_break_gap_s=self._settings.line_break_gap_s,
            commit_lag_s=self._settings.commit_lag_s,
            head_guard_s=self._settings.head_guard_s,
            decode_left_context_s=self._settings.decode_left_context_s,
            commit_jitter_margin_s=self._settings.commit_jitter_margin_s,
        )
        self._worker.set_device(device_index)
        try:
            self._worker.set_net_source(self._net_source)
        except ValueError as exc:
            self.statusBar().showMessage(f"LAN 経由入力の指定が不正です: {exc}")
            self._worker = None
            return
        self._worker.set_record_callback(self._recorder.add_block)
        self._worker.level_changed.connect(self.level_meter.set_level)
        self._worker.audio_block_received.connect(self.spectrogram.add_audio_block)
        self._worker.status.connect(self.statusBar().showMessage)
        self._worker.error.connect(self._on_worker_error)
        self._worker.committed_text_changed.connect(self._on_committed_text)
        self._worker.provisional_text_changed.connect(self._on_provisional_text)
        self._worker.stream_diag.connect(self._on_stream_diag)
        self._worker.current_mode_changed.connect(self._on_current_mode)
        self._worker.received_wpm_changed.connect(self._on_received_wpm)
        # UI → ワーカー はキューイング接続必須
        self.request_set_mode.connect(self._worker.set_mode)
        self.request_set_threshold.connect(self._worker.set_confidence_threshold)
        self.request_set_squelch.connect(self._worker.set_squelch_threshold)
        self.request_set_decoding.connect(self._worker.set_decoding)
        self.request_set_bpf_enabled.connect(self._worker.set_bpf_enabled)
        self.request_set_bpf_params.connect(self._worker.set_bpf_params)
        self.request_clear_decode.connect(self._worker.clear)
        self.level_meter.threshold_changed.connect(self.request_set_squelch)
        self.bpf_check.toggled.connect(self.request_set_bpf_enabled)
        self.bpf_center_spin.valueChanged.connect(self._on_bpf_changed)
        self.bpf_bw_spin.valueChanged.connect(self._on_bpf_changed)

        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.start)
        self._worker_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.decode_toggle_btn.setEnabled(True)
        self.device_combo.setEnabled(False)
        self.statusBar().showMessage("音声受信中... デコードを開始するには「● デコード開始」を押してください")

    def _on_stop(self) -> None:
        if self._worker is None:
            return
        # 録音中なら保存
        if self._recorder.is_recording:
            self.record_btn.setChecked(False)
        self._worker.stop()
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        self._worker = None
        self._worker_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.decode_toggle_btn.setChecked(False)
        self.decode_toggle_btn.setEnabled(False)
        self.device_combo.setEnabled(not self._net_source)
        self.statusBar().showMessage("停止しました")

    # ---- 設定変更 ----
    def _current_mode(self) -> DisplayMode:
        """画面が今表示しているモード. **設定ファイルの値ではない。**

        戻り値を ``DisplayMode`` にしてあるのは、``auto`` を落とす受け手を
        型で見つけられるようにするため (送信ダイアログの型の絞り込みが
        ``auto`` を未知として扱い、型が 9/10 消えていた)。
        """
        modes: tuple[DisplayMode, ...] = ("european", "japanese", "auto")
        return modes[self.mode_combo.currentIndex()]

    def _on_mode_changed(self, _index: int) -> None:
        mode = self._current_mode()
        if self._worker is not None:
            # スレッド安全な signal 経由で送信
            self.request_set_mode.emit(mode)
        self.statusBar().showMessage(f"モード: {mode}")

    def _on_threshold_changed(self, value: int) -> None:
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
        if self._worker is not None:
            self.request_set_threshold.emit(threshold)

    def _on_load_checkpoint(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "チェックポイント選択", "models", "Checkpoint (*.pt)"
        )
        if not path_str:
            return
        try:
            new_engine = InferenceEngine.from_checkpoint(
                path_str, device=self._engine.device
            )
        except (RuntimeError, OSError, KeyError) as exc:
            self.statusBar().showMessage(f"読込失敗: {exc!r}")
            return
        was_running = self._worker is not None
        if was_running:
            self._on_stop()
        self._engine = new_engine
        self._set_ckpt_label(path_str)
        self.ckpt_label.setStyleSheet("color: #ccc;")
        self.statusBar().showMessage(f"チェックポイント読込: {path_str}")
        if was_running:
            self._on_start()

    def _on_record_toggled(self, checked: bool) -> None:
        if checked:
            self._recorder.start()
            self.record_btn.setText("■ 録音停止")
            self.statusBar().showMessage("録音中…")
        else:
            wav_path = self._recorder.save_and_reset(
                decoded_text=self._committed_text,
                mode=self._current_mode(),
            )
            self.record_btn.setText("● 録音開始")
            if wav_path is not None:
                self.statusBar().showMessage(f"録音保存: {wav_path}")
            else:
                self.statusBar().showMessage("録音内容なし")

    # ---- デコードトグル ----
    def _on_decode_toggled(self, on: bool) -> None:
        self.decode_toggle_btn.setText("■ デコード停止" if on else "● デコード開始")
        if self._worker is not None:
            self.request_set_decoding.emit(on)

    # ---- デコード本文クリア ----
    def _on_clear_decode(self) -> None:
        """デコード本文 (確定/暫定テキスト) をクリアする. 清書パネルは消さない."""
        if self._worker is not None:
            # ワーカー側の蓄積も初期化 (これだけだと再 emit で戻るため Signal 経由)
            self.request_clear_decode.emit()
        # 未起動時・即時反映のためローカル表示も消す
        self._committed_text = ""
        self._committed_spans = ()
        self._provisional_text = ""
        self._refresh_decode_display()
        # クリア後に自動清書が再び発火できるよう基準もリセット
        self._auto_refine_state = AutoRefineState()
        self.statusBar().showMessage("デコード本文をクリアしました")

    # ---- ライブ連続モード表示 ----
    def _init_live_display_state(self) -> None:
        """確定/暫定テキストの状態を初期化する."""
        self._committed_text: str = ""
        # 辞書補正で触った範囲 (_committed_text 上の文字位置). 表示の色分けに使う.
        self._committed_spans: tuple[CorrectedSpan, ...] = ()
        # 清書結果の積み上げ (増分方式). 手動ボタンのときは置き換える
        self._llm_refined_html: list[str] = []
        # どこまで送ったか (応答が返ったら refined_len に確定させる)
        self._llm_sent_len: int = 0
        self._provisional_text: str = ""
        self._auto_submode: str = "european"
        # 受信信号から測った速度 (WPM)。測れていないときは None。
        # 送信ダイアログの「受信に合わせる」に渡す。
        self._received_wpm: float | None = None

    def _on_committed_text(self, text: str) -> None:
        """確定テキスト全体を受信し、辞書補正をかけてから表示を更新する.

        補正後のテキストを ``_committed_text`` に入れるので、LLM 清書にも
        録音ラベルにも補正済みの文が渡る (LLM が直す誤りが減る)。

        **確定列の末尾の語だけは後から書き換わりうる。** 確定境界が語の途中に
        落ちると ``CQD`` → 次の hop で ``CQDE`` と伸び、そこで初めて ``CQ DE``
        に切れるため。補正は文字列だけの純粋関数なので、同じ文字列に対しては
        常に同じ結果になる (作り直しても表示が揺れない)。
        """
        if self.word_correct_check.isChecked():
            result = correct_text(text)
            self._committed_text = result.text
            self._committed_spans = result.spans
        else:
            self._committed_text = text
            self._committed_spans = ()
        self._refresh_decode_display()
        self._maybe_auto_refine()

    def _on_provisional_text(self, text: str) -> None:
        """暫定テキストを受信して表示を更新する."""
        self._provisional_text = text
        self._refresh_decode_display()

    def _on_stream_diag(self, diag: dict) -> None:
        """ストリーミング診断情報をステータスバーに表示する (auto モードでは現在モードも)."""
        msg = (
            f"window={diag['window']:.0f}s "
            f"hop={diag['hop']:.0f}s "
            f"lag={diag['lag']:.1f}s "
            f"decode={diag['decode_ms']:.0f}ms"
        )
        if self._current_mode() == "auto":
            label = {"european": "欧文", "japanese": "和文"}.get(
                self._auto_submode, self._auto_submode
            )
            msg = f"自動(現在:{label}) " + msg
        self.statusBar().showMessage(msg)

    def _on_received_wpm(self, wpm: object) -> None:
        """受信信号の速度を受け取って表示する. ``None`` なら消す.

        **測れないときは消すこと。** 前の値を残すと、相手が変わっても古い速度が
        出たままになり、それを見て送信速度を合わせると外す
        (``src/infer/wpm.py`` 参照)。

        値は送信ダイアログの「受信に合わせる」にも渡す。
        """
        self._received_wpm = float(wpm) if wpm is not None else None
        if self._received_wpm is None:
            self.wpm_label.setText("")
        else:
            self.wpm_label.setText(f"受信 {self._received_wpm:.0f} WPM")

    def _on_current_mode(self, mode: str) -> None:
        """auto モード中のサブモード (欧文/和文) を保持する (表示は _on_stream_diag が統合)."""
        self._auto_submode = mode

    def _current_display_html(self) -> str:
        """確定 (黒) + 暫定 (グレー) を HTML 文字列として返す.

        プロサイン (<KN>, <BT> 等) の角括弧が HTML タグとして解釈されないよう
        html.escape() を両テキストに適用する.
        """
        html_parts = [self._committed_html()]
        if self._settings.show_provisional:
            provisional_escaped = html.escape(self._provisional_text)
            html_parts.append(
                f'<span style="color:#999999;">{provisional_escaped}</span>'
            )
        return "".join(html_parts)

    def _committed_html(self) -> str:
        """確定テキストを HTML 化する. 辞書補正した範囲だけ橙色にする.

        **エスケープは範囲ごとに行う。** 先に全体を escape すると ``<`` が
        ``&lt;`` に伸びて文字位置がずれ、色をつける範囲が別の場所になる。
        """
        text, spans = self._committed_text, self._committed_spans

        def piece(fragment: str) -> str:
            # 改行を <br> に変換する。setHtml で描画するため、変換しないと HTML が
            # 改行を潰して「無音による改行」が画面上まったく効かない。
            # escape の後に置換すること (先に置換すると <br> 自体がエスケープされる)。
            return html.escape(fragment).replace("\n", "<br>")

        parts: list[str] = []
        cursor = 0
        for span in spans:
            if span.start > cursor:
                parts.append(
                    f'<span style="color:#000000;">{piece(text[cursor:span.start])}</span>'
                )
            parts.append(
                f'<span style="color:#c05000;" title="元: {html.escape(span.original)}">'
                f'{piece(text[span.start:span.end])}</span>'
            )
            cursor = span.end
        if cursor < len(text):
            parts.append(f'<span style="color:#000000;">{piece(text[cursor:])}</span>')
        return "".join(parts)

    def _refresh_decode_display(self) -> None:
        """text_view を確定/暫定 HTML で更新し、末尾までスクロールする."""
        self.text_view.setHtml(self._current_display_html())
        # setHtml は先頭に戻すので、毎回末尾へ送る。受信中は常に最新行を見たい。
        scroll = self.text_view.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def _on_show_provisional_toggled(self, checked: bool) -> None:
        """未確定テキストの表示切替 (即座に反映する)."""
        self._settings.show_provisional = checked
        self._refresh_decode_display()

    def _on_bpf_changed(self, _value: int) -> None:
        if self._worker is not None:
            self.request_set_bpf_params.emit(
                float(self.bpf_center_spin.value()),
                float(self.bpf_bw_spin.value()),
            )

    def _on_worker_error(self, message: str) -> None:
        # _on_stop() は最後に無条件で「停止しました」を表示するため、先に
        # 停止処理を済ませてからエラー本文を表示する。逆順だと停止処理の
        # メッセージでエラー本文が上書きされ、接続失敗の原因が画面から
        # 消えてしまう (--net-source 経路で最も起きやすい失敗が見えなくなる)。
        self._on_stop()
        self.statusBar().showMessage(message)

    # ---- LLM 清書 ----
    def _init_llm_worker(self) -> None:
        """LLM ワーカーを別スレッドで起動し、プロバイダを設定する."""
        self._llm_worker = LLMWorker(timeout_s=self._settings.llm_timeout_s)
        self._llm_worker.set_compact(self._settings.llm_compact_prompt)
        self._llm_thread = QThread()
        self._llm_worker.moveToThread(self._llm_thread)
        self._llm_worker.result_ready.connect(self._on_llm_result)
        self._llm_worker.error.connect(self._on_llm_error)
        self._llm_worker.busy_changed.connect(self._on_llm_busy)
        self.request_llm_transform.connect(self._llm_worker.request_transform)
        self.request_set_llm_provider.connect(self._llm_worker.set_provider)
        self._llm_thread.start()
        self._refresh_llm_provider()

    def _refresh_llm_provider(self) -> None:
        """現在の設定からプロバイダを生成しワーカーへ渡す. 失敗はステータス表示."""
        self._settings.llm_provider = self.llm_provider_combo.currentText()
        self._settings.llm_model = self.llm_model_edit.currentText()
        try:
            provider = create_provider(self._settings)
            self.request_set_llm_provider.emit(provider)
        except LLMError as exc:
            self.request_set_llm_provider.emit(None)
            self.statusBar().showMessage(f"LLM 設定: {exc}")

    def _on_llm_provider_changed(self, provider: str) -> None:
        # プロバイダに合わせてモデル候補を入れ替え、既定モデルを選ぶ
        # (M8: claude のまま llama3.1 を送って 404 になる罠を防ぐ)。
        models = self._models_for(provider)
        if models:
            self.llm_model_edit.clear()
            self.llm_model_edit.addItems(models)
            self.llm_model_edit.setCurrentText(models[0])
        self._refresh_llm_provider()

    def _on_llm_model_changed(self, *_args) -> None:
        self._refresh_llm_provider()

    def _on_refine_clicked(self) -> None:
        """**まとめて清書**: 未清書分を 1 回で全部送る.

        最初からやり直したいときは「クリア」を押してから押す
        (クリアで ``refined_len`` が 0 に戻る)。
        """
        if self._llm_busy:
            return
        request = plan_refine_all(self._committed_text, self._auto_refine_state)
        if request is None:
            self.statusBar().showMessage("清書していない確定テキストがありません")
            return
        self._send_refine(request)

    def _send_refine(self, request: RefineRequest) -> None:
        """清書要求をワーカーへ送る (2 つのモードの共通経路)."""
        self._llm_sent_len = request.refined_len
        self._auto_refine_state.last_time = time.monotonic()
        self.request_llm_transform.emit(
            request.text, self._current_mode(), request.lead
        )

    def _on_llm_result(self, text: str) -> None:
        """清書結果を表示する.

        **2 つのモードとも増分なので、結果は常に積み上げる。**
        最初からやり直したいときは「クリア」を押す。
        """
        cleaned = text.strip()
        if cleaned:
            self._llm_refined_html.append(cleaned)
        self._refresh_llm_display()
        # 送った分までを清書済みにする (楽観更新した値をここで確定させる)
        self._auto_refine_state.refined_len = self._llm_sent_len
        self._auto_refine_state.last_time = time.monotonic()

        # 改行で区切った場合、まだ未清書の行が残っていることがある
        # (1 回の更新で複数行ぶん確定するとき)。続きがあれば次を送る
        self._maybe_auto_refine()

    def _refresh_llm_display(self) -> None:
        """積み上げた清書結果を表示する (推測の赤表示は切替可)."""
        highlight = self.llm_highlight_check.isChecked()
        self.llm_text_view.setHtml(
            "<br>".join(markup.to_html(part, highlight) for part in self._llm_refined_html)
        )
        scroll = self.llm_text_view.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def _on_compact_toggled(self, checked: bool) -> None:
        """短いプロンプトの切替をワーカーへ伝える."""
        self._settings.llm_compact_prompt = checked
        if getattr(self, "_llm_worker", None) is not None:
            self._llm_worker.set_compact(checked)

    def _on_llm_clear(self) -> None:
        """清書パネルの表示を消去する (生デコードは消さない)."""
        self.llm_text_view.clear()
        self._llm_refined_html = []
        # 積み上げを消したので、次は最初から清書し直す
        self._auto_refine_state = AutoRefineState()
        self.statusBar().showMessage("清書をクリアしました")

    def _on_llm_error(self, message: str) -> None:
        self.statusBar().showMessage(f"LLM: {message}")

    def _on_llm_busy(self, busy: bool) -> None:
        self._llm_busy = busy
        self.llm_refine_btn.setEnabled(not busy)
        self.llm_refine_btn.setText("清書中…" if busy else "まとめて清書")

    def _maybe_auto_refine(self) -> None:
        """確定テキスト更新時に呼ばれ、**改行が入った時点でその行までを清書する**.

        改行は無音 3 秒以上で入るので (``src/infer/line_break.py``)、実質
        「送信の切れ目」である。文として完結したところで区切れるので、
        途中で切るより自然な日本語になる。改行が来ない長い送信のために
        ``llm_auto_interval_s`` 経過でも送る保険を持つ。
        """
        if not self.llm_auto_check.isChecked() or self._llm_busy:
            return
        request = plan_auto_refine(
            self._committed_text,
            self._auto_refine_state,
            now=time.monotonic(),
            interval_s=self._settings.llm_auto_interval_s,
        )
        if request is None:
            return
        # busy_changed(True) が往復する前に確定テキストが再更新されても
        # 二重発火しないよう、``last_time`` は _send_refine で先行更新する.
        self._send_refine(request)

    # ---- ジオメトリ復元・保存 ----
    def _restore_geometry(self) -> None:
        g = self._settings.window_geometry
        if all(k in g for k in ("x", "y", "w", "h")):
            self.setGeometry(g["x"], g["y"], g["w"], g["h"])

    def _save_settings(self) -> None:
        g = self.geometry()
        self._settings.window_geometry = {
            "x": g.x(), "y": g.y(), "w": g.width(), "h": g.height(),
        }
        self._settings.mode = self._current_mode()
        self._settings.confidence_threshold = self.threshold_slider.value() / 100.0
        self._settings.input_device = self.device_combo.currentData()
        self._settings.show_spectrogram = self.show_spectrogram_check.isChecked()
        self._settings.show_provisional = self.show_provisional_check.isChecked()
        self._settings.word_correct_enabled = self.word_correct_check.isChecked()
        self._settings.squelch_threshold_db = self.level_meter.threshold_db()
        self._settings.bpf_enabled = self.bpf_check.isChecked()
        self._settings.bpf_center_hz = float(self.bpf_center_spin.value())
        self._settings.bpf_bandwidth_hz = float(self.bpf_bw_spin.value())
        self._settings.llm_provider = self.llm_provider_combo.currentText()
        self._settings.llm_model = self.llm_model_edit.currentText()
        self._settings.llm_auto = self.llm_auto_check.isChecked()
        self._settings.llm_highlight_guesses = self.llm_highlight_check.isChecked()
        self._settings.llm_compact_prompt = self.llm_compact_check.isChecked()
        try:
            save_settings(self._settings, self._config_path)
        except OSError:
            pass

    def _open_tx_dialog(self) -> None:
        """送信ダイアログを開く.

        **打鍵はこの PC ではしない。** 無線機を繋いだ PC の CLI が行う
        (この PC には COM ポートが無い)。

        ``TxDialog`` の import を関数の中に置くのは、送信を使わない人に
        ``pykakasi`` の読み込みを負わせないため (``reading.py`` の遅延読み込みと
        同じ理由)。

        受信テキストは**清書済みがあればそちらを渡す**。生のデコードより
        誤りが減っており、拾える欄の精度が上がる (設計書 §5.1)。

        **モードは画面の今の値を渡す** (``_current_mode()``)。
        ``self._settings.mode`` は ``_save_settings`` でしか書き戻さない
        ので、画面を和文にしても設定は欧文のままであり、和文の型が一覧から
        消えていた (2026-08-11 レビュー I1)。``auto`` はそのまま渡す —
        ``templates_for_mode`` が両方の型を出す。``_auto_submode`` (自動
        切替の今の側) は渡さない。交信の途中で勝手に裏返るので、それで
        一覧を絞ると**運用者が選ぼうとした型が直前に消える**ことになる。
        """
        from src.app.tx_dialog import TxDialog

        # **二枚目を作らない。** 画面が固まらなくなった以上 [送信…] は何度でも
        # 押せる。二枚目を開くと、打鍵側は 1 つしか繋がないので後から開いた
        # ほうが**自分自身の接続**に busy で撥ねられる。
        if self._tx_dialog is not None:
            self._tx_dialog.raise_()
            self._tx_dialog.activateWindow()
            return

        received = "\n".join(self._llm_refined_html) or self._committed_text
        dialog = TxDialog(
            self._settings,
            parent=self,
            received_text=received,
            mode=self._current_mode(),
            received_wpm=self._received_wpm,
        )
        self._tx_dialog = dialog
        # **開いている間もメイン画面を操作できる** (運用者の要望、2026-08-12)。
        # ``exec()`` はメイン画面を固めるので ``show()`` にする。受信を見ながら
        # 返信を書けること、送信中でもデコードを止められることが目的である。
        #
        # **後片付けは ``finished`` に移す。** ``exec()`` の後ろに書いていた
        # ものを ``show()`` の後ろに残すと、**開いた瞬間に畳んでしまう**。
        dialog.finished.connect(lambda _result: self.close_tx_dialog())
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def close_tx_dialog(self) -> None:
        """送信ダイアログを畳んで捨てる. **何度呼んでも安全。**

        畳まないと、画面が消えた後も接続と 3 秒の繋ぎ直しタイマが生き続け、
        次に開いたときに**自分自身の古い接続**に busy で撥ねられる
        (「別の運用者が使用中です」という嘘の理由が出て、アプリを再起動する
        まで送信できなくなる)。
        """
        dialog = self._tx_dialog
        self._tx_dialog = None
        if dialog is None:
            return
        dialog.shutdown()
        dialog.deleteLater()
        self._save_settings()          # ダイアログが tx_endpoint / tx_wpm を書き換えている

    def closeEvent(self, event) -> None:  # noqa: N802
        # **送信ダイアログを置き去りにしない。** モードレスになったので、
        # メイン画面だけ閉じると打鍵の接続と 3 秒タイマが残る
        if self._tx_dialog is not None:
            self._tx_dialog.close()
            self.close_tx_dialog()
        self._on_stop()
        if getattr(self, "_llm_thread", None) is not None:
            self._llm_thread.quit()
            # 進行中の LLM リクエストは最大 llm_timeout_s 秒で必ず終了する。
            # それを超える猶予を持って待機し、実行中スレッドの破棄 (segfault) を防ぐ。
            self._llm_thread.wait(int(self._settings.llm_timeout_s * 1000) + 2000)
        self._save_settings()
        super().closeEvent(event)


def resolve_device(preference: str) -> torch.device:
    """設定値からデコード用デバイスを決める.

    ``"auto"`` のときだけ CUDA を使う。**既定は cpu** で、GPU はローカル LLM に
    空けておく (デコーダは小さく CPU で hop に間に合う)。
    CUDA が無いのに ``"cuda"`` を指定された場合は cpu に落とす。
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preference)


def main(
    checkpoint_path: str | None = None,
    net_source: str | None = None,
    device: str | None = None,
) -> int:
    """エントリポイント."""
    from PySide6.QtWidgets import QApplication
    import sys

    settings = load_settings()
    if checkpoint_path:
        settings.checkpoint_path = checkpoint_path

    if device:
        settings.decode_device = device
    torch_device = resolve_device(settings.decode_device)
    if torch_device.type == "cpu" and settings.decode_threads > 0:
        torch.set_num_threads(settings.decode_threads)
    print(
        f"[decode] device={torch_device} threads={torch.get_num_threads()}"
        if torch_device.type == "cpu" else f"[decode] device={torch_device}"
    )

    if settings.checkpoint_path and Path(settings.checkpoint_path).exists():
        engine = InferenceEngine.from_checkpoint(
            settings.checkpoint_path, device=torch_device
        )
    else:
        engine = InferenceEngine.untrained(device=torch_device)

    app = QApplication.instance() or QApplication(sys.argv)
    window = CWDecoderWindow(engine, settings, net_source=net_source)
    window.show()
    return app.exec()


__all__ = ["CWDecoderWindow", "main", "resolve_device"]
