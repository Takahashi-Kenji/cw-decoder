"""設定画面.

**主画面をコンパクトに保つため、ほとんどの設定をここへ集める**
(運用者の要望、2026-08-14)。主画面に残すのは交信中に触るものだけ:

    モード / 開始・停止 / デコード / クリア / 録音 / 送信 /
    まとめて清書 / 清書クリア / 設定… / レベルメータ+スケルチ /
    スペクトルの濃さ・幅 / 受信 WPM

反映のタイミングが 2 種類ある
----------------------------
**すぐ効くもの**と**次回の開始時に効くもの**が混ざっている。後者は
``SlidingWindowDecoder`` を作るときに固まる値なので、受信中に変えても効かない。
**黙って効かないのが一番困る**ので、該当する項目には印を付ける。

危ない値に歯止めを置く
----------------------
``commit_lag`` を 0 にする、リングを 1 秒にするといった操作で受信が壊れないよう、
各項目に妥当な範囲を持たせる。

実効右文脈を見せる
------------------
``commit_lag`` と ``hop`` は**和で効く** (``lag + hop/2``、目標 2.25 秒)。
片方だけ動かすと右文脈が静かに失われる。過去に実際に踏んでいるので、
**計算結果をその場に表示**して気づけるようにする。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.infer.settings import AppSettings

# 次回の開始時にしか効かない項目に付ける印。
_LATER = "  ⟳"
_LATER_NOTE = "⟳ … 次回の「開始」から反映されます"

# 次回の開始時にしか効かない項目と、通知に出す短い名前。
#
# **ここが唯一の一覧。** 主画面 (``_deferred_setting_names``) はこれを読んで
# 「何項目が持ち越されたか」を知らせる。画面の印 (``_LATER``) と一覧がずれると
# 「印が無いのに黙って効かない」項目ができる。2026-08-16 に ``2 段階確定`` が
# まさにそれになっているのが見つかった (印 12 個・通知 13 項目)。
# 両者の一致は ``tests/test_settings_dialog.py`` の ``TestDeferredMarks`` が見る。
DEFERRED_SETTING_LABELS: dict[str, str] = {
    "sample_rate": "サンプルレート",
    "checkpoint_path": "モデル",
    "decode_device": "デコード装置",
    "decode_threads": "スレッド数",
    "hop_s": "デコード間隔",
    "commit_lag_s": "確定までの待ち",
    "window_s": "デコード用リング",
    "decode_left_context_s": "左文脈",
    "head_guard_s": "先頭で捨てる長さ",
    "low_confidence_extra_lag_s": "読めない文字の猶予",
    "line_break_gap_s": "改行する無音",
    "two_stage_commit_enabled": "2 段階確定",
    "refine_capacity_s": "清書用バッファ",
}


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("color: #888; font-size: 11px;")
    label.setWordWrap(True)
    return label


class SettingsDialog(QDialog):
    """``AppSettings`` を編集する画面.

    **元の設定は書き換えない。** ``result_settings`` に新しい値を組んで返し、
    呼び出し側が受け取ってから反映する (取り消しで元に戻せるようにするため)。
    """

    def __init__(
        self,
        settings: AppSettings,
        parent: QWidget | None = None,
        lexicon_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.resize(520, 560)
        self._settings = settings
        self._lexicon_path = lexicon_path
        self.result_settings: AppSettings | None = None

        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        self.tabs.addTab(self._build_input_tab(), "入力")
        self.tabs.addTab(self._build_decode_tab(), "デコード")
        self.tabs.addTab(self._build_commit_tab(), "確定")
        self.tabs.addTab(self._build_correct_tab(), "補正")
        self.tabs.addTab(self._build_llm_tab(), "清書")
        self.tabs.addTab(self._build_display_tab(), "表示")

        root.addWidget(_note(_LATER_NOTE))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ---- 部品 ----
    @staticmethod
    def _spin(
        value: float, low: float, high: float, step: float, suffix: str, decimals: int = 2
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setSuffix(suffix)
        box.setValue(value)
        return box

    @staticmethod
    def _int_spin(value: int, low: int, high: int, suffix: str = "") -> QSpinBox:
        box = QSpinBox()
        box.setRange(low, high)
        box.setSuffix(suffix)
        box.setValue(value)
        return box

    # ---- 入力 ----
    def _build_input_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        s = self._settings

        self.sample_rate = self._int_spin(s.sample_rate, 8000, 48000, " Hz")
        self.sample_rate.setSingleStep(8000)
        form.addRow("サンプルレート" + _LATER, self.sample_rate)

        self.bpf_enabled = QCheckBox("帯域通過フィルタを使う")
        self.bpf_enabled.setChecked(s.bpf_enabled)
        form.addRow(self.bpf_enabled)
        self.bpf_center = self._spin(s.bpf_center_hz, 200, 2000, 10, " Hz", 0)
        form.addRow("中心周波数", self.bpf_center)
        self.bpf_bw = self._spin(s.bpf_bandwidth_hz, 50, 2000, 10, " Hz", 0)
        form.addRow("帯域幅", self.bpf_bw)

        self.squelch_hold = self._spin(s.squelch_hold_sec, 0.0, 10.0, 0.1, " 秒")
        self.squelch_hold.setToolTip(
            "スケルチが閉じるまでの保持時間。閾値そのものは\n"
            "レベルメータ上をドラッグして決めます。"
        )
        form.addRow("スケルチ保持時間", self.squelch_hold)

        self.recording_enabled = QCheckBox("受信を自動で録音する")
        self.recording_enabled.setChecked(s.recording_enabled)
        form.addRow(self.recording_enabled)
        self.recording_dir = QLineEdit(s.recording_dir)
        form.addRow("録音の保存先", self.recording_dir)
        return page

    # ---- デコード ----
    def _build_decode_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        s = self._settings

        self.checkpoint_path = QLineEdit(s.checkpoint_path or "")
        self.checkpoint_path.setPlaceholderText("models/full/best_infer.pt")
        form.addRow("モデル" + _LATER, self.checkpoint_path)

        self.decode_device = QComboBox()
        self.decode_device.addItems(["cpu", "cuda", "auto"])
        self.decode_device.setCurrentText(s.decode_device)
        self.decode_device.setToolTip(
            "実測では **CPU 4 スレッドが CUDA より速い** (112 ms 対 139 ms)。\n"
            "モデルが小さいので GPU はレイヤごとの起動費用が勝ちます。\n"
            "GPU はローカル LLM の清書に空けておく方が全体として速くなります。"
        )
        form.addRow("デコードに使う装置" + _LATER, self.decode_device)

        self.decode_threads = self._int_spin(s.decode_threads, 0, 32)
        self.decode_threads.setToolTip("0 なら PyTorch の既定に任せます。実測では 4 が最速。")
        form.addRow("CPU スレッド数" + _LATER, self.decode_threads)

        self.confidence_threshold = self._spin(
            s.confidence_threshold, 0.0, 1.0, 0.05, ""
        )
        self.confidence_threshold.setToolTip(
            "これ未満の確信度の文字は「読めなかった印」(_) になります。\n"
            "**下げてはいけません。** 0.5 → 0.0 にすると held-out の CER が\n"
            "19.74% → 23.64% と 3.9pt 悪化しました。? の裏に正解が隠れて\n"
            "いるのではなく、ほとんど間違った文字が出てくるだけでした。"
        )
        form.addRow("確信度の閾値", self.confidence_threshold)

        self.prosign_threshold = self._spin(s.prosign_threshold, 0.0, 1.0, 0.05, "")
        self.prosign_threshold.setToolTip(
            "和文開始 (ホレ)・終了 (ラタ) などのプロサインにだけ適用する閾値。\n"
            "通常より低くしてあるのは、モード切替を取り逃がすと以降の和文が\n"
            "すべて欧文表で読まれて全滅するためです。"
        )
        form.addRow("プロサインの閾値", self.prosign_threshold)

        self.switch_on_japanese_only = QCheckBox("和文にしかない符号でもモードを切り替える")
        self.switch_on_japanese_only.setChecked(s.switch_on_japanese_only)
        self.switch_on_japanese_only.setToolTip(
            "ホレ/ラタ が実信号で取れないことがあるため、\n"
            "単一の符号に頼らない冗長な経路として用意しています。"
        )
        form.addRow(self.switch_on_japanese_only)
        return page

    # ---- 確定 ----
    def _build_commit_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        s = self._settings

        self.hop_s = self._spin(s.hop_s, 0.1, 2.0, 0.1, " 秒")
        self.commit_lag_s = self._spin(s.commit_lag_s, 0.5, 5.0, 0.25, " 秒")
        form.addRow("デコード間隔 (hop)" + _LATER, self.hop_s)
        form.addRow("確定までの待ち (lag)" + _LATER, self.commit_lag_s)

        # **和で効くので、計算結果をその場に見せる** (片方だけ動かす事故を防ぐ)
        self.effective_label = QLabel()
        self.effective_label.setStyleSheet("font-weight: bold;")
        form.addRow("実効右文脈", self.effective_label)
        form.addRow(_note(
            "実効右文脈 = lag + hop ÷ 2。**目標は 2.25 秒**です。"
            "held-out の実測で最良で、3.0 秒より速くて精度も上でした。"
            "縮めると 1.5 秒で +1.8pt、1.0 秒で +5.7pt 悪化します。"
            "**hop と lag は和で効くので、片方だけ動かさないでください。**"
        ))
        self.hop_s.valueChanged.connect(self._update_effective)
        self.commit_lag_s.valueChanged.connect(self._update_effective)
        self._update_effective()

        self.window_s = self._spin(s.window_s, 5.0, 120.0, 5.0, " 秒", 1)
        self.window_s.setToolTip(
            "デコード用リングの長さ。2 段階確定はこの範囲の音しか読み直せません。\n"
            "**広げると 2 段階確定の 1 回が長くなり、受信を取りこぼします。**"
        )
        form.addRow("デコード用リング" + _LATER, self.window_s)

        self.decode_left_context_s = self._spin(
            s.decode_left_context_s, 1.0, 30.0, 0.5, " 秒", 1
        )
        form.addRow("左文脈" + _LATER, self.decode_left_context_s)
        self.head_guard_s = self._spin(s.head_guard_s, 0.0, 5.0, 0.5, " 秒", 1)
        self.head_guard_s.setToolTip(
            "デコード区間の先頭で捨てる長さ。窓の切断で符号が分断されるため。\n"
            "交信の冒頭では無効になります (1 文字目を捨てないため)。"
        )
        form.addRow("先頭で捨てる長さ" + _LATER, self.head_guard_s)

        self.low_confidence_extra_lag_s = self._spin(
            s.low_confidence_extra_lag_s, 0.0, 5.0, 0.5, " 秒", 1
        )
        self.low_confidence_extra_lag_s.setToolTip(
            "読めなかった印 (_) になる文字だけ確定を遅らせ、右文脈が増えてから\n"
            "読み直します。held-out で 1.5 秒のとき TER -2.01pt、和文は -3.84pt。\n"
            "0 にすると従来の挙動です。"
        )
        form.addRow("読めない文字の猶予" + _LATER, self.low_confidence_extra_lag_s)

        self.line_break_gap_s = self._spin(s.line_break_gap_s, 0.0, 10.0, 0.5, " 秒", 1)
        self.line_break_gap_s.setToolTip(
            "この長さ以上の無音で改行します。ターンの切れ目の定義でもあり、\n"
            "2 段階確定はこの単位で読み直します。0 にすると改行しません。"
        )
        form.addRow("改行する無音の長さ" + _LATER, self.line_break_gap_s)

        # **印はチェックボックス自身の文字に付ける** (この行にはラベルが無い)。
        # 切替は ``SlidingWindowDecoder`` を作るときに固まるので、受信中は効かない。
        self.two_stage_commit_enabled = QCheckBox(
            "2 段階確定を行う (ターン終了時に読み直して置き換える)" + _LATER
        )
        self.two_stage_commit_enabled.setChecked(s.two_stage_commit_enabled)
        self.two_stage_commit_enabled.setToolTip(
            "held-out で TER 27.4% → 25.1%。ただしターンの音がリングに残って\n"
            "いる場合だけ走るので、長い送信では諦めます。"
        )
        form.addRow(self.two_stage_commit_enabled)
        return page

    def _update_effective(self) -> None:
        effective = self.commit_lag_s.value() + self.hop_s.value() / 2
        diff = effective - 2.25
        mark = "" if abs(diff) < 0.01 else f"  (目標 2.25 秒から {diff:+.2f})"
        self.effective_label.setText(f"{effective:.2f} 秒{mark}")

    # ---- 補正 ----
    def _build_correct_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        s = self._settings

        # **3 択で見せる** (運用者、2026-08-17)。
        #
        # 内部は 2 つの真偽値だが、**取りうる状態は 3 つしかない**。和文の補正は
        # 欧文の補正の上に載るので「和文だけ」は選べない。チェックボックス 2 つだと
        # 選べない組み合わせが画面に残るうえ、「和文にも使う」が何に対する
        # 「にも」なのか読めない、という指摘を受けた。
        group = QGroupBox("辞書補正")
        choices = QVBoxLayout(group)
        self.correct_off = QRadioButton("使わない")
        self.correct_european = QRadioButton("欧文だけに使う")
        self.correct_both = QRadioButton("欧文と和文の両方に使う")
        self.correct_european.setToolTip(
            "CW の定型語彙で語を切り直し・寄せします (LLM 不要・即時)。\n"
            "寄せ先は符号 (・-) の距離で選ぶので、点 1 個の差という\n"
            "CW の誤りの構造が保たれます。"
        )
        self.correct_both.setToolTip(
            "和文の補正は曖昧一致つきの分割を伴い、欧文より踏み込んだ処理です。\n"
            "1 文字が要素の途中で切れて 2 文字になった誤り (ロ+ム=テ) も戻せます。\n"
            "日常で使わないカナ (ヱ→イマ) の置き換えもここに含まれます。"
        )
        for button in (self.correct_off, self.correct_european, self.correct_both):
            choices.addWidget(button)
        # **親の真偽値が切れていれば「使わない」。** 子の値は覚えたままにする
        # (「使わない」を選んだだけで和文の選択を捨てない)
        if not s.word_correct_enabled:
            self.correct_off.setChecked(True)
        elif s.word_correct_ja_enabled:
            self.correct_both.setChecked(True)
        else:
            self.correct_european.setChecked(True)
        choices.addWidget(_note(
            "和文はカナの切り直しを伴うので、欧文より踏み込んだ処理です。"
            "日常で使わないカナの置き換え (ヱ → イマ) も、和文を選んだときだけ働きます。"
        ))
        layout.addWidget(group)

        if self._lexicon_path is not None:
            layout.addSpacing(8)
            layout.addWidget(_note(
                f"和文の語彙を足すには次のファイルを編集してください。\n{self._lexicon_path}\n"
                "組み込みの語彙と同じ形式 {\"カテゴリ\": [\"語\", ...]} です。"
                "**起動時に 1 回だけ読み込みます。**"
            ))
            row = QHBoxLayout()
            self.open_lexicon_btn = QPushButton("語彙ファイルの場所を開く")
            row.addWidget(self.open_lexicon_btn)
            row.addStretch(1)
            layout.addLayout(row)

        layout.addStretch(1)
        return page

    # ---- 清書 ----
    def _build_llm_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        s = self._settings

        self.llm_provider = QComboBox()
        self.llm_provider.addItems(["ollama", "openai", "claude"])
        self.llm_provider.setCurrentText(s.llm_provider)
        form.addRow("プロバイダ", self.llm_provider)
        self.llm_model = QLineEdit(s.llm_model)
        form.addRow("モデル", self.llm_model)
        self.ollama_endpoint = QLineEdit(s.ollama_endpoint)
        form.addRow("Ollama の宛先", self.ollama_endpoint)

        self.llm_auto = QCheckBox("自動で清書する")
        self.llm_auto.setChecked(s.llm_auto)
        form.addRow(self.llm_auto)
        self.llm_auto_interval_s = self._spin(
            s.llm_auto_interval_s, 5.0, 300.0, 5.0, " 秒", 0
        )
        form.addRow("自動清書の間隔", self.llm_auto_interval_s)
        self.llm_timeout_s = self._spin(s.llm_timeout_s, 10.0, 600.0, 10.0, " 秒", 0)
        form.addRow("応答待ちの上限", self.llm_timeout_s)

        self.llm_compact_prompt = QCheckBox("短いプロンプトを使う")
        self.llm_compact_prompt.setChecked(s.llm_compact_prompt)
        self.llm_compact_prompt.setToolTip(
            "**小さいローカルモデルではこちらが良い。** 重いプロンプトだと\n"
            "例文をそのまま返したり、内容を捏造したりする実測があります。"
        )
        form.addRow(self.llm_compact_prompt)
        self.llm_highlight_guesses = QCheckBox("推測した箇所を赤で表示する")
        self.llm_highlight_guesses.setChecked(s.llm_highlight_guesses)
        form.addRow(self.llm_highlight_guesses)

        self.refine_redecode_enabled = QCheckBox("清書の前に全体を読み直す")
        self.refine_redecode_enabled.setChecked(s.refine_redecode_enabled)
        self.refine_redecode_enabled.setToolTip(
            "清書の直前に、清書用バッファを別スレッドで丸ごと読み直してから\n"
            "LLM に渡します。**画面の確定テキストは変わりません。**\n"
            "自動清書では未清書分だけ、まとめて清書では全体を読み直します。"
        )
        form.addRow(self.refine_redecode_enabled)
        self.refine_capacity_s = self._spin(
            s.refine_capacity_s, 30.0, 900.0, 30.0, " 秒", 0
        )
        self.refine_capacity_s.setToolTip(
            "清書用に貯めておく音声の長さ。デコード用リングとは別です。\n"
            "300 秒で約 9.6 MB、読み直しに約 1.3 秒かかります (別スレッド)。"
        )
        form.addRow("清書用バッファ" + _LATER, self.refine_capacity_s)
        return page

    # ---- 表示 ----
    def _build_display_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        s = self._settings

        self.show_spectrogram = QCheckBox("スペクトルを表示する")
        self.show_spectrogram.setChecked(s.show_spectrogram)
        layout.addWidget(self.show_spectrogram)

        self.show_provisional = QCheckBox("未確定の文字を表示する")
        self.show_provisional.setChecked(s.show_provisional)
        self.show_provisional.setToolTip(
            "右文脈が足りない状態の読みなので**ほとんど間違っています**。\n"
            "受信が動いていることの手応えとして出しています。\n"
            "計算量は増えません (確定と同じ結果の末尾を出しているだけ)。"
        )
        layout.addWidget(self.show_provisional)

        layout.addSpacing(8)
        layout.addWidget(_note(
            "スペクトルの濃さと表示幅は、主画面のスペクトル下のスライダで\n"
            "見ながら調整してください。"
        ))
        layout.addStretch(1)
        return page

    # ---- 確定 ----
    def _on_accept(self) -> None:
        self.result_settings = replace(
            self._settings,
            sample_rate=int(self.sample_rate.value()),
            bpf_enabled=self.bpf_enabled.isChecked(),
            bpf_center_hz=self.bpf_center.value(),
            bpf_bandwidth_hz=self.bpf_bw.value(),
            squelch_hold_sec=self.squelch_hold.value(),
            recording_enabled=self.recording_enabled.isChecked(),
            recording_dir=self.recording_dir.text().strip() or "data/real",
            checkpoint_path=self.checkpoint_path.text().strip() or None,
            decode_device=self.decode_device.currentText(),
            decode_threads=int(self.decode_threads.value()),
            confidence_threshold=self.confidence_threshold.value(),
            prosign_threshold=self.prosign_threshold.value(),
            switch_on_japanese_only=self.switch_on_japanese_only.isChecked(),
            hop_s=self.hop_s.value(),
            commit_lag_s=self.commit_lag_s.value(),
            window_s=self.window_s.value(),
            decode_left_context_s=self.decode_left_context_s.value(),
            head_guard_s=self.head_guard_s.value(),
            low_confidence_extra_lag_s=self.low_confidence_extra_lag_s.value(),
            line_break_gap_s=self.line_break_gap_s.value(),
            two_stage_commit_enabled=self.two_stage_commit_enabled.isChecked(),
            word_correct_enabled=not self.correct_off.isChecked(),
            # 「使わない」のときは**元の値を保つ**。3 択に畳んだ都合で子の値が
            # 見えなくなるだけなので、開いて OK を押しただけで捨ててはいけない
            word_correct_ja_enabled=(
                self._settings.word_correct_ja_enabled
                if self.correct_off.isChecked()
                else self.correct_both.isChecked()
            ),
            llm_provider=self.llm_provider.currentText(),
            llm_model=self.llm_model.text().strip(),
            ollama_endpoint=self.ollama_endpoint.text().strip(),
            llm_auto=self.llm_auto.isChecked(),
            llm_auto_interval_s=self.llm_auto_interval_s.value(),
            llm_timeout_s=self.llm_timeout_s.value(),
            llm_compact_prompt=self.llm_compact_prompt.isChecked(),
            llm_highlight_guesses=self.llm_highlight_guesses.isChecked(),
            refine_redecode_enabled=self.refine_redecode_enabled.isChecked(),
            refine_capacity_s=self.refine_capacity_s.value(),
            show_spectrogram=self.show_spectrogram.isChecked(),
            show_provisional=self.show_provisional.isChecked(),
        )
        self.accept()


__all__ = ["SettingsDialog"]
