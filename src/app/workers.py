"""音声キャプチャ → ストリーミング推論を別スレッドで実行する Qt ワーカー."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from src.infer.audio import AudioCapture
from src.infer.engine import InferenceEngine
from src.infer.net_audio import NetworkAudioCapture, parse_endpoint
from src.infer.sliding_window import (
    DEFAULT_LOW_CONFIDENCE_EXTRA_LAG_S,
    DecodeView,
    SlidingWindowDecoder,
)
from src.infer.line_break import DEFAULT_LINE_BREAK_GAP_S, render_committed
from src.infer.refine_buffer import DEFAULT_REFINE_CAPACITY_S, RefineBuffer
from src.infer.squelch import Squelch
from src.infer.wpm import estimate_wpm
from src.tokens.converter import TokenConverter


class _StreamingBPF:
    """ステートフル多相バターワース帯域通過フィルタ (scipy SOS)."""

    def __init__(
        self,
        sample_rate: int = 8000,
        center_hz: float = 600.0,
        bandwidth_hz: float = 400.0,
        order: int = 4,
    ) -> None:
        from scipy.signal import butter, sosfilt_zi
        self.sample_rate = sample_rate
        nyq = sample_rate / 2.0
        low = max(20.0, center_hz - bandwidth_hz / 2.0) / nyq
        high = min(0.99, (center_hz + bandwidth_hz / 2.0) / nyq)
        self._sos = butter(order, [low, high], btype="bandpass", output="sos")
        self._zi = sosfilt_zi(self._sos) * 0.0   # 初期状態 ゼロ

    def process(self, block: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt
        out, self._zi = sosfilt(self._sos, block, zi=self._zi)
        return out.astype(block.dtype, copy=False)

    def reset(self) -> None:
        from scipy.signal import sosfilt_zi
        self._zi = sosfilt_zi(self._sos) * 0.0


# 受信 WPM を測り直す間隔 (秒)。**hop (0.5 秒) ごとには測らない。**
# 値が細かく揺れて読みづらく、目的 (相手の速さの目安) にも合わない。
WPM_INTERVAL_S = 3.0

# 1 回の測定に使う音の長さ (秒)。短いと標本が足りず測れず、長いと
# 速度が変わったときの追従が遅れる。窓長 (既定 30 秒) を超えては遡れない。
WPM_WINDOW_S = 12.0


class AudioInferenceWorker(QObject):
    """audio capture → SlidingWindowDecoder → TokenConverter を 1 ワーカーで実行.

    UI スレッドとは Qt シグナルで通信:

    - ``committed_text_changed(str)``: 確定テキスト全体 (黒表示)
    - ``provisional_text_changed(str)``: 暫定テキスト (グレー表示)
    - ``current_mode_changed(str)``: auto モードの現在サブモード ("european"/"japanese")
    - ``audio_block_received(np.ndarray)``: スペクトログラム/レベルメータ用の生波形
    - ``level_changed(float)``: dBFS
    - ``status(str)``: 任意ステータス文字列
    - ``error(str)``: エラー
    """

    audio_block_received = Signal(object)   # np.ndarray
    level_changed = Signal(float)
    status = Signal(str)
    error = Signal(str)
    committed_text_changed = Signal(str)    # 確定テキスト全体 (黒表示)
    provisional_text_changed = Signal(str)  # 暫定テキスト (グレー表示)
    current_mode_changed = Signal(str)      # auto モードの現在サブモード ("european"/"japanese")
    stream_diag = Signal(dict)              # {window, hop, lag, decode_ms}
    # 受信信号の速度 (WPM). 測れないときは None を流す。
    # **測れないことも伝える。** 前の値が残り続けると、相手が変わっても古い
    # 速度が出たままになる (それを見て送信速度を合わせると外す)。
    received_wpm_changed = Signal(object)   # float | None

    def __init__(
        self,
        engine: InferenceEngine,
        sample_rate: int = 8000,
        mode: str = "european",
        confidence_threshold: float = 0.5,
        squelch_threshold_db: float = -60.0,
        squelch_hold_sec: float = 1.0,
        bpf_enabled: bool = True,
        prosign_threshold: float = 0.35,
        switch_on_japanese_only: bool = True,
        bpf_center_hz: float = 600.0,
        bpf_bandwidth_hz: float = 400.0,
        window_s: float = 30.0,
        # AppSettings の既定と揃えること (実効右文脈 commit_lag + hop/2 = 2.25 秒)。
        # 実運用では main_window から settings の値が渡されるが、ここがずれていると
        # 直接生成したときだけ挙動が変わる
        hop_s: float = 0.5,
        commit_lag_s: float = 2.0,
        # この長さ以上の無音で改行する (0 以下なら改行しない)
        line_break_gap_s: float = DEFAULT_LINE_BREAK_GAP_S,
        head_guard_s: float = 1.0,
        decode_left_context_s: float = 5.0,
        commit_jitter_margin_s: float = 0.02,
        # 読めなかった印になる文字に与える余分な猶予
        low_confidence_extra_lag_s: float = DEFAULT_LOW_CONFIDENCE_EXTRA_LAG_S,
        # 清書用に貯めておく長さ (デコード用リングとは別)
        refine_capacity_s: float = DEFAULT_REFINE_CAPACITY_S,
        # 2 段階確定 (ターン終了時の置き換え) を行うか
        two_stage_commit_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.two_stage_commit_enabled = two_stage_commit_enabled
        # **清書専用の長いバッファ。** 別スレッドの再デコードが読む
        self.refine_buffer = RefineBuffer(
            capacity_s=refine_capacity_s, sample_rate=sample_rate
        )
        self.sample_rate = sample_rate
        self.mode: str = mode
        self.confidence_threshold = confidence_threshold
        self.squelch_threshold_db = squelch_threshold_db
        self.squelch_hold_sec = squelch_hold_sec
        self.bpf_enabled = bpf_enabled
        self.bpf_center_hz = bpf_center_hz
        self.bpf_bandwidth_hz = bpf_bandwidth_hz
        self._bpf = _StreamingBPF(
            sample_rate=sample_rate,
            center_hz=bpf_center_hz,
            bandwidth_hz=bpf_bandwidth_hz,
        )
        self.prosign_threshold = prosign_threshold
        self.switch_on_japanese_only = switch_on_japanese_only
        self._converter = TokenConverter(
            mode=mode,
            confidence_threshold=confidence_threshold,
            prosign_threshold=prosign_threshold,
            switch_on_japanese_only=switch_on_japanese_only,
        )
        self._capture: AudioCapture | NetworkAudioCapture | None = None
        self._timer: QTimer | None = None
        self._device_index: int | None = None
        # LAN 経由入力 (--net-source)。None ならローカルデバイスを使う。
        self._net_endpoint: tuple[str, int] | None = None
        self._running = False
        self._record_callback = None
        # スキッシュ状態管理
        self._squelch = Squelch(
            threshold_db=squelch_threshold_db,
            hold_sec=squelch_hold_sec,
            sample_rate=sample_rate,
        )
        # 診断統計
        self._fed_block_count = 0
        self._token_count = 0
        self._diag_tick = 0
        # デコード状態フラグ
        self._decoding = False
        # --- ライブ連続モード ---
        self.window_s = window_s
        self.hop_s = hop_s
        self.commit_lag_s = commit_lag_s
        self._sliding = SlidingWindowDecoder(
            engine,
            window_s=window_s, hop_s=hop_s, commit_lag_s=commit_lag_s,
            head_guard_s=head_guard_s, decode_left_context_s=decode_left_context_s,
            commit_jitter_margin_s=commit_jitter_margin_s, sample_rate=sample_rate,
            # 読めなかった印になる文字は確定を遅らせて読み直す。閾値は表示側
            # (TokenConverter) と同じものを渡すこと。ずれると「画面では ``_`` なのに
            # 待たせていない」「読めているのに待たせている」が起きる
            low_confidence_threshold=confidence_threshold,
            low_confidence_extra_lag_s=low_confidence_extra_lag_s,
        )
        self._samples_since_redecode = 0
        self._hop_samples = int(hop_s * sample_rate)
        self._line_break_gap_samples = int(line_break_gap_s * sample_rate)
        # 暫定トークンが残っているか. 残っている間は無音でも redecode を続け、
        # 送信終了後 commit_lag 経過時に末尾を確定させる (レビュー #4).
        self._has_pending_provisional = False
        # 受信 WPM の測定。**再デコードとは別の間隔で回す。**
        # 10 秒ぶんの測定が 6 ms 程度なので 3 秒ごとでも負荷は無視できる。
        # hop (0.5 秒) ごとに測ると値が細かく揺れて読みづらい、という理由もある。
        self._samples_since_wpm = 0
        self._wpm_interval_samples = int(WPM_INTERVAL_S * sample_rate)

    # ---- 設定変更 ----
    @Slot(str)
    def set_mode(self, mode: str) -> None:
        if mode not in ("european", "japanese", "auto"):
            return
        self.mode = mode
        self._converter = TokenConverter(
            mode=mode,
            confidence_threshold=self.confidence_threshold,
            prosign_threshold=self.prosign_threshold,
            switch_on_japanese_only=self.switch_on_japanese_only,
        )
        # モード切替時に旧モードの確定トークンが新モード変換表で再 emit されないよう
        # ライブ連続モード用の状態も完全リセットする (Bug B2).
        self._sliding.reset()
        self._samples_since_redecode = 0
        self._has_pending_provisional = False
        # モード変更で旧テキスト (旧変換表の結果) を画面から消す.
        self.committed_text_changed.emit("")
        self.provisional_text_changed.emit("")
        self.status.emit(f"mode -> {mode}")

    @Slot()
    def clear(self) -> None:
        """確定/暫定テキストとライブ連続モードの状態をリセットする (本文クリア用).

        モードは変えず、蓄積された確定テキストとスライディングウィンドウだけを
        初期化する. set_mode と同じリセット手順を踏むため、クリア後に再開しても
        旧テキストが再 emit されない.
        """
        self._sliding.reset()
        self._samples_since_redecode = 0
        self._has_pending_provisional = False
        self.committed_text_changed.emit("")
        self.provisional_text_changed.emit("")

    @Slot(float)
    def set_confidence_threshold(self, threshold: float) -> None:
        self.confidence_threshold = threshold
        self._converter = TokenConverter(
            mode=self.mode,
            confidence_threshold=threshold,
            prosign_threshold=self.prosign_threshold,
            switch_on_japanese_only=self.switch_on_japanese_only,
        )
        self.status.emit(f"threshold -> {threshold:.2f}")

    @Slot(object)
    def set_device(self, device_index: object) -> None:
        self._device_index = device_index if isinstance(device_index, int) else None

    @Slot(object)
    def set_net_source(self, endpoint: object) -> None:
        """LAN 経由の転送元を設定する。``None`` でローカルデバイスに戻す.

        Raises:
            ValueError: ``"host:port"`` として解釈できない場合.
        """
        if endpoint is None or (isinstance(endpoint, str) and not endpoint.strip()):
            self._net_endpoint = None
            return
        self._net_endpoint = parse_endpoint(str(endpoint))

    @Slot(float)
    def set_squelch_threshold(self, threshold_db: float) -> None:
        self.squelch_threshold_db = float(threshold_db)
        # Squelch は生成時に閾値を受け取るので、ここで反映しないと UI の変更が効かない
        self._squelch.threshold_db = float(threshold_db)
        self.status.emit(f"squelch -> {threshold_db:.1f} dBFS")

    @Slot(bool)
    def set_bpf_enabled(self, enabled: bool) -> None:
        self.bpf_enabled = bool(enabled)
        if enabled:
            self._bpf.reset()
        self.status.emit(f"BPF {'ON' if enabled else 'OFF'}")

    @Slot(float, float)
    def set_bpf_params(self, center_hz: float, bandwidth_hz: float) -> None:
        self.bpf_center_hz = float(center_hz)
        self.bpf_bandwidth_hz = float(bandwidth_hz)
        self._bpf = _StreamingBPF(
            sample_rate=self.sample_rate,
            center_hz=center_hz,
            bandwidth_hz=bandwidth_hz,
        )
        self.status.emit(f"BPF {center_hz:.0f}Hz ±{bandwidth_hz/2:.0f}Hz")

    @Slot(bool)
    def set_decoding(self, on: bool) -> None:
        # ON: 前セッションの状態と表示を破棄して新規開始 (QSO ごとに押し直す UX).
        # OFF: 末尾の暫定を確定に昇格して残す (stop() と同様; グレー文字の凍結を防ぐ).
        if on:
            self._decoding = True
            self._sliding.reset()
            self._samples_since_redecode = 0
            self._has_pending_provisional = False
            self.committed_text_changed.emit("")
            self.provisional_text_changed.emit("")
            self.status.emit("デコード中")
        else:
            self._decoding = False
            if self._has_pending_provisional:
                final = self._sliding.finalize()
                self._has_pending_provisional = False
                self._emit_live_view(final, 0.0)
            self.status.emit("デコード停止")

    def set_record_callback(self, callback) -> None:
        """``callback(block: np.ndarray)`` 形式のフックを登録."""
        self._record_callback = callback

    # ---- ライフサイクル ----
    @Slot()
    def start(self) -> None:
        try:
            if self._net_endpoint is not None:
                host, port = self._net_endpoint
                self._capture = NetworkAudioCapture(
                    host, port, target_sample_rate=self.sample_rate,
                )
                self._capture.start()
                sr = self._capture.source_sample_rate
                self.status.emit(
                    f"capture started @ {sr} Hz → {self.sample_rate} Hz "
                    f"(LAN {host}:{port})"
                )
            else:
                self._capture = AudioCapture(
                    device_index=self._device_index,
                    target_sample_rate=self.sample_rate,
                    block_duration_s=0.05,
                )
                self._capture.start()
                sr = self._capture.source_sample_rate
                self.status.emit(f"capture started @ {sr} Hz → {self.sample_rate} Hz")
            self._running = True
            self._decoding = False
            self._timer = QTimer()
            self._timer.timeout.connect(self._tick)
            self._timer.start(20)
        except Exception as exc:                       # noqa: BLE001
            # str(exc) を使う (repr だと NetworkCaptureError の複数行ガイダンスの
            # 改行が \n という literal のまま出て、ステータスバーで読めなくなる)。
            # 改行は " / " に畳んで 1 行に収める。
            detail = " / ".join(str(exc).splitlines())
            self.error.emit(f"start failed: {detail}")

    def _feed_live_block(self, block: np.ndarray) -> None:
        """ライブ連続モード: ブロックを窓に投入し、hop ごとに再デコード."""
        self._sliding.push(block)
        # **清書用には別に長く貯める。** デコード用リングを広げると
        # refine_closed_turns が長い区間を同期デコードして音声スレッドを
        # 止めるため (src/infer/refine_buffer.py の説明を参照)
        self.refine_buffer.push(block)
        self._samples_since_redecode += block.size
        if self._samples_since_redecode < self._hop_samples:
            return
        self._samples_since_redecode = 0
        # 2 段階確定: 終わったターンを全文脈でデコードし直す (改行と同じ
        # gap でターンを定義する)。直したら無音中でも表示を作り直す。
        # 走るのはターンが閉じた直後の 1 回だけで、負荷は 30 秒ぶんの
        # デコード 1 回 (実測 112 ms) が上限。
        refined = (
            self._sliding.refine_closed_turns(self._line_break_gap_samples)
            if self.two_stage_commit_enabled
            else False
        )
        # CPU 節約: 直近 hop 区間が無音 かつ 未確定 (暫定) が残っていない場合のみ
        # 再デコードをスキップ. 暫定が残っているうちは送信終了後も redecode を
        # 続け、commit_lag 経過時に末尾を確定させる (レビュー #4: 無音突入で
        # 末尾数文字が暫定のまま永久放置されるのを防ぐ).
        ring = self._sliding._ring
        recent = ring[-self._hop_samples:] if ring.size >= self._hop_samples else ring
        if recent.size:
            rms = float(np.sqrt(np.mean(recent * recent)))
            recent_db = 20.0 * np.log10(rms) if rms > 1e-6 else -120.0
            recent_silent = recent_db < self.squelch_threshold_db
        else:
            recent_silent = True
        if recent_silent and not self._has_pending_provisional and not refined:
            return
        t0 = time.perf_counter()
        view: DecodeView = self._sliding.redecode()
        decode_ms = (time.perf_counter() - t0) * 1000.0
        self._has_pending_provisional = bool(view.provisional)
        self._emit_live_view(view, decode_ms)

    def _maybe_measure_wpm(self, block_size: int) -> None:
        """受信信号の速度を測り直す (``WPM_INTERVAL_S`` ごと).

        **音は溜め直さず窓から取る** (``SlidingWindowDecoder.recent_audio``)。
        測れないときは ``None`` を流す — 前の値を残すと、相手が変わっても
        古い速度が出たままになる。
        """
        self._samples_since_wpm += block_size
        if self._samples_since_wpm < self._wpm_interval_samples:
            return
        self._samples_since_wpm = 0
        wave = self._sliding.recent_audio(WPM_WINDOW_S)
        est = estimate_wpm(wave, self._sliding.sample_rate)
        self.received_wpm_changed.emit(est.wpm if est is not None else None)

    def _emit_live_view(self, view: DecodeView, decode_ms: float) -> None:
        """確定/暫定テキストをモード引き継ぎで変換して emit.

        確定列を変換 → final_mode で暫定列を変換 → current_mode を emit.
        auto モードではホレ/ラタ プロサインで確定列が和文に切り替わった場合、
        暫定列も継続して和文として変換される.
        """
        # 確定列は常に欧文起点で変換 (auto モードのみ initial_mode を尊重; 固定モードは無視).
        # 一定以上の無音があれば改行を入れる (送信のターンの切れ目で行を分ける)。
        # 判定はトークンの時刻差だけで決まるので、作り直しても結果が動かない
        # (= 確定した文字が書き換わらない) — 詳細は src/infer/line_break.py。
        committed_text, final_mode = render_committed(
            view.committed, self._converter, self._line_break_gap_samples,
            initial_mode="european",
        )
        prov_ids = [t.token_id for t in view.provisional]
        prov_confs = [t.confidence for t in view.provisional]
        # 確定列と暫定列の境界が語間に落ちるとスペースが消える
        # ("GL 73 CQ" → "GL 73CQ")。確定列が既にスペースで終わっていない場合だけ
        # 暫定列の先頭スペースを残す (二重スペースを避ける)。
        keep_space = bool(committed_text) and not committed_text.endswith(" ")
        # 暫定列は区切らない。末尾の数秒しかなく区切りが入る余地がほとんど無い。
        res_p = self._converter.convert(
            prov_ids, prov_confs,
            initial_mode=final_mode,
            keep_leading_space=keep_space,
        )
        prov_text = res_p.text
        self.committed_text_changed.emit(committed_text)
        self.provisional_text_changed.emit(prov_text)
        self.current_mode_changed.emit(final_mode)   # ステータス表示用
        self.stream_diag.emit({
            "window": self.window_s, "hop": self.hop_s,
            "lag": self.commit_lag_s, "decode_ms": round(decode_ms, 1),
        })

    @Slot()
    def stop(self) -> None:
        self._running = False
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._capture is not None:
            try:
                self._capture.stop()
            except Exception:                          # noqa: BLE001
                pass
            self._capture = None
        # ライブ連続モード: 残暫定を最終確定 (commit_lag を無視) し、
        # 2 段階確定で最後のターンまで直してから見せる (もう音は増えない)
        had_pending = self._has_pending_provisional
        if had_pending:
            self._sliding.finalize()
            self._has_pending_provisional = False
        refined = (
            self._sliding.refine_closed_turns(
                self._line_break_gap_samples, all_turns=True
            )
            if self.two_stage_commit_enabled
            else False
        )
        # **何も変わっていなければ emit しない。** 無条件に emit すると、
        # 何もデコードしていない状態の停止で空文字が流れ、画面が白紙になる
        if had_pending or refined:
            final = DecodeView(committed=list(self._sliding._committed))
            self._emit_live_view(final, 0.0)
        self.status.emit("capture stopped")

    # ---- 内部 ----
    def _tick(self) -> None:
        if not self._running or self._capture is None:
            return
        blocks = self._capture.drain()
        if not blocks:
            return
        level_db = self._capture.level_db_rms

        for block in blocks:
            # BPF を最前段で適用 (生音声に対して)
            if self.bpf_enabled:
                proc_block = self._bpf.process(block)
            else:
                proc_block = block
            self.audio_block_received.emit(proc_block)

            # スキッシュ: BPF 後のレベルで判定し、閉じている間は無音に置き換える。
            # 判定と根拠は src/infer/squelch.py を参照 (捨てずに置き換えるのは
            # 時間軸を止めないため)。
            #
            # 以前はここで squelch_open を計算するだけで実際には使っておらず、
            # _feed_live_block が再デコードを飛ばすだけだった。飛ばしても音は窓に
            # 残るので、次に信号が来た時点でノイズ由来のトークンが出ていた。
            proc_block = self._squelch.process(proc_block)

            if self._decoding:
                self._feed_live_block(proc_block)
                self._maybe_measure_wpm(proc_block.size)

            if self._record_callback is not None:
                try:
                    self._record_callback(proc_block)
                except Exception:                   # noqa: BLE001
                    pass
        self.level_changed.emit(level_db)
        # 診断: 1秒に1回統計を吐く
        self._diag_tick += 1
        if self._diag_tick >= 50:
            self._diag_tick = 0
            rec_mark = "DEC" if self._decoding else "..."
            self.status.emit(f"lvl={level_db:.1f}dB [{rec_mark}]")


def create_worker_thread(worker: AudioInferenceWorker) -> QThread:
    """ワーカーを QThread に移動して開始 (start_capture は別途呼ぶ)."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.start()
    return thread


__all__ = ["AudioInferenceWorker", "create_worker_thread"]
