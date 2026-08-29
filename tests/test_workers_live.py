"""ライブ連続モードのワーカー結線テスト (オフスクリーン Qt)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from src.infer.engine import InferenceEngine
from src.app.workers import AudioInferenceWorker

_app = QApplication.instance() or QApplication([])


def _worker() -> AudioInferenceWorker:
    eng = InferenceEngine.untrained("cpu")
    return AudioInferenceWorker(
        eng, sample_rate=8000,
        window_s=5.0, hop_s=1.0, commit_lag_s=1.0, head_guard_s=0.5,
        squelch_threshold_db=-60.0,
    )


def test_worker_has_sliding_decoder() -> None:
    w = _worker()
    assert w._sliding is not None


def test_redecode_triggers_committed_signal() -> None:
    w = _worker()
    received: list[str] = []
    w.committed_text_changed.connect(received.append)
    w.set_decoding(True)
    # 6 秒分の無音でない疑似信号を push して redecode を 1 回以上発火させる
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(8000 * 6) * 0.2).astype(np.float32)
    # 50ms ブロックに分けて push + redecode ロジックを直接駆動
    for i in range(0, sig.size, 400):
        w._feed_live_block(sig[i:i + 400])
    # 非無音入力で committed_text_changed が 1 回以上 emit されること
    assert len(received) >= 1


def test_set_mode_resets_sliding_decoder() -> None:
    """Bug B2: set_mode 後に _sliding の状態がリセットされること."""
    w = _worker()
    w.set_decoding(True)
    rng = np.random.default_rng(1)
    sig = (rng.standard_normal(8000 * 3) * 0.2).astype(np.float32)
    for i in range(0, sig.size, 400):
        w._feed_live_block(sig[i:i + 400])
    # 何らかの状態が溜まっていること (committed か ring か total_consumed)
    assert w._sliding._total_consumed > 0
    w.set_mode("japanese")
    assert w._sliding._total_consumed == 0
    assert w._sliding._committed == []
    assert w._sliding._last_commit_end is None
    assert w._has_pending_provisional is False


def test_set_mode_accepts_auto() -> None:
    """set_mode が 'auto' を受け付け、滑らかにリセットされること."""
    w = _worker()
    rng = np.random.default_rng(3)
    sig = (rng.standard_normal(8000 * 2) * 0.2).astype(np.float32)
    for i in range(0, sig.size, 400):
        w._feed_live_block(sig[i:i + 400])
    w.set_mode("auto")
    assert w.mode == "auto"
    assert w._sliding._total_consumed == 0
    assert w._has_pending_provisional is False


def test_auto_mode_provisional_inherits_committed_mode() -> None:
    """確定列がホレで和文に入ったら、暫定列も和文で変換される."""
    from src.tokens.converter import TokenConverter
    from src.tokens.morse_tokens import TOKEN_TO_ID

    conv = TokenConverter(mode="auto")
    hore = TOKEN_TO_ID["-・・---"]
    a_i = TOKEN_TO_ID["・-"]

    res_c = conv.convert([hore], initial_mode="european")
    res_p = conv.convert([a_i], initial_mode=res_c.final_mode)
    assert res_c.text == "[ホレ]"
    assert res_c.final_mode == "japanese"
    assert res_p.text == "イ"   # european 開始なら "A" になってしまう


def test_set_mode_clears_displayed_text() -> None:
    """set_mode がモード変更後に committed/provisional 両シグナルで "" を emit すること."""
    w = _worker()
    committed: list[str] = []
    provisional: list[str] = []
    w.committed_text_changed.connect(committed.append)
    w.provisional_text_changed.connect(provisional.append)
    w.set_mode("japanese")
    assert committed and committed[-1] == ""
    assert provisional and provisional[-1] == ""


def test_set_decoding_on_clears_text() -> None:
    """set_decoding(True) が committed/provisional 両シグナルで "" を emit し、
    _decoding フラグが True になること."""
    w = _worker()
    committed: list[str] = []
    provisional: list[str] = []
    w.committed_text_changed.connect(committed.append)
    w.provisional_text_changed.connect(provisional.append)
    w.set_decoding(True)
    assert w._decoding is True
    assert committed and committed[-1] == ""
    assert provisional and provisional[-1] == ""


def test_set_decoding_off_finalizes_pending_provisional() -> None:
    """set_decoding(False) が _has_pending_provisional == True のとき
    finalize() を呼んでフラグをクリアし、_decoding を False にすること."""
    w = _worker()
    # 暫定フラグを手動で立てる (finalize 対象が空の状態でも例外なく動作する)
    w._has_pending_provisional = True
    w.set_decoding(False)
    assert w._decoding is False
    assert w._has_pending_provisional is False


def test_current_mode_changed_signal_emitted() -> None:
    """_emit_live_view が current_mode_changed を emit すること."""
    w = _worker()
    modes: list[str] = []
    w.current_mode_changed.connect(modes.append)
    rng = np.random.default_rng(4)
    sig = (rng.standard_normal(8000 * 6) * 0.2).astype(np.float32)
    for i in range(0, sig.size, 400):
        w._feed_live_block(sig[i:i + 400])
    # current_mode_changed が 1 回以上 emit されること
    assert len(modes) >= 1
    # 値は "european" か "japanese" のいずれか
    assert all(m in ("european", "japanese") for m in modes)


def test_operating_point_signal_emits_none_for_noise() -> None:
    """雑音だけなら動作点は測れず None が流れる (古い推奨が残らないように)."""
    w = _worker()
    got: list[object] = []
    w.operating_point_changed.connect(got.append)
    w.set_decoding(True)
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(8000 * 7) * 0.2).astype(np.float32)
    # 音声スレッドのブロック処理と同じ順で駆動する (窓へ投入 → 速度/動作点の測定)。
    # `_maybe_measure_wpm` は `_feed_live_block` の中からは呼ばれない。
    for i in range(0, sig.size, 400):
        w._feed_live_block(sig[i:i + 400])
        w._maybe_measure_wpm(400)
    assert len(got) >= 1
    assert all(g is None for g in got)


def test_operating_point_signal_emits_recommendation_for_cw() -> None:
    """CW が入れば (OperatingPoint, Recommendation) が流れる. **自動切替はしない (表示のみ)**."""
    from src.infer.model_recommend import Recommendation
    from src.infer.operating_point import OperatingPoint
    from src.synth.keying import KeyingParams
    from src.synth.synthesizer import SynthConfig, synthesize_from_text

    w = _worker()
    got: list[object] = []
    w.operating_point_changed.connect(got.append)
    w.set_decoding(True)
    cfg = SynthConfig(mode="european", snr_db=20.0, snr_is_effective=True,
                      keying=KeyingParams(wpm=20.0, tone_freq_hz=600.0))
    sig = synthesize_from_text("CQ CQ CQ DE JA1ABC JA1ABC K K", cfg,
                               np.random.default_rng(0)).samples
    for i in range(0, sig.size - 400, 400):
        w._feed_live_block(sig[i:i + 400])
        w._maybe_measure_wpm(400)
    payloads = [g for g in got if g is not None]
    assert payloads, "CW を流したのに動作点が一度も測れなかった"
    op, rec = payloads[-1]
    assert isinstance(op, OperatingPoint) and isinstance(rec, Recommendation)
    assert op.wpm == pytest.approx(20.0, rel=0.25)


def test_auto_mode_word_break_bias_follows_submode() -> None:
    """自動モードの語間バイアスは、確定列の末尾が和文の中かで切り替わる.

    2026-08-25 の「自動モードは和文の値 (−5) で固定」は欧文の語間を消した
    (運用者の報告 2026-08-28)。p2b ではラタの認識率が 17% → 83% なので追従させる。
    """
    from src.infer.sliding_window import CommittedToken, DecodeView
    from src.tokens.morse_tokens import HORE_CODE, RATA_CODE, TOKEN_TO_ID

    eng = InferenceEngine.untrained("cpu")
    w = AudioInferenceWorker(
        eng, sample_rate=8000, mode="auto",
        word_break_bias_european=-1.0, word_break_bias_japanese=-5.0,
        window_s=5.0, hop_s=1.0, commit_lag_s=1.0, head_guard_s=0.5,
        squelch_threshold_db=-60.0,
    )
    assert eng.word_break_bias == -1.0            # 開始は欧文 (CQ は欧文で始まる)

    def tok(code: str, i: int) -> CommittedToken:
        return CommittedToken(TOKEN_TO_ID[code], 1.0, i * 800, i * 800 + 400)

    a = TOKEN_TO_ID["・-"]                          # A / イ
    w._emit_live_view(DecodeView(committed=[tok(HORE_CODE, 0), tok("・-", 1)]), 0.0)
    assert eng.word_break_bias == -5.0            # ホレの後は和文の値

    w._emit_live_view(DecodeView(committed=[tok(HORE_CODE, 0), tok("・-", 1),
                                            tok(RATA_CODE, 2), tok("・-", 3)]), 0.0)
    assert eng.word_break_bias == -1.0            # ラタの後は欧文の値に戻る


def test_fixed_mode_word_break_bias_is_unchanged() -> None:
    """固定モードは従来どおり (欧文 −1 / 和文 −5)."""
    eng = InferenceEngine.untrained("cpu")
    kw = dict(sample_rate=8000, word_break_bias_european=-1.0, word_break_bias_japanese=-5.0,
              window_s=5.0, hop_s=1.0, commit_lag_s=1.0, head_guard_s=0.5,
              squelch_threshold_db=-60.0)
    AudioInferenceWorker(eng, mode="european", **kw)
    assert eng.word_break_bias == -1.0
    AudioInferenceWorker(eng, mode="japanese", **kw)
    assert eng.word_break_bias == -5.0


def test_european_stream_signal_follows_committed_tokens() -> None:
    """確定列を常に欧文表で読んだ 1 行が european_stream_changed で流れること.

    和文モードでも欧文表で読む (欧文ストリームライン、2026-08-29 要望)。
    """
    from src.infer.sliding_window import CommittedToken, DecodeView
    from src.tokens.morse_tokens import TOKEN_TO_ID

    w = _worker()
    w.set_mode("japanese")
    got: list[str] = []
    w.european_stream_changed.connect(got.append)
    committed = [
        CommittedToken(TOKEN_TO_ID["-・-・"], 0.9, 0, 100),      # C
        CommittedToken(TOKEN_TO_ID["・-・-"], 0.9, 200, 300),    # ロ (欧文表に無い)
    ]
    w._emit_live_view(DecodeView(committed=committed), decode_ms=0.0)
    assert got and got[-1] == "C_"


def test_european_stream_clears_with_committed() -> None:
    """クリア (set_decoding) のとき欧文ストリームも空で流れること."""
    w = _worker()
    got: list[str] = []
    w.european_stream_changed.connect(got.append)
    w.set_decoding(True)
    assert got and got[-1] == ""
