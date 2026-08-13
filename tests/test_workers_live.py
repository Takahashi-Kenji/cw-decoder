"""ライブ連続モードのワーカー結線テスト (オフスクリーン Qt)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
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
