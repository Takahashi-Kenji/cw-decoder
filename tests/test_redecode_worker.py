"""清書前の全体再デコードワーカーのテスト.

Qt のスレッドは使わず、``request_redecode`` を直接呼んで振る舞いだけ見る。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.app.redecode_worker import RedecodeWorker
from src.infer.engine import FrameToken
from src.tokens.morse_tokens import TOKEN_TO_ID


class _Recorder:
    def __init__(self, worker: RedecodeWorker) -> None:
        self.results: list[tuple[str, int]] = []
        self.errors: list[str] = []
        self.busy: list[bool] = []
        worker.result_ready.connect(lambda t, n: self.results.append((t, n)))
        worker.error.connect(self.errors.append)
        worker.busy_changed.connect(self.busy.append)


def _worker(tokens: list[FrameToken] | None = None) -> RedecodeWorker:
    w = RedecodeWorker(checkpoint_path="dummy.pt")

    class _FakeEngine:
        def decode_chunk(self, wave):          # noqa: ANN001, ANN202
            return list(tokens or [])

    w._engine = _FakeEngine()                   # type: ignore[assignment]
    return w


def _tok(code: str, start: int, conf: float = 0.99) -> FrameToken:
    return FrameToken(
        token_id=TOKEN_TO_ID[code], confidence=conf,
        frame_start=start, frame_end=start + 3,
    )


def test_empty_audio_returns_empty_without_decoding() -> None:
    w = _worker()
    rec = _Recorder(w)
    w.request_redecode(np.zeros(0, dtype=np.float32), 1234, "japanese", 0.5, True, True)
    assert rec.results == [("", 1234)]
    assert rec.busy == []                       # 走っていないので busy も出さない


def test_decodes_and_returns_end_sample() -> None:
    """**末尾の絶対位置をそのまま返す。** 呼び出し側が「どこまで清書したか」を
    時間で覚えるために使う (再デコードで文字位置は当てにならない)."""
    w = _worker([_tok("・-・--", 0), _tok("・-・-・", 10), _tok("-・-・・", 20)])
    rec = _Recorder(w)
    w.request_redecode(np.zeros(8000, dtype=np.float32), 999, "japanese", 0.5, False, False)
    text, end = rec.results[0]
    assert end == 999
    assert text.replace(" ", "") == "テンキ"


def test_word_correction_is_applied_when_enabled() -> None:
    """辞書補正はこの段で掛ける (清書に渡る前に直しておく)."""
    w = _worker([_tok("・-・--", 0), _tok("・-・-・", 10), _tok("-・-・・", 20),
                 _tok("-・・・", 30)])
    rec = _Recorder(w)
    w.request_redecode(np.zeros(8000, dtype=np.float32), 0, "japanese", 0.5, True, True)
    assert rec.results[0][0].startswith("テンキ ハ")


def test_word_correction_can_be_disabled() -> None:
    w = _worker([_tok("・-・--", 0), _tok("・-・-・", 10), _tok("-・-・・", 20),
                 _tok("-・・・", 30)])
    rec = _Recorder(w)
    w.request_redecode(np.zeros(8000, dtype=np.float32), 0, "japanese", 0.5, False, False)
    assert rec.results[0][0].replace(" ", "") == "テンキハ"


def test_failure_is_reported_without_raising() -> None:
    """**清書は補助機能なので落とさない。** 失敗しても受信は続く."""
    w = RedecodeWorker(checkpoint_path="dummy.pt")

    class _Broken:
        def decode_chunk(self, wave):          # noqa: ANN001, ANN202
            raise RuntimeError("壊れた")

    w._engine = _Broken()                       # type: ignore[assignment]
    rec = _Recorder(w)
    w.request_redecode(np.zeros(80, dtype=np.float32), 0, "japanese", 0.5, False, False)
    assert rec.results == []
    assert len(rec.errors) == 1
    assert rec.busy == [True, False]            # 失敗しても busy は必ず戻す


def test_engine_is_loaded_lazily() -> None:
    """清書を使わない運用で 17 MB を無駄に確保しない."""
    w = RedecodeWorker(checkpoint_path="dummy.pt")
    assert w._engine is None


def test_multi_dimensional_audio_is_flattened() -> None:
    w = _worker([_tok("・-・--", 0)])
    rec = _Recorder(w)
    w.request_redecode(np.zeros((100, 1), dtype=np.float32), 0, "japanese", 0.5, False, False)
    assert rec.results[0][0] == "テ"


@pytest.mark.parametrize("mode", ["european", "japanese", "auto"])
def test_modes_are_accepted(mode: str) -> None:
    w = _worker([_tok("・-", 0)])
    rec = _Recorder(w)
    w.request_redecode(np.zeros(80, dtype=np.float32), 0, mode, 0.5, False, False)
    assert len(rec.results) == 1
