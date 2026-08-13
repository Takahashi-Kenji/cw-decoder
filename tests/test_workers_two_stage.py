"""ワーカーと 2 段階確定の繋ぎ目.

**単体で通っても層をまたぐと壊れる** (このリポジトリで繰り返し踏んでいる)。
デコーダ側の単体は ``tests/test_sliding_window.py``。ここでは
``AudioInferenceWorker`` が実際に呼ぶこと・表示し直すことを見る。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.app.workers import AudioInferenceWorker
from src.infer.engine import FrameToken
from src.infer.sliding_window import DEFAULT_LEAD_IN_S, CommittedToken
from src.tokens.morse_tokens import EUROPEAN_CHAR_TO_CODE, TOKEN_TO_ID


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class _Engine:
    """refine の呼び出しを記録する偽エンジン."""

    frame_hop_samples = 40

    def __init__(self, result=None):
        self.result = result or []
        self.calls: list[int] = []

    def decode_chunk(self, wave):
        self.calls.append(int(wave.size))
        return list(self.result)


def _plant(worker: AudioInferenceWorker, token_id: int) -> None:
    """閉じたターンを 1 つ植える (8000..16000。10 秒押し込めば閉じている)."""
    worker._sliding.push(np.zeros(80000, dtype=np.float32))
    worker._sliding._committed = [
        CommittedToken(token_id=token_id, confidence=0.9,
                       absolute_sample_start=8000, absolute_sample_end=16000)
    ]


def test_hop_ごとに_refine_が呼ばれる(qapp) -> None:
    """無音の hop でも、閉じたターンがあれば直しにいく."""
    eng = _Engine()                          # 空を返す = 置き換えは起きない
    worker = AudioInferenceWorker(eng, sample_rate=8000)
    worker._decoding = True
    _plant(worker, TOKEN_TO_ID[EUROPEAN_CHAR_TO_CODE["T"]])

    worker._feed_live_block(np.zeros(4000, dtype=np.float32))   # ちょうど 1 hop

    assert len(eng.calls) >= 1               # refine が音声を取りにきた


def test_停止時に最後のターンを直して表示し直す(qapp) -> None:
    e_id = TOKEN_TO_ID[EUROPEAN_CHAR_TO_CODE["E"]]
    # **助走の無音より後ろに置く。** 助走の中に出たトークンは幻覚として
    # 捨てられる (`DEFAULT_LEAD_IN_S`)。0 に置くと捨てられて空になる
    lead = int(DEFAULT_LEAD_IN_S * 8000)
    at = (lead + 8000) // _Engine.frame_hop_samples
    eng = _Engine([FrameToken(token_id=e_id, confidence=0.9,
                              frame_start=at, frame_end=at + 1)])
    worker = AudioInferenceWorker(eng, sample_rate=8000)
    _plant(worker, TOKEN_TO_ID[EUROPEAN_CHAR_TO_CODE["T"]])
    got: list[str] = []
    worker.committed_text_changed.connect(got.append)

    worker.stop()

    assert got, "書き直したのに表示が更新されていない"
    assert got[-1] == "E"                    # T が全文脈の E に置き換わった


def test_何も無い停止で画面を白紙にしない(qapp) -> None:
    """**無条件に emit すると、空文字が流れて画面が消える** (実装中に踏んだ)."""
    worker = AudioInferenceWorker(_Engine(), sample_rate=8000)
    got: list[str] = []
    worker.committed_text_changed.connect(got.append)

    worker.stop()

    assert got == []
