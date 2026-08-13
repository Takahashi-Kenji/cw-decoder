"""受信信号の速度 (WPM) 推定.

**合成音だけで判断しないこと。** この機能では一度、合成音で誤差 +0.5% まで
改善した補正が実録音では +32% と悪化した (``src/infer/wpm.py`` の docstring)。
ここのテストは**仕組みが壊れていないこと**を見るためのもので、精度の根拠は
運用者の実録音 90 件での実測 (誤差 中央 +1.6%、±10% に 78/90) である。
実録音は git 管理外なので、ここでは合成音を使う。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.infer.wpm import (
    LIVE_CONTRAST_MIN,
    MAX_WPM,
    MIN_WPM,
    detect_tone,
    envelope_on_off,
    estimate_wpm,
    split_dot_dash,
)
from src.synth.keying import KeyingParams
from src.synth.synthesizer import SynthConfig, synthesize_from_text

TEXT = "CQ CQ CQ DE JH0ILL JH0ILL K NOW THE QUICK BROWN FOX 599 599"


def _wave(wpm: float, *, snr_db: float = 10.0, jitter: float = 0.0, tone: float = 600.0):
    rng = np.random.default_rng(7)
    cfg = SynthConfig(
        mode="european",
        snr_db=snr_db,
        keying=KeyingParams(
            wpm=wpm, element_jitter_sigma_ratio=jitter, tone_freq_hz=tone
        ),
    )
    res = synthesize_from_text(TEXT, cfg, rng)
    return res.samples, res.sample_rate


class TestEstimateWpm:
    @pytest.mark.parametrize("wpm", [12.0, 16.0, 20.0, 28.0])
    def test_公称値に近い値が出る(self, wpm: float) -> None:
        """**包絡線法は速いほど高めに出る** (ランプぶん短点が短く測れるため)。

        偏りは ``ランプ長 / 短点長`` に比例するので、28 WPM で +15% 程度まで
        伸びる。ここでは「向きも桁も合っている」ことだけを見る。
        """
        est = estimate_wpm(*_wave(wpm))
        assert est is not None
        assert wpm * 0.9 <= est.wpm <= wpm * 1.25

    def test_速い信号ほど大きい値になる(self) -> None:
        """順序が保たれること. **目盛りが合うことより、こちらが本質。**"""
        slow = estimate_wpm(*_wave(12.0))
        fast = estimate_wpm(*_wave(28.0))
        assert slow is not None and fast is not None
        assert fast.wpm > slow.wpm * 1.8

    def test_雑音が多くても測れる(self) -> None:
        est = estimate_wpm(*_wave(20.0, snr_db=0.0, jitter=0.15))
        assert est is not None
        assert 15.0 <= est.wpm <= 27.0

    def test_トーンを取れる(self) -> None:
        est = estimate_wpm(*_wave(20.0, tone=750.0))
        assert est is not None
        assert 700.0 <= est.tone_hz <= 800.0

    def test_短点と長音の数を返す(self) -> None:
        est = estimate_wpm(*_wave(20.0))
        assert est is not None
        assert est.n_dot >= 6
        assert est.n_dash >= 3


class TestReturnsNone:
    """**嘘の数字を出すくらいなら何も出さない。**"""

    def test_短すぎる入力(self) -> None:
        assert estimate_wpm(np.zeros(4000, dtype=np.float32), 8000) is None

    def test_無音(self) -> None:
        assert estimate_wpm(np.zeros(80000, dtype=np.float32), 8000) is None

    def test_雑音だけ(self) -> None:
        rng = np.random.default_rng(3)
        noise = rng.normal(0.0, 0.1, 8000 * 10).astype(np.float32)
        est = estimate_wpm(noise, 8000)
        # 測れてしまった場合でも、範囲外は弾かれているはず
        assert est is None or MIN_WPM <= est.wpm <= MAX_WPM

    def test_標本が足りなければ測らない(self) -> None:
        """短点の数が ``min_dots`` に満たなければ ``None``."""
        assert estimate_wpm(*_wave(20.0), min_dots=10_000) is None

    def test_範囲外の速度は捨てる(self) -> None:
        """**測定の破綻を数字として出さない。**

        5〜40 WPM は実運用の CW が収まる範囲 (送信側の設定範囲と揃えてある)。
        """
        wave, sr = _wave(20.0)
        assert MIN_WPM < 20.0 < MAX_WPM        # 前提の確認
        # 4 倍に間引くと見かけの速度が 4 倍になり、範囲を超える
        assert estimate_wpm(wave[::4].copy(), sr) is None


class TestEnvelope:
    def test_歯止めを外すと全区間を使う(self) -> None:
        wave, sr = _wave(20.0)
        seg = envelope_on_off(wave, sr, contrast_min=LIVE_CONTRAST_MIN)
        assert seg.clean_sec == pytest.approx(wave.size / sr, rel=0.01)

    def test_ONとOFFが交互に取れる(self) -> None:
        wave, sr = _wave(20.0)
        seg = envelope_on_off(wave, sr, contrast_min=LIVE_CONTRAST_MIN)
        assert seg.on_sec.size >= 20
        assert seg.off_sec.size >= 20

    def test_トーン検出(self) -> None:
        wave, sr = _wave(20.0, tone=500.0)
        assert 450.0 <= detect_tone(wave, sr) <= 550.0


class TestSplitDotDash:
    def test_空なら空を返す(self) -> None:
        empty = np.array([], dtype=np.float64)
        dot, dash = split_dot_dash(empty)
        assert dot.size == 0
        assert dash.size == 0

    def test_境界を明示できる(self) -> None:
        on = np.array([0.06, 0.06, 0.18, 0.18])
        dot, dash = split_dot_dash(on, 0.12)
        assert dot.tolist() == [0.06, 0.06]
        assert dash.tolist() == [0.18, 0.18]


class TestRecentAudio:
    """``SlidingWindowDecoder`` から音を取って測る経路.

    **単体で通っても層をまたぐと壊れる。** このリポジトリでは繰り返し
    踏んでいるので、リングバッファ → 測定を実際に通す。
    """

    @staticmethod
    def _decoder():
        from src.infer.sliding_window import SlidingWindowDecoder

        class _NullEngine:                     # デコードはしないので中身は要らない
            pass

        return SlidingWindowDecoder(_NullEngine(), window_s=30.0, sample_rate=8000)

    def test_直近の音を取れる(self) -> None:
        dec = self._decoder()
        dec.push(np.arange(8000 * 5, dtype=np.float32))
        got = dec.recent_audio(2.0)
        assert got.size == 8000 * 2
        assert got[-1] == pytest.approx(8000 * 5 - 1)

    def test_溜まっている以上には遡れない(self) -> None:
        dec = self._decoder()
        dec.push(np.zeros(8000, dtype=np.float32))
        assert dec.recent_audio(12.0).size == 8000

    def test_空なら空を返す(self) -> None:
        assert self._decoder().recent_audio(5.0).size == 0

    def test_書き換えてもリングバッファを壊さない(self) -> None:
        """**コピーを返すこと。** ビューを返すと呼び出し側の加工が窓を汚す."""
        dec = self._decoder()
        dec.push(np.ones(8000, dtype=np.float32))
        got = dec.recent_audio(1.0)
        got[:] = 0.0
        assert float(dec.recent_audio(1.0).mean()) == pytest.approx(1.0)

    def test_窓から取った音で速度を測れる(self) -> None:
        """リングバッファ → ``recent_audio`` → ``estimate_wpm`` を通す."""
        wave, sr = _wave(20.0)
        dec = self._decoder()
        for i in range(0, wave.size, 1024):     # 実際と同じくブロックで投入する
            dec.push(wave[i:i + 1024])
        est = estimate_wpm(dec.recent_audio(12.0), sr)
        assert est is not None
        assert 18.0 <= est.wpm <= 25.0


class TestWiring:
    """画面までの繋ぎ目が揃っていること.

    ``MainWindow`` はモデルと音声デバイスが要るので組み立てられない。
    **名前の食い違いだけでも黙って動かなくなる**ので、そこだけ固定する
    (シグナル名・スロット名・引数名のいずれかを直したら、ここが落ちる)。
    """

    def test_ワーカーがシグナルを持つ(self) -> None:
        from src.app.workers import WPM_INTERVAL_S, WPM_WINDOW_S, AudioInferenceWorker

        assert hasattr(AudioInferenceWorker, "received_wpm_changed")
        assert hasattr(AudioInferenceWorker, "_maybe_measure_wpm")
        # 窓は間隔より長いこと (短いと測るたびに音が足りなくなる)
        assert WPM_WINDOW_S > WPM_INTERVAL_S

    def test_画面がスロットを持つ(self) -> None:
        from src.app.main_window import CWDecoderWindow

        assert hasattr(CWDecoderWindow, "_on_received_wpm")

    def test_送信ダイアログが受け取れる(self) -> None:
        import inspect

        from src.app.tx_dialog import TxDialog

        params = inspect.signature(TxDialog.__init__).parameters
        assert "received_wpm" in params
        assert params["received_wpm"].default is None
        assert hasattr(TxDialog, "match_received_wpm")
