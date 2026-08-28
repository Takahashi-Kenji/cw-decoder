"""実受信機チェーン収集 (無線機 A で送信 → 無線機 B で受信) のテスト.

**送ったテキストがそのままラベルになる**のがこの経路の値打ちだが、
「打鍵計画」と「実際に出た電波」は違いうる (2026-08-03 に、原稿にあるのに
オートキーヤーが送出しなかったと**誤って**判断してラベルを壊した経緯がある)。
だからここでは打鍵側の実測 (elements_sent / aborted) と突き合わせ、
**食い違ったら保存しない**。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.finetune.collect import (
    CollectConfig,
    PlanItem,
    RejectReason,
    build_session_plan,
    collect_item,
    collect_with_retry,
    dry_run_error,
    fingerprint_error,
    level_verdict,
    load_plan_file,
    measure_recording,
    write_sample,
)
from src.finetune.dataset import discover_real_samples
from src.tx.net_key import NetKeyError as _NetKeyError
from src.tx.net_key import NetKeyRejected as _NetKeyRejected


class _FakeCheck:
    def __init__(self, elements: int, seconds: float) -> None:
        self.chars = 0
        self.elements = elements
        self.seconds = seconds


class _FakeSend:
    def __init__(self, elements_sent: int, *, aborted: bool = False, seconds: float = 1.0) -> None:
        self.elements_sent = elements_sent
        self.aborted = aborted
        self.watchdog_tripped = False
        self.seconds = seconds
        self.max_error_ms = 1.0
        self.mean_error_ms = 0.5
        self.reason = "stop" if aborted else None


class _FakeKeyer:
    """打鍵側の代役. 実際には電波を出さない."""

    def __init__(self, elements: int = 20, *, sent: int | None = None, aborted: bool = False) -> None:
        self.elements = elements
        self.sent = elements if sent is None else sent
        self.aborted = aborted
        self.calls: list[tuple[str, float]] = []

    def check(self, text: str, wpm: float) -> _FakeCheck:
        return _FakeCheck(self.elements, seconds=2.0)

    def send(self, text: str, wpm: float) -> _FakeSend:
        self.calls.append((text, wpm))
        return _FakeSend(self.sent, aborted=self.aborted)


class _FakeCapture:
    """音声側の代役. drain のたびに一定量の波形を返す.

    ``dropped`` は「**この項目の収集中に**落ちるブロック数」。実物の
    ``dropped_blocks`` は積算値なので、収集の前後で増えたかどうかで判定する
    (前の項目で落ちた分を今の項目のせいにしない)。
    """

    def __init__(self, amplitude: float = 0.3, dropped: int = 0) -> None:
        self.amplitude = amplitude
        self._dropped = 7          # 過去に落ちた分 (増えたかどうかだけが問題)
        self._drop_during = dropped
        self.drain_calls = 0

    def drain(self) -> list[np.ndarray]:
        self.drain_calls += 1
        if self.drain_calls == 1:
            return []              # 溜まりを捨てる 1 回目
        self._dropped += self._drop_during
        t = np.arange(8000) / 8000.0
        return [self.amplitude * np.sin(2 * np.pi * 600.0 * t)]

    @property
    def dropped_blocks(self) -> int:
        return self._dropped


def _cfg(**kw) -> CollectConfig:
    base = dict(tail_s=0.0, gap_s=0.0, min_level_db=-40.0, sample_rate=8000)
    base.update(kw)
    return CollectConfig(**base)


class TestBuildSessionPlan:
    def test_件数ぶん作る(self) -> None:
        plan = build_session_plan(count=5, modes=("european",), wpm_list=(20.0,), seed=1)
        assert len(plan) == 5

    def test_同じ_seed_なら同じ計画(self) -> None:
        a = build_session_plan(count=4, modes=("european", "japanese"), wpm_list=(18.0, 24.0), seed=7)
        b = build_session_plan(count=4, modes=("european", "japanese"), wpm_list=(18.0, 24.0), seed=7)
        assert [(i.text, i.wpm, i.mode) for i in a] == [(i.text, i.wpm, i.mode) for i in b]

    def test_速度は順に巡る(self) -> None:
        """**乱数で引かない。** 各速度が同じ回数だけ出るようにする
        (夜通し回して 30 WPM が 3 件しか無い、を避ける)."""
        plan = build_session_plan(count=6, modes=("european",), wpm_list=(15.0, 20.0, 25.0), seed=1)
        assert [i.wpm for i in plan] == [15.0, 20.0, 25.0, 15.0, 20.0, 25.0]

    def test_モードも順に巡る(self) -> None:
        plan = build_session_plan(count=4, modes=("european", "japanese"), wpm_list=(20.0,), seed=1)
        assert [i.mode for i in plan] == ["european", "japanese", "european", "japanese"]

    def test_名前は連番で衝突しない(self) -> None:
        plan = build_session_plan(count=3, modes=("european",), wpm_list=(20.0,), seed=1,
                                  session_id="s01")
        names = [i.name for i in plan]
        assert names == ["s01_0001_european", "s01_0002_european", "s01_0003_european"]
        assert len(set(names)) == 3

    def test_本文はモードの表で打てる(self) -> None:
        """打てない文字が混ざっていたら収集そのものが無駄になる."""
        from src.tokens.morse_tokens import text_to_codes
        for item in build_session_plan(count=6, modes=("european", "japanese"),
                                       wpm_list=(20.0,), seed=3):
            assert text_to_codes(item.text, item.mode)


class TestCollectItem:
    def _item(self) -> PlanItem:
        return PlanItem(name="s_0001_european", text="CQ TEST", mode="european", wpm=20.0)

    def test_正常なら波形とラベルが返る(self) -> None:
        got = collect_item(self._item(), _FakeKeyer(), _FakeCapture(), _cfg(), sleep=lambda s: None)
        assert got.rejected is None
        assert got.wave is not None and len(got.wave) > 0

    def test_中止されたら保存しない(self) -> None:
        got = collect_item(self._item(), _FakeKeyer(aborted=True), _FakeCapture(),
                           _cfg(), sleep=lambda s: None)
        assert got.rejected is RejectReason.ABORTED

    def test_打った要素数が食い違ったら保存しない(self) -> None:
        """**原稿はラベルではない。** 途中までしか出ていない電波に
        全文のラベルを付けると、物差しを壊す."""
        got = collect_item(self._item(), _FakeKeyer(elements=20, sent=12), _FakeCapture(),
                           _cfg(), sleep=lambda s: None)
        assert got.rejected is RejectReason.ELEMENT_MISMATCH

    def test_音が小さすぎたら保存しない(self) -> None:
        """無線機 B が A を聞けていない状態で一晩回すのを防ぐ."""
        got = collect_item(self._item(), _FakeKeyer(), _FakeCapture(amplitude=0.0005),
                           _cfg(), sleep=lambda s: None)
        assert got.rejected is RejectReason.TOO_QUIET

    def test_音が落ちていたら保存しない(self) -> None:
        """受信バッファが溢れると波形に穴が空く。ラベルは全文のままなので
        **穴の空いた音に正しいラベル**という最悪の組み合わせになる."""
        got = collect_item(self._item(), _FakeKeyer(), _FakeCapture(dropped=1),
                           _cfg(), sleep=lambda s: None)
        assert got.rejected is RejectReason.DROPPED_AUDIO

    def test_送信前に溜まりを捨てる(self) -> None:
        """前の項目の残響を次のラベルに混ぜない."""
        cap = _FakeCapture()
        collect_item(self._item(), _FakeKeyer(), cap, _cfg(), sleep=lambda s: None)
        assert cap.drain_calls >= 2      # 捨てる 1 回 + 収穫 1 回以上

    def test_打鍵の実測を持ち帰る(self) -> None:
        got = collect_item(self._item(), _FakeKeyer(), _FakeCapture(), _cfg(), sleep=lambda s: None)
        assert got.send is not None and got.send.elements_sent == 20


class TestWriteSample:
    def test_学習側が読める形式で書く(self, tmp_path: Path) -> None:
        item = PlanItem(name="s_0001_japanese", text="コンニチハ", mode="japanese", wpm=20.0)
        write_sample(tmp_path, item, np.ones(8000) * 1000.0, 8000, meta={"agc": "fast"})
        found = discover_real_samples(tmp_path)
        assert len(found) == 1
        assert found[0].mode == "japanese"
        assert found[0].text == "コンニチハ"

    def test_セッションの条件をヘッダに残す(self, tmp_path: Path) -> None:
        """どの設定で録った音かが分からないと、後から切り分けられない."""
        item = PlanItem(name="s_0001_european", text="CQ", mode="european", wpm=25.0)
        write_sample(tmp_path, item, np.ones(8000) * 1000.0, 8000,
                     meta={"agc": "slow", "filter_hz": "500"})
        text = (tmp_path / "s_0001_european.txt").read_text(encoding="utf-8")
        assert "agc: slow" in text and "filter_hz: 500" in text and "wpm: 25" in text

    def test_ラベルがトークン化できないなら書かない(self, tmp_path: Path) -> None:
        item = PlanItem(name="x_0001_european", text="", mode="european", wpm=20.0)
        with pytest.raises(ValueError):
            write_sample(tmp_path, item, np.ones(800), 8000, meta={})


class TestPlanFromFile:
    """**打つ内容を外から全部指定できること。**

    弱点 (recall 0% の記号、デ/テ/。 の取り違え) を狙った原稿を作って
    ぶつけられるようにする。
    """

    def _write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "plan.txt"
        path.write_text(body, encoding="utf-8")
        return path

    def test_本文だけの行はモードと速度を巡らせる(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "CQ TEST\nQRZ?\n")
        plan = load_plan_file(path, session_id="s", modes=("european",), wpm_list=(20.0, 25.0))
        assert [(i.text, i.wpm) for i in plan] == [("CQ TEST", 20.0), ("QRZ?", 25.0)]

    def test_モードを行ごとに指定できる(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "japanese\tコンニチハ\n")
        plan = load_plan_file(path, session_id="s", modes=("european",), wpm_list=(20.0,))
        assert plan[0].mode == "japanese" and plan[0].text == "コンニチハ"

    def test_速度も行ごとに指定できる(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "japanese\t30\tデス\n")
        plan = load_plan_file(path, session_id="s", modes=("european",), wpm_list=(20.0,))
        assert plan[0].wpm == 30.0 and plan[0].mode == "japanese"

    def test_空行とコメントは飛ばす(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "# 弱点ねらい\n\nCQ\n")
        assert len(load_plan_file(path, session_id="s", modes=("european",), wpm_list=(20.0,))) == 1

    def test_打てない文字はその場でエラー(self, tmp_path: Path) -> None:
        """**一晩回してから気づくのでは遅い。**"""
        path = self._write(tmp_path, "european\tこんにちは\n")
        with pytest.raises(SystemExit):
            load_plan_file(path, session_id="s", modes=("european",), wpm_list=(20.0,))

    def test_名前は連番(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "CQ\nDE\n")
        plan = load_plan_file(path, session_id="w", modes=("european",), wpm_list=(20.0,))
        assert [i.name for i in plan] == ["w_0001_european", "w_0002_european"]


class TestFingerprintGate:
    """**両 PC の符号表が一致していないと集めてはいけない。**

    無線機 PC は別の複製を読む。表がずれていると、こちらのラベルと実際に出る
    電波が食い違ったデータが**静かに**溜まる。集めてから気づくと全部使えない。
    """

    def test_一致していれば通す(self) -> None:
        assert fingerprint_error(matches=True, allow=False) is None

    def test_食い違えば止める(self) -> None:
        message = fingerprint_error(matches=False, allow=False)
        assert message is not None
        assert "符号表" in message

    def test_明示的に許せば通すが警告を返す(self) -> None:
        """欧文だけ集めるなど、承知の上で進める場合の逃げ道."""
        assert fingerprint_error(matches=False, allow=True) is None


class TestLevelVerdict:
    """無音か、信号が来ているか、飽和しているかを言い分ける.

    **「繋がっている」と「聞こえている」は違う。** 接続は成立するのに
    レベルが -119 dB だった実例がある (取り込むデバイスが違っていた)。
    """

    def test_無音は無音と言う(self) -> None:
        assert level_verdict(-90.0).startswith("無音")

    def test_音が来ていれば正常(self) -> None:
        """**RMS だけでノイズ床と信号は区別できない。**
        受信機は常に何か鳴っているので、言い切らずに範囲だけ言う."""
        for db in (-45.0, -30.0, -12.0):
            assert "正常" in level_verdict(db)

    def test_飽和は警告(self) -> None:
        assert "過大" in level_verdict(-2.0)


class TestDryRunGate:
    """**打鍵側が dry-run のまま集めない。**

    dry-run では電鍵が動かないので、録れるのはノイズ床だけ。それに全文の
    ラベルが付いた「音の無い学習データ」が静かに溜まる。
    """

    def test_dry_run_なら止める(self) -> None:
        assert dry_run_error(dry_run=True) is not None

    def test_本番なら通す(self) -> None:
        assert dry_run_error(dry_run=False) is None


class TestMeasureRecording:
    """録音ごとの実測値をラベルに残す.

    **信号をノイズにどれだけ埋めたかは、後から音を見ないと分からない。**
    セッションの条件 (出力・アッテネータ) と実際の録れ高は一致しないので、
    録れた音そのものから測って残す。学習データを SNR で切り分けたいときに要る。
    """

    @staticmethod
    def _keyed(sr: int = 8000, tone: float = 600.0) -> np.ndarray:
        """0.1 秒鳴って 0.1 秒休むを繰り返す波形."""
        t = np.arange(sr * 2) / sr
        gate = ((t * 5).astype(int) % 2 == 0).astype(float)
        return 0.3 * gate * np.sin(2 * np.pi * tone * t)

    def test_トーンを測る(self) -> None:
        got = measure_recording(self._keyed(tone=580.0), 8000)
        assert abs(float(got["tone_hz"]) - 580.0) < 25.0

    def test_レベルを測る(self) -> None:
        got = measure_recording(self._keyed(), 8000)
        assert -30.0 < float(got["level_db"]) < 0.0

    def test_鳴りっぱなしよりキーイングの方がコントラストが高い(self) -> None:
        t = np.arange(16000) / 8000
        steady = 0.3 * np.sin(2 * np.pi * 600.0 * t)
        keyed = measure_recording(self._keyed(), 8000)
        flat = measure_recording(steady, 8000)
        assert float(keyed["contrast_db"]) > float(flat["contrast_db"]) + 10.0

    def test_値は文字列で返す(self) -> None:
        """TXT ヘッダにそのまま書けること."""
        for value in measure_recording(self._keyed(), 8000).values():
            assert isinstance(value, str)


class TestReconnect:
    """一晩の収集では、途中で切れても繋ぎ直して続ける.

    打鍵側は**待機中の繋ぎ直しを許している** (送信中は許さない)。
    50 件目で切れて朝に 50 件、では 8 時間が無駄になる。
    """

    class _FlakyKeyer(_FakeKeyer):
        def __init__(self, fail_times: int) -> None:
            super().__init__()
            self.fail_times = fail_times
            self.connects = 0

        def send(self, text: str, wpm: float):
            if self.fail_times > 0:
                self.fail_times -= 1
                raise _NetKeyError("切断")
            return super().send(text, wpm)

    def _item(self) -> PlanItem:
        return PlanItem(name="s_0001_european", text="CQ", mode="european", wpm=20.0)

    def test_一度切れても繋ぎ直して録る(self) -> None:
        keyer = self._FlakyKeyer(fail_times=1)
        got = collect_with_retry(
            self._item(), keyer, _FakeCapture(), _cfg(),
            sleep=lambda s: None, reconnect=lambda: setattr(keyer, "connects", keyer.connects + 1),
            max_retries=3, error_types=(_NetKeyError,),
        )
        assert got.rejected is None
        assert keyer.connects == 1

    def test_繋ぎ直しの回数に上限がある(self) -> None:
        """繋がらないものを一晩叩き続けない."""
        keyer = self._FlakyKeyer(fail_times=99)
        with pytest.raises(_NetKeyError):
            collect_with_retry(
                self._item(), keyer, _FakeCapture(), _cfg(),
                sleep=lambda s: None, reconnect=lambda: None,
                max_retries=2, error_types=(_NetKeyError,),
            )

    def test_決定的な拒否は繰り返さない(self) -> None:
        """打てない文字を含む等、何度やっても同じ失敗は即座に投げ直す."""
        class _Rejecting(_FakeKeyer):
            def __init__(self) -> None:
                super().__init__()
                self.attempts = 0

            def send(self, text: str, wpm: float):
                self.attempts += 1
                raise _NetKeyRejected("bad_chars", "打てない文字")

        keyer = _Rejecting()
        with pytest.raises(_NetKeyRejected):
            collect_with_retry(
                self._item(), keyer, _FakeCapture(), _cfg(),
                sleep=lambda s: None, reconnect=lambda: None,
                max_retries=3, error_types=(_NetKeyError,),
                no_retry_types=(_NetKeyRejected,),
            )
        assert keyer.attempts == 1

    def test_切れなければ繋ぎ直さない(self) -> None:
        keyer = self._FlakyKeyer(fail_times=0)
        collect_with_retry(
            self._item(), keyer, _FakeCapture(), _cfg(),
            sleep=lambda s: None, reconnect=lambda: setattr(keyer, "connects", keyer.connects + 1),
            max_retries=3, error_types=(_NetKeyError,),
        )
        assert keyer.connects == 0


class TestLeadIn:
    """**送信の前にノイズ床を録る。**

    打鍵の直前にバッファを空にすると、音が「無音 → いきなり符号」で始まる。
    実運用では符号の前に必ずノイズ床がある。1200 件の収録で **93% の録音で
    1 文字目が脱落**した (脱落 1470 個のうち 1188 個が先頭)。
    録れた音を後から貼り合わせても直らない (継ぎ目で別の誤りが出る)。
    """

    def _item(self) -> PlanItem:
        return PlanItem(name="s_0001_european", text="CQ", mode="european", wpm=20.0)

    def test_送信の前に待つ(self) -> None:
        waits: list[float] = []
        cfg = CollectConfig(tail_s=0.0, gap_s=0.0, min_level_db=-40.0, lead_in_s=1.5)
        collect_item(self._item(), _FakeKeyer(), _FakeCapture(), cfg, sleep=waits.append)
        assert 1.5 in waits

    def test_待つのは溜まりを捨てた後(self) -> None:
        """捨てる前に待つと、前の項目の残りがラベルに混ざる."""
        order: list[str] = []

        class _Cap(_FakeCapture):
            def drain(self):
                order.append("drain")
                return super().drain()

        class _Keyer(_FakeKeyer):
            def send(self, text: str, wpm: float):
                order.append("send")
                return super().send(text, wpm)

        cfg = CollectConfig(tail_s=0.0, gap_s=0.0, min_level_db=-40.0, lead_in_s=1.0)
        collect_item(self._item(), _Keyer(), _Cap(), cfg,
                     sleep=lambda s: order.append(f"wait{s}"))
        assert order[:3] == ["drain", "wait1.0", "send"]

    def test_ゼロなら待たない(self) -> None:
        waits: list[float] = []
        cfg = CollectConfig(tail_s=0.0, gap_s=0.0, min_level_db=-40.0, lead_in_s=0.0)
        collect_item(self._item(), _FakeKeyer(), _FakeCapture(), cfg, sleep=waits.append)
        assert waits == []
