"""打鍵エンジンとシリアル層のテスト.

**実機は使わない。** 電波が出るため。``RecordingKeySink`` と偽の時計で検証する。
"""
from __future__ import annotations

import threading

import pytest

from src.tx.keyer import Keyer, RecordingKeySink
from src.tx.serial_key import SerialKeyConfig, SerialKeyError, SerialKeySink


class FakeClock:
    """呼ばれるたびに進む偽の時計. ``sleep`` で任意に進められる."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        # 空回り (sleep(0)) でも必ず少し進める。実機でも 0 秒では返らないし、
        # 進まないと待ちループが終わらない
        self.now += max(seconds, 1e-6)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def keyer(clock: FakeClock) -> Keyer:
    return Keyer(clock=clock, sleep=clock.sleep)


class TestKeyingSequence:
    def test_key_follows_the_element_states(self, keyer: Keyer) -> None:
        sink = RecordingKeySink()
        keyer.send([0.06, 0.06, 0.18], [True, False, True], sink, use_ptt=False)
        # 末尾に必ず OFF が入る (上げっぱなしにしない)
        assert sink.key_states[:3] == [True, False, True]
        assert sink.key_states[-1] is False

    def test_reports_how_many_were_sent(self, keyer: Keyer) -> None:
        sink = RecordingKeySink()
        report = keyer.send([0.06] * 5, [True] * 5, sink, use_ptt=False)
        assert report.elements_sent == 5

    def test_ends_with_both_lines_low(self, keyer: Keyer) -> None:
        sink = RecordingKeySink()
        keyer.send([0.06], [True], sink)
        assert sink.ended_safely()

    def test_mismatched_lengths_are_rejected(self, keyer: Keyer) -> None:
        with pytest.raises(ValueError, match="長さが違います"):
            keyer.send([0.06, 0.06], [True], RecordingKeySink())

    def test_empty_sequence(self, keyer: Keyer) -> None:
        sink = RecordingKeySink()
        report = keyer.send([], [], sink, use_ptt=False)
        assert report.elements_sent == 0
        assert sink.ended_safely()


class TestTiming:
    """**絶対時刻で刻む。** sleep の積み上げは誤差が累積する."""

    def test_targets_do_not_drift(self, clock: FakeClock, keyer: Keyer) -> None:
        keyer.send([0.06] * 100, [True, False] * 50, RecordingKeySink(), use_ptt=False)
        # 100 要素 * 60 ms = 6.0 秒。積み上げの誤差が乗らないこと
        assert clock.now == pytest.approx(6.0, abs=1e-9)

    def test_a_late_element_is_recovered(self, clock: FakeClock) -> None:
        """1 要素が遅れても、後続が絶対時刻へ戻すこと.

        ``sleep(要素の長さ)`` を積み上げる方式なら遅れがそのまま残り、
        全体が 0.65 秒になる。絶対時刻で刻むと後続が追いつき 0.6 秒に戻る。
        **遅れが累積しないのがこの方式の要点。**
        """
        calls = {"n": 0}

        def slow_sleep(seconds: float) -> None:
            clock.now += max(seconds, 1e-6)
            calls["n"] += 1
            if calls["n"] == 1 and seconds > 0:
                clock.now += 0.05          # 1 要素目だけ 50 ms 遅らせる

        keyer = Keyer(clock=clock, sleep=slow_sleep)
        keyer.send([0.06] * 10, [True] * 10, RecordingKeySink(), use_ptt=False)
        assert clock.now == pytest.approx(0.6, abs=1e-6)

    def test_report_carries_measured_error(self, keyer: Keyer) -> None:
        report = keyer.send([0.06] * 3, [True] * 3, RecordingKeySink(), use_ptt=False)
        assert report.max_error_s >= 0.0
        assert report.max_error_ms == pytest.approx(report.max_error_s * 1000.0)


class TestAbort:
    """[中止] は送信中いつでも効く."""

    def test_abort_stops_early(self, keyer: Keyer) -> None:
        sink = RecordingKeySink()
        abort = threading.Event()
        abort.set()
        report = keyer.send([0.06] * 10, [True] * 10, sink, abort=abort)
        assert report.aborted is True
        assert report.elements_sent == 0

    def test_abort_still_drops_both_lines(self, keyer: Keyer) -> None:
        sink = RecordingKeySink()
        abort = threading.Event()
        abort.set()
        keyer.send([0.06] * 10, [True] * 10, sink, abort=abort)
        assert sink.ended_safely()


class TestWatchdog:
    """キー ON が続いてよい上限. 最長の長音でも 1 秒未満である."""

    def test_trips_on_an_absurdly_long_key_down(self, clock: FakeClock) -> None:
        keyer = Keyer(key_watchdog_s=5.0, clock=clock, sleep=clock.sleep)
        sink = RecordingKeySink()
        report = keyer.send([10.0], [True], sink, use_ptt=False)
        assert report.watchdog_tripped is True
        assert sink.ended_safely()

    def test_normal_elements_pass(self, keyer: Keyer) -> None:
        report = keyer.send([0.18], [True], RecordingKeySink(), use_ptt=False)
        assert report.watchdog_tripped is False

    def test_long_key_up_is_fine(self, keyer: Keyer) -> None:
        """OFF が長いのは無音であって異常ではない."""
        report = keyer.send([10.0], [False], RecordingKeySink(), use_ptt=False)
        assert report.watchdog_tripped is False


class TestPtt:
    def test_ptt_wraps_the_transmission(self, keyer: Keyer) -> None:
        sink = RecordingKeySink()
        keyer.send([0.06], [True], sink, ptt_lead_s=0.05, ptt_tail_s=0.1)
        names = [name for name, _ in sink.events]
        assert names[0] == "ptt"
        assert names[-1] == "ptt"
        assert sink.events[0][1] is True
        assert sink.events[-1][1] is False

    def test_ptt_can_be_disabled(self, keyer: Keyer) -> None:
        """無線機のブレークインに任せる構成."""
        sink = RecordingKeySink()
        keyer.send([0.06], [True], sink, use_ptt=False)
        assert not [name for name, _ in sink.events if name == "ptt"]

    def test_lead_time_elapses_before_keying(self, clock: FakeClock, keyer: Keyer) -> None:
        keyer.send([0.06], [True], RecordingKeySink(), ptt_lead_s=0.05, ptt_tail_s=0.0)
        assert clock.now == pytest.approx(0.05 + 0.06, abs=1e-9)


class TestSinkFailure:
    """線を操作できなくなっても上げっぱなしにしない."""

    def test_exception_mid_send_still_drops_lines(self, keyer: Keyer) -> None:
        class Failing(RecordingKeySink):
            def key(self, on: bool) -> None:
                super().key(on)
                if len([e for e in self.events if e[0] == "key"]) == 2:
                    raise SerialKeyError("USB が抜かれた")

        sink = Failing()
        with pytest.raises(SerialKeyError):
            keyer.send([0.06] * 5, [True] * 5, sink, use_ptt=False)
        assert sink.ended_safely()


class TestSerialConfig:
    def test_same_line_for_key_and_ptt_is_rejected(self) -> None:
        """打鍵のたびに送受信が切り替わってしまう."""
        config = SerialKeyConfig(port="COM1", key_line="DTR", ptt_line="DTR")
        with pytest.raises(SerialKeyError, match="同じ線"):
            config.validate()

    def test_unknown_line_is_rejected(self) -> None:
        with pytest.raises(SerialKeyError, match="電鍵の線"):
            SerialKeyConfig(port="COM1", key_line="CTS").validate()

    def test_ptt_none_is_allowed(self) -> None:
        config = SerialKeyConfig(port="COM1", key_line="DTR", ptt_line="NONE")
        config.validate()
        assert config.uses_ptt is False

    def test_valid_config_passes(self) -> None:
        SerialKeyConfig(port="COM1", key_line="DTR", ptt_line="RTS").validate()


class TestSerialSinkSafety:
    """**開いた瞬間に線が上がる変換器がある。** 直後に落とすこと."""

    def _fake_serial_module(self, monkeypatch, opened: dict):
        class FakeSerial:
            def __init__(self, port, dsrdtr=True, rtscts=True, timeout=None):
                opened["kwargs"] = {"dsrdtr": dsrdtr, "rtscts": rtscts}
                # 変換器の実際の挙動を真似る: 開いた瞬間に両線が上がる
                self.dtr = True
                self.rts = True
                self.closed = False
                opened["obj"] = self

            def close(self) -> None:
                self.closed = True

        import sys
        import types

        module = types.ModuleType("serial")
        module.Serial = FakeSerial          # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "serial", module)
        return FakeSerial

    def test_lines_are_dropped_right_after_open(self, monkeypatch) -> None:
        opened: dict = {}
        self._fake_serial_module(monkeypatch, opened)
        sink = SerialKeySink(SerialKeyConfig(port="COM1"))
        sink.open()
        assert opened["obj"].dtr is False
        assert opened["obj"].rts is False
        sink.close()

    def test_handshake_is_disabled_when_opening(self, monkeypatch) -> None:
        """dsrdtr/rtscts を切らないと、開いた瞬間に線が上がる."""
        opened: dict = {}
        self._fake_serial_module(monkeypatch, opened)
        SerialKeySink(SerialKeyConfig(port="COM1")).open()
        assert opened["kwargs"] == {"dsrdtr": False, "rtscts": False}

    def test_close_drops_lines_before_closing(self, monkeypatch) -> None:
        opened: dict = {}
        self._fake_serial_module(monkeypatch, opened)
        sink = SerialKeySink(SerialKeyConfig(port="COM1"))
        sink.open()
        sink.key(True)
        sink.close()
        assert opened["obj"].dtr is False
        assert opened["obj"].closed is True

    def test_context_manager_closes(self, monkeypatch) -> None:
        opened: dict = {}
        self._fake_serial_module(monkeypatch, opened)
        with SerialKeySink(SerialKeyConfig(port="COM1")):
            pass
        assert opened["obj"].closed is True

    def test_invert_flips_the_level(self, monkeypatch) -> None:
        opened: dict = {}
        self._fake_serial_module(monkeypatch, opened)
        sink = SerialKeySink(SerialKeyConfig(port="COM1", key_invert=True))
        sink.open()
        sink.key(True)
        assert opened["obj"].dtr is False       # 負論理
        sink.close()

    def test_keying_without_open_is_an_error(self) -> None:
        sink = SerialKeySink(SerialKeyConfig(port="COM1"))
        with pytest.raises(SerialKeyError, match="開いていません"):
            sink.key(True)
