"""打鍵エンジン — 要素列を時間どおりに叩く.

ハードウェアを知らない
----------------------
``KeySink`` という窓口 (``key(on)`` と ``ptt(on)`` だけ) に向かって叩く。
本番はシリアル、テストは記録するだけの偽物を差す。
**これで打鍵タイミングをハード無しで検証できる。**

タイミングをどう担保するか
--------------------------
Windows の既定タイマ精度は約 15.6 ms。25 WPM の短点は 48 ms なので、
何もしなければ短点の 3 割が揺れる。4 つ重ねて対処する。

1. **``timeBeginPeriod(1)``** — タイマ精度を 1 ms に上げる。**送信中だけ**
2. **絶対時刻で刻む** — ``sleep(要素の長さ)`` を積み上げると誤差が累積する。
   開始時刻からの累積目標時刻を持ち、そこへ向けて待つ。1 要素の遅れが次に伝播しない
3. **混合待ち** — 目標の少し手前まで ``sleep``、最後は空回り
4. 呼び出し側が専用スレッドで回す

実測のずれを ``KeyingReport`` で返す。**数字が出ていれば環境の悪化に気づける。**

安全
----
* **どの経路で抜けても両線を落とす** (``finally``)
* **番犬**: キー ON が続いてよい上限を超えたら異常とみなし停止する。
  最長の長音でも 1 秒未満なので、数秒続くのはバグかフリーズである
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

# 目標時刻のこの手前まで sleep し、残りは空回りで待つ (秒)。
# sleep の分解能が 1 ms でも、実際の復帰は数百 µs 遅れる。
SPIN_MARGIN_S = 0.002

# 空回りの上限 (秒)。**キーを下げたまま固まらないための最後の砦。**
SPIN_GUARD_S = 1.0

# 空回りの回数上限。時計が凍ると時刻での上限も効かないため、回数でも縛る。
# 1 ms 待つのに数千回程度なので、これに達するのは異常である。
SPIN_MAX_ITERATIONS = 5_000_000


class KeySink(Protocol):
    """電鍵と PTT の線を上下させる窓口."""

    def key(self, on: bool) -> None: ...
    def ptt(self, on: bool) -> None: ...


@dataclass
class KeyingReport:
    """送信の結果と、実測したタイミングのずれ."""

    elements_sent: int
    aborted: bool
    watchdog_tripped: bool
    max_error_s: float
    mean_error_s: float
    total_s: float

    @property
    def max_error_ms(self) -> float:
        return self.max_error_s * 1000.0

    @property
    def mean_error_ms(self) -> float:
        return self.mean_error_s * 1000.0


class RecordingKeySink:
    """記録するだけの ``KeySink`` (テスト用)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, bool]] = []

    def key(self, on: bool) -> None:
        self.events.append(("key", on))

    def ptt(self, on: bool) -> None:
        self.events.append(("ptt", on))

    @property
    def key_states(self) -> list[bool]:
        return [on for name, on in self.events if name == "key"]

    def ended_safely(self) -> bool:
        """最後に両線が落ちているか (上げっぱなしになっていないか)."""
        last_key = [on for name, on in self.events if name == "key"]
        last_ptt = [on for name, on in self.events if name == "ptt"]
        return (not last_key or last_key[-1] is False) and (
            not last_ptt or last_ptt[-1] is False
        )


class _TimerResolution:
    """送信中だけ Windows のタイマ精度を 1 ms に上げる.

    Windows 以外、または呼び出しに失敗した環境では何もしない
    (精度は落ちるが動作はする)。
    """

    def __init__(self, milliseconds: int = 1) -> None:
        self._ms = milliseconds
        # Windows 以外では ``ctypes.WinDLL`` 自体が存在しないので型は Any
        self._winmm: Any = None

    def __enter__(self) -> _TimerResolution:
        try:
            import ctypes

            self._winmm = ctypes.WinDLL("winmm")     # type: ignore[attr-defined]
            self._winmm.timeBeginPeriod(self._ms)
        except Exception:
            self._winmm = None
        return self

    def __exit__(self, *exc: object) -> None:
        if self._winmm is not None:
            with contextlib.suppress(Exception):
                self._winmm.timeEndPeriod(self._ms)
            self._winmm = None


def _wait_until(target: float, clock, sleep) -> None:
    """``target`` (clock の値) まで待つ. 手前まで sleep、最後は細かく刻む.

    **空回りに上限を置く。** 時計が止まる・巻き戻ることは通常起きない
    (``perf_counter`` は単調) が、**ここで無限ループするとキーを下げたまま
    固まる**。番犬は要素の長さしか見ないので、この経路は番犬でも救えない。

    最後を ``sleep(0)`` で刻むのは、CPU を焼かないためと、テストが偽の時計を
    差せるようにするため (純粋な ``pass`` の空回りだと時計が進まない)。
    """
    remaining = target - clock()
    if remaining > SPIN_MARGIN_S:
        sleep(remaining - SPIN_MARGIN_S)
    deadline = clock() + max(remaining, 0.0) + SPIN_GUARD_S
    # 時計が凍った場合は時刻の上限も効かないので、回数でも縛る
    for _ in range(SPIN_MAX_ITERATIONS):
        if clock() >= target or clock() > deadline:
            return
        sleep(0)


class Keyer:
    """要素列を時間どおりに ``KeySink`` へ叩き込む.

    Args:
        key_watchdog_s: キー ON が続いてよい上限 (秒)。超えたら停止する。
        clock: 時刻源 (テストで差し替える)。
        sleep: 待機関数 (テストで差し替える)。
    """

    def __init__(
        self,
        key_watchdog_s: float = 5.0,
        clock=time.perf_counter,
        sleep=time.sleep,
    ) -> None:
        self.key_watchdog_s = key_watchdog_s
        self._clock = clock
        self._sleep = sleep

    def send(
        self,
        durations: Sequence[float],
        is_on: Sequence[bool],
        sink: KeySink,
        abort: threading.Event | None = None,
        ptt_lead_s: float = 0.0,
        ptt_tail_s: float = 0.0,
        use_ptt: bool = True,
    ) -> KeyingReport:
        """要素列を送信する.

        **どの経路で抜けても両線を落とす。**
        """
        if len(durations) != len(is_on):
            raise ValueError("durations と is_on の長さが違います")
        abort = abort or threading.Event()
        errors: list[float] = []
        sent = 0
        aborted = False
        watchdog = False

        with _TimerResolution():
            try:
                if use_ptt:
                    sink.ptt(True)
                    if ptt_lead_s > 0:
                        _wait_until(
                            self._clock() + ptt_lead_s, self._clock, self._sleep
                        )

                start = self._clock()
                elapsed = 0.0
                # 長さが違えば冒頭で ValueError にしているので strict=True で問題ない
                for duration, on in zip(durations, is_on, strict=True):
                    if abort.is_set():
                        aborted = True
                        break
                    if on and duration > self.key_watchdog_s:
                        # 最長の長音でも 1 秒未満。これはバグかフリーズである
                        watchdog = True
                        break
                    sink.key(bool(on))
                    elapsed += float(duration)
                    target = start + elapsed
                    _wait_until(target, self._clock, self._sleep)
                    errors.append(abs(self._clock() - target))
                    sent += 1

                sink.key(False)
                if use_ptt and ptt_tail_s > 0 and not aborted:
                    _wait_until(self._clock() + ptt_tail_s, self._clock, self._sleep)
            finally:
                # **上げっぱなしにしない。** 例外でも中止でもここを通る
                try:
                    sink.key(False)
                finally:
                    if use_ptt:
                        sink.ptt(False)

        return KeyingReport(
            elements_sent=sent,
            aborted=aborted,
            watchdog_tripped=watchdog,
            max_error_s=max(errors) if errors else 0.0,
            mean_error_s=(sum(errors) / len(errors)) if errors else 0.0,
            total_s=sum(float(d) for d in durations[:sent]),
        )


__all__ = [
    "SPIN_GUARD_S",
    "SPIN_MARGIN_S",
    "SPIN_MAX_ITERATIONS",
    "KeySink",
    "Keyer",
    "KeyingReport",
    "RecordingKeySink",
]
