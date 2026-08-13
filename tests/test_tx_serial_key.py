"""シリアル窓口のテスト.

**実機は要らない。** ``sys.modules["serial"]`` に偽物を差し込み、DTR/RTS へ
書かれた値をそのまま記録して確かめる。

ここで見るのは**「落とす」処理が本当に切になっているか**である。
反転結線 (``key_invert=True`` — キー ON = DTR L) では ``dtr = False`` が
「キー ON」を意味するので、安全のための後始末が逆に送信機を鍵付けする。
最終レビューで実測された欠陥 (C2) の回帰防止。
"""
from __future__ import annotations

import sys
import types

import pytest

from src.tx.serial_key import SerialKeyConfig, SerialKeyError, SerialKeySink


class FakeSerial:
    """DTR/RTS への代入を記録するだけの偽シリアル."""

    def __init__(self, port: str, dsrdtr: bool = True, rtscts: bool = True, timeout: float | None = None) -> None:
        self.port = port
        self.dsrdtr = dsrdtr
        self.rtscts = rtscts
        self.timeout = timeout
        self.closed = False
        self.writes: list[tuple[str, bool]] = []
        self._dtr = False
        self._rts = False

    @property
    def dtr(self) -> bool:
        return self._dtr

    @dtr.setter
    def dtr(self, value: bool) -> None:
        self._dtr = value
        self.writes.append(("dtr", value))

    @property
    def rts(self) -> bool:
        return self._rts

    @rts.setter
    def rts(self, value: bool) -> None:
        self._rts = value
        self.writes.append(("rts", value))

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def fake_serial(monkeypatch):
    """``import serial`` を偽物に差し替え、作られたポートを集める."""
    created: list[FakeSerial] = []

    def factory(port, dsrdtr=True, rtscts=True, timeout=None):
        port_obj = FakeSerial(port, dsrdtr, rtscts, timeout)
        created.append(port_obj)
        return port_obj

    module = types.ModuleType("serial")
    module.Serial = factory                      # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", module)
    return created


def key_writes(port: FakeSerial, line: str) -> list[bool]:
    """指定の線に書かれた値だけを順に取り出す."""
    return [value for name, value in port.writes if name == line]


def test_通常結線では開いた直後に両線がL(fake_serial) -> None:
    sink = SerialKeySink(SerialKeyConfig(port="COM3"))
    sink.open()
    port = fake_serial[0]
    assert port.dtr is False
    assert port.rts is False


def test_反転結線では開いた直後にキーが切になる(fake_serial) -> None:
    """**C2 の核心。** 反転では「切」は L ではなく H である.

    以前は ``open()`` が ``dtr=False`` を直に書いており、反転結線では
    そのまま「キー ON」だった (PTT 先行 50 ms のあいだ押されっぱなし)。
    """
    sink = SerialKeySink(SerialKeyConfig(port="COM3", key_invert=True))
    sink.open()
    port = fake_serial[0]
    # 反転では キー ON = dtr False。開いた直後は必ず「切」= True でなければならない
    assert port.dtr is True
    # 開いた直後に書かれた値が key(True) と同じであってはならない
    sink.key(True)
    assert key_writes(port, "dtr")[0] != key_writes(port, "dtr")[-1]


def test_反転結線では閉じるときもキーが切になる(fake_serial) -> None:
    """``Keyer`` の ``finally`` が落とした線を、閉じる処理が打ち消さないこと."""
    sink = SerialKeySink(SerialKeyConfig(port="COM3", key_invert=True))
    sink.open()
    port = fake_serial[0]
    sink.key(True)
    sink.key(False)
    sink.close()
    assert port.closed is True
    assert key_writes(port, "dtr")[-1] is True            # 切 (反転なので H)


def test_PTTの反転も落とす処理に効く(fake_serial) -> None:
    sink = SerialKeySink(SerialKeyConfig(port="COM3", ptt_invert=True))
    sink.open()
    port = fake_serial[0]
    assert port.dtr is False                              # 電鍵は非反転 → L が切
    assert port.rts is True                               # PTT は反転 → H が切
    sink.close()
    assert key_writes(port, "rts")[-1] is True


def test_PTTがNONEなら余った線は素のLにする(fake_serial) -> None:
    """割り当てていない線は極性の概念が無い。**触らず放置はしない。**"""
    sink = SerialKeySink(SerialKeyConfig(port="COM3", ptt_line="NONE", key_invert=True))
    sink.open()
    port = fake_serial[0]
    assert port.dtr is True                               # 電鍵 (反転) は切 = H
    assert port.rts is False                              # 余り線は素の L
    assert key_writes(port, "rts") == [False]             # PTT としては触らない


def test_落とす処理は書き込みが失敗しても例外にしない(fake_serial) -> None:
    """閉じる途中で呼ぶので、ここで投げると後片付けが止まる."""
    sink = SerialKeySink(SerialKeyConfig(port="COM3"))
    sink.open()
    port = fake_serial[0]

    def explode(self: FakeSerial, value: bool) -> None:
        raise OSError("USB が抜かれた")

    type(port).dtr = property(lambda self: False, explode)  # type: ignore[assignment]
    try:
        sink.close()                                      # 例外が漏れてはいけない
    finally:
        type(port).dtr = FakeSerial.dtr                   # 後始末 (クラス属性を戻す)
    assert port.closed is True


def test_開けなければSerialKeyErrorになる(monkeypatch) -> None:
    module = types.ModuleType("serial")

    def factory(*args, **kwargs):
        raise OSError("そんなポートは無い")

    module.Serial = factory                      # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", module)
    with pytest.raises(SerialKeyError):
        SerialKeySink(SerialKeyConfig(port="COM9")).open()
