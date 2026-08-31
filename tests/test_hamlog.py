"""Hamlog 連携のテスト.

**本物の Hamlog は動かさない。** 入力ウィンドウの代役を渡して、
「何番の欄に何を入れたか」だけを見る。Windows メッセージを実際に送る部分
(``_Win32Window``) は薄く保ち、ここでは検証しない。
"""
from __future__ import annotations

import datetime as _dt

import pytest

from src.tx.hamlog import (
    FIELD_COMMANDS,
    CLEAR_COMMAND,
    HamlogNotRunning,
    QsoEntry,
    register,
)


class FakeWindow:
    """Hamlog の入力ウィンドウの代役. **入れられた順に覚える。**"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True

    def send_field(self, command: int, text: str) -> None:
        self.calls.append((command, text))


@pytest.fixture()
def entry() -> QsoEntry:
    return QsoEntry(
        call="JA1ABC",
        date="26/08/30",
        time="14:05",
        rst_sent="599",
        rst_received="579",
        freq_mhz="7.026",
        mode="CW",
        name="タロウ",
        qth="神奈川県横浜市",
        remarks1="CW デコーダから登録",
        remarks2="鶴見区",
    )


class TestQsoEntry:
    def test_今の時刻から作れる(self) -> None:
        """**JST で作る。** PC の時計が別の地域でも和暦の交信記録は JST."""
        moment = _dt.datetime(2026, 8, 30, 5, 6, tzinfo=_dt.timezone.utc)  # JST 14:06
        e = QsoEntry.at(
            moment, call="JA1ABC", rst_sent="599", rst_received="579", freq_mhz="7.026"
        )
        assert e.date == "26/08/30"
        assert e.time == "14:06"

    def test_モードの既定はCW(self) -> None:
        e = QsoEntry.at(
            _dt.datetime.now(_dt.timezone.utc),
            call="JA1ABC", rst_sent="599", rst_received="579", freq_mhz="7.026",
        )
        assert e.mode == "CW"

    def test_コールサインは大文字にする(self) -> None:
        e = QsoEntry.at(
            _dt.datetime.now(_dt.timezone.utc),
            call=" ja1abc ", rst_sent="599", rst_received="579", freq_mhz="7.026",
        )
        assert e.call == "JA1ABC"


class TestRegister:
    """**入力欄に入れるだけ。** 確定 (ログへの書き込み) はしない.

    2026-08-30 の運用者の判断。値を間違えたまま書き込むと Hamlog 側で消す手間が
    かかる。運用者が入力ウィンドウを見てから Enter を押す。
    """

    def test_全部の欄に入る(self, entry: QsoEntry) -> None:
        window = FakeWindow()
        register(entry, window)
        # コールサインにはフラグが乗るので、下位 16 ビットで引き直す
        sent = {command & 0xFFFF: text for command, text in window.calls}
        assert sent[FIELD_COMMANDS["call"]] == "JA1ABC"
        assert sent[FIELD_COMMANDS["date"]] == "26/08/30"
        assert sent[FIELD_COMMANDS["time"]] == "14:05"
        assert sent[FIELD_COMMANDS["rst_sent"]] == "599"
        assert sent[FIELD_COMMANDS["rst_received"]] == "579"
        assert sent[FIELD_COMMANDS["freq_mhz"]] == "7.026"
        assert sent[FIELD_COMMANDS["mode"]] == "CW"
        assert sent[FIELD_COMMANDS["name"]] == "タロウ"
        assert sent[FIELD_COMMANDS["qth"]] == "神奈川県横浜市"
        assert sent[FIELD_COMMANDS["remarks1"]] == "CW デコーダから登録"
        assert sent[FIELD_COMMANDS["remarks2"]] == "鶴見区"

    def test_先に入力欄を消す(self, entry: QsoEntry) -> None:
        """**前の交信の残りを混ぜない。** 消してから入れる."""
        window = FakeWindow()
        register(entry, window)
        assert window.cleared is True

    def test_空の欄は送らない(self) -> None:
        """名前や住所が分からない交信のほうが多い. 空で上書きしない."""
        window = FakeWindow()
        register(
            QsoEntry(
                call="JA1ABC", date="26/08/30", time="14:05",
                rst_sent="599", rst_received="579", freq_mhz="7.026",
            ),
            window,
        )
        sent = {command & 0xFFFF: text for command, text in window.calls}
        assert FIELD_COMMANDS["name"] not in sent
        assert FIELD_COMMANDS["qth"] not in sent
        assert FIELD_COMMANDS["remarks1"] not in sent
        assert FIELD_COMMANDS["remarks2"] not in sent

    def test_コールサインを最初に入れる(self, entry: QsoEntry) -> None:
        """Hamlog はコールサインで自分の記録を引く. 先に入れておく."""
        assert window_first_command(entry) & 0xFFFF == FIELD_COMMANDS["call"]

    def test_保存は押さない(self, entry: QsoEntry) -> None:
        """**保存のコマンドは絶対に送らない** (運用者が Hamlog 側で [Save] を押す)."""
        from src.tx.hamlog import SAVE_COMMAND

        window = FakeWindow()
        register(entry, window)
        assert all(command != SAVE_COMMAND for command, _ in window.calls)

    def test_コールサインだけEnterで確定させる(self, entry: QsoEntry) -> None:
        """Enter が無いと Hamlog は交信として扱わず [Save] が働かない (実機で確認).

        **Enter は保存ではない** — 運用者がコールサイン欄で押すのと同じ、
        重複チェックとユーザーリスト取り込みのための確定である。
        """
        from src.tx.hamlog import FIELD_COMMANDS, THW_ENTER

        window = FakeWindow()
        register(entry, window)
        with_enter = [c for c, _ in window.calls if c & THW_ENTER]
        assert len(with_enter) == 1
        assert with_enter[0] & 0xFFFF == FIELD_COMMANDS["call"]

    def test_コールサイン以外はEnterを付けない(self, entry: QsoEntry) -> None:
        from src.tx.hamlog import FIELD_COMMANDS, THW_ENTER

        window = FakeWindow()
        register(entry, window)
        for command, _ in window.calls:
            if command & 0xFFFF != FIELD_COMMANDS["call"]:
                assert command & THW_ENTER == 0

    def test_窓が無ければ知らせる(self, entry: QsoEntry) -> None:
        """**黙って捨てない。** Hamlog が起動していないことを運用者に見せる."""
        with pytest.raises(HamlogNotRunning):
            register(entry, None)


def window_first_command(entry: QsoEntry) -> int:
    window = FakeWindow()
    register(entry, window)
    return window.calls[0][0]


class TestFieldCommands:
    """番号は公式仕様 (``HamlogMs.txt``) とヘルプの並びから確定したもの.

    仕様書が名指しするのは 1=コールサイン / 2=日付 / 12=QTH / 13,14=Remarks と
    「1～14 まで入力欄の並びのとおり」だけ。並びはヘルプ (入力環境設定) の
    「Date, Time, His, My, Freq, Mode, Code, GL, QSL の項目のフォント」が示す。
    **His は送る RST、My はもらった RST。** 詳しくは ``src/tx/hamlog.py`` の
    docstring を参照 (2026-08-30、Ver5.47c で確認)。
    """

    @pytest.mark.parametrize(
        ("field", "command"),
        [
            ("call", 1), ("date", 2), ("time", 3),
            ("rst_sent", 4), ("rst_received", 5),
            ("freq_mhz", 6), ("mode", 7), ("name", 11), ("qth", 12),
            ("remarks1", 13), ("remarks2", 14),
        ],
    )
    def test_公式仕様の番号(self, field: str, command: int) -> None:
        assert FIELD_COMMANDS[field] == command

    def test_番号は重複しない(self) -> None:
        assert len(set(FIELD_COMMANDS.values())) == len(FIELD_COMMANDS)

    def test_クリアは16番(self) -> None:
        assert CLEAR_COMMAND == 16
