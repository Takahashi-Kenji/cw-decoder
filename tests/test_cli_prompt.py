"""起動時の対話 (配布版の打鍵サーバ / 音声送出) のテスト.

**exe をダブルクリックした人が、コマンドを覚えずに始められること**が目的
(運用者の指示、2026-08-31)。COM ポート番号も音声デバイス番号もその PC でしか
分からないので、一覧を出して番号で選ばせ、選んだ値を覚える。

ここはハードウェアに触らない。候補・前回値・入力を受け取って選択を返すだけの
純粋な関数にしてあるので、実機が無くても全部試せる。
"""
from __future__ import annotations

import json

import pytest

from src.cli.prompt import (
    Choice,
    ask_choice,
    ask_yes,
    load_remembered,
    save_remembered,
)


def _fake_io(answers: list[str]):
    """入力を順に返す関数と、出た文字を溜める関数を作る.

    **``read`` に渡された文言も溜める。** 本物の ``input(prompt)`` は
    その文言を画面に出すので、捨ててしまうと「利用者に何が見えているか」を
    試したことにならない。
    """
    written: list[str] = []
    it = iter(answers)

    def read(prompt: str = "") -> str:
        written.append(prompt)
        return next(it)

    return read, written.append, written


# **値を見せたいものは ``label`` の先頭に自分で入れる** (COM ポートは値そのものが
# 名前なので出す)。一覧は ``label`` しか出さない。
PORTS = (
    Choice(value="COM3", label="COM3  USB-SERIAL CH340"),
    Choice(value="COM5", label="COM5  Prolific USB-to-Serial"),
)


class TestAskChoice:
    def test_number_selects(self) -> None:
        read, write, out = _fake_io(["2"])
        got = ask_choice("シリアルポート", PORTS, None, read, write)
        assert got.value == "COM5"
        # 候補は番号付きで見えていること
        text = "".join(out)
        assert "1) COM3" in text and "2) COM5" in text

    def test_empty_uses_remembered(self) -> None:
        """2 回目からは Enter だけで前回の値で始まること."""
        read, write, out = _fake_io([""])
        got = ask_choice("シリアルポート", PORTS, "COM5", read, write)
        assert got.value == "COM5"
        assert "前回: COM5" in "".join(out)

    def test_empty_without_memory_uses_first(self) -> None:
        read, write, _ = _fake_io([""])
        assert ask_choice("シリアルポート", PORTS, None, read, write).value == "COM3"

    def test_forgotten_value_is_ignored(self) -> None:
        """前回の機器が繋がっていないとき、その値を既定にしない."""
        read, write, out = _fake_io([""])
        got = ask_choice("シリアルポート", PORTS, "COM9", read, write)
        assert got.value == "COM3"
        assert "前回: COM9" not in "".join(out)

    @pytest.mark.parametrize("bad", ["0", "3", "abc", "-1"])
    def test_bad_input_asks_again(self, bad: str) -> None:
        read, write, out = _fake_io([bad, "1"])
        assert ask_choice("シリアルポート", PORTS, None, read, write).value == "COM3"
        assert "もう一度" in "".join(out)

    def test_single_candidate_still_shown(self) -> None:
        """1 つしか無くても黙って選ばない (どれを使うか目で見せる)."""
        read, write, out = _fake_io([""])
        one = (Choice(value="COM3", label="COM3  USB-SERIAL CH340"),)
        assert ask_choice("シリアルポート", one, None, read, write).value == "COM3"
        assert "1) COM3" in "".join(out)

    def test_no_candidates_raises(self) -> None:
        read, write, _ = _fake_io([])
        with pytest.raises(LookupError):
            ask_choice("シリアルポート", (), None, read, write)


class TestAskYes:
    """**電波が出る前の確認。** 既定は「進まない」側に倒す."""

    @pytest.mark.parametrize("answer", ["", "y", "Y", "はい"])
    def test_enter_or_yes_proceeds(self, answer: str) -> None:
        read, write, _ = _fake_io([answer])
        assert ask_yes("準備はできましたか", read, write) is True

    @pytest.mark.parametrize("answer", ["n", "N", "いいえ"])
    def test_no_stops(self, answer: str) -> None:
        read, write, _ = _fake_io([answer])
        assert ask_yes("準備はできましたか", read, write) is False


class TestRemembered:
    def test_round_trip(self, tmp_path) -> None:
        path = tmp_path / "key_server.json"
        save_remembered(path, {"port": "COM3", "key_line": "DTR"})
        assert load_remembered(path) == {"port": "COM3", "key_line": "DTR"}

    def test_missing_file_is_empty(self, tmp_path) -> None:
        assert load_remembered(tmp_path / "nope.json") == {}

    def test_broken_file_is_empty(self, tmp_path) -> None:
        """壊れていても起動を止めない (対話でやり直せる)."""
        path = tmp_path / "key_server.json"
        path.write_text("{壊れている", encoding="utf-8")
        assert load_remembered(path) == {}

    def test_creates_parent_directory(self, tmp_path) -> None:
        path = tmp_path / "できていない" / "key_server.json"
        save_remembered(path, {"port": "COM3"})
        assert json.loads(path.read_text(encoding="utf-8"))["port"] == "COM3"


class TestOnlyOneNumberPerLine:
    """**行に数字を 2 つ並べないこと.**

    2026-08-31、運用者が音声デバイスの一覧で ``2`` を選んだのに 1 番の機器が
    使われた、と報告した。原因は表示がこうなっていたこと::

        2) 1  マイク (Creative Live! Cam Sync 10
        ↑     ↑
      行番号  デバイス番号

    行番号のとおりに動いてはいたが、**どちらを打てばよいのか読み取れない**。
    値そのものが名前になっている COM ポート (``COM3``) では起きず、値が
    ただの番号である音声デバイスでだけ起きる。

    直しかた: 一覧に出すのは ``label``。値は画面に出さない (``short`` で
    「前回」の表示にだけ使う)。
    """

    def test_audio_device_line_has_no_second_number(self) -> None:
        choice = Choice(value="1", label="マイク (Creative Live! Cam Sync)", short="マイク (Creative)")
        line = choice.line(2)
        assert line == "    2) マイク (Creative Live! Cam Sync)", line
        # 行番号より後ろに、選択と紛らわしい裸の番号が出ていないこと
        assert ") 1 " not in line

    def test_serial_port_line_still_shows_the_port(self) -> None:
        """COM ポートは**値そのものが名前**なので今までどおり出す."""
        choice = Choice(value="COM3", label="COM3  USB-SERIAL CH340")
        assert choice.line(1) == "    1) COM3  USB-SERIAL CH340"

    def test_remembered_hint_uses_the_short_name(self) -> None:
        """``[前回: 1]`` では何の機器か分からない."""
        devices = (
            Choice(value="0", label="サウンド マッパー", short="サウンド マッパー"),
            Choice(value="1", label="マイク (Creative)", short="マイク (Creative)"),
        )
        read, write, out = _fake_io([""])
        got = ask_choice("入力デバイス", devices, "1", read, write)
        assert got.value == "1"
        assert "前回: マイク (Creative)" in "".join(out)
        assert "前回: 1]" not in "".join(out)
