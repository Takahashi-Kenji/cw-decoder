"""経歴と読み辞書のテスト.

2026-08-12 に構造を変えた。``display`` + ``reading`` (同じものの 2 通りの書き方)
から、``european`` + ``japanese`` (**独立した 2 つの値**) へ。
``「FT991」`` は ``FT991`` の読みではなく別の値である。
"""
from __future__ import annotations

import json

import pytest

from src.tx.profile import (
    UNKNOWN,
    BilingualField,
    OperatorProfile,
    load_profile,
    save_profile,
)


class TestBilingualField:
    def test_和文の型では和文側(self) -> None:
        assert BilingualField("FT991", "「FT991」").for_mode("japanese") == "「FT991」"

    def test_欧文の型では欧文側(self) -> None:
        assert BilingualField("FT991", "「FT991」").for_mode("european") == "FT991"

    def test_anyの型では欧文側(self) -> None:
        """``any`` の型は欧文の略語を主とする (設計書 §3)."""
        assert BilingualField("FT991", "「FT991」").for_mode("any") == "FT991"

    @pytest.mark.parametrize("mode", ["european", "japanese", "any"])
    def test_もう一方で代用しない(self, mode: str) -> None:
        """**片側が空でも、もう一方の値は使わない。**

        書き忘れと区別がつかなくなる。空は空のまま返し、``?`` に倒すのは
        差し込み側の仕事である。
        """
        only_european = BilingualField(european="FT991", japanese="")
        if mode == "japanese":
            assert only_european.for_mode(mode) == ""
        else:
            assert only_european.for_mode(mode) == "FT991"


class TestNoFieldReadings:
    """**``field_readings()`` は廃止した** (2026-08-12).

    「表示形 → 読み」の辞書を本文全体に当てていたため、欧文の本文の
    ``TARO`` が ``タロウ`` に化けてモードが和文に倒れ、文中の欧文がまるごと
    送れなくなっていた。2 値にしたので、差し込みのときにモードで選べば済む。
    """

    def test_経歴は変換辞書を持たない(self) -> None:
        assert not hasattr(OperatorProfile(), "field_readings")

    def test_読み辞書は残る(self) -> None:
        """運用者が意図して入れたものは残す (警告は編集画面が出す)."""
        p = OperatorProfile(reading_dictionary={"東京": "トウキョウ"})
        assert p.reading_dictionary == {"東京": "トウキョウ"}


class TestCallsign:
    def test_コールサインは1つだけ(self) -> None:
        """和文の交信でもコールサインは欧文で送るので、分ける意味が無い."""
        assert OperatorProfile(callsign="JH0ILL").callsign == "JH0ILL"

    def test_既定は空文字(self) -> None:
        assert OperatorProfile().callsign == ""


class TestRoundTrip:
    def test_保存して読み直せる(self, tmp_path) -> None:
        path = tmp_path / "operator.json"
        p = OperatorProfile(
            callsign="JH0ILL",
            name=BilingualField("TARO", "タロウ"),
            rig=BilingualField("FT991", "「FT991」"),
            notes="和文中心",
            reading_dictionary={"東京": "トウキョウ"},
        )
        save_profile(p, path)

        loaded = load_profile(path)
        assert loaded.callsign == "JH0ILL"
        assert loaded.name.european == "TARO"
        assert loaded.name.japanese == "タロウ"
        assert loaded.rig.japanese == "「FT991」"
        assert loaded.notes == "和文中心"
        assert loaded.reading_dictionary == {"東京": "トウキョウ"}

    def test_人が読める_utf8_で書く(self, tmp_path) -> None:
        """運用者が直接編集できること."""
        path = tmp_path / "operator.json"
        save_profile(OperatorProfile(name=BilingualField("TARO", "タロウ")), path)
        text = path.read_text(encoding="utf-8")
        assert "タロウ" in text          # \\u エスケープされていない
        assert json.loads(text)["name"]["japanese"] == "タロウ"

    def test_無いファイルは空の経歴(self, tmp_path) -> None:
        assert load_profile(tmp_path / "nope.json").callsign == ""

    def test_壊れたファイルは空の経歴(self, tmp_path) -> None:
        """壊れていてもアプリは起動できる必要がある."""
        path = tmp_path / "operator.json"
        path.write_text("{ broken", encoding="utf-8")
        assert load_profile(path).callsign == ""

    def test_壊れたファイルを上書きしない(self, tmp_path) -> None:
        """**読んだだけでは消さない** (手で直せる余地を残す。設計書 §6)."""
        path = tmp_path / "operator.json"
        path.write_text("{ broken", encoding="utf-8")
        load_profile(path)
        assert path.read_text(encoding="utf-8") == "{ broken"

    def test_知らない鍵は無視する(self, tmp_path) -> None:
        path = tmp_path / "operator.json"
        path.write_text('{"callsign": "JH0ILL", "zzz": 1}', encoding="utf-8")
        assert load_profile(path).callsign == "JH0ILL"

    def test_旧い形は空として読む(self, tmp_path) -> None:
        """**移行は書かない** (設計書 §2.4)。書き直せば済む.

        旧い形は ``{"display": ..., "reading": ...}``。``european``/``japanese``
        を持たないので空の欄になる。**落ちないこと**が要件である。
        """
        path = tmp_path / "operator.json"
        path.write_text(
            '{"callsign": {"display": "JH0ILL"},'
            ' "name": {"display": "\\u592a\\u90ce", "reading": "\\u30bf\\u30ed\\u30a6"}}',
            encoding="utf-8",
        )
        loaded = load_profile(path)
        assert loaded.callsign == ""
        assert loaded.name.european == ""
        assert loaded.name.japanese == ""

    def test_フォルダを作る(self, tmp_path) -> None:
        path = tmp_path / "deep" / "operator.json"
        save_profile(OperatorProfile(), path)
        assert path.exists()

    def test_保存が失敗しても既存を壊さない(self, tmp_path, monkeypatch) -> None:
        """**一時ファイルに書いて置き換える** (型の保存と同じ流儀)."""
        path = tmp_path / "operator.json"
        save_profile(OperatorProfile(callsign="JH0ILL"), path)

        import json as json_mod

        def boom(*args, **kwargs):
            raise OSError("書き込み失敗")

        monkeypatch.setattr(json_mod, "dump", boom)
        with pytest.raises(OSError):
            save_profile(OperatorProfile(callsign="XXXXX"), path)

        assert load_profile(path).callsign == "JH0ILL"


class TestUnknownMarker:
    def test_値が無いときの印(self) -> None:
        """``?`` は和文表にも欧文表にもあるので、どちらのモードでも送れる."""
        assert UNKNOWN == "?"
