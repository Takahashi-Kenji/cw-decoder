"""緩い分割 (`segment_european_loose`) — 間隔で切れない局を語彙で切る."""

from __future__ import annotations

from src.infer.word_correct import segment_european_loose


class TestLoose:
    def test_運用者の録音の並びを切る(self) -> None:
        assert segment_european_loose("CQCQCQDEJA1ABC/3") == ["CQ", "CQ", "CQ", "DE", "JA1ABC/3"]

    def test_コールサインを2回(self) -> None:
        got = segment_european_loose("DEJA1ABC/3JA1ABC/3K")
        assert got[:5] == ["DE", "JA1ABC/3", "JA1ABC/3", "K"] or got == ["DE", "JA1ABC/3", "JA1ABC/3", "K"]

    def test_数字列を切る(self) -> None:
        assert segment_european_loose("URRST599599NAME") == ["UR", "RST", "599", "599", "NAME"]

    def test_短ければ触らない(self) -> None:
        assert segment_european_loose("CQDE") == ["CQDE"]

    def test_RSTの略記(self) -> None:
        """運用者の実受信: GA5NNBK が切れなかった (7 文字・5NN が語彙に無い)."""
        assert segment_european_loose("GA5NNBK") == ["GA", "5NN", "BK"]

    def test_未知だらけなら触らない(self) -> None:
        """何にでも切れてしまうのを防ぐ."""
        assert segment_european_loose("XZQWVXZQWV") == ["XZQWVXZQWV"]

    def test_スペース入りは触らない(self) -> None:
        assert segment_european_loose("CQ CQ DE JA1ABC") == ["CQ CQ DE JA1ABC"]

    def test_未知が続けばまとめる(self) -> None:
        got = segment_european_loose("CQCQDEJH")
        assert got[:3] == ["CQ", "CQ", "DE"] and got[-1] == "JH"


class TestCorrectTextIntegration:
    def test_つながった欧文が語彙で切れる(self) -> None:
        from src.infer.word_correct import correct_text
        got = correct_text("CQCQCQDEJA1ABC/3JA1ABC/3K").text
        assert got.startswith("CQ CQ CQ DE JA1ABC/3 JA1ABC/3")

    def test_既にスペースがあれば従来どおり(self) -> None:
        from src.infer.word_correct import correct_text
        assert correct_text("CQ CQ DE JA1ABC JA1ABC K").text == "CQ CQ DE JA1ABC JA1ABC K"

    def test_和文には掛からない(self) -> None:
        from src.infer.word_correct import correct_text
        s = "オハヨウゴザイマスキョウハハレデス"
        assert " " not in correct_text(s, japanese_enabled=False).text

    def test_分割のみモードは寄せをしない(self) -> None:
        from src.infer.word_correct import correct_text
        # NAM は寄せが働けば NAME になるが、segment_only では触らない
        assert correct_text("NAM", segment_only=True).text == "NAM"
        assert correct_text("CQCQCQDEJA1ABC/3", segment_only=True).text == "CQ CQ CQ DE JA1ABC/3"
