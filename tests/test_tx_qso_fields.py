"""受信テキストから欄を拾うテスト.

**拾えなかったら空のままにする。** 当てにならないものを黙って埋めると、
運用者が気づかないまま誤った電波が出る。
"""
from __future__ import annotations

from src.tx.qso_fields import QsoFields, extract_fields, strip_guess_marks


class TestStripGuessMarks:
    def test_清書の推測マーカーを外す(self) -> None:
        """清書結果には ⟦…⟧ が入っている (src/llm/markup.py)."""
        assert strip_guess_marks("CQ DE ⟦JA1ABC⟧ K") == "CQ DE JA1ABC K"

    def test_マーカーが無ければそのまま(self) -> None:
        assert strip_guess_marks("CQ DE JA1ABC K") == "CQ DE JA1ABC K"


class TestTheirCall:
    def test_DEの次を採る(self) -> None:
        """CW は <相手> DE <自分> の順で送る。**DE の次が送信者**."""
        assert extract_fields("JH0ILL DE JA1ABC K").their_call == "JA1ABC"

    def test_小文字でも拾う(self) -> None:
        assert extract_fields("jh0ill de ja1abc k").their_call == "JA1ABC"

    def test_最後のDEを採る(self) -> None:
        """一度の送信に DE が複数出ることがある。**最後が今の送信者**."""
        text = "JH0ILL DE JA1ABC JA1ABC K JH0ILL DE JH2XYZ K"
        assert extract_fields(text).their_call == "JH2XYZ"

    def test_DEが無ければ自局でないコールを採る(self) -> None:
        assert extract_fields("JH0ILL JA1ABC K", my_call="JH0ILL").their_call == "JA1ABC"

    def test_DEが無く自局も分からなければ空(self) -> None:
        """**どちらが相手か決められないなら埋めない。**"""
        assert extract_fields("JH0ILL JA1ABC K").their_call == ""

    def test_コールが無ければ空(self) -> None:
        assert extract_fields("{HORE}コンニチハ{RATA}").their_call == ""

    def test_DEの次がコールの形でなければ空(self) -> None:
        assert extract_fields("CQ DE CQ").their_call == ""

    def test_数字を含む局も拾う(self) -> None:
        assert extract_fields("JH0ILL DE 7K1ABC K").their_call == "7K1ABC"

    def test_移動局のスラッシュ付きコールも拾う(self) -> None:
        """JI1ABC/1 のような移動局表記も DE の次に来ることがある."""
        assert extract_fields("JH0ILL DE JI1ABC/1 K").their_call == "JI1ABC/1"

    def test_実在の呼出符号形式を広く拾う(self) -> None:
        """1〜2 文字プレフィックス (W1AW) と 2 文字プレフィックス (VK2ABC 等) も拾う."""
        for call in ("JA1ABC", "JH0ILL", "7K1ABC", "W1AW", "VK2ABC", "DL1XYZ"):
            assert extract_fields(f"CQ DE {call} K").their_call == call

    def test_DEが最後の語でも落ちない(self) -> None:
        """送信途中の断片などで DE の後に続く語が無いことがある。"""
        assert extract_fields("JH0ILL DE").their_call == ""

    def test_DEが最後の語で自局が分かっていても落ちない(self) -> None:
        assert extract_fields("JH0ILL DE", my_call="JH0ILL").their_call == ""


class TestTheirName:
    def test_NAMEの次を採る(self) -> None:
        assert extract_fields("UR RST 599 NAME TARO K").their_name == "TARO"

    def test_OPの次を採る(self) -> None:
        assert extract_fields("OP JOHN QTH TOKYO").their_name == "JOHN"

    def test_和文のナマエハの次を採る(self) -> None:
        assert extract_fields("{HORE}ナマエ ハ タロウ、ヨロシク{RATA}").their_name == "タロウ"

    def test_見つからなければ空(self) -> None:
        """**当てにならないので黙って埋めない。**"""
        assert extract_fields("JH0ILL DE JA1ABC K").their_name == ""

    def test_OPの次がQコードなら拾わない(self) -> None:
        """OP QRT は「運用者が終了する」の意で、QRT は名前ではない."""
        assert extract_fields("OP QRT").their_name == ""

    def test_OPの次がQコードでも他の手掛かりがあれば拾う(self) -> None:
        """OP QRT で誤爆させないための除外が、他の正しい手掛かりまで潰さないか."""
        assert extract_fields("OP QRT NAME TARO K").their_name == "TARO"


class TestQsoFields:
    def test_何も無ければ全部空(self) -> None:
        assert extract_fields("") == QsoFields()

    def test_清書マーカー入りでも拾える(self) -> None:
        assert extract_fields("JH0ILL DE ⟦JA1ABC⟧ K").their_call == "JA1ABC"

    def test_空白だけでも落ちない(self) -> None:
        assert extract_fields("   ") == QsoFields()

    def test_非常に長い入力でも落ちない(self) -> None:
        text = "JH0ILL DE JA1ABC K " * 5000
        result = extract_fields(text)
        assert result.their_call == "JA1ABC"
