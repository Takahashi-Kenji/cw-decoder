"""「。ス」→「デス」の局所修正のテスト (2026-08-29 運用者報告).

**辞書補正には頼らない。** 運用者の指摘は「和文の補正は強すぎて使えない。
間違ったところを補完してあげればいいだけ」。実際、現状の辞書補正は
``ヨウカイデス`` を ``マイク デス`` にする (符号の遠い語へ寄せてしまう)。

ここで直すのは**符号として 1 要素違いで、日本語として明らかにおかしい**
一点だけ:

    デ = テ(・-・--) + 濁点(・・) = ・-・--・・   (7 要素)
    。 =                          ・-・-・・     (6 要素)

実測 (data/real 和文 18 件、2026-08-29): 「。」の直後の文字は「ス」が 10 回で
断然 1 位 (2 位以下は 1〜2 回)。10 件すべて「デス」として自然で、
正当な「。」+「ス始まりの語」は 1 件も無かった。

音の実測でも、該当箇所は ``・-・--・・`` の **7 要素**が正しく打たれており
(短点 64ms / 長点 200ms)、**打鍵は正しくモデルが誤読している**。
確信度は 0.99〜1.00 なので閾値では救えない。
"""
from __future__ import annotations

from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import TOKEN_TO_ID


def _ids(*codes: str) -> list[int]:
    return [TOKEN_TO_ID[c] for c in codes]


DANRAKU = "・-・-・・"
SU = "---・-"
NE = "--・-"
DAKUTEN = "・・"
KO = "----"
SHI = "--・-・"
A = "--・--"


def _convert(codes: list[str], mode: str = "japanese") -> str:
    conv = TokenConverter(mode)
    ids = [TOKEN_TO_ID[c] for c in codes]
    return conv.convert(ids, [1.0] * len(ids), initial_mode="japanese").text


class TestDanrakuSuBecomesDesu:
    def test_bare_danraku_su(self) -> None:
        assert _convert([DANRAKU, SU]) == "デス"

    def test_desu_ne(self) -> None:
        assert _convert([DANRAKU, SU, NE]) == "デスネ"

    def test_after_a_word(self) -> None:
        # アツイ。ス → アツイデス (運用者が報告した実例)
        assert _convert(["--・--", "・--・", "・-", DANRAKU, SU]) == "アツイデス"

    def test_confidences_stay_aligned(self) -> None:
        """トークンが 1 個増えるので確信度もずれないこと (長さ検査に落ちない)."""
        conv = TokenConverter("japanese")
        ids = _ids(DANRAKU, SU)
        result = conv.convert(ids, [0.9, 0.8], initial_mode="japanese")
        assert result.text == "デス"


class TestGuardrails:
    """**直しすぎない。** 強い補正こそが運用者に「使えない」と言われた原因."""

    def test_danraku_followed_by_other_kana_is_kept(self) -> None:
        # 「。」の直後が「ス」以外なら本物の段落として残す
        assert _convert([DANRAKU, KO]) == "。コ"

    def test_trailing_danraku_is_kept(self) -> None:
        assert _convert([A, DANRAKU]) == "ア。"

    def test_su_word_after_danraku_is_kept(self) -> None:
        """段落の直後に「スコシ」が来るのは自然な日本語なので触らない."""
        codes = [DANRAKU, SU, KO, SHI]      # 。スコシ
        assert _convert(codes).startswith("。ス")

    def test_european_mode_is_untouched(self) -> None:
        """欧文表に「。」も「テ」も無い。欧文では触らない (壊すだけ)."""
        conv = TokenConverter("european")
        ids = _ids(DANRAKU, SU)
        text = conv.convert(ids, [1.0, 1.0]).text
        assert "デ" not in text

    def test_only_the_first_of_two_danraku_is_examined(self) -> None:
        """本物の段落が続くときに巻き込まないこと (。。ス → 。デス)."""
        assert _convert([DANRAKU, DANRAKU, SU]) == "。デス"
