"""「。」を「デ」に直す局所修正のテスト (2026-08-29 運用者の指示).

**辞書には頼らない。** 運用者:「和文の補正は補正が強すぎて、使えない。
間違ったところを補完してあげればいいだけだ」。実際、辞書補正は
``ヨウカイデス`` を ``マイク デス`` にする。

判定は**間隔だけ**で行う。運用者の指示:

    「。」の後、改行する時間ぶん何も文字がなければ「。」。そうでなければ「デ」
    (「改行する時間を 1 秒に設定すればいいだけだから、その改行時間で判定する」)

**新しい設定は増やさない。** ``line_break_gap_s`` をそのまま使う。
``split_at_gaps`` が既にその時間で区切っているので、判定は
**「区間の末尾の『。』だけが段落、それ以外は全部デ」**で済む。

デ = テ(・-・--) + 濁点(・・) = ・-・--・・ (7 要素) と
。 =                        ・-・-・・   (6 要素) は 1 要素しか違わない。
音の実測では 7 要素が正しく打たれており、確信度 0.99〜1.00 でモデルが
誤読していた (閾値では救えない)。

実データ (data/real 和文 18 件) で「。」の後の間隔は 160ms〜1950ms。
文脈で判定できた 18 件のうち 15 件 (83%) が「デ」だった
(ノイウ。ス→デス、ホシイデスヘヘ。ハ→デハ、オリマセン。シタガ→デシタガ、
。キナクナツテ→デキナクナッテ 等)。
"""
from __future__ import annotations

from src.infer.line_break import render_committed
from src.infer.sliding_window import CommittedToken
from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import TOKEN_TO_ID

DANRAKU = "・-・-・・"
SU = "---・-"
NE = "--・-"
A = "--・--"
KO = "----"
HE = "・"
SO = "---・"
NI = "-・-・"
TA = "-・"
HA = "-・・・"
RU = "-・--・"
NU = "・・・・"

_SR = 8000
_GAP_S = 1.0                    # 運用者が設定する改行時間 (1 秒)
_GAP = int(_GAP_S * _SR)
_TOKEN = 400


def _tokens(*items: str) -> list[CommittedToken]:
    """符号を並べる。``"g<秒>"`` を挟むとその秒数の無音を空ける."""
    out: list[CommittedToken] = []
    pos = 0
    for item in items:
        if item.startswith("g"):
            pos += int(float(item[1:]) * _SR)
            continue
        out.append(CommittedToken(
            token_id=TOKEN_TO_ID[item], confidence=0.9,
            absolute_sample_start=pos, absolute_sample_end=pos + _TOKEN,
        ))
        pos += _TOKEN
    return out


def _render(*items: str, mode: str = "japanese", gap: int = _GAP,
            merge_he: bool = False) -> str:
    text, _ = render_committed(
        _tokens(*items), TokenConverter(mode), gap, initial_mode="japanese",
        merge_stray_dot=merge_he,
    )
    return text


class TestDanrakuBecomesDe:
    """改行時間ぶん空いていない「。」は「デ」の読み違い."""

    def test_immediately_followed(self) -> None:
        assert _render(DANRAKU, SU) == "デス"

    def test_mid_sentence(self) -> None:
        # 運用者が報告した「あつい。す」の形
        assert _render(A, DANRAKU, SU) == "アデス"

    def test_any_following_kana_not_just_su(self) -> None:
        """**「ス」に限らない。** 実データでは デハ / デシタ / デキ もあった."""
        assert _render(A, DANRAKU, KO) == "アデコ"

    def test_just_under_the_gap(self) -> None:
        assert _render(A, DANRAKU, "g0.9", SU) == "アデス"

    def test_several_in_one_line(self) -> None:
        assert _render(A, DANRAKU, SU, A, DANRAKU, NE) == "アデスアデネ"


class TestRealParagraphSurvives:
    """改行時間を越えた「。」は本物の段落として残す."""

    def test_gap_at_the_threshold(self) -> None:
        assert _render(A, DANRAKU, "g1.0", SU).startswith("ア。")

    def test_long_gap(self) -> None:
        assert _render(A, DANRAKU, "g1.5", SU).startswith("ア。")

    def test_trailing_danraku_is_kept(self) -> None:
        """末尾の「。」は次が無いので段落のまま."""
        assert _render(A, DANRAKU) == "ア。"

    def test_paragraph_still_breaks_the_line(self) -> None:
        """段落として残ったものは改行の印としても働き続ける."""
        assert "\n" in _render(A, DANRAKU, "g1.5", SU)


class TestGuardrails:
    def test_european_mode_is_untouched(self) -> None:
        """欧文表に「。」も「テ」も無い。欧文では触らない (壊すだけ)."""
        assert "デ" not in _render(DANRAKU, SU, mode="european")

    def test_other_tokens_are_untouched(self) -> None:
        assert _render(A, KO) == "アコ"

    def test_disabled_when_line_break_is_off(self) -> None:
        """改行を切っている (gap<=0) ときは触らない.

        判定の物差しが無いので、勝手に「。」を消さない方に倒す。
        (段落そのものによる改行は gap とは別に働くので、ここでは
         「デ」に変わっていないことだけを見る)
        """
        out = _render(A, DANRAKU, SU, gap=0)
        assert "デ" not in out
        assert "。" in out


# ---- 単独の「ヘ」を次の文字と合成する (2026-08-29 運用者の提案) ----
#
# 運用者:「癖のある打点は、短点(ヘ)とそれにつづく文字の組み合わせで 1 つの文字に
# なる。補正でヘが出てきたら、次の文字と合成してモールスコードを正規に補正したら
# どうか」。
#
# 実録音 20260829_153839 の測定:
#   * 「ヘ」の後の間隔は中央 4.8 短点、**その他の文字間は中央 8.2 短点**。
#     この局は文字間を広く取る癖があり、その中で「ヘ」の後だけ半分 =
#     文字間ではなく要素間である
#   * 「ヘ」+次の符号を繋ぐと **19/19 が正規の和文符号**になった
#   * 合成すると「センタクモノハコイランドリ」「コンナコトヲシ」と意味が通る
#     (合成前は「ヘソヘニタクモノハコヘムラヘニドリ」)


class TestMergeStrayHe:
    def test_he_plus_so_becomes_se(self) -> None:
        # ヘ(・) + ソ(---・) = ・---・ = セ
        assert _render(HE, SO, merge_he=True) == "セ"

    def test_he_plus_ni_becomes_n(self) -> None:
        # ヘ(・) + ニ(-・-・) = ・-・-・ = ン
        assert _render(HE, NI, merge_he=True) == "ン"

    def test_in_a_sentence(self) -> None:
        assert _render(HE, SO, HE, NI, TA, merge_he=True) == "センタ"

    def test_untouched_when_not_a_valid_code(self) -> None:
        """繋いでも和文表に無いなら触らない (68 文字中 34 文字が該当)."""
        # ヘ(・) + ル(-・--・) = ・-・--・ は表に無い
        assert _render(HE, RU, merge_he=True) == "ヘル"


class TestLaughterSurvives:
    """**「ヌヘヘ」は笑い (HI HI) なのでそのまま残す** (運用者の指示)."""

    def test_he_he_is_not_merged(self) -> None:
        # ヘ + ヘ = ・・ = 濁点。合成すると笑いが壊れる
        assert _render(NU, HE, HE, merge_he=True) == "ヌヘヘ"

    def test_bare_he_he(self) -> None:
        assert _render(HE, HE, merge_he=True) == "ヘヘ"

    def test_three_he(self) -> None:
        assert _render(HE, HE, HE, merge_he=True) == "ヘヘヘ"


class TestMergeIsOptOut:
    """**既定では合成しない。** 設定で選んだときだけ効く (運用者の指示)."""

    def test_off_by_default(self) -> None:
        assert _render(HE, SO) == "ヘソ"
