"""自動モード変換器のテスト (ホレ/ラタによる欧文⇄和文切替)."""
from src.tokens.converter import FALLBACK_CHAR, TokenConverter
from src.tokens.morse_tokens import TOKEN_TO_ID

HORE = TOKEN_TO_ID["-・・---"]   # 和文開始
RATA = TOKEN_TO_ID["・・・-・"]  # 和文終了 / 欧文 SN (同符号, id 共通)
A_I = TOKEN_TO_ID["・-"]         # 欧文 A / 和文 イ


def _conv(ids):
    return TokenConverter(mode="auto").convert(ids)


def test_starts_in_european():
    # 初期は欧文: ・- は A
    assert _conv([A_I]).text == "A"


def test_hore_switches_to_japanese():
    # ホレ以降は和文表: ・- は イ
    res = _conv([HORE, A_I])
    assert res.text == "[ホレ]イ"
    assert res.final_mode == "japanese"


def test_rata_switches_back_to_european():
    # ホレ→和文(イ)→ラタ→欧文(A)
    res = _conv([HORE, A_I, RATA, A_I])
    assert res.text == "[ホレ]イ[ラタ]A"
    assert res.final_mode == "european"


def test_rata_code_in_european_is_sn_no_switch():
    # 欧文中の ・・・-・ は SN 表示でモード切替しない
    res = _conv([RATA, A_I])
    assert res.text == "[SN]A"
    assert res.final_mode == "european"


def test_hore_idempotent_when_already_japanese():
    res = _conv([HORE, HORE, A_I])
    assert res.text == "[ホレ][ホレ]イ"
    assert res.final_mode == "japanese"


def test_initial_mode_continues_from_japanese():
    # 暫定引き継ぎ: 和文で開始すると ・- は イ
    res = TokenConverter(mode="auto").convert([A_I], initial_mode="japanese")
    assert res.text == "イ"
    assert res.final_mode == "japanese"


def test_dakuten_composes_only_in_japanese_segment():
    # 和文セグメント内: ハ(-・・・) + 濁点(・・) → バ
    ha = TOKEN_TO_ID["-・・・"]
    dak = TOKEN_TO_ID["・・"]
    res = _conv([HORE, ha, dak])
    assert res.text == "[ホレ]バ"


def test_fixed_european_unchanged_and_final_mode():
    res = TokenConverter(mode="european").convert([A_I])
    assert res.text == "A"
    assert res.final_mode == "european"


def test_fixed_japanese_unchanged_and_final_mode():
    res = TokenConverter(mode="japanese").convert([A_I])
    assert res.text == "イ"
    assert res.final_mode == "japanese"


def test_low_confidence_hore_does_not_switch():
    # 低確信度のホレは ? になり、モードを切替えない (ノイズでモード破壊を防ぐ)
    res = TokenConverter(mode="auto", confidence_threshold=0.8).convert(
        [HORE, A_I], confidences=[0.5, 1.0]
    )
    assert res.text == FALLBACK_CHAR + "A"
    assert res.final_mode == "european"


# --- プロサイン専用の確信度閾値 ---
#
# 実録音でホレの確信度が 0.42〜0.46 に集まり、既定の閾値 0.50 で棄却されて
# 「?」になり、自動モード切替が働かなかった (2026-08-04 実測)。
# ホレ/ラタは 1 トークンの誤りが後続の全文字の解釈を変えるため、
# 通常のトークンとは別の閾値で扱えるようにする。


def test_low_confidence_prosign_is_rejected_by_default():
    """既定 (prosign_threshold=None) では従来どおり confidence_threshold で判定する."""
    conv = TokenConverter(mode="auto", confidence_threshold=0.5)
    res = conv.convert([HORE, A_I], [0.45, 0.99])
    assert res.text == FALLBACK_CHAR + "A", "閾値未満のホレは ? になり、モードは欧文のまま"
    assert res.final_mode == "european"


def test_prosign_threshold_lets_low_confidence_hore_switch_mode():
    """プロサイン専用閾値を下げるとホレが通り、和文へ切り替わる."""
    conv = TokenConverter(mode="auto", confidence_threshold=0.5, prosign_threshold=0.4)
    res = conv.convert([HORE, A_I], [0.45, 0.99])
    assert res.text == "[ホレ]イ"
    assert res.final_mode == "japanese"


def test_prosign_threshold_applies_to_rata_too():
    conv = TokenConverter(mode="auto", confidence_threshold=0.5, prosign_threshold=0.4)
    res = conv.convert([HORE, A_I, RATA, A_I], [0.99, 0.99, 0.45, 0.99])
    assert res.text == "[ホレ]イ[ラタ]A"
    assert res.final_mode == "european"


def test_prosign_threshold_does_not_loosen_other_tokens():
    """プロサイン以外のトークンは confidence_threshold のまま判定される."""
    conv = TokenConverter(mode="auto", confidence_threshold=0.5, prosign_threshold=0.1)
    res = conv.convert([HORE, A_I], [0.99, 0.45])
    assert res.text == f"[ホレ]{FALLBACK_CHAR}", "イ は 0.45 < 0.5 なので読めなかった印のまま"


def test_prosign_below_its_own_threshold_is_still_rejected():
    conv = TokenConverter(mode="auto", confidence_threshold=0.5, prosign_threshold=0.4)
    res = conv.convert([HORE, A_I], [0.35, 0.99])
    assert res.text == FALLBACK_CHAR + "A"
    assert res.final_mode == "european"


def test_prosign_threshold_ignored_in_fixed_mode():
    """固定モードではモード切替が無いので、プロサインも通常トークンとして扱う."""
    conv = TokenConverter(
        mode="japanese", confidence_threshold=0.5, prosign_threshold=0.1
    )
    res = conv.convert([HORE], [0.45])
    assert res.text == FALLBACK_CHAR, "固定和文では ・・--- は表に無く、低確信度なら ?"


def test_prosign_threshold_rejects_out_of_range():
    import pytest

    with pytest.raises(ValueError):
        TokenConverter(mode="auto", prosign_threshold=1.5)
    with pytest.raises(ValueError):
        TokenConverter(mode="auto", prosign_threshold=-0.1)


# --- 欧文表に無い符号を見たら和文へ切り替える ---
#
# 自動切替は実質的に和文のためにあるが、ホレ 1 個の検出に賭けるのは脆い
# (実運用でホレが拾えず切り替わらない事例が発生, 2026-08-04)。
# 和文にしかない符号は 23 種あり、コ・ソ・ロ・ノ・ス・ア・シ・ヒ 等の高頻度カナを
# 含むため、和文の送信なら数文字で必ず当たる。しかもこれらは欧文モードでは
# TABLE_MISS で「?」になっており、切り替えたほうが確実に良くなる。

KO = TOKEN_TO_ID["----"]     # 和文 コ (欧文表に無い)
RO = TOKEN_TO_ID["・-・-"]    # 和文 ロ (欧文表に無い)
SK = TOKEN_TO_ID["・・・-・-"]  # 欧文 [SK] (和文表に無い)


def test_japanese_only_code_switches_to_japanese():
    """欧文モード中に和文専用符号が来たら和文へ切り替え、その文字も和文で出す."""
    res = TokenConverter(mode="auto").convert([KO, A_I])
    assert res.text == "コイ", "コ で切替、続く ・- は イ"
    assert res.final_mode == "japanese"


def test_japanese_only_code_without_switch_is_table_miss():
    """無効化すると従来どおり ? になる."""
    conv = TokenConverter(mode="auto", switch_on_japanese_only=False)
    res = conv.convert([KO, A_I])
    assert res.text == FALLBACK_CHAR + "A"
    assert res.final_mode == "european"


def test_switch_happens_only_when_european_table_misses():
    """欧文表にある符号では切り替えない (・- は A のまま)."""
    res = TokenConverter(mode="auto").convert([A_I, A_I])
    assert res.text == "AA"
    assert res.final_mode == "european"


def test_low_confidence_japanese_only_code_does_not_switch():
    """閾値を満たさない符号では切り替えない (誤検出でモードが飛ぶのを防ぐ)."""
    conv = TokenConverter(mode="auto", confidence_threshold=0.5)
    res = conv.convert([KO, A_I], [0.3, 0.99])
    assert res.text == FALLBACK_CHAR + "A"
    assert res.final_mode == "european"


def test_rata_still_returns_to_european_after_auto_switch():
    """和文専用符号で入った場合も、ラタで欧文へ戻れる."""
    res = TokenConverter(mode="auto").convert([RO, RATA, A_I])
    assert res.text == "ロ[ラタ]A"
    assert res.final_mode == "european"


def test_european_only_code_stays_european():
    """欧文にしかない符号 ([SK] 等) では和文へ行かない."""
    res = TokenConverter(mode="auto").convert([SK, A_I])
    assert res.text == "[SK]A"
    assert res.final_mode == "european"


def test_switch_is_ignored_in_fixed_mode():
    """固定欧文モードでは切り替えない (従来どおり ?)."""
    res = TokenConverter(mode="european").convert([KO])
    assert res.text == FALLBACK_CHAR
