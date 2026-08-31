"""受信テキストから返信に使う欄を拾う.

拾えなかったら空のままにする
----------------------------
**当てにならないものを黙って埋めない。** 運用者が気づかないまま誤った電波が
出るほうが、欄が空で手で打つより悪い。

相手の RST は拾わない
---------------------
返信に書くのは「**こちらが相手に与える RST**」であって、相手からもらった値では
ない。混同すると事故になるので、そもそも拾わない (設計書 §4.1)。

入力は清書済みテキスト
----------------------
生のデコードより誤りが減っている。清書結果には推測箇所の ``⟦…⟧`` が入っている
ので、:func:`strip_guess_marks` で外してから拾う。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 清書が付ける推測箇所のマーカー (src/llm/markup.py の OPEN_MARK / CLOSE_MARK)。
_GUESS_MARKS = str.maketrans("", "", "⟦⟧")

# アマチュア無線のコールサインのおおまかな形 (JA1ABC / JH0ILL / 7K1ABC など)。
# 移動運用局は "/1" "/P" のようなスラッシュ付き符号が続くことがあるので、
# 任意のスラッシュ接尾辞も許す (JI1ABC/1 など)。これが無いと DE の次に来た
# 移動局コールをまるごと取りこぼし、fullmatch が失敗して their_call が
# 空になってしまう。
# **厳密な判定はしない。** 拾えなければ空にするので、緩くても害が小さい。
_CALL_RE = re.compile(r"\b[A-Z0-9]{1,3}[0-9][A-Z]{1,4}(?:/[A-Z0-9]{1,4})?\b")

# 名前の手掛かり。**当てにならないので、手掛かりが無ければ拾わない。**
_NAME_RES = (
    re.compile(r"\bNAME\s+([A-Z0-9]+)\b"),
    re.compile(r"\bOP\s+([A-Z0-9]+)\b"),
    re.compile(r"ナマエ\s*ハ\s*([ァ-ヴー]+)"),
)

# NAME / OP の次に来ても名前ではない語 (Qコード・略語・プロサインなど)。
# 例えば "OP QRT" は「運用者は終了する」の意味で、QRT は名前ではない。
# ここに載っている語を拾ってしまったときは名前無しとして次の手掛かりを探す
# (誤ったコールサインが電波に出るのと同じ理屈で、誤った名前を出すよりましなため)。
_NAME_STOPWORDS = frozenset(
    {
        "QRT", "QRZ", "QRM", "QRN", "QRP", "QRO", "QSA", "QSB", "QSL", "QSO", "QSY", "QTH",
        "RST", "UR", "DE", "HR", "HERE", "ES", "K", "KN", "AR", "SK", "BK", "TU",
        "GL", "GA", "GM", "GE", "GN", "CUL", "HW", "VY", "FB", "TNX", "PSE", "NW",
        "AGN", "WX", "RIG", "ANT", "PWR", "NAME", "CALL",
    }
)


@dataclass(frozen=True)
class QsoFields:
    """受信テキストから拾えた欄. **拾えなかったものは空文字。**"""

    their_call: str = ""
    their_name: str = ""


#: RST に使う略号数字 (cut numbers)。CW では数字を短い符号の文字で代える習慣が
#: あり、RST では ``5NN`` = ``599``、``57N`` = ``579`` のように **9 を N と打つ**
#: のが日常的である。
#:
#: **N=9 だけにしてある** (2026-08-30 の運用者の指示「N があれば 9 にしてもらえれば
#: いい」)。理由は 2 つある:
#:
#: * **RST に 0 は現れない** (R は 1〜5、S と T は 1〜9)。``T``=0 を入れても
#:   RST では使い道が無く、危険なだけである
#: * 文字が数字に化ける規則なので、**広げるほど語を RST と取り違えやすくなる**
#:   (``TU`` = ありがとう を ``02`` にしては事故になる)
#:
#: 増やすときは、その文字が RST の位置に本当に現れるかを確かめること。
CUT_NUMBERS: dict[str, str] = {"N": "9"}

_WORD_RE = re.compile(r"[0-9A-Z]+")


def _as_rst(token: str) -> str:
    """その塊が RST として読めるなら数字に直して返す. 読めなければ空文字.

    **数字を 1 つも含まない塊は RST にしない。** ``TU`` (ありがとう) は
    T も U も略号数字だが、これを ``02`` にしては事故になる。
    """
    if not 2 <= len(token) <= 3:
        return ""
    if not any(ch.isdigit() for ch in token):
        return ""
    if any(not ch.isdigit() and ch not in CUT_NUMBERS for ch in token):
        return ""
    return "".join(CUT_NUMBERS.get(ch, ch) for ch in token)


def normalise_rst(text: str) -> str:
    """RST の欄に入れる形にする. **略号数字を数字に直す** (``5NN`` → ``599``).

    受信画面や欧文ストリームから選んだ文字をそのまま渡せるように、前後の語が
    付いていても RST らしい塊を探す (``UR 5NN K`` → ``599``)。

    **語間が出ないことがある** (実運用でよくある) ので、``UR5NN`` のように
    語と RST がくっついた塊は末尾 3 文字・2 文字も試す。

    **末尾を試すのは「RST に使えない文字を含む塊」だけ。** ``5NN9`` のように
    全部が RST に使える文字なのに長さが合わないものは、どこを切っても当て推量に
    なる (末尾 3 文字なら ``999``)。**当てにならないものを黙って埋めない** —
    このモジュールの方針である。RST らしい塊が無ければ空文字を返す。
    """
    for token in _WORD_RE.findall(text.upper()):
        found = _as_rst(token)
        if found:
            return found
        # 語と RST がくっついた塊か (RST に使えない文字を含むか) を見てから切る
        if any(not ch.isdigit() and ch not in CUT_NUMBERS for ch in token):
            for candidate in (token[-3:], token[-2:]):
                found = _as_rst(candidate)
                if found:
                    return found
    return ""


def strip_guess_marks(text: str) -> str:
    """清書が付けた推測箇所のマーカー ``⟦…⟧`` を外す."""
    return text.translate(_GUESS_MARKS)


def _their_call(text: str, my_call: str) -> str:
    """相手のコールを拾う.

    **``DE`` を手掛かりにする。** CW は ``<相手> DE <自分>`` の順で送るので、
    ``DE`` の次に来るのが送信者である。一度の送信に ``DE`` が複数出ることが
    あるので**最後のものを採る** (今の送信者がそれ)。``DE`` が最後の語で
    次に何も無い場合は、走査範囲がそこまで届かないので自然に見つからない
    (例外にはならない)。

    ``DE`` が無いときは、コールの形をした語のうち**自局でないほう**を採る。
    自局が分からなければ**どちらが相手か決められないので空にする**。
    """
    tokens = text.split()
    for index in range(len(tokens) - 1, 0, -1):
        if tokens[index - 1] == "DE":
            candidate = tokens[index]
            if _CALL_RE.fullmatch(candidate):
                return candidate
            return ""
    if not my_call:
        return ""
    for token in tokens:
        if _CALL_RE.fullmatch(token) and token != my_call:
            return token
    return ""


def _their_name(text: str) -> str:
    """相手の名前を拾う.

    ``NAME`` / ``OP`` / 和文「ナマエハ」を手掛かりにする。
    **当てにならないので、手掛かりが無ければ拾わない。**
    ``OP`` の直後にはQコード等が来ることもある (``OP QRT`` など) ので、
    それらしくない語 (:data:`_NAME_STOPWORDS`) を拾ってしまったときは、
    同じ手掛かりの中で次の出現を探し、それでも無ければ次の手掛かりに移る。
    """
    for pattern in _NAME_RES:
        for match in pattern.finditer(text):
            candidate = match.group(1)
            if candidate not in _NAME_STOPWORDS:
                return candidate
    return ""


def extract_fields(text: str, my_call: str = "") -> QsoFields:
    """受信テキストから欄を拾う.

    Args:
        text: 清書済みテキスト (``⟦…⟧`` が入っていてよい)。
        my_call: 自局のコール。``DE`` が無いときの手掛かりに使う。
    """
    cleaned = strip_guess_marks(text).upper()
    return QsoFields(
        their_call=_their_call(cleaned, my_call.upper()),
        their_name=_their_name(cleaned),
    )


__all__ = ["QsoFields", "extract_fields", "strip_guess_marks"]
