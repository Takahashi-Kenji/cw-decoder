"""送信テキスト → 符号列 → 打鍵の要素列.

和文と欧文が混ざる
------------------
**和文の交信でも、コールサインと RST は欧文で送るのが作法である。** 和文の本文だけを
``{HORE}`` … ``{RATA}`` で囲み、その外は欧文で送る。

    JH0ILL DE JA1ABC {HORE}コンニチハ、テンキ ハ ハレ{RATA} K

``text_to_codes`` はモードを 1 つしか取らないので、ここでマーカーを見て切り替える。
外は欧文表、内は和文表。**マーカー自体も符号として送る** (受信側はこれでモードを
切り替える)。

検証を符号化の前に置く
----------------------
``text_to_codes`` は表に無い文字で ``KeyError`` を投げる (``char_to_code[ch]`` の
直接引き)。落ちてから慌てるのではなく、**先に検証して、送れない文字を位置つきで
返す**。呼び出し側はそれを赤く出して送信を止める。

要素列は合成器と共有する
------------------------
符号列から「いつキーを ON/OFF するか」を作るのは ``build_element_sequence``
(``src/synth/keying.py``)。**合成器と送信で同じものを使う。** 片方だけに実装を
持つと必ず食い違い、「学習データの音」と「実際に出る電波」がずれる
(そのずれは、自分が送った符号を自分のデコーダが読めない、という形で現れる)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.synth.keying import build_element_sequence
from src.tokens.morse_tokens import (
    EUROPEAN_CHAR_TO_CODE,
    JAPANESE_CHAR_TO_CODES,
    SPECIAL_INPUT_MARKERS,
    TX_INPUT_MARKERS,
    TX_ONLY_EUROPEAN_CHAR_TO_CODE,
    WORD_BREAK_CODE,
    text_to_codes,
)
from src.tx.reading import BadChar

HORE = "{HORE}"
RATA = "{RATA}"

# 欧文区間の印。**和文の中に短い欧文を打つときに使う。**
# `「` `」` はどちらもこの語彙に符号を持たないので、文字としては出せない
# (docs/superpowers/specs/2026-08-12-european-span-design.md)。
SPAN_OPEN = "「"
SPAN_CLOSE = "」"

# 段落。**和文の中で区間を閉じるときに実際に電波へ出るのはこれ。**
# 以前は `」` を出しているつもりだったが、`・-・-・・` を `」` と誤記していた
# だけで、**電波に出ていたのは最初から段落だった** (2026-08-12 に運用者が発見)。
# 表記を直しても電波は変わらない。運用者は引き続き `「…」` と書けばよい。
#
# 本物の上向き括弧 `」` は `・-・・-・` だが、足すとトークン ID がずれて
# 再学習が要る (`src/tokens/morse_tokens.py` の注記)。**区切りとしては段落で
# 用が足りる**ので、当面はこれで出す。
DANRAKU = "。"

# 送信テキストの中の `」` は**段落を意味する**。区間の閉じでも、
# `コンニチハ」` のように単独で書かれたときでも同じ。
# **読み替えは符号化の直前だけで行う。** `split_segments` の段階で書き換えると
# ``find_unsendable`` の位置解決 (``text.index(segment.text, offset)``) が壊れる。
_JAPANESE_TX_CHARS: frozenset[str] = frozenset(JAPANESE_CHAR_TO_CODES) | {SPAN_CLOSE}

# 欧文の送信可能文字。**符号表を書き写さない** (アーキテクチャ原則 2)。受信語彙の
# EUROPEAN_CHAR_TO_CODE に、送信専用の記号表 (TX_ONLY_EUROPEAN_CHAR_TO_CODE) を足す。
# 受信は従来どおり EUROPEAN_CHAR_TO_CODE だけなので、これらの符号を受けたときは
# ? (TABLE_MISS) になる (RX 側は変更しない)。
_EUROPEAN_TX_CHARS: frozenset[str] = frozenset(EUROPEAN_CHAR_TO_CODE) | frozenset(
    TX_ONLY_EUROPEAN_CHAR_TO_CODE
)


def _as_danraku(text: str) -> str:
    """符号化の直前に `」` を段落へ読み替える (1 文字 → 1 文字)."""
    return text.replace(SPAN_CLOSE, DANRAKU)


# 送信側は {KAKKO}/{TOJI} (送信専用の括弧マーカー) も通す。受信語彙には無いので
# SPECIAL_INPUT_MARKERS だけの合成器・受信側には影響しない。
_MARKER_RE = re.compile("|".join(re.escape(m) for m in TX_INPUT_MARKERS))

# 和文表にしか無い文字 (カタカナ・句読点・濁点等)。モード推定の手掛かりにする。
# **符号表を書き写さず、そこから作る** (アーキテクチャ原則 2)。数字や `-` のように
# 両方の表にある文字は入らないので、曖昧なものでモードが決まることはない。
_JAPANESE_ONLY_RE = re.compile(
    "[" + re.escape("".join(sorted(_JAPANESE_TX_CHARS - set(EUROPEAN_CHAR_TO_CODE)))) + "]"
)

# 対になった欧文区間 (``「…」``、次の ``」`` までを 1 区間とする。split_segments と同じ解釈)。
# `_initial_mode` が中身を見るときに取り除く。区間の中身は常に欧文であり、
# 閉じ括弧 `」` も区間の印にすぎないので、周りのモードの手掛かりにしてはいけない
# (取り除かないと `「FT991」` の `」` を和文にしか無い文字と誤認し、
# 前後が欧文でも和文開始と誤判定していた)。
_SPAN_RE = re.compile(re.escape(SPAN_OPEN) + "[^" + re.escape(SPAN_CLOSE) + "]*" + re.escape(SPAN_CLOSE))

# 教科書どおりの間隔。送信は機械的に正確であるべきなのでジッタはかけない。
DASH_DOT_RATIO = 3.0
INTER_CHAR_SPACE_UNITS = 3.0
INTER_WORD_SPACE_UNITS = 7.0
INTRA_ELEMENT_SPACE_UNITS = 1.0


@dataclass(frozen=True)
class Segment:
    """同じモードで送る一続き."""

    text: str
    mode: str          # "european" | "japanese"


@dataclass(frozen=True)
class ElementSequence:
    """打鍵の要素列."""

    durations: np.ndarray      # 各要素の長さ (秒)
    is_on: np.ndarray          # 各要素のキー状態
    # 元の符号列 (・-)。打鍵には使わないが、ログと不具合調査で
    # 「何を送ったのか」を要素列より人が読める形で残せるようにしている
    codes: tuple[str, ...]

    @property
    def total_seconds(self) -> float:
        return float(self.durations.sum()) if self.durations.size else 0.0


def _mode_evidence(text: str) -> str:
    """モード判定の手掛かりを探すために、対になった ``「…」`` を中身ごと取り除く.

    区間の中身は常に欧文であり、閉じ括弧 ``」`` も区間の印にすぎない。
    取り除かないと ``」`` を「和文にしか無い文字」と誤認し、``「FT991」`` の
    ような欧文だけの文でも和文と判定してしまう (中身の妥当性は
    ``find_unsendable`` が別途見る)。

    **``_initial_mode`` と ``needs_japanese_wrap`` の両方がここを通ること。**
    以前はそれぞれが別々に前処理していて ``needs_japanese_wrap`` だけこの
    除去を欠いており、``「FT991」`` だけの本文にホレ・ラタが静かに付く退行を
    生んだ (2026-08-12 の最終レビューで指摘)。判定そのもの
    (``_JAPANESE_ONLY_RE``) は 1 か所のままなので、ここでも書き写さない。
    """
    return _SPAN_RE.sub("", text)


def _initial_mode(text: str) -> str:
    """最初のマーカーより前の中身から、始まりのモードを決める.

    **モードは「どちらの符号表で読むか」であり、ホレ/ラタを送るかとは別物である。**
    以前は ``{HORE}`` を見つけるまで欧文のままだったので、囲みを外すとカタカナが
    まるごと「送れない文字」になっていた (2026-08-11 に実運用で判明)。
    符号は作れるのだから送れてよい、というのが運用者の判断である。

    **曖昧なものでモードを決めない。** 数字・``-``・``?``・``@`` は両方の表に
    あるので手掛かりにしない。和文にしか無い文字 (カタカナ・``、``・``」``・
    濁点等) があるときだけ和文で始める。

    これで「和文の途中から書き始め ``{RATA}`` で欧文に戻す」書き方も通る。

    始まりの中身にある ``「…」`` は :func:`_mode_evidence` で取り除いてから見る
    (``needs_japanese_wrap`` と共通)。
    """
    head = text
    for marker in (HORE, RATA):
        position = head.find(marker)
        if position >= 0:
            head = head[:position]
    head = _mode_evidence(head)
    return "japanese" if any(_JAPANESE_ONLY_RE.match(ch) for ch in head) else "european"


def split_segments(text: str) -> list[Segment]:
    """``{HORE}`` … ``{RATA}`` と ``「…」`` を境にモードを切り替えて刻む.

    マーカーは**それが属する側**の segment に入れる。``{HORE}`` は欧文側の
    終わりに置く (欧文の符号として送られ、受信側が和文へ切り替える)。

    ``「…」`` は**欧文区間**である。運用者は和文の中に短い欧文 (リグ名など) を
    打つときこう書き、**ホレ・ラタは使わない** (2026-08-12 の聞き取り)。

    * ``「`` は出さない (どちらの符号表にも無い)
    * 中身は欧文の符号表で送る
    * ``」`` は**周りが和文なら和文の符号として出す** (区切りとして届く)。
      **周りが欧文なら落とす** (``」`` は和文表にしかないため)
    * **対になっているときだけ印として扱う。** 単独の ``」`` は従来どおり
      和文の終わりであり、閉じていない ``「`` は印にしない (書き間違いが
      「送信できない文字」として見える)

    始まりのモードは中身から決める (:func:`_initial_mode`)。
    """
    segments: list[Segment] = []
    mode = _initial_mode(text)
    buffer = ""
    index = 0
    while index < len(text):
        if text.startswith(HORE, index):
            buffer += HORE
            segments.append(Segment(buffer, mode))
            buffer, mode = "", "japanese"
            index += len(HORE)
        elif text.startswith(RATA, index):
            buffer += RATA
            segments.append(Segment(buffer, mode))
            buffer, mode = "", "european"
            index += len(RATA)
        elif text.startswith(SPAN_OPEN, index):
            close = text.find(SPAN_CLOSE, index + len(SPAN_OPEN))
            inner = text[index + len(SPAN_OPEN) : close] if close >= 0 else ""
            if close < 0 or any(marker in inner for marker in SPECIAL_INPUT_MARKERS):
                # 閉じていない、または区間の中に {HORE}/{RATA} 等のマーカーが
                # またがっている。**どちらも印として扱わない。** ホレ・ラタは
                # 区間の中では使わない書き方であり (規則 5)、マーカーが混じって
                # いるのは書き間違いである。`「` をそのまま buffer へ積むので、
                # 符号化のときに「送信できない文字」として見える
                buffer += text[index]
                index += len(SPAN_OPEN)
                continue
            # 区間の前までを今のモードで確定させる
            if buffer:
                segments.append(Segment(buffer, mode))
                buffer = ""
            # ``「`` ``」`` のすぐ内側の空白は**飾りとして書かれた余白**であり、
            # 語間ではない。取り除かないと、``「`` 自体は符号を出さないので
            # 直前の語間と重なって二重にカウントされる
            # (``ア 「 A 」 イ`` で実際に踏んだ。実運用の UI は必ず
            # ``to_sendable_kana`` を経由するので届かないが、``key_server`` に
            # 直接テキストが来る経路では届く。2026-08-12 の最終レビューで判明)。
            # 区間の前後にある本物の語間 (``RIG 「FT991」ANT`` の間など) は
            # buffer 側の末尾空白・次の segment の先頭空白として別に扱われるので、
            # ここで削っても消えない。
            segments.append(Segment(inner.strip(), "european"))
            # **区間の閉じは周りが和文のときだけ出す。** 欧文の中では落とす
            # (段落は和文表にしか無いので、残すと送れなくなる)。
            # **ここで `。` に書き換えないこと。** ``find_unsendable`` は
            # segment が元テキストの部分文字列であることを前提に位置を求め直す。
            # 段落への読み替えは ``_as_danraku`` が符号化の直前に行う。
            if mode == "japanese":
                segments.append(Segment(SPAN_CLOSE, "japanese"))
            index = close + len(SPAN_CLOSE)
        else:
            buffer += text[index]
            index += 1
    if buffer:
        segments.append(Segment(buffer, mode))
    # **空文字列の segment だけを落とす。空白だけの segment は残す。**
    # 空文字列は空の区間 (``「」``) の inner から生じる、意味の無いものである。
    # 一方、空白だけの segment は連続する区間の間の語間 ("「A」 「B」" の
    # 間の 1 文字) であり、これを ``.strip()`` で落とすと語間ごと消えてしまう
    # (2026-08-12 のレビューで判明。encode() 側の語間補完は segments 配列の
    # 要素しか見ないので、ここで落ちると復元できない)。
    return [s for s in segments if s.text]


def find_unsendable(text: str) -> tuple[BadChar, ...]:
    """送信できない文字を位置つきで返す (モードを見て判定する).

    **符号表を書き写さない。** ``EUROPEAN_CHAR_TO_CODE`` と
    ``JAPANESE_CHAR_TO_CODES`` から判定する (アーキテクチャ原則 2)。
    """
    bad: list[BadChar] = []
    offset = 0
    for segment in split_segments(text):
        # segment の開始位置を元テキスト上で求め直す (マーカーを含むため)
        start = text.index(segment.text, offset)
        offset = start + len(segment.text)
        # 和文側は `」` を段落の別名として通す (``_JAPANESE_TX_CHARS``)。
        # 欧文側は送信専用の記号 (``TX_ONLY_EUROPEAN_CHAR_TO_CODE``) も通す
        # (``_EUROPEAN_TX_CHARS``)。
        table = _EUROPEAN_TX_CHARS if segment.mode == "european" else _JAPANESE_TX_CHARS
        i = 0
        while i < len(segment.text):
            marker = _MARKER_RE.match(segment.text, i)
            if marker:
                i = marker.end()
                continue
            ch = segment.text[i]
            probe = ch.upper() if segment.mode == "european" else ch
            if not ch.isspace() and probe not in table:
                bad.append(BadChar(index=start + i, char=ch))
            i += 1
    return tuple(bad)


def encode(text: str) -> list[str]:
    """送信テキストを符号列にする.

    Raises:
        ValueError: 送信できない文字が含まれるとき。
            **呼び出し側は先に ``find_unsendable`` で確認すること。**
    """
    bad = find_unsendable(text)
    if bad:
        chars = "".join(b.char for b in bad)
        raise ValueError(f"送信できない文字が含まれています: {chars!r}")
    segments = split_segments(text)
    codes: list[str] = []
    for i, segment in enumerate(segments):
        codes.extend(text_to_codes(  # type: ignore[arg-type]
            _as_danraku(segment.text), segment.mode, include_tx_only=True
        ))
        # ``「…」`` の直前など、segment が空白で終わるとき、text_to_codes は
        # その末尾の WORD_BREAK を「意味なし」として削ってしまう (符号表側の
        # 既存仕様。1 segment だけを見ているので、後ろに続きがあることを
        # 知らない)。次の segment が続くならその語間は本物なので打ち直す
        # (``RIG 「FT991」`` で実際に踏んだ。segment.text を書き換えて直すと
        # find_unsendable の位置解決が壊れるので、ここで補う)。
        if (
            i + 1 < len(segments)
            and segment.text.endswith(" ")
            and (not codes or codes[-1] != WORD_BREAK_CODE)
        ):
            codes.append(WORD_BREAK_CODE)
    # **打ち直した WORD_BREAK が、次の segment 自身の先頭空白による WORD_BREAK と
    # 重なることがある。** 上の打ち直しは「次の segment が始まったときに何を
    # 出すか」まで知らないので、次の segment が空白で始まる (``「 A 」`` の
    # ように ``「`` の直後に空白を書いた場合など) と WORD_BREAK が連続する。
    # text_to_codes は 1 segment の中でしか連続する空白を畳まないので
    # (符号表側の既存仕様)、segment をまたぐ重複はここで畳む
    # (``ア 「 A 」 イ`` で実際に踏んだ。2026-08-12 の最終レビューで判明)。
    deduped: list[str] = []
    for code in codes:
        if code == WORD_BREAK_CODE and deduped and deduped[-1] == WORD_BREAK_CODE:
            continue
        deduped.append(code)
    return deduped


def build_sequence(text: str, wpm: float) -> ElementSequence:
    """送信テキストから打鍵の要素列を作る.

    Raises:
        ValueError: 送信できない文字があるとき、または ``wpm`` が 0 以下のとき。
    """
    if wpm <= 0:
        raise ValueError(f"wpm must be > 0, got {wpm}")
    codes = encode(text)
    if not codes:
        return ElementSequence(
            durations=np.zeros(0, dtype=np.float64),
            is_on=np.zeros(0, dtype=bool),
            codes=(),
        )
    durations, is_on, _, _ = build_element_sequence(
        codes,
        word_break_after=(),          # WORD_BREAK は codes に入っている
        dot_sec=1.2 / wpm,
        dash_dot_ratio=DASH_DOT_RATIO,
        inter_char_space_units=INTER_CHAR_SPACE_UNITS,
        inter_word_space_units=INTER_WORD_SPACE_UNITS,
        intra_element_space_units=INTRA_ELEMENT_SPACE_UNITS,
    )
    return ElementSequence(durations=durations, is_on=is_on, codes=tuple(codes))


def needs_japanese_wrap(text: str) -> bool:
    """その本文は ``{HORE}`` … ``{RATA}`` で囲む必要があるか (中身で決める).

    **囲むかどうかは「型が和文と名乗っているか」ではなく「中身が和文か」で
    決めなければならない。** 型の ``mode`` で決めていたため、``any`` の型に
    和文の本文を書くと囲みが外れた。符号としては和文の符号が並ぶので警告も
    出ないが、**受信側はモードを切り替えられず化けたまま届く** — この機能で
    唯一の「警告なしで誤った電波が出る」経路だった (2026-08-11 レビュー I5)。

    判定は :func:`_initial_mode` と同じ ``_JAPANESE_ONLY_RE`` を、同じ
    :func:`_mode_evidence` (対になった ``「…」`` を中身ごと取り除く) を通してから
    使う (符号表を書き写さない。アーキテクチャ原則 2)。数字・``-``・``?`` の
    ように両方の表にある文字ではモードを決めない。

    **``「…」`` を取り除かずに判定すると、``「FT991」`` だけの本文でも
    ``」`` を和文の手掛かりと誤認し、ホレ・ラタが静かに付く。** 受信側は
    ホレで和文モードに入ってから欧文の符号を受け取ることになり、警告も
    出ないまま化けた電波が出る (2026-08-12 の最終レビューで判明した退行。
    `_initial_mode` はこの除去をしていたが、ここだけ欠けていた)。

    既に ``{HORE}`` があるなら型が自分で囲んでいるので偽を返す
    (``wrap_japanese`` も二重には囲まないが、画面のチェックボックスの状態を
    「囲む」にしておくと、その後の編集で意味が食い違う)。
    """
    if HORE in text:
        return False
    return bool(_JAPANESE_ONLY_RE.search(_mode_evidence(text)))


def wrap_japanese(text: str) -> str:
    """和文の本文を ``{HORE}`` … ``{RATA}`` で囲む.

    既に囲まれていれば何もしない。
    """
    stripped = text.strip()
    if not stripped or HORE in stripped:
        return text
    return f"{HORE}{stripped}{RATA}"


__all__ = [
    "DANRAKU",
    "HORE",
    "RATA",
    "SPAN_CLOSE",
    "SPAN_OPEN",
    "ElementSequence",
    "Segment",
    "build_sequence",
    "encode",
    "find_unsendable",
    "needs_japanese_wrap",
    "split_segments",
    "wrap_japanese",
]
