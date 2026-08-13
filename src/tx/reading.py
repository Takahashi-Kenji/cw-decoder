"""日本語 → 送信可能なカタカナへの変換.

なぜ変換が要るか
----------------
和文モールスの符号表にあるのは**カタカナ・数字・``、``・``」``・``?``・``-``・``@``・
長音 ``ー``・濁点 ``゛``・半濁点 ``゜`` だけ** (92 文字)。漢字もひらがなも
小書き文字 (ャュョッェ) も無い。

読みやすい日本語で編集し、送信直前にカタカナへ変換する。

変換の順序
----------
1. **読み辞書** — 運用者が育てる ``語 → 読み``。最優先
2. **経歴の読み** — 名前・QTH 等、経歴の欄に併記された読み
3. **pykakasi** — 残りを機械変換
4. **小書き → 大書き**
5. **句読点の対応**

**固有名詞の誤読が最も痛い。** 1 と 2 が先に来るのはそのためである。

黙って落とさない
----------------
符号表に無い文字は**位置つきで返す**。呼び出し側はそれを赤く出し、送信を止める。
落として送ると、相手には意味の通らない符号が届く。

``text_to_codes`` は表に無い文字で ``KeyError`` を投げる (``char_to_codes_ja[ch]``
の直接引き)。**検証は必ず符号化の前に行うこと。**
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.tokens.morse_tokens import JAPANESE_CHAR_TO_CODES, TX_INPUT_MARKERS
from src.tx.profile import OperatorProfile

# 小書き文字は和文モールスに無いので大書きへ倒す。
# 「キヤク」と「キャク」は符号上どちらも同じであり、これは仕様である。
SMALL_KANA_MAP: dict[str, str] = {
    "ャ": "ヤ", "ュ": "ユ", "ョ": "ヨ", "ッ": "ツ", "ヮ": "ワ",
    "ァ": "ア", "ィ": "イ", "ゥ": "ウ", "ェ": "エ", "ォ": "オ",
    "ヵ": "カ", "ヶ": "ケ",
}

# 和文の区切りは ``、`` (区切点)、文の終わりは ``。`` (段落)。
# **``。`` は倒さない。** 以前は「符号表に無い」として ``、`` に倒していたが、
# それは ``・-・-・・`` を ``」`` と誤記していたためで、段落は元から符号表にある
# (2026-08-12 に運用者が発見)。``．`` も ``。`` に倒す (同じ意味の全角記号)。
# **`「` は落とさない。** `「…」` は「ここは欧文で打つ」という印であり
# (docs/superpowers/specs/2026-08-12-european-span-design.md)、区間として
# 解釈するのは encoder.split_segments の仕事である。ここで消すと印が
# 届く前に無くなる。`」` は元から残している (和文の終わりでもあるため)。
# **`『』` (全角二重括弧) は `「」` に倒す。** IME で二重括弧を打つ運用者もいる。
# 以前は `』` だけ `」` に倒し `『` は空文字に落としていたため非対称で、
# `『FT991』` が `FT991」` になり `FT` が「送信できない文字」として弾かれていた
# (2026-08-12 の最終レビューで指摘)。
PUNCTUATION_MAP: dict[str, str] = {
    "．": "。", "，": "、", "！": "、", "!": "、",
    "・": "、", "；": "、", ";": "、", "：": "、", ":": "、",
    "？": "?", "『": "「", "』": "」",
    "（": "", "）": "", "(": "", ")": "",
    "　": " ",          # 全角スペースは語間へ
    "〜": "-", "～": "-", "―": "-", "—": "-",
}

# 送信可能な文字の集合。**符号表を書き写さず、そこから作る** (原則 2)。
# `「` `」` はどちらも符号表に無いが、`「…」` の印として通す (判定は encoder が
# 行う)。**`」` を足したのは 2026-08-12。** それまでは `・-・-・・` を `」` と
# 誤記していたおかげで偶然ここを通っていた。段落を `。` に直した以上、
# `」` は明示的に通さないと `「FT991」` の閉じが送れなくなる。
_SENDABLE: frozenset[str] = frozenset(JAPANESE_CHAR_TO_CODES) | frozenset(" 「」")

# {KAKKO}/{TOJI} (送信専用の括弧マーカー) も素通りさせる。変換自体はこの層の
# 責務ではなく、符号化 (encoder.encode) がまとめて解釈する。
_MARKER_RE = re.compile("|".join(re.escape(m) for m in TX_INPUT_MARKERS))


@dataclass(frozen=True)
class BadChar:
    """送信できない文字と、その位置 (変換後テキスト上)."""

    index: int
    char: str


@dataclass
class ConversionResult:
    """変換結果."""

    text: str
    bad_chars: tuple[BadChar, ...] = ()
    # どの語が辞書で当たったか (画面で根拠を示すため)
    dictionary_hits: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def sendable(self) -> bool:
        return not self.bad_chars


def _split_by_dictionary(
    text: str, lookup: dict[str, str]
) -> tuple[list[tuple[str, bool]], list[tuple[str, str]]]:
    """辞書語で本文を刻む.

    Returns:
        (断片, 辞書由来か) の並びと、当たった (語, 読み) の一覧。

    **辞書で置いた語を塊として扱う必要がある。** 置換してから機械変換に渡すと、
    ``ヨコハマシ`` が ``ヨコハマ|シ`` と再分割され、語間が入ってしまう
    (機械変換の区切りを語間として使うため)。

    長い語から当てる。そうしないと ``神奈川県`` が ``神奈川`` + ``県`` に割れる。
    """
    hits: list[tuple[str, str]] = []
    pieces: list[tuple[str, bool]] = [(text, False)]
    for word in sorted(lookup, key=len, reverse=True):
        if not word:
            continue
        reading = lookup[word]
        nxt: list[tuple[str, bool]] = []
        matched = False
        for piece, is_dict in pieces:
            if is_dict or word not in piece:
                nxt.append((piece, is_dict))
                continue
            matched = True
            parts = piece.split(word)
            for i, part in enumerate(parts):
                if i:
                    nxt.append((reading, True))
                if part:
                    nxt.append((part, False))
        pieces = nxt
        if matched:
            hits.append((word, reading))
    return pieces, hits


def _kana_words(text: str) -> list[str]:
    """pykakasi でカタカナにし、**語の区切りごとに分けて**返す.

    機械変換は形態素に近い単位で区切りを返す。和文 CW は語間を置いて送るので
    (``テンキ ハ ハレ デス``)、この区切りをそのまま語間に使う。
    区切りを捨てて繋げると、受信側が読みづらい 1 本の長い符号列になる。

    **必ず 1 行ずつ渡すこと。** pykakasi は改行をまたいでも内部の蓄積を捨てず、
    改行の後に「それまでの全文」をもう一度返す::

        convert("こんにちは\\nこんにちは")
          [0] orig="こんにちは"           kana="コンニチハ"
          [1] orig="\\n"                  kana=""
          [2] orig="こんにちはこんにちは"  kana="コンニチハコンニチハ"   <- 蓄積

    まとめて渡していたため、画面では「Enter を押すたびに送信文が増えていく」
    形で現れた (2026-08-11 に実運用で発覚)。**CW に改行は無いので、行の
    切れ目は語間 1 つとして扱う** (呼び出し側が ``" ".join`` する)。

    pykakasi は import が重いので遅延読み込みにする (送信を使わない人に
    起動の遅さを負わせない)。
    """
    import pykakasi

    converter = pykakasi.kakasi()
    words: list[str] = []
    for line in text.splitlines() or [text]:
        words.extend(part["kana"] for part in converter.convert(line) if part["kana"])
    return words


def normalise_sendable(text: str) -> str:
    """小書きと句読点を送信可能な形へ倒す (カタカナ化の後に行う)."""
    out: list[str] = []
    for ch in text:
        if ch in PUNCTUATION_MAP:
            out.append(PUNCTUATION_MAP[ch])
        else:
            out.append(SMALL_KANA_MAP.get(ch, ch))
    return "".join(out)


def find_bad_chars(text: str) -> tuple[BadChar, ...]:
    """符号表に無い文字を位置つきで返す.

    ``{HORE}`` などのマーカーは符号に展開されるので通す。
    """
    bad: list[BadChar] = []
    index = 0
    length = len(text)
    while index < length:
        marker = _MARKER_RE.match(text, index)
        if marker:
            index = marker.end()
            continue
        ch = text[index]
        if ch not in _SENDABLE:
            bad.append(BadChar(index=index, char=ch))
        index += 1
    return tuple(bad)


def to_sendable_kana(
    text: str, profile: OperatorProfile | None = None
) -> ConversionResult:
    """日本語を送信可能なカタカナに変換する.

    Args:
        text: 日本語 (漢字かな交じり可)。
        profile: 経歴。**読み辞書だけ**を使う (欄は差し込み側で選ぶ)。
            ``None`` なら機械変換のみ。

    Returns:
        変換後テキストと、送信できない文字の位置。
    """
    if not text:
        return ConversionResult("")

    lookup: dict[str, str] = {}
    if profile is not None:
        # **経歴の欄はここに注がない。** 以前は `field_readings()` が
        # 「表示形 → 読み」の辞書を作って本文全体に当てており、欧文の本文の
        # `TARO` が `タロウ` に化けてモードが和文に倒れ、**文中の欧文が
        # まるごと送れなくなっていた** (2026-08-11 に実測)。
        # 経歴を欧文用/和文用の 2 値にしたので、差し込みのときにモードで
        # 選べば済む (`templates.profile_values`)。**この仕組み自体が要らない。**
        #
        # 運用者が育てた `reading_dictionary` は残る。意図して入れたものであり、
        # 編集画面が「欧文の語がある」と警告する (設計書 §4.3)。
        lookup.update(profile.reading_dictionary)

    pieces, hits = _split_by_dictionary(text, lookup)
    words: list[str] = []
    for piece, is_dictionary in pieces:
        if is_dictionary:
            words.append(piece)          # 辞書の読みは塊のまま (再分割させない)
        else:
            words.extend(_kana_words(piece))

    normalised = normalise_sendable(" ".join(words))
    # 句読点の前後に語間は要らない ("ハレ 、 アツイ" → "ハレ、アツイ")。
    # 実録音のラベルもこの形である ("テンキ ハ ハレ、アツイ デス")。
    # **`?` はここに入れない。** 句読点として直前の空白を削っていたが、
    # `?` は 2026-08-12 から「値が無い」の印でもある (`profile.UNKNOWN`)。
    # 削ると `ナマエ ハ ? デス` が `ナマエ ハ? デス` になり、`ハ` と `?` が
    # 語間なしで繋がって別の語として届く (実測)。
    # `HW?` のように運用者が空白なしで書いたものはそのまま残る。
    normalised = re.sub(r"\s+([、。」])", r"\1", normalised)
    # pykakasi は `「` を独立した語として返すので、直後に語間の空白が入る
    # ("「 FT991" となる)。`「` は中身に直接くっつくので、そこだけ削る
    # (2026-08-12 のレビューで判明。`「…」` を欧文区間として使う経路で実際に踏んだ)。
    normalised = re.sub(r"(「)\s+", r"\1", normalised)
    # `、` `。` の前後には語間が要らない (通常の句読点として扱う)。
    # **`。` を足したのは 2026-08-12。** それまで `。` は `、` に倒されていたので
    # ここに書く必要が無かった。段落として残すようにした以上、`、` と同じ扱いに
    # しないと `コンニチハ 。 テンキ` と語間が 2 つ増える (実際に踏んだ)。
    # **`」` は違う。** `「…」` の区間が終わったあとに続く語との間には語間が要る
    # ("「FT991」 デス" の `」` と `デス` の間)。以前は `」` も句読点として
    # 直後の空白を削っていたが、`」` が欧文区間の終わりマーカーにもなった今、
    # それでは区間の直後の語間が消えてしまう (2026-08-12 のレビューで判明)。
    # `」` が本当に文末で後ろに何も続かない場合は、この後の `.strip()` で
    # 末尾の空白ごと落ちるので問題ない。
    normalised = re.sub(r"([、。])\s+", r"\1", normalised)
    # 連続する空白は語間 1 つに畳む (符号上は同じで、見た目だけ揺れるため)
    normalised = re.sub(r" {2,}", " ", normalised).strip()
    return ConversionResult(
        text=normalised,
        bad_chars=find_bad_chars(normalised),
        dictionary_hits=tuple(hits),
    )


__all__ = [
    "PUNCTUATION_MAP",
    "SMALL_KANA_MAP",
    "BadChar",
    "ConversionResult",
    "find_bad_chars",
    "normalise_sendable",
    "to_sendable_kana",
]
