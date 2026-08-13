"""モールス符号トークン定義と変換表.

「音 → 符号 → 文字」分離アーキテクチャの「符号」層:

- NN 語彙 = 統合符号トークン集合 (全モードユニーク符号 + CTC blank)
- 文字割当 (欧文 / 和文) は EUROPEAN_TABLE / JAPANESE_TABLE で決定

符号表記:

- ドット: ``・`` (U+30FB KATAKANA MIDDLE DOT)
- ダッシュ: ``-`` (U+002D HYPHEN-MINUS)

照合参照: 総務省「無線局運用規則 別表第一号 (モールス符号)」(e-Gov 法令検索)
照合テスト: ``tests/test_tokens.py`` + ``tests/data/eGov_morse_reference.py``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

Mode = Literal["european", "japanese"]

# UI 表示モード: 固定 2 モード + 自動切替.
DisplayMode = Literal["european", "japanese", "auto"]

# 和文開始/終了プロサインの符号 (自動モード切替のトリガ).
# ラタは欧文プロサイン SN と完全に同符号 (・・・-・).
HORE_CODE: Final[str] = "-・・---"   # 和文開始 (→ japanese)
RATA_CODE: Final[str] = "・・・-・"  # 和文終了 (japanese 中のみ → european)

DOT: Final[str] = "・"   # ・
DASH: Final[str] = "-"


@dataclass(frozen=True)
class Token:
    """符号トークン.

    Attributes:
        code: ``・`` と ``-`` からなる符号文字列. blank は ``"<BLANK>"``.
        id: 学習・推論で用いる連番 ID (0 は CTC blank).
    """

    code: str
    id: int


# ============================================================
# 欧文符号表 (符号 → 表示文字)
# ============================================================
# 26 英字 + 10 数字 + 8 記号 + 4 プロサイン = 48 エントリ
EUROPEAN_TABLE: Final[dict[str, str]] = {
    # 英字
    "・-": "A",
    "-・・・": "B",
    "-・-・": "C",
    "-・・": "D",
    "・": "E",
    "・・-・": "F",
    "--・": "G",
    "・・・・": "H",
    "・・": "I",
    "・---": "J",
    "-・-": "K",
    "・-・・": "L",
    "--": "M",
    "-・": "N",
    "---": "O",
    "・--・": "P",
    "--・-": "Q",
    "・-・": "R",
    "・・・": "S",
    "-": "T",
    "・・-": "U",
    "・・・-": "V",
    "・--": "W",
    "-・・-": "X",
    "-・--": "Y",
    "--・・": "Z",
    # 数字
    "・----": "1",
    "・・---": "2",
    "・・・--": "3",
    "・・・・-": "4",
    "・・・・・": "5",
    "-・・・・": "6",
    "--・・・": "7",
    "---・・": "8",
    "----・": "9",
    "-----": "0",
    # 記号
    "・-・-・-": ".",
    "--・・--": ",",
    "・・--・・": "?",
    "-・・-・": "/",
    "-・・・-": "=",
    "・-・-・": "+",          # 同符号: AR プロサイン, 和文 ン
    "-・・・・-": "-",
    "・--・-・": "@",
    # プロサイン (単一文字非対応のものは [XX] 表記)
    "・・・-・-": "[SK]",     # 交信終了
    "・・・-・": "[SN]",      # 了解 (同符号: 和文 ラタ)
    "-・--・": "[KN]",       # 指定局応答 (同符号: 和文 ル)
    "・・・・・・・・": "[HH]",  # 訂正 (ドット 8 個)
}

# ============================================================
# 和文符号表 (符号 → 表示文字)
# ============================================================
# 48 カナ + 5 補助記号 (濁点/半濁点/長音/区切点/段落) + 2 和文専用記号 ([ホレ]/[ラタ])
#   + 10 数字 + 3 その他記号 (?/-/@) = 68 エントリ
# (この数字は e-Gov 規則と表を目で突き合わせるときの検算値なので、表を増減したら
#  必ず更新すること。実測値は tests で len(JAPANESE_TABLE) として参照できる)
JAPANESE_TABLE: Final[dict[str, str]] = {
    # カナ (要件 §3.1.3)
    "・-": "イ",
    "・-・-": "ロ",
    "-・・・": "ハ",
    "-・-・": "ニ",
    "-・・": "ホ",
    "・": "ヘ",
    "・・-・・": "ト",
    "・・-・": "チ",
    "--・": "リ",
    "・・・・": "ヌ",
    "-・--・": "ル",
    "・---": "ヲ",
    "-・-": "ワ",
    "・-・・": "カ",
    "--": "ヨ",
    "-・": "タ",
    "---": "レ",
    "---・": "ソ",
    "・--・": "ツ",
    "--・-": "ネ",
    "・-・": "ナ",
    "・・・": "ラ",
    "-": "ム",
    "・・-": "ウ",
    "・-・・-": "ヰ",
    "・・--": "ノ",
    "・-・・・": "オ",
    "・・・-": "ク",
    "・--": "ヤ",
    "-・・-": "マ",
    "-・--": "ケ",
    "--・・": "フ",
    "----": "コ",
    "-・---": "エ",
    "・-・--": "テ",
    "--・--": "ア",
    "-・-・-": "サ",
    "-・-・・": "キ",
    "-・・--": "ユ",
    "-・・・-": "メ",
    "・・-・-": "ミ",
    "--・-・": "シ",
    "・--・・": "ヱ",
    "--・・-": "ヒ",
    "-・・-・": "モ",
    "・---・": "セ",
    "---・-": "ス",
    "・-・-・": "ン",
    # 補助記号
    "・・": "゛",           # 濁点 (同符号: 欧文 I)
    "・・--・": "゜",       # 半濁点
    "・--・-": "ー",        # 長音
    "・-・-・-": "、",      # 区切点 (同符号: 欧文 .)
    # **段落は `。` である。`」` ではない。**
    # 2026-06-11 の定義当初から `」` と書かれていた (2026-08-12 に運用者が発見)。
    # 段落の記号の字形が `」` に似ているため、表を書き写すときに閉じ括弧と
    # 取り違えたものと思われる。受信のたび、相手の文末が `」` と表示されていた。
    # 本物の括弧は和文表に存在するが、**この語彙には入れていない** —
    # 下向き括弧 `「` = `-・--・-`、上向き括弧 `」` = `・-・・-・`。
    # トークン ID は (符号長, 符号文字列) の辞書順なので、足すと
    # **それ以降の ID が全部ずれて学習済みモデルが無意味になる**。
    # 足すのは再学習の機会に合わせること (`tests/test_tokens.py`
    # `TestDanrakuAndBrackets` が歯止めになっている)。
    "・-・-・・": "。",      # 段落
    # 和文専用記号
    "-・・---": "[ホレ]",   # 和文開始
    "・・・-・": "[ラタ]",   # 和文終了 (同符号: 欧文 [SN])
    # 数字 (和文 QSO でも RST レポート・周波数等で頻出するため共通)
    "・----": "1",
    "・・---": "2",
    "・・・--": "3",
    "・・・・-": "4",
    "・・・・・": "5",
    "-・・・・": "6",
    "--・・・": "7",
    "---・・": "8",
    "----・": "9",
    "-----": "0",
    # その他 和文モードでも使われる記号 (欧文と同符号、和文表に未割当てのもの)
    "・・--・・": "?",      # クエスチョン
    "-・・・・-": "-",     # ハイフン
    "・--・-・": "@",      # アットマーク
}


# ============================================================
# 濁点・半濁点合成テーブル
# ============================================================
DAKUTEN_CHAR: Final[str] = "゛"
HANDAKUTEN_CHAR: Final[str] = "゜"

# プレーンカナ + 濁点 → 合成カナ
DAKUTEN_COMPOSE: Final[dict[str, str]] = {
    "カ": "ガ", "キ": "ギ", "ク": "グ", "ケ": "ゲ", "コ": "ゴ",
    "サ": "ザ", "シ": "ジ", "ス": "ズ", "セ": "ゼ", "ソ": "ゾ",
    "タ": "ダ", "チ": "ヂ", "ツ": "ヅ", "テ": "デ", "ト": "ド",
    "ハ": "バ", "ヒ": "ビ", "フ": "ブ", "ヘ": "ベ", "ホ": "ボ",
    "ウ": "ヴ",
}
# プレーンカナ + 半濁点 → 合成カナ
HANDAKUTEN_COMPOSE: Final[dict[str, str]] = {
    "ハ": "パ", "ヒ": "ピ", "フ": "プ", "ヘ": "ペ", "ホ": "ポ",
}


# ============================================================
# 統合トークン集合 (NN 語彙)
# ============================================================
BLANK_TOKEN_ID: Final[int] = 0
BLANK_CODE: Final[str] = "<BLANK>"
# 語間 (7 dot 無音) を表す擬似符号. 実符号としては表に登場しないため
# 中黒/ハイフン以外の文字でユニーク化.
WORD_BREAK_CODE: Final[str] = "<WORDBREAK>"


def _build_unified_tokens() -> tuple[
    tuple[Token, ...], dict[str, int], dict[int, Token]
]:
    """欧文・和文の全ユニーク符号 + 語間 WORDBREAK を統合して語彙を構築.

    ID 0 を CTC blank に予約し、ID 1〜N に実符号を割り当てる.
    実符号の順序は (符号長, 符号文字列) の辞書順で安定化.
    末尾に ``WORD_BREAK_CODE`` を 1 つ追加.
    """
    codes: set[str] = set(EUROPEAN_TABLE.keys()) | set(JAPANESE_TABLE.keys())
    sorted_codes: list[str] = sorted(codes, key=lambda c: (len(c), c))
    tokens: list[Token] = [Token(code=BLANK_CODE, id=BLANK_TOKEN_ID)]
    for i, code in enumerate(sorted_codes, start=1):
        tokens.append(Token(code=code, id=i))
    # WORDBREAK は実符号の後 (末尾)
    tokens.append(Token(code=WORD_BREAK_CODE, id=len(tokens)))
    tokens_tuple: tuple[Token, ...] = tuple(tokens)
    token_to_id: dict[str, int] = {t.code: t.id for t in tokens_tuple}
    id_to_token: dict[int, Token] = {t.id: t for t in tokens_tuple}
    return tokens_tuple, token_to_id, id_to_token


UNIFIED_TOKENS, TOKEN_TO_ID, ID_TO_TOKEN = _build_unified_tokens()
VOCAB_SIZE: Final[int] = len(UNIFIED_TOKENS)
WORD_BREAK_TOKEN_ID: Final[int] = TOKEN_TO_ID[WORD_BREAK_CODE]


# ============================================================
# 逆引き: 文字 → 符号列 (合成器が利用)
# ============================================================
def _invert_table_for_synthesis(table: dict[str, str]) -> dict[str, str]:
    """符号表を逆引き. ``[XX]`` プロサイン表示は合成入力から除外."""
    return {char: code for code, char in table.items() if not char.startswith("[")}


EUROPEAN_CHAR_TO_CODE: Final[dict[str, str]] = _invert_table_for_synthesis(EUROPEAN_TABLE)


def _build_japanese_char_to_codes() -> dict[str, tuple[str, ...]]:
    """文字 → 符号列. 合成カナは (プレーンカナ符号, 濁点/半濁点符号) の 2 符号."""
    inverse: dict[str, str] = _invert_table_for_synthesis(JAPANESE_TABLE)
    result: dict[str, tuple[str, ...]] = {char: (code,) for char, code in inverse.items()}
    dakuten_code = inverse[DAKUTEN_CHAR]
    handakuten_code = inverse[HANDAKUTEN_CHAR]
    for plain, composed in DAKUTEN_COMPOSE.items():
        result[composed] = (inverse[plain], dakuten_code)
    for plain, composed in HANDAKUTEN_COMPOSE.items():
        result[composed] = (inverse[plain], handakuten_code)
    return result


JAPANESE_CHAR_TO_CODES: Final[dict[str, tuple[str, ...]]] = _build_japanese_char_to_codes()


# ============================================================
# 送信専用の補助表 (受信語彙に無い符号。送信は NN を通らないので送れる)
# ============================================================
# **トークン集合には入れない。** 入れると ID がずれて学習済みモデルが無意味になる
# (tests/test_tx_only_chars.py の TestTokenSetUnchanged が歯止め)。
# 出典は tests/data/eGov_morse_reference.py の EGOV_EUROPEAN_PUNCT_TX_ONLY と
# EGOV_JAPANESE_BRACKETS_NOT_IN_VOCAB を参照 (照合テストあり)。
# 受信側は従来どおり: これらの符号を受けたときは ? (TABLE_MISS) になる。
TX_ONLY_EUROPEAN_CHAR_TO_CODE: Final[dict[str, str]] = {
    ":": "---・・・",
    "'": "・----・",
    '"': "・-・・-・",   # 和文の上向き括弧と同符号
    "(": "-・--・",      # プロサイン KN と同符号
    ")": "-・--・-",     # 和文の下向き括弧と同符号
    "×": "-・・-",       # 乗算記号。X と同符号
}

# 和文の本物の括弧。文字 「」 は欧文区間マーカー・段落の別名として既に
# 使われているため、{HORE} と同じ中括弧マーカーで入力する。
TX_ONLY_MARKERS: Final[dict[str, str]] = {
    "{KAKKO}": "-・--・-",   # 下向き括弧 「
    "{TOJI}": "・-・・-・",   # 上向き括弧 」
}


# ============================================================
# 特殊マーカー (合成入力用)
# ============================================================
# プロサインや和文専用記号は ``[XX]`` 表記のため、表の逆引きには含まれない.
# 合成入力では ``{HORE}`` / ``{RATA}`` / ``{SK}`` の中括弧マーカーで指定する.
SPECIAL_INPUT_MARKERS: Final[dict[str, str]] = {
    "{HORE}": "-・・---",        # 和文開始
    "{RATA}": "・・・-・",       # 和文終了 (= 欧文プロサイン SN と同符号)
    "{SK}":   "・・・-・-",      # 交信終了
}

# 送信側が使うマーカー全集合。**合成器は SPECIAL_INPUT_MARKERS だけを見る**
# ({KAKKO}/{TOJI} の符号は語彙に無く、ラベル生成に流れると TOKEN_TO_ID で落ちる)。
TX_INPUT_MARKERS: Final[dict[str, str]] = {**SPECIAL_INPUT_MARKERS, **TX_ONLY_MARKERS}


# ============================================================
# 公開 API
# ============================================================
def text_to_codes(
    text: str, mode: Mode, emit_word_breaks: bool = True, include_tx_only: bool = False
) -> list[str]:
    """テキストを符号列に変換 (合成器用).

    - 欧文モード: 文字を大文字化して 1 文字 = 1 符号
    - 和文モード: 合成カナは 2 符号に展開
    - ``{HORE}`` / ``{RATA}`` / ``{SK}`` の中括弧マーカーは特殊符号に展開
    - スペース: ``emit_word_breaks=True`` (デフォルト) なら ``WORD_BREAK_CODE``
      を 1 つ emit. ``False`` ならスキップ (旧挙動).
    - ``include_tx_only=True`` (送信側専用) なら送信専用表とマーカーもマージして解釈
    """
    codes: list[str] = []
    markers = TX_INPUT_MARKERS if include_tx_only else SPECIAL_INPUT_MARKERS
    char_to_code: dict[str, str] | None = None
    if mode == "european":
        char_to_code = (
            {**EUROPEAN_CHAR_TO_CODE, **TX_ONLY_EUROPEAN_CHAR_TO_CODE}
            if include_tx_only
            else EUROPEAN_CHAR_TO_CODE
        )
    char_to_codes_ja = JAPANESE_CHAR_TO_CODES if mode == "japanese" else None
    # 大文字化はマーカーが既に大文字なので safe
    work = text.upper() if mode == "european" else text
    i = 0
    n = len(work)
    while i < n:
        # 特殊マーカーを優先マッチ
        matched = False
        for marker, code in markers.items():
            if work.startswith(marker, i):
                codes.append(code)
                i += len(marker)
                matched = True
                break
        if matched:
            continue
        ch = work[i]
        i += 1
        if ch.isspace():
            if emit_word_breaks:
                # 連続するスペースは 1 つの WORD_BREAK に集約
                if not codes or codes[-1] != WORD_BREAK_CODE:
                    codes.append(WORD_BREAK_CODE)
            continue
        if mode == "european":
            assert char_to_code is not None
            codes.append(char_to_code[ch])
        else:
            assert char_to_codes_ja is not None
            codes.extend(char_to_codes_ja[ch])
    # 末尾の WORD_BREAK は意味なし
    if codes and codes[-1] == WORD_BREAK_CODE:
        codes.pop()
    return codes


def lookup_display(code: str, mode: Mode) -> str | None:
    """符号を表示文字に変換. 該当無しは ``None``."""
    table = EUROPEAN_TABLE if mode == "european" else JAPANESE_TABLE
    return table.get(code)


__all__ = [
    "BLANK_CODE",
    "BLANK_TOKEN_ID",
    "DAKUTEN_CHAR",
    "DAKUTEN_COMPOSE",
    "DASH",
    "DisplayMode",
    "DOT",
    "EUROPEAN_CHAR_TO_CODE",
    "HORE_CODE",
    "RATA_CODE",
    "SPECIAL_INPUT_MARKERS",
    "WORD_BREAK_CODE",
    "WORD_BREAK_TOKEN_ID",
    "EUROPEAN_TABLE",
    "HANDAKUTEN_CHAR",
    "HANDAKUTEN_COMPOSE",
    "ID_TO_TOKEN",
    "JAPANESE_CHAR_TO_CODES",
    "JAPANESE_TABLE",
    "Mode",
    "TOKEN_TO_ID",
    "TX_INPUT_MARKERS",
    "TX_ONLY_EUROPEAN_CHAR_TO_CODE",
    "TX_ONLY_MARKERS",
    "Token",
    "UNIFIED_TOKENS",
    "VOCAB_SIZE",
    "lookup_display",
    "text_to_codes",
]
