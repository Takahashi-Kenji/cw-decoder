"""LLM 清書プロンプトの構築と、カナ→欧文対応表の実行時生成.

カナ→欧文対応表は morse_tokens.py を唯一の真正ソースとして生成し、
二重定義しない (アーキテクチャ原則 2).
"""
from __future__ import annotations

from src.infer.word_correct import EUROPEAN_LEXICON
from src.llm.markup import CLOSE_MARK, OPEN_MARK
from src.tokens.morse_tokens import (
    DAKUTEN_CHAR,
    DisplayMode,
    EUROPEAN_TABLE,
    HANDAKUTEN_CHAR,
    JAPANESE_TABLE,
)


def build_kana_to_european() -> dict[str, str]:
    """同一符号を共有する和文カナ → 欧文文字の対応表を生成する.

    濁点・半濁点は合成用記号のため除外する.
    """
    table: dict[str, str] = {}
    for code, kana in JAPANESE_TABLE.items():
        if kana in (DAKUTEN_CHAR, HANDAKUTEN_CHAR):
            continue
        eu = EUROPEAN_TABLE.get(code)
        if eu is not None:
            table[kana] = eu
    return table


# 共通ヘッダ + 誤り訂正 (全モード共通). 項目 2 (整形) はモード別に差し込む.
_SYSTEM_HEADER = """あなたはアマチュア無線 CW (モールス信号) のデコード結果を校正する専門家です。
入力は AI デコーダの生出力で、誤り・脱落 (?) が含まれます。次を行ってください。

1. デコード誤りの訂正: 文脈から D↔B, K↔T, Y↔A, 9→O, I→E 等の系統誤りや
   ? (脱落) を推測して補正する。"""

# 短縮数字 (cut numbers)。CW では数字を短い符号の文字で送る慣習がある。
# **5NN = 599 が圧倒的に多い** (RST レポート)。デコーダはそのまま文字として出すので、
# 清書側で数字に戻す。運用者の要望 (2026-08-08)。
#
# 辞書補正 (word_correct) はこれを直せない。数字を含む語は「触らない」と決めており、
# それはコールサインを守るための規則だからである。文脈が要るのでここは LLM の仕事。
CUT_NUMBERS: dict[str, str] = {
    "T": "0", "A": "1", "U": "2", "V": "3",
    "B": "7", "D": "8", "N": "9",
}

_CUT_NUMBER_RULE = (
    "短縮数字 (文字で送られた数字) を数字に戻す: 5NN→599、NN→99、"
    + ", ".join(f"{k}→{v}" for k, v in CUT_NUMBERS.items())
    + "。"
)

# 欧文モード: 欧文のまま整形し、日本語へ翻訳しない.
#
# **歯止めの 2 行は実測で必要と分かったもの** (2026-08-08、held-out 欧文)。
# 短縮数字の指示だけを入れたところ、正しく取れていた 559 を「599 の方が普通だから」と
# 書き換え、RST を丸ごと落とす例が出た。
_EU_CLEANUP = f"""2. 読みやすく整形する: 略語 (CQ, RST, QTH, OM, TNX, 73 等) はそのまま、
   または一般的な表記に整える。出力は欧文のままとし、日本語に翻訳しないこと。
   {_CUT_NUMBER_RULE}
   **文字を数字にするだけで、既にある数字は絶対に変えない** (559 を 599 にしない)。
   コールサインの中の文字も変えない。**語を消さない** (RST や QTH を落とさない)。"""

# 和文 / auto モード: 読みやすい日本語へ清書する.
#
# **電文がカタカナなのは和文モールスの仕様上どうにもならない。** 読む側は
# 漢字かな交じりの方が楽なので、そこへ直すのが清書の主目的である (運用者の指摘)。
# 「読みやすい日本語にする」だけではカタカナのまま並ぶことがあったため、
# 漢字とひらがなへ直すことを例つきで明示する。
_JP_CLEANUP = """2. 読みやすい日本語への清書: **漢字かな交じりの自然な日本語**にする。
   カタカナのまま並べず、適切に漢字とひらがなへ直すこと
   (例: 「コンニチハ、テンキ ハ ハレ デス」→「こんにちは。天気は晴れです」)。
   略語 (CQ, RST, QTH, OM, TNX, 73 等) も適度に展開する。
   多少の推測違いは許容する。読めない断片をカタカナのまま残すより、
   自然な日本語として読める形にすることを優先する。"""

# 増分清書で直前のやり取りを渡すときのルール。
# 自動清書は確定テキスト全体ではなく未清書分だけを送るため、話のつながりを
# 見せる目的で直前部分を添える。これを出力に混ぜられると重複するので明示的に禁じる。
_LEAD_RULE = """
参考として直前のやり取りを添えることがある。文脈の把握にだけ使い、
**参考部分は出力しないこと**。出力するのは「ここから整えてください」以降の分だけ。"""

# f-string: OPEN_MARK / CLOSE_MARK をビルド時に展開する.
#
# 項目 4 の方針は**モードで変える**。欧文は「確実さ優先」、和文は「読みやすさ優先」。
# 従来は全モードで「確信ありげな誤りより不確実だと分かる出力を優先」と指示しており、
# これが「カタカナのままでは読めない」という和文の要望と正面から矛盾していた。
#
# **欧文側を読みやすさ優先にしてはいけない。** 欧文はコールサインや RST を捏造されると
# 実害が出る (辞書補正が数字を含む語に触らないのと同じ理由)。
_MARKER_HEAD = f"""3. あなたが推測・補正・展開した箇所は必ず {OPEN_MARK} と {CLOSE_MARK} で囲む。
   直接読めた確実な箇所は囲まない。マーカー以外の記号で囲ってはいけない。
"""

_EU_ACCURACY = f"""4. 自信を持って正しく復元できない箇所は、それらしい語を捏造しないこと。読めた音を
   そのまま残すか不明な箇所として示し、必ず {OPEN_MARK}…{CLOSE_MARK} で囲む。
   「確信ありげな誤り」より「不確実だと分かる出力」を優先する。
"""

_JP_READABILITY = f"""4. 読みやすさを優先する。多少の推測違いは許容されるので、
   カタカナの断片をそのまま残さず、文脈から最もありそうな日本語に直すこと。
   ただし**推測した箇所は必ず {OPEN_MARK}…{CLOSE_MARK} で囲む** (どこが推測かは
   読み手が判断できる必要がある)。まったく手掛かりが無い箇所だけは
   読めた音を残し、同じく {OPEN_MARK}…{CLOSE_MARK} で囲む。
"""

_MARKER_TAIL = f"""

出力は清書後テキストのみ。入力がどれほど壊れていても、謝罪・言い訳や
「校正不可能」「破損が著しい」「音声を確認してください」等のメタコメント・助言・
前置き・解説は一切出力しないこと。復元しきれない部分は {OPEN_MARK}…{CLOSE_MARK} で
囲んでそのまま残し、必ず最善の清書結果だけを返す。"""

# 欧文モード用の参考語彙: 有効な略語/プロサインを「壊さない」ためのヒント.
# 日本語訳は付けない (翻訳を誘発しないため).
#
# 語彙は word_correct.EUROPEAN_LEXICON を唯一の真正ソースとする。ここに書き写すと
# 辞書補正と LLM で「有効な語」の認識がずれる (アーキテクチャ原則 2)。
# 73/88 は数字なので辞書補正の語彙には入れていないが、LLM に「壊すな」と伝える
# 分には有効なのでここで足す。
_EU_GLOSSARY = """

参考: 次は有効な欧文 CW 略語/プロサイン。これらは誤りではないので壊さずそのまま残す:
""" + " ".join([*EUROPEAN_LEXICON, "73", "88"])

# プロサイン (運用記号) の説明。
#
# **これが無いと [ホレ] を「晴れ」「こんにちは」と読み替えてしまう** (実測で
# gemma3:4b / gemma4:e4b / gemma3:12b すべてが誤読した)。角括弧付きの表記は
# 語ではなく運用記号なので、そのまま残させる。
_PROSIGN_RULE = """

参考: 角括弧で囲まれた次の表記は**運用記号**であって語ではない。
日本語に訳さず、そのままの表記で残すこと:
[ホレ]=和文の開始  [ラタ]=和文の終わり  [SK]=交信終了  [KN]=特定局へどうぞ
[SN]=了解  [HH]=訂正"""

# 和文 / auto モード用の参考語彙: 定型表現の認識を助ける.
_JP_GLOSSARY = """

参考: 和文 CW でよく使われる定型表現 (文脈に合えば優先的に当てはめる):
こちらは / どうぞ / 了解 / お願いします / よろしくお願いします / ありがとうございました /
さようなら / お休みなさい / また明日 / またお会いしましょう / 失礼します / お元気で /
お名前は / 信号は / 天気は / アンテナ / リグ(無線機) / 出力 / 所在地"""

# プレーン文字列: .format() で {open}/{close}/{table} を後から埋める (和文 / auto のみ).
_JP_EXTRA = """

5. 欧文化: 和文 (カタカナ) として出力されているが、欧文 (コールサイン・RST・数字・Q コード等)
   として意味が通る箇所は、次のカナ→欧文対応表に従って欧文化する。
   変換した箇所も {open}…{close} で囲む。表に無い変換は行わない。

カナ→欧文対応表:
{table}"""


# ---- 短いプロンプト (小さいローカルモデル向け) ----
#
# **重いプロンプトは小さいモデルに害になる。** 2026-08-08 の実測:
#
#   * schroneko/gemma-2-2b-jpn-it は入力を無視し、``_JP_CLEANUP`` の**例文を
#     そのまま返した** (全 4 件が「こんにちは。天気は晴れです」)
#   * gemma4:e4b は重いプロンプトだと「チューリップ」を捏造し、「おはよう」を
#     落とし、「曇り」を「雲」にした。短いプロンプトではどれも正しく出た
#
# 規則 5 項目 + カナ→欧文対応表 (48 行) + 語彙 + プロサイン説明は、
# クラウドの大きいモデルには効くが 4B 前後には多すぎる。
#
# **プロサインは箇条書き 1 行にすると効く。** 同じ内容を段落で書いた重い版では
# ``[ホレ]`` が「これは」「晴れ」になったが、1 行版では消えた。
_COMPACT_JP = f"""カタカナで書かれた無線の電文を、漢字かな交じりの読みやすい日本語に直してください。

規則:
- 出力は直した日本語の文だけ。説明・前置き・絵文字は書かない。
- 意味が取れない部分はカタカナのまま残す。
- 内容を足さない。書かれていないことを書かない。
- 推測して直した箇所は {OPEN_MARK} と {CLOSE_MARK} で囲む。確実な箇所は囲まない。
- [ホレ] [ラタ] [SK] [KN] [SN] は運用記号。訳さずそのまま残す。"""

def _format_kana_table() -> str:
    """カナ→欧文対応表をスペース区切りの文字列に整形する."""
    items = sorted(build_kana_to_european().items())
    return " ".join(f"{kana}={eu}" for kana, eu in items)


def build_messages(
    raw_text: str,
    mode: DisplayMode,
    lead_text: str | None = None,
    compact: bool = False,
) -> list[dict[str, str]]:
    """全プロバイダ共通の中立メッセージ形式を構築する.

    欧文モードは欧文のまま整形し日本語に翻訳しない。和文 / auto モードは
    読みやすい日本語へ清書し、欧文として意味が通る箇所を欧文化する
    (auto でもデコーダは和文を出力しうるため欧文化表を含める. 仕様 §4, §6.3)。

    Args:
        raw_text: 今回清書する分。自動清書では**未清書の増分だけ**を渡す。
        mode: 表示モード。
        lead_text: 直前のやり取り。話のつながりを見せるためだけに渡し、出力させない。
            ``None`` なら従来どおり ``raw_text`` のみを送る。
        compact: 短いプロンプトを使う。**ローカルの小さいモデルではこちらが良い**
            (重いプロンプトだと例文をそのまま返したり捏造したりする)。
    """
    is_japanese = mode in ("japanese", "auto")
    # **短い版は和文だけ。** 欧文は重い版を使う。理由は測定ではなく構造による:
    # 欧文で必要なのは「触らない規律」で、それは重い版の捏造禁止・確実さ優先に
    # 書いてある。短縮数字の規則と無線語彙も重い版に置いてある。
    #
    # (2026-08-08 に短い版と重い版を欧文で比べたが、**あの比較はノイズだった**。
    #  同じプロンプトの 2 回の実行で CER が 16.32% と 20.92% に割れた。
    #  差 4.6pt は効果より大きい。温度を 0 に固定して初めて測れるようになった。
    #  → src/llm/providers/claude.py の temperature)
    #
    # **欧文の LLM 清書はそもそも CER を悪化させる** (温度 0・3 回平均で
    # 17.15% → 18.27%、+1.12pt)。辞書補正が既に機械的な直しを済ませており、
    # LLM は誤りを足す側に回る。欧文で清書を使うかは読みやすさとの取引である。
    if compact and is_japanese:
        return _compact_messages(raw_text, lead_text)
    cleanup = _JP_CLEANUP if is_japanese else _EU_CLEANUP
    glossary = _JP_GLOSSARY if is_japanese else _EU_GLOSSARY
    accuracy = _JP_READABILITY if is_japanese else _EU_ACCURACY
    system = (
        f"{_SYSTEM_HEADER}\n{cleanup}\n{_MARKER_HEAD}{accuracy}"
        f"{_MARKER_TAIL}{glossary}{_PROSIGN_RULE}"
    )
    if is_japanese:
        system += _JP_EXTRA.format(
            open=OPEN_MARK, close=CLOSE_MARK, table=_format_kana_table()
        )
    lead = (lead_text or "").strip()
    if lead:
        system += _LEAD_RULE
        user = (
            "参考 (直前のやり取り。出力しないでください):\n"
            f"{lead}\n\n"
            "ここから整えてください:\n"
            f"{raw_text}"
        )
    else:
        user = raw_text
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _compact_messages(
    raw_text: str, lead_text: str | None
) -> list[dict[str, str]]:
    """短いプロンプト版のメッセージを組む (和文専用)."""
    system = _COMPACT_JP
    lead = (lead_text or "").strip()
    if lead:
        system += _LEAD_RULE
        user = (
            "参考 (直前のやり取り。出力しないでください):\n"
            f"{lead}\n\n"
            "ここから整えてください:\n"
            f"{raw_text}"
        )
    else:
        user = raw_text
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


__all__ = ["build_kana_to_european", "build_messages"]
