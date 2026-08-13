"""実信号ファインチューニング用の打鍵原稿コーパス (手書き).

``keying_scripts.generate_keying_script`` は乱数でフレーズを組み合わせるため、
6要素符号 (``?`` ``,`` ``.`` ``-`` ``@`` ``、`` ``。``) のような**低頻度だが認識が
壊滅しているトークン**の出現回数を設計値どおりに保証できない。本モジュールは
本文を直接持つことでカバレッジを固定する。

用途の分離:

- 既存の ``script_eu_01..10`` / ``script_ja_01..10`` (録音済み) = 評価用 keyed_val
- 本コーパス由来の ``script_eu_t01..`` / ``script_ja_t01..`` = 学習用 (train)

``t`` 付きの命名により、学習と評価の取り違えをファイル名の時点で防ぐ。

設計値 (欧文30件):

- 実運用文 18 件 / 弱点トークン重点 12 件
- WPM は実運用実測 (14.6〜19.4) に寄せて 16 / 20 / 24 に配分
- 1 件の打鍵時間は 8〜20 秒 (CTC の 3〜30 秒制約の内側)
- 語間の過剰検出 (打鍵20件の挿入誤り53件中33件) 対策として、短語の連続・
  長語・語間なしの連続 (コールサイン/数字列) を意図的に混ぜる
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from src.tokens.morse_tokens import Mode, SPECIAL_INPUT_MARKERS, text_to_codes


@dataclass(frozen=True)
class KeyingScript:
    """打鍵原稿 1 件."""

    name: str
    mode: Mode
    wpm: float
    text: str


def resolve_code(key: str, mode: Mode) -> str:
    """入力表記 (``?`` ``、`` ```` 等) を単一の符号に解決する.

    カバレッジ目標を読みやすい表記で書くための補助。濁点付きカナのように
    2 符号に展開される表記は目標キーに使えないため ``ValueError`` とする。
    """
    if key in SPECIAL_INPUT_MARKERS:
        return SPECIAL_INPUT_MARKERS[key]
    codes = text_to_codes(key, mode)
    if len(codes) != 1:
        raise ValueError(f"単一符号に解決できません: {key!r} ({mode}) -> {codes}")
    return codes[0]


def code_histogram(scripts: tuple[KeyingScript, ...]) -> Counter[str]:
    """符号単位の出現数を数える.

    濁点・半濁点は独立トークンなので文字単位では数えられない。NN が実際に
    学習する単位に合わせて符号で数える。
    """
    counter: Counter[str] = Counter()
    for script in scripts:
        counter.update(text_to_codes(script.text, script.mode))
    return counter


# 実運用文 (18 件): CQ・レポート交換・ラグチュー・クロージング。
# 実際の QSO に出る語順・語彙で構成し、ドメインを合わせる役割を持つ。
_EUROPEAN_OPERATIONAL: tuple[KeyingScript, ...] = (
    KeyingScript("eu_t01", "european", 16, "CQ CQ CQ DE JA1ABC K"),
    KeyingScript("eu_t02", "european", 16, "CQ DX DE JH0XYZ/1 PSE K"),
    KeyingScript("eu_t03", "european", 16, "JA1ABC DE JR2GHI GM OM +"),
    KeyingScript("eu_t04", "european", 20, "UR RST 569 569 = QTH KANAGAWA"),
    KeyingScript("eu_t05", "european", 16, "NAME TARO TARO HW CPY?"),
    KeyingScript("eu_t06", "european", 20, "RIG IC7300 PWR 50W ES ANT GP"),
    KeyingScript("eu_t07", "european", 20, "QTH YOKOHAMA CITY KANAGAWA JAPAN"),
    KeyingScript("eu_t08", "european", 20, "ANT DIPOLE UP 10M AGL = PWR 5W"),
    KeyingScript("eu_t09", "european", 20, "WX FINE HR TEMP 24C ES WIND SW"),
    KeyingScript("eu_t10", "european", 16, "TNX VY FB QSO OM 73 ES GL"),
    KeyingScript("eu_t11", "european", 16, "QRZ? QRZ? DE JA1ABC PSE K"),
    KeyingScript("eu_t12", "european", 20, "SRI QRM QSY 7046 KHZ PSE +"),
    KeyingScript("eu_t13", "european", 20, "PSE QRS VY TNX OM ES HW CPY?"),
    KeyingScript("eu_t14", "european", 20, "JA1ABC/P DE JR2GHI R R FB TNX"),
    KeyingScript("eu_t15", "european", 16, "GE OM UR RST 469 NAME TAK +"),
    KeyingScript("eu_t16", "european", 20, "QSL VIA BURO TNX 73 ES GD DX"),
    KeyingScript("eu_t17", "european", 16, "HW? BK BK PSE RPT UR CALL?"),
    KeyingScript("eu_t18", "european", 20, "ANT 3EL YAGI UP 12M = PWR 100W"),
)

# 弱点トークン重点 (12 件): 6要素符号と数字を高頻度で含む。
# recall 26.7% の 6要素符号を実打鍵で学習させることが目的で、出現頻度は
# 意図的に実運用より高い。
_EUROPEAN_WEAK_FOCUS: tuple[KeyingScript, ...] = (
    KeyingScript("eu_t19", "european", 20, "E-MAIL JA1ABC@JARL.COM PSE"),
    KeyingScript("eu_t20", "european", 24, "RIG IC-7300 ANT HB-9CV UP 8M"),
    KeyingScript("eu_t21", "european", 24, "QSO DATE 2026-08-03 1200Z +"),
    KeyingScript("eu_t22", "european", 16, "HW? QSL? QRZ? TU? OK?"),
    KeyingScript("eu_t23", "european", 24, "QRA TOKYO, OSAKA, NAGOYA, KOBE"),
    KeyingScript("eu_t24", "european", 20, "QSL INFO@JARL.OR.JP TNX 73"),
    KeyingScript("eu_t25", "european", 24, "MY CALL JA1ABC/QRP = HW CPY?"),
    KeyingScript("eu_t26", "european", 16, "TU DR OM. 73. GL. TNX."),
    KeyingScript("eu_t27", "european", 24, "FREQ 7,025.5 KHZ QSY? PSE RPT"),
    KeyingScript("eu_t28", "european", 24, "NR 1-2-3, 4-5-6, 7-8-9, 0"),
    KeyingScript("eu_t29", "european", 24, "MY MAIL JR3GHI@JARL.COM, UR?"),
    KeyingScript("eu_t30", "european", 24, "SRI QRT. MAIL JH4JKL@JARL.COM"),
)

EUROPEAN_SCRIPTS: tuple[KeyingScript, ...] = (
    *_EUROPEAN_OPERATIONAL,
    *_EUROPEAN_WEAK_FOCUS,
)

# 和文 実運用文 (36 件): 挨拶・レポート交換・リグ/アンテナ・天候・ラグチュー・
# クロージング。
#
# 表記の注意: 和文モールスの符号表に小書き文字 (ャ ュ ョ ッ ェ 等) は無く、
# 拗音・促音も大書きで送る。よって原稿も「リヨウカイ」「ワツト」表記になる。
_JAPANESE_OPERATIONAL: tuple[KeyingScript, ...] = (
    KeyingScript("ja_t01", "japanese", 16, "コンニチハ、ドウゾ"),
    KeyingScript("ja_t02", "japanese", 16, "オハヨウ ゴザイマス、ドウゾ"),
    KeyingScript("ja_t03", "japanese", 20, "コンバンハ、ヨロシク オネガイ シマス"),
    KeyingScript("ja_t04", "japanese", 20, "シンゴウ ハ 599 デス、ヨク トレマス"),
    KeyingScript("ja_t05", "japanese", 20, "シンゴウ ハ ヨワイ デス ガ トレマス"),
    KeyingScript("ja_t06", "japanese", 20, "ナマエ ハ タロウ、タロウ デス"),
    KeyingScript("ja_t07", "japanese", 16, "バシヨ ハ カナガワケン ヨコハマ デス"),
    KeyingScript("ja_t08", "japanese", 16, "ワタシ ノ バシヨ ハ ヨコハマシ デス"),
    KeyingScript("ja_t09", "japanese", 20, "リグ ハ アイシー、デンリヨク ハ 50 ワツト"),
    KeyingScript("ja_t10", "japanese", 16, "アンテナ ハ ダイポール デス"),
    KeyingScript("ja_t11", "japanese", 24, "アンテナ ハ ジーピー、タカサ ハ 10 メートル"),
    KeyingScript("ja_t12", "japanese", 20, "テンキ ハ ハレ、キオン ハ 24 ド デス"),
    KeyingScript("ja_t13", "japanese", 24, "コチラ ハ アメ デス、スコシ サムイ デス"),
    KeyingScript("ja_t14", "japanese", 20, "テンキ ハ クモリ、カゼ ガ ツヨイ デス"),
    KeyingScript("ja_t15", "japanese", 16, "マイニチ アツクテ タイヘン デス ネ"),
    KeyingScript("ja_t16", "japanese", 16, "キヨウ ハ シゴト ガ ヤスミ デス"),
    KeyingScript("ja_t17", "japanese", 16, "ムセン ハ タノシイ デス ネ"),
    KeyingScript("ja_t18", "japanese", 16, "コンデイシヨン ガ ワルイ デス ネ"),
    KeyingScript("ja_t19", "japanese", 24, "サイキン ハ ワブン ノ レンシユウ ヲ シテ イマス"),
    KeyingScript("ja_t20", "japanese", 16, "キヨウ ハ アリガトウ ゴザイマシタ"),
    KeyingScript("ja_t21", "japanese", 20, "マタ ヨロシク オネガイ シマス、サヨウナラ"),
    KeyingScript("ja_t22", "japanese", 16, "コレデ オワリ マス、73"),
    KeyingScript("ja_t23", "japanese", 16, "ドウモ アリガトウ、マタ アイマシヨウ"),
    KeyingScript("ja_t24", "japanese", 24, "ハジメマシテ、ヨロシク オネガイ シマス"),
    KeyingScript("ja_t25", "japanese", 20, "オナマエ ヲ モウ イチド オネガイ シマス"),
    KeyingScript("ja_t26", "japanese", 20, "バシヨ ヲ モウ イチド オネガイ シマス"),
    KeyingScript("ja_t27", "japanese", 16, "リヨウカイ シマシタ、アリガトウ"),
    KeyingScript("ja_t28", "japanese", 20, "スコシ ハヤイ デス、モウ スコシ ユツクリ"),
    KeyingScript("ja_t29", "japanese", 24, "ザンネン デス ガ ノイズ ガ オオイ デス"),
    KeyingScript("ja_t30", "japanese", 20, "コチラ ノ シンゴウ ハ イカガ デス カ?"),
    KeyingScript("ja_t31", "japanese", 16, "アンテナ ヲ アタラシク シマシタ"),
    KeyingScript("ja_t32", "japanese", 16, "ヤマ ノ ウエ カラ デテ イマス"),
    KeyingScript("ja_t33", "japanese", 16, "ホンジツ ハ ヨイ テンキ デス"),
    KeyingScript("ja_t34", "japanese", 16, "ミナサン オゲンキ デ、サヨウナラ"),
    KeyingScript("ja_t35", "japanese", 16, "イマ カラ シヨクジ ニ イキマス"),
    KeyingScript("ja_t36", "japanese", 16, "ソレデハ マタ、ドウゾ?"),
)

# 和文 弱点トークン重点 (24 件): 区切点 ``、`` / 段落 ``。`` / ``?`` ``-`` ``@``
# 長音 ``ー`` 濁点 ``゛`` 半濁点 ``゜`` と数字。
_JAPANESE_WEAK_FOCUS: tuple[KeyingScript, ...] = (
    KeyingScript("ja_t37", "japanese", 24, "バシヨ ハ カナガワケン、ヨコハマシ、ツルミ。"),
    KeyingScript("ja_t38", "japanese", 20, "シユウハスウ ハ 7-0-3-5 キロヘルツ"),
    KeyingScript("ja_t39", "japanese", 24, "デンワ ハ 026-123-4567。"),
    KeyingScript("ja_t40", "japanese", 20, "メール ハ タロウ@ジヤール、ムセン@ヤマ"),
    KeyingScript("ja_t41", "japanese", 20, "ドウゾ?、モウ イチド?、ワカリマセン?"),
    KeyingScript("ja_t42", "japanese", 24, "ガギグゲゴ、ザジズゼゾ、ダヂヅデド"),
    KeyingScript("ja_t43", "japanese", 24, "バビブベボ、パピプペポ、ヴ。"),
    KeyingScript("ja_t44", "japanese", 20, "ナガノ、ニイガタ、トヤマ、イシカワ、フクイ"),
    KeyingScript("ja_t45", "japanese", 20, "イチ、ニ、サン、ヨン、ゴ、ロク、ナナ"),
    KeyingScript("ja_t46", "japanese", 24, "ハチ、キユウ、ゼロ、1234567890。"),
    KeyingScript("ja_t47", "japanese", 20, "レポート ハ 599、599。"),
    KeyingScript("ja_t48", "japanese", 24, "ケータイ ハ 090-1234-5678 デス"),
    KeyingScript("ja_t49", "japanese", 20, "ジコク ハ 1200、ドウゾ。"),
    KeyingScript("ja_t50", "japanese", 20, "コーヒー、ケーキ、ビール、ラーメン"),
    KeyingScript("ja_t51", "japanese", 24, "スーパー デ カーテン ヲ カイマシタ。"),
    KeyingScript("ja_t52", "japanese", 20, "ドウゾ?、リヨウカイ?、オワリ?、ドウゾ?"),
    KeyingScript("ja_t53", "japanese", 20, "ヰ、ヱ、ヌ、ヲ、ン、ヌノ、イヌ"),
    KeyingScript("ja_t54", "japanese", 24, "サイシユウ-テキ-ニ、コレ-デ、オワリ。"),
    KeyingScript("ja_t55", "japanese", 20, "ニホン-トウキヨウ-シブヤ。"),
    KeyingScript("ja_t56", "japanese", 24, "2026-08-03、1200 デス"),
    KeyingScript("ja_t57", "japanese", 20, "ツルミ@ヨコハマ、ドウゾ?。"),
    KeyingScript("ja_t58", "japanese", 24, "ゼロ、イチ、ニ、サン-ヨン-ゴ-ロク-ナナ-ハチ"),
    KeyingScript("ja_t59", "japanese", 20, "オワリ マス。、73、サヨウナラ"),
    KeyingScript("ja_t60", "japanese", 24, "ペン、ポスト、プリン、ピアノ。"),
)

JAPANESE_SCRIPTS: tuple[KeyingScript, ...] = (
    *_JAPANESE_OPERATIONAL,
    *_JAPANESE_WEAK_FOCUS,
)

ALL_SCRIPTS: tuple[KeyingScript, ...] = (*EUROPEAN_SCRIPTS, *JAPANESE_SCRIPTS)

# 打鍵時間の許容範囲 (秒). CTC の 3〜30 秒制約に余裕を持たせた内側の値。
DURATION_MIN_SEC = 8.0
DURATION_MAX_SEC = 20.0

# 欧文コーパスが満たすべき最低出現数。6要素符号 (recall 26.7%) を中心に、
# 30 件という規模で現実的に確保できる水準として設定する。
EUROPEAN_MIN_OCCURRENCES: dict[str, int] = {
    "?": 12,
    ".": 10,
    ",": 8,
    "-": 11,
    "@": 4,
    "/": 3,
    "=": 4,
    "+": 4,
}

# 数字は 10 種すべてを最低 3 回。RST・周波数・時刻・番号で自然に散らす。
EUROPEAN_MIN_DIGIT_OCCURRENCES = 3

# 和文コーパス (60 件) が満たすべき最低出現数。
# ``@`` は和文の実運用でほぼ使われないため少ない (欧文側で 4 回確保している)。
JAPANESE_MIN_OCCURRENCES: dict[str, int] = {
    "、": 60,
    "。": 12,
    "?": 10,
    "-": 19,
    "@": 3,
    "ー": 17,
    "゛": 100,
    "゜": 13,
}

# 和文の数字は欧文より出番が少ないため 3 回。
JAPANESE_MIN_DIGIT_OCCURRENCES = 3

# 清音 48 字。1 度も出ない字を作らないための検査用 (ヰ・ヱは実運用で稀なため 1 回)。
JAPANESE_BASE_KANA = "イロハニホヘトチリヌルヲワカヨタレソツネナラムウヰノオクヤマケフコエテアサキユメミシヱヒモセスン"

__all__ = [
    "ALL_SCRIPTS",
    "DURATION_MAX_SEC",
    "DURATION_MIN_SEC",
    "EUROPEAN_MIN_DIGIT_OCCURRENCES",
    "EUROPEAN_MIN_OCCURRENCES",
    "EUROPEAN_SCRIPTS",
    "JAPANESE_BASE_KANA",
    "JAPANESE_MIN_DIGIT_OCCURRENCES",
    "JAPANESE_MIN_OCCURRENCES",
    "JAPANESE_SCRIPTS",
    "KeyingScript",
    "code_histogram",
    "resolve_code",
]
