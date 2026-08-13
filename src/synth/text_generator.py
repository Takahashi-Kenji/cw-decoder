"""学習用テキスト生成 (欧文 / 和文).

実運用パターン (コールサイン形式・RST・Q 符号・CQ 呼出・QSO 定型・和文
ラグチュー風) を混合し、ランダム文字列のみに偏らないテキストを生成する.
すべての関数は ``np.random.Generator`` を受け取り再現可能.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np

# ============================================================
# 欧文パターン
# ============================================================
_EUROPEAN_LETTERS: Final[str] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EUROPEAN_DIGITS: Final[str] = "0123456789"

# 主要 Q 符号 (アマチュア無線で頻出). 末尾 ? は問合せ形
_Q_CODES: Final[tuple[str, ...]] = (
    "QRZ", "QRM", "QRN", "QSB", "QSY", "QTH", "QSL", "QRP", "QRO", "QSO",
    "QRT", "QRX", "QSK", "QRL",
)
_Q_CODES_INTERROGATIVE: Final[tuple[str, ...]] = (
    "QRZ?", "QSL?", "QTH?", "QRP?", "QRO?", "QSY?", "QRL?", "QRV?",
)
# 交信終了のクロージング (SK プロサイン入り)
_QSO_CLOSINGS: Final[tuple[str, ...]] = (
    "73 TU {SK}",
    "73 GL {SK}",
    "TNX QSO 73 {SK}",
    "TU ES 73 {SK}",
    "GB 73 {SK}",
)
# RST レポートの代表値
_RST_REPORTS: Final[tuple[str, ...]] = (
    "599", "579", "569", "559", "479", "559", "339",
)
_NAMES: Final[tuple[str, ...]] = (
    "TARO", "JIRO", "KEN", "HIRO", "TOSHI", "YOSHI", "AKI", "TETSU",
)
_QTH: Final[tuple[str, ...]] = (
    "TOKYO", "OSAKA", "KANAGAWA", "YOKOHAMA", "SENDAI", "SAPPORO", "FUKUOKA",
)

EuropeanPattern = Literal[
    "random", "callsign", "rst", "qcode", "qcode_q", "cq", "qso_stamp", "qso_close"
]

DEFAULT_EUROPEAN_WEIGHTS: Final[dict[EuropeanPattern, float]] = {
    "random": 0.22,
    "callsign": 0.18,
    "rst": 0.08,
    "qcode": 0.08,
    "qcode_q": 0.10,      # 新規: ? 付き Q 符号
    "cq": 0.14,
    "qso_stamp": 0.10,
    "qso_close": 0.10,    # 新規: {SK} で終わる交信終了
}


def _gen_callsign(rng: np.random.Generator) -> str:
    """日本のアマチュアコールサイン (例: JA0XYZ) 風の文字列を生成."""
    prefix = rng.choice(["JA", "JH", "JR", "JE", "JF", "JG", "JI", "JK", "JL", "JM", "JN", "JO", "JP", "JQ", "JS", "7K", "7L", "7M", "7N"])
    area = rng.integers(0, 10)
    suffix = "".join(rng.choice(list(_EUROPEAN_LETTERS), size=int(rng.integers(2, 4))))
    return f"{prefix}{area}{suffix}"


def _gen_random_european(rng: np.random.Generator, length: int) -> str:
    pool = _EUROPEAN_LETTERS + _EUROPEAN_DIGITS
    return "".join(rng.choice(list(pool), size=length))


def _gen_cq(rng: np.random.Generator) -> str:
    call = _gen_callsign(rng)
    return f"CQ CQ CQ DE {call} {call} K"


def _gen_qso_stamp(rng: np.random.Generator) -> str:
    name = rng.choice(_NAMES)
    qth = rng.choice(_QTH)
    rst = rng.choice(_RST_REPORTS)
    return f"UR RST {rst} NAME {name} QTH {qth}"


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights sum must be positive")
    return {k: v / total for k, v in weights.items()}


def generate_european_text(
    rng: np.random.Generator,
    length_range: tuple[int, int] = (5, 30),
    pattern_weights: dict[EuropeanPattern, float] | None = None,
) -> str:
    """欧文学習テキストを 1 サンプル生成."""
    weights = _normalize_weights(
        dict(pattern_weights or DEFAULT_EUROPEAN_WEIGHTS)
    )
    keys = list(weights.keys())
    probs = list(weights.values())
    pattern = rng.choice(keys, p=probs)

    if pattern == "random":
        length = int(rng.integers(length_range[0], length_range[1] + 1))
        return _gen_random_european(rng, length)
    if pattern == "callsign":
        return _gen_callsign(rng)
    if pattern == "rst":
        return f"RST {rng.choice(_RST_REPORTS)}"
    if pattern == "qcode":
        return rng.choice(_Q_CODES)
    if pattern == "qcode_q":
        return str(rng.choice(_Q_CODES_INTERROGATIVE))
    if pattern == "cq":
        return _gen_cq(rng)
    if pattern == "qso_stamp":
        return _gen_qso_stamp(rng)
    if pattern == "qso_close":
        return str(rng.choice(_QSO_CLOSINGS))
    raise ValueError(f"Unknown pattern: {pattern}")


# ============================================================
# 和文パターン
# ============================================================
_PLAIN_KANA: Final[tuple[str, ...]] = tuple(
    "イロハニホヘトチリヌルヲワカヨタレソツネナラムウヰノオクヤマケフコエテアサキユメミシヱヒモセスン"
)
# 濁音化可能なカナ (和文モールスの標準範囲. ヴ は非標準のため除外)
_DAKUTEN_BASE: Final[tuple[str, ...]] = tuple(
    "カキクケコサシスセソタチツテトハヒフヘホ"
)
_HANDAKUTEN_BASE: Final[tuple[str, ...]] = tuple("ハヒフヘホ")
_DAKUTEN_KANA: Final[tuple[str, ...]] = tuple(
    "ガギグゲゴザジズゼゾダヂヅデドバビブベボ"
)
_HANDAKUTEN_KANA: Final[tuple[str, ...]] = tuple("パピプペポ")
_CHOON: Final[str] = "ー"
_KUTEN: Final[str] = "、"
_DANRAKU: Final[str] = "。"

# ラグチュー風定型文
_JA_GREETINGS: Final[tuple[str, ...]] = (
    "コンバンハ", "コンニチハ", "オハヨウ", "ヨロシク",
)
_JA_WEATHER: Final[tuple[str, ...]] = (
    "テンキ ハ ハレ", "テンキ ハ アメ", "テンキ ハ クモリ", "サムイ デス", "アツイ デス",
)
_JA_RIG: Final[tuple[str, ...]] = (
    "リグ ハ アイシー", "リグ ハ ヤエス", "アンテナ ハ ダイポール",
    "ワツテージ ハ コ", "シカタ ナイ", "アンテナ ハ ロングワイヤ",
)

JapanesePattern = Literal[
    "random", "greeting", "weather", "rig", "lagchew", "horerata"
]

DEFAULT_JAPANESE_WEIGHTS: Final[dict[JapanesePattern, float]] = {
    "random": 0.25,
    "greeting": 0.12,
    "weather": 0.12,
    "rig": 0.12,
    "lagchew": 0.19,
    "horerata": 0.20,     # ホレ/ラタ 含むパターンの比率を増加
}


def _gen_random_japanese(
    rng: np.random.Generator,
    length: int,
    dakuten_prob: float = 0.15,
    choon_prob: float = 0.05,
    kuten_prob: float = 0.03,
    danraku_prob: float = 0.02,
) -> str:
    """ランダム和文 (濁音・半濁音・長音・区切点・段落を混入)."""
    out: list[str] = []
    for _ in range(length):
        r = rng.random()
        if r < dakuten_prob * 0.7:
            out.append(str(rng.choice(_DAKUTEN_KANA)))
        elif r < dakuten_prob:
            out.append(str(rng.choice(_HANDAKUTEN_KANA)))
        elif r < dakuten_prob + choon_prob and out and out[-1] in _PLAIN_KANA:
            out.append(_CHOON)
        elif r < dakuten_prob + choon_prob + kuten_prob:
            out.append(_KUTEN)
        elif r < dakuten_prob + choon_prob + kuten_prob + danraku_prob:
            out.append(_DANRAKU)
        else:
            out.append(str(rng.choice(_PLAIN_KANA)))
    return "".join(out)


def _gen_lagchew(rng: np.random.Generator) -> str:
    parts = [
        str(rng.choice(_JA_GREETINGS)),
        str(rng.choice(_JA_WEATHER)),
        str(rng.choice(_JA_RIG)),
    ]
    return _KUTEN.join(parts)


def _gen_horerata(rng: np.random.Generator) -> str:
    """ホレ ... ラタ で囲まれた和文サンプル (運用通り)."""
    pattern_choice = rng.random()
    if pattern_choice < 0.4:
        body = str(rng.choice(_JA_GREETINGS + _JA_WEATHER))
    elif pattern_choice < 0.7:
        body = _gen_lagchew(rng)
    else:
        body_len = int(rng.integers(3, 12))
        body = _gen_random_japanese(rng, body_len)
    # 半分は両側、それ以外は片側のみ
    side = rng.random()
    if side < 0.6:
        return f"{{HORE}}{body}{{RATA}}"
    elif side < 0.8:
        return f"{{HORE}}{body}"
    else:
        return f"{body}{{RATA}}"


def generate_japanese_text(
    rng: np.random.Generator,
    length_range: tuple[int, int] = (5, 30),
    pattern_weights: dict[JapanesePattern, float] | None = None,
) -> str:
    """和文学習テキストを 1 サンプル生成."""
    weights = _normalize_weights(
        dict(pattern_weights or DEFAULT_JAPANESE_WEIGHTS)
    )
    keys = list(weights.keys())
    probs = list(weights.values())
    pattern = rng.choice(keys, p=probs)

    if pattern == "random":
        length = int(rng.integers(length_range[0], length_range[1] + 1))
        return _gen_random_japanese(rng, length)
    if pattern == "greeting":
        return str(rng.choice(_JA_GREETINGS))
    if pattern == "weather":
        return str(rng.choice(_JA_WEATHER))
    if pattern == "rig":
        return str(rng.choice(_JA_RIG))
    if pattern == "lagchew":
        return _gen_lagchew(rng)
    if pattern == "horerata":
        return _gen_horerata(rng)
    raise ValueError(f"Unknown pattern: {pattern}")


# ============================================================
# モード横断ジェネレータ
# ============================================================
@dataclass
class TextGenConfig:
    """テキスト生成設定."""

    european_length_range: tuple[int, int] = (5, 30)
    japanese_length_range: tuple[int, int] = (5, 30)
    european_pattern_weights: dict[EuropeanPattern, float] = field(
        default_factory=lambda: dict(DEFAULT_EUROPEAN_WEIGHTS)
    )
    japanese_pattern_weights: dict[JapanesePattern, float] = field(
        default_factory=lambda: dict(DEFAULT_JAPANESE_WEIGHTS)
    )


def generate_text(
    rng: np.random.Generator,
    mode: Literal["european", "japanese"],
    config: TextGenConfig | None = None,
) -> str:
    cfg = config or TextGenConfig()
    if mode == "european":
        return generate_european_text(
            rng,
            length_range=cfg.european_length_range,
            pattern_weights=cfg.european_pattern_weights,
        )
    if mode == "japanese":
        return generate_japanese_text(
            rng,
            length_range=cfg.japanese_length_range,
            pattern_weights=cfg.japanese_pattern_weights,
        )
    raise ValueError(f"Unknown mode: {mode}")


__all__ = [
    "DEFAULT_EUROPEAN_WEIGHTS",
    "DEFAULT_JAPANESE_WEIGHTS",
    "EuropeanPattern",
    "JapanesePattern",
    "TextGenConfig",
    "generate_european_text",
    "generate_japanese_text",
    "generate_text",
]
