"""動作点から**どのモデルが得意か**を推す (計画 v5 P4).

**自動で切り替えない。推奨を表示するだけ。** 運用者が選び、その記録が溜まって
「推奨と選択が一致する」と分かってから自動化する (計画 v5 P4-3)。

判別は P3 で測った 3 動作点の重心への最近傍 (2 軸: 文字間・コントラスト、
各軸を全体の標準偏差で正規化)。leave-one-out で **98.8%** (2026-08-27、
`scripts/measure_operating_points.py`)。**WPM は使わない** — L4 は意図的に
15〜30 WPM を混ぜており、合成が 8〜50 WPM を覆うのでどのモデルも速度には対応する。

重心は測定日の値を定数で持つ。**モデルや収集条件が変わったら測り直して更新すること。**
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.infer.operating_point import OperatingPoint


class OperatingClass(str, Enum):
    """動作点の 3 クラス. 値は表示名."""

    WEAK_TIGHT = "弱信号・狭い間隔"
    STRONG_TIGHT = "強信号・狭い間隔"
    STRONG_WIDE = "強信号・広い間隔"


@dataclass(frozen=True)
class ClassCentroid:
    cls: OperatingClass
    char_gap_units: float
    contrast_db: float
    # 何で測ったか (更新のときに出典を追えるように)
    source: str


# 2026-08-27 の実測 (`scripts/measure_operating_points.py --axes model`, 各集合 ≤120 件)。
# 弱・狭 = 旧録音 72 + L4 5W 119 / 強・狭 = L4 50W 120 / 強・広 = held-out 20。
CENTROIDS: tuple[ClassCentroid, ...] = (
    ClassCentroid(OperatingClass.WEAK_TIGHT, 3.42, 8.8, "旧録音 + L4 8/24 5W (2026-08-27)"),
    ClassCentroid(OperatingClass.STRONG_TIGHT, 3.26, 16.4, "L4 8/25 50W (2026-08-27)"),
    ClassCentroid(OperatingClass.STRONG_WIDE, 4.38, 28.4, "held-out data/keying_scripts (2026-08-27)"),
)

# 正規化の尺度 = 全集合を合わせた各軸の標準偏差 (同日の測定)。
# 文字間は 0.5 dot、コントラストは 7 dB が「1 単位」。
AXIS_SCALE: tuple[float, float] = (0.5, 7.0)

# クラス → 得意なモデル (表示用ラベル, 既定のファイル)。ファイルは設定で上書きできる。
#
# 弱・狭の既定は **p2b** (2026-08-28)。運用者が実受信で wabun / p2c / 他と聴き比べ、
# 「最も誤り率がなく、正確度が高く信頼できる」と判断した。3 集合の合格 (held-out
# 16.31% / L4 弱 0.80% / 先頭誤り 7) と実受信の判断が一致した初めてのモデル。
DEFAULT_MODEL_FOR_CLASS: dict[OperatingClass, tuple[str, str]] = {
    OperatingClass.WEAK_TIGHT: ("p2b", "models/p2b_presilence/cw_p2b.onnx"),
    OperatingClass.STRONG_TIGHT: ("full_v5", "models/full_v5/best_infer.pt"),
    OperatingClass.STRONG_WIDE: ("ft_v5_wide", "models/ft_v5_wide/cw_v5wide.onnx"),
}


# 既定以外にも切替リストに出す候補 (ラベル, ファイル, 得意な動作点)。
# 実受信で比べるための候補。**推奨 (describe) には使わない** — 既定が推奨。
EXTRA_MODELS: tuple[tuple[str, str, OperatingClass | None], ...] = (
    ("wabun (p2b の前身)", "models/wabun/best_infer.pt", OperatingClass.WEAK_TIGHT),
    ("p2c (先頭無音+文字間+要素間)", "models/p2c_all/cw_p2c.onnx", OperatingClass.WEAK_TIGHT),
    ("baseline (第 4 版まで配布)", "models/baseline_v4/best_infer.pt", None),
)


@dataclass(frozen=True)
class RegisteredModel:
    """アプリの切替リストに出す 1 件."""

    label: str                  # 例: "wabun"
    path: str                   # 例: "models/wabun/best_infer.pt"
    cls: OperatingClass | None  # 得意な動作点 (None = 未分類)

    @property
    def display(self) -> str:
        return f"{self.label} — {self.cls.value}" if self.cls else self.label


def registered_models(root: "Path | str | None" = None) -> list[RegisteredModel]:
    """`DEFAULT_MODEL_FOR_CLASS` のうち**ファイルが実在するもの**を切替リスト用に返す.

    存在しないものは出さない (配布版には同梱されないモデルがある)。
    """
    from pathlib import Path
    base = Path(root) if root is not None else Path.cwd()
    out: list[RegisteredModel] = []
    entries = [(label, rel, cls) for cls, (label, rel) in DEFAULT_MODEL_FOR_CLASS.items()]
    entries += list(EXTRA_MODELS)
    for label, rel, cls in entries:
        candidate = base / rel
        if candidate.exists():
            out.append(RegisteredModel(label=label, path=str(candidate), cls=cls))
    return out


@dataclass(frozen=True)
class Recommendation:
    cls: OperatingClass
    distance: float             # 正規化した重心までの距離 (小さいほど確信)
    margin: float               # 2 位との差 (小さいほど境界に近い)

    @property
    def confident(self) -> bool:
        """境界から十分離れているか. 0.5 単位未満なら「どちらとも言えない」."""
        return self.margin >= 0.5


def classify(op: OperatingPoint) -> Recommendation:
    """動作点を 3 クラスに振り分ける (最近傍重心)."""
    x = op.model_axes()
    scale = np.asarray(AXIS_SCALE, dtype=np.float64)
    dists = sorted(
        (float(np.linalg.norm((x - np.array([c.char_gap_units, c.contrast_db])) / scale)), c.cls)
        for c in CENTROIDS
    )
    best_d, best_cls = dists[0]
    second_d = dists[1][0] if len(dists) > 1 else float("inf")
    return Recommendation(cls=best_cls, distance=best_d, margin=second_d - best_d)


def describe(op: OperatingPoint, rec: Recommendation) -> str:
    """ステータスバー向けの 1 行. 例: ``文字間 3.3 / 9 dB → 弱信号・狭い間隔 (wabun)``."""
    label, _ = DEFAULT_MODEL_FOR_CLASS[rec.cls]
    tail = f"{rec.cls.value} ({label})" if rec.confident else f"{rec.cls.value}? ({label})"
    return f"文字間 {op.char_gap_units:.1f} / {op.contrast_db:.0f} dB → {tail}"


__all__ = [
    "AXIS_SCALE",
    "CENTROIDS",
    "DEFAULT_MODEL_FOR_CLASS",
    "EXTRA_MODELS",
    "ClassCentroid",
    "OperatingClass",
    "Recommendation",
    "RegisteredModel",
    "registered_models",
    "classify",
    "describe",
]
