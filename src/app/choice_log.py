"""モデルの選択と推奨を記録する (計画 v5 P4-3).

**自動切替に進むかどうかは、この記録で決める。** 推奨 (`model_recommend`) と
運用者の実際の選択が一致していれば自動化する価値があり、食い違っていれば
推奨側 (重心・軸) を直す。どちらかを判断できる記録が無いまま自動化しない。

1 行 1 JSON (`logs/model_choice.jsonl`)。書けなくても**アプリを止めない**。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_PATH = Path("logs") / "model_choice.jsonl"


@dataclass(frozen=True)
class ChoiceRecord:
    """1 件の記録. ``event`` は ``"select"`` (運用者が選んだ) か ``"recommend"`` (推奨が変わった)."""

    timestamp: str
    event: str
    model_path: str | None          # select: 選んだモデル / recommend: そのとき使っていたモデル
    recommended_class: str | None   # 推奨クラス (表示名)。測れていなければ None
    recommended_label: str | None   # 推奨モデルのラベル
    confident: bool | None
    char_gap_units: float | None
    contrast_db: float | None
    wpm: float | None


class ChoiceLog:
    def __init__(self, path: Path | str = DEFAULT_LOG_PATH) -> None:
        self.path = Path(path)
        self._last_recommended: str | None = None

    def _write(self, record: ChoiceRecord) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except OSError:
            # **記録が書けなくてもデコードを止めない。**
            pass

    def select(self, model_path: str, payload: object) -> None:
        """運用者がモデルを選んだ. ``payload`` は直近の (OperatingPoint, Recommendation) か None."""
        self._write(self._record("select", model_path, payload))

    def recommend(self, model_path: str | None, payload: object) -> None:
        """推奨が変わったときだけ書く (3 秒ごとに同じ行を積まない)."""
        rec = self._record("recommend", model_path, payload)
        key = f"{rec.recommended_class}|{rec.confident}"
        if key == self._last_recommended:
            return
        self._last_recommended = key
        self._write(rec)

    @staticmethod
    def _record(event: str, model_path: str | None, payload: object) -> ChoiceRecord:
        now = datetime.now().isoformat(timespec="seconds")
        if payload is None:
            return ChoiceRecord(now, event, model_path, None, None, None, None, None, None)
        op, rec = payload
        from src.infer.model_recommend import DEFAULT_MODEL_FOR_CLASS
        label, _ = DEFAULT_MODEL_FOR_CLASS[rec.cls]
        return ChoiceRecord(
            timestamp=now, event=event, model_path=model_path,
            recommended_class=rec.cls.value, recommended_label=label,
            confident=bool(rec.confident),
            char_gap_units=float(op.char_gap_units), contrast_db=float(op.contrast_db),
            wpm=float(op.wpm),
        )


def agreement(path: Path | str = DEFAULT_LOG_PATH) -> tuple[int, int]:
    """記録から「選択が推奨と一致した回数 / 推奨があった選択の回数」を返す.

    自動切替へ進む判断材料。一致の定義は「選んだモデルのファイル名に推奨ラベルを含む」。
    """
    p = Path(path)
    if not p.exists():
        return 0, 0
    hit = total = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") != "select" or not d.get("recommended_label") or not d.get("model_path"):
            continue
        total += 1
        if d["recommended_label"] in Path(d["model_path"]).as_posix():
            hit += 1
    return hit, total


__all__ = ["DEFAULT_LOG_PATH", "ChoiceLog", "ChoiceRecord", "agreement"]
