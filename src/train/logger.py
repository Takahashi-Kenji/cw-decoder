"""学習指標の CSV ロガー."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CSVLogger:
    """1 行 1 ステップの CSV ロガー.

    フィールドは初回 ``log`` 呼び出し時のキーで固定. 以降は同じキーセットで
    ログすること.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open(encoding="utf-8") as f:
                first = f.readline().strip()
                if first:
                    self._fields = first.split(",")

    def log(self, **fields: Any) -> None:
        new_file = self._fields is None
        if new_file:
            self._fields = list(fields.keys())
            with self.path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(self._fields)
        with self.path.open("a", encoding="utf-8", newline="") as f:
            row = [fields.get(k, "") for k in (self._fields or [])]
            csv.writer(f).writerow(row)


__all__ = ["CSVLogger"]
