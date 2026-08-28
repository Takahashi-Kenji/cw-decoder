"""モデル選択の記録 (`src/app/choice_log.py`)."""

from __future__ import annotations

import json
from pathlib import Path

from src.app.choice_log import ChoiceLog, agreement
from src.infer.model_recommend import classify
from src.infer.operating_point import OperatingPoint


def payload(char_gap: float, contrast: float):
    op = OperatingPoint(wpm=24.0, dash_dot_ratio=3.0, intra_gap_units=1.0,
                        char_gap_units=char_gap, contrast_db=contrast, n_dot=20, n_dash=10)
    return op, classify(op)


class TestChoiceLog:
    def test_選択を記録する(self, tmp_path: Path) -> None:
        log = ChoiceLog(tmp_path / "c.jsonl")
        log.select("models/p2b_presilence/cw_p2b.onnx", payload(3.3, 9.0))
        rows = [json.loads(l) for l in (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows[0]["event"] == "select"
        assert rows[0]["recommended_label"] == "p2b"
        assert rows[0]["char_gap_units"] == 3.3

    def test_推奨が測れていなくても選択は記録する(self, tmp_path: Path) -> None:
        log = ChoiceLog(tmp_path / "c.jsonl")
        log.select("models/x.pt", None)
        row = json.loads((tmp_path / "c.jsonl").read_text(encoding="utf-8"))
        assert row["recommended_class"] is None and row["model_path"] == "models/x.pt"

    def test_同じ推奨は繰り返し書かない(self, tmp_path: Path) -> None:
        """3 秒ごとに同じ行を積まない."""
        log = ChoiceLog(tmp_path / "c.jsonl")
        for _ in range(5):
            log.recommend("m.pt", payload(3.3, 9.0))
        log.recommend("m.pt", payload(4.4, 28.0))
        lines = (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_書けなくても落ちない(self, tmp_path: Path) -> None:
        bad = tmp_path / "file_not_dir"
        bad.write_text("x", encoding="utf-8")
        log = ChoiceLog(bad / "c.jsonl")      # 親がファイルなので作れない
        log.select("m.pt", None)              # 例外にならないこと


class TestAgreement:
    def test_一致率を数える(self, tmp_path: Path) -> None:
        log = ChoiceLog(tmp_path / "c.jsonl")
        log.select("models/p2b_presilence/cw_p2b.onnx", payload(3.3, 9.0))  # 推奨 p2b → 一致
        log.select("models/full_v5/best_infer.pt", payload(3.3, 9.0))       # 推奨 p2b → 不一致
        log.select("models/x.pt", None)                                  # 推奨なし → 数えない
        assert agreement(tmp_path / "c.jsonl") == (1, 2)

    def test_無ければゼロ(self, tmp_path: Path) -> None:
        assert agreement(tmp_path / "none.jsonl") == (0, 0)
