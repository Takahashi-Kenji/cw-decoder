"""動作点 → 推奨モデル (`src/infer/model_recommend.py`)."""

from __future__ import annotations

from src.infer.model_recommend import (
    CENTROIDS,
    DEFAULT_MODEL_FOR_CLASS,
    OperatingClass,
    classify,
    describe,
)
from src.infer.operating_point import OperatingPoint


def op(char_gap: float, contrast: float) -> OperatingPoint:
    return OperatingPoint(wpm=24.0, dash_dot_ratio=3.0, intra_gap_units=1.0,
                          char_gap_units=char_gap, contrast_db=contrast, n_dot=20, n_dash=10)


class TestClassify:
    def test_重心そのものはそのクラス(self) -> None:
        for c in CENTROIDS:
            rec = classify(op(c.char_gap_units, c.contrast_db))
            assert rec.cls == c.cls
            assert rec.distance == 0.0
            assert rec.confident

    def test_実測の代表値が期待どおりに落ちる(self) -> None:
        """2026-08-27 の 4 集合の中央値 (WPM は無視される)."""
        assert classify(op(3.15, 8.4)).cls == OperatingClass.WEAK_TIGHT      # 旧録音
        assert classify(op(3.25, 9.0)).cls == OperatingClass.WEAK_TIGHT      # L4 5W
        assert classify(op(3.26, 16.4)).cls == OperatingClass.STRONG_TIGHT   # L4 50W
        assert classify(op(4.38, 28.4)).cls == OperatingClass.STRONG_WIDE    # held-out

    def test_境界の近くは確信なし(self) -> None:
        """弱・狭 (8.8 dB) と 強・狭 (16.4 dB) の中間."""
        rec = classify(op(3.3, 12.6))
        assert not rec.confident

    def test_全クラスにモデルが割り当てられている(self) -> None:
        assert set(DEFAULT_MODEL_FOR_CLASS) == set(OperatingClass)


class TestDescribe:
    def test_表示文字列(self) -> None:
        o = op(3.3, 9.0)
        text = describe(o, classify(o))
        assert text.startswith("文字間 3.3 / 9 dB → ")
        assert "p2b" in text        # 弱・狭の既定 (運用者が実受信で選んだ)

    def test_確信がなければ疑問符(self) -> None:
        o = op(3.3, 12.6)
        assert "?" in describe(o, classify(o))


class TestRegisteredModels:
    def test_実在するものだけ返す(self, tmp_path) -> None:
        from src.infer.model_recommend import registered_models
        (tmp_path / "models" / "p2b_presilence").mkdir(parents=True)
        (tmp_path / "models" / "p2b_presilence" / "cw_p2b.onnx").write_bytes(b"x")
        got = registered_models(tmp_path)
        assert [m.label for m in got] == ["p2b"]
        assert got[0].cls == OperatingClass.WEAK_TIGHT
        assert "弱信号" in got[0].display

    def test_何も無ければ空(self, tmp_path) -> None:
        from src.infer.model_recommend import registered_models
        assert registered_models(tmp_path) == []

    def test_追加候補も実在すれば出る(self, tmp_path) -> None:
        from src.infer.model_recommend import registered_models
        (tmp_path / "models" / "wabun").mkdir(parents=True)
        (tmp_path / "models" / "wabun" / "best_infer.pt").write_bytes(b"x")
        got = registered_models(tmp_path)
        assert len(got) == 1 and got[0].label.startswith("wabun")
        assert got[0].cls == OperatingClass.WEAK_TIGHT

    def test_未分類の候補は表示名がラベルだけ(self, tmp_path) -> None:
        from src.infer.model_recommend import registered_models
        (tmp_path / "models" / "baseline_v4").mkdir(parents=True)
        (tmp_path / "models" / "baseline_v4" / "best_infer.pt").write_bytes(b"x")
        got = registered_models(tmp_path)
        assert got[0].cls is None and got[0].display == got[0].label
