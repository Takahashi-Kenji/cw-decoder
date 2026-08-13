"""report 比較ロジックのテスト."""
from __future__ import annotations

from src.eval.compare import compare_reports


def _report(overall_ter, eff, mode, tokens):
    return {
        "synth_val": {
            "overall": {"ter": overall_ter, "cer": overall_ter, "n_samples": 10},
            "by_eff_snr": eff,
            "by_mode": mode,
            "token_errors": tokens,
        }
    }


class TestCompareReports:
    def test_overall_ter_delta(self) -> None:
        base = _report(0.45, {}, {}, [])
        cur = _report(0.31, {}, {}, [])
        out = compare_reports(base, cur)
        d = out["synth_val"]["overall"]
        assert d["ter_baseline"] == 0.45
        assert d["ter_current"] == 0.31
        assert round(d["ter_delta"], 2) == -0.14   # 減少 = 改善

    def test_by_eff_snr_delta_and_missing_bin(self) -> None:
        base = _report(0.4, {"0.0": {"ter": 0.45}, "5.0": {"ter": 0.03}}, {}, [])
        cur = _report(0.3, {"0.0": {"ter": 0.22}}, {}, [])  # 5.0 が current に無い
        out = compare_reports(base, cur)["synth_val"]["by_eff_snr"]
        assert round(out["0.0"]["ter_delta"], 2) == -0.23
        assert out["5.0"]["ter_delta"] == "n/a"   # 欠損ビン

    def test_by_mode_delta(self) -> None:
        base = _report(0.4, {}, {"european": {"ter": 0.30}, "japanese": {"ter": 0.55}}, [])
        cur = _report(0.3, {}, {"european": {"ter": 0.20}, "japanese": {"ter": 0.40}}, [])
        out = compare_reports(base, cur)["synth_val"]["by_mode"]
        assert round(out["european"]["ter_delta"], 2) == -0.10
        assert round(out["japanese"]["ter_delta"], 2) == -0.15

    def test_token_recall_improvements_sorted(self) -> None:
        base = _report(0.4, {}, {}, [
            {"token_id": 10, "code": "・・・-・", "recall": 0.0},
            {"token_id": 11, "code": "・", "recall": 0.9},
        ])
        cur = _report(0.3, {}, {}, [
            {"token_id": 10, "code": "・・・-・", "recall": 0.6},
            {"token_id": 11, "code": "・", "recall": 0.92},
        ])
        imp = compare_reports(base, cur)["synth_val"]["token_recall_improvements"]
        # token 10 の改善 (+0.6) が token 11 (+0.02) より上位
        assert imp[0]["token_id"] == 10
        assert round(imp[0]["recall_delta"], 2) == 0.60

    def test_missing_section_skipped(self) -> None:
        # keyed_val が baseline に無い → keyed_val は結果に出ない
        base = _report(0.4, {}, {}, [])
        cur = {**_report(0.3, {}, {}, []), "keyed_val": {"overall": {"ter": 0.2}}}
        out = compare_reports(base, cur)
        assert "synth_val" in out
        assert "keyed_val" not in out
