"""評価指標のテスト."""
from __future__ import annotations

import pytest

from src.train.metrics import (
    AggregateMetrics,
    DetailedEvalReport,
    EvalRecord,
    EvalReport,
    bin_snr,
    bin_wpm,
    error_rate,
    levenshtein_distance,
)


class TestLevenshtein:
    @pytest.mark.parametrize("a,b,expected", [
        ([1, 2, 3], [1, 2, 3], 0),
        ([1, 2, 3], [1, 2, 4], 1),    # 1 substitution
        ([1, 2, 3], [1, 2], 1),       # 1 deletion
        ([1, 2, 3], [1, 2, 3, 4], 1), # 1 insertion
        ("kitten", "sitting", 3),
        ([], [], 0),
        ([1, 2], [], 2),
        ([], [1, 2], 2),
    ])
    def test_distance(self, a, b, expected) -> None:
        assert levenshtein_distance(a, b) == expected


class TestErrorRate:
    def test_basic(self) -> None:
        assert error_rate("ABCD", "ABCD") == 0.0
        assert error_rate("ABXD", "ABCD") == pytest.approx(0.25)
        assert error_rate("AB", "ABCD") == pytest.approx(0.5)

    def test_empty_reference(self) -> None:
        assert error_rate("", "") == 0.0
        assert error_rate("ABC", "") == 1.0


class TestAggregateMetrics:
    def test_combine_two_samples(self) -> None:
        m = AggregateMetrics()
        rec1 = EvalRecord(
            ref_tokens=[1, 2, 3], pred_tokens=[1, 2, 3],
            ref_text="ABC", pred_text="ABC",
        )
        rec2 = EvalRecord(
            ref_tokens=[4, 5], pred_tokens=[4, 6],
            ref_text="DE", pred_text="DX",
        )
        m.add(rec1)
        m.add(rec2)
        # token errors: 0 + 1 = 1, ref tokens = 5 → TER = 0.2
        assert m.ter == pytest.approx(0.2)
        assert m.cer == pytest.approx(0.2)
        assert m.n_samples == 2


class TestEvalReport:
    def test_breakdown_by_snr(self) -> None:
        report = EvalReport()
        rec_good = EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[1, 2],
            ref_text="AB", pred_text="AB",
            snr_db=10.0,
        )
        rec_bad = EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[3, 4],
            ref_text="AB", pred_text="XY",
            snr_db=0.0,
        )
        report.add(rec_good, snr_bin=10.0)
        report.add(rec_bad, snr_bin=0.0)
        assert report.by_snr[10.0].ter == 0.0
        assert report.by_snr[0.0].ter == 1.0
        assert report.overall.ter == 0.5

    def test_summary_lines_formatted(self) -> None:
        report = EvalReport()
        rec = EvalRecord(
            ref_tokens=[1], pred_tokens=[1],
            ref_text="A", pred_text="A",
        )
        report.add(rec, snr_bin=10.0, wpm_bin=20.0)
        lines = report.summary_lines()
        assert any("Overall" in l for l in lines)
        assert any("SNR" in l for l in lines)
        assert any("WPM" in l for l in lines)

    def test_breakdown_by_eff_snr(self) -> None:
        report = EvalReport()
        rec_good = EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[1, 2],
            ref_text="AB", pred_text="AB", eff_snr_db=9.5,
        )
        rec_bad = EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[3, 4],
            ref_text="AB", pred_text="XY", eff_snr_db=-0.5,
        )
        report.add(rec_good, eff_snr_bin=10.0)
        report.add(rec_bad, eff_snr_bin=0.0)
        assert report.by_eff_snr[10.0].ter == 0.0
        assert report.by_eff_snr[0.0].ter == 1.0
        assert report.overall.ter == 0.5

    def test_eff_snr_defaults_none_and_optional(self) -> None:
        # eff_snr を渡さなくても既存どおり動く
        report = EvalReport()
        rec = EvalRecord(ref_tokens=[1], pred_tokens=[1], ref_text="A", pred_text="A")
        assert rec.eff_snr_db is None
        report.add(rec, snr_bin=10.0)
        assert report.by_eff_snr == {}


class TestBinning:
    @pytest.mark.parametrize("snr,expected", [
        (10.0, 10.0),
        (12.5, 15.0),    # 12.5 rounds to nearest 5 = 10 or 15 (banker's rounding handled)
        (-3.0, -5.0),
        (0.5, 0.0),
    ])
    def test_bin_snr(self, snr: float, expected: float) -> None:
        result = bin_snr(snr)
        # round-half-to-even may give 10 or 15 for 12.5; accept either
        assert result in (expected, expected - 5.0, expected + 5.0)
        assert abs(result - snr) <= 2.5 + 0.001

    @pytest.mark.parametrize("wpm,expected", [
        (20.0, 20.0),
        (17.0, 15.0),
        (23.0, 25.0),
    ])
    def test_bin_wpm(self, wpm: float, expected: float) -> None:
        assert bin_wpm(wpm) == expected


class TestDetailedReportByModeEffSnr:
    def _report(self) -> DetailedEvalReport:
        d = DetailedEvalReport()
        # 欧文2件 (1件 TER0, 1件 TER1), eff_snr 10.0
        d.add(EvalRecord(ref_tokens=[1, 2], pred_tokens=[1, 2], ref_text="AB", pred_text="AB",
                         eff_snr_db=10.0), name="e1", mode="european", eff_snr_bin=10.0)
        d.add(EvalRecord(ref_tokens=[1, 2], pred_tokens=[3, 4], ref_text="AB", pred_text="XY",
                         eff_snr_db=0.0), name="e2", mode="european", eff_snr_bin=0.0)
        # 和文1件 TER0, eff_snr 10.0
        d.add(EvalRecord(ref_tokens=[5], pred_tokens=[5], ref_text="C", pred_text="C",
                         eff_snr_db=10.0), name="j1", mode="japanese", eff_snr_bin=10.0)
        return d

    def test_by_mode_separates_modes(self) -> None:
        bm = self._report().by_mode()
        assert bm["european"].n_samples == 2
        assert bm["japanese"].n_samples == 1
        assert bm["european"].ter == 0.5   # (0 + 2 errors) / 4 ref tokens
        assert bm["japanese"].ter == 0.0

    def test_to_dict_has_by_eff_snr_and_by_mode(self) -> None:
        import json
        payload = self._report().to_dict()
        # 既存キーは維持
        assert "overall" in payload and "totals" in payload and "token_errors" in payload
        # 新キー
        assert payload["by_mode"]["european"]["n_samples"] == 2
        assert payload["by_mode"]["japanese"]["ter"] == 0.0
        assert payload["by_eff_snr"]["10.0"]["n_samples"] == 2
        assert payload["by_eff_snr"]["0.0"]["ter"] == 1.0
        json.dumps(payload, ensure_ascii=False)  # JSON 化できる

    def test_summary_lines_has_effsnr_section(self) -> None:
        lines = self._report().summary_lines()
        assert any("By EffSNR" in l for l in lines)

    def test_empty_report_has_empty_by_mode(self) -> None:
        d = DetailedEvalReport()
        payload = d.to_dict()
        assert payload["by_mode"] == {}
        assert payload["by_eff_snr"] == {}
        assert not any("By EffSNR" in l for l in d.summary_lines())
