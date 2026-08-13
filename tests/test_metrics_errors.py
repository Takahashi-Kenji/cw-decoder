"""token 別エラー分析・confusion matrix のテスト (Phase A 評価基盤).

参照: CODEX_REVIEW_RESULT.md §5 優先度 4「token 別エラー分析」
"""
from __future__ import annotations

import json

import pytest

from src.tokens.morse_tokens import TOKEN_TO_ID, WORD_BREAK_TOKEN_ID
from src.train.metrics import (
    DetailedEvalReport,
    EditOp,
    EvalRecord,
    TokenErrorAnalysis,
    align_sequences,
    describe_token,
    levenshtein_distance,
    token_label,
)


def _kinds(ops: list[EditOp]) -> list[str]:
    return [op.kind for op in ops]


class TestAlignSequences:
    def test_identical_sequences_are_all_equal(self) -> None:
        ops = align_sequences([1, 2, 3], [1, 2, 3])
        assert _kinds(ops) == ["equal", "equal", "equal"]
        assert all(op.ref_token == op.pred_token for op in ops)

    def test_substitution(self) -> None:
        ops = align_sequences([1, 9, 3], [1, 2, 3])
        assert _kinds(ops) == ["equal", "substitution", "equal"]
        sub = ops[1]
        assert sub.ref_token == 2
        assert sub.pred_token == 9
        assert sub.ref_index == 1
        assert sub.pred_index == 1

    def test_insertion_is_extra_token_in_pred(self) -> None:
        # pred に余分な 9 がある → insertion
        ops = align_sequences([1, 9, 2], [1, 2])
        assert _kinds(ops) == ["equal", "insertion", "equal"]
        ins = ops[1]
        assert ins.pred_token == 9
        assert ins.ref_token is None
        assert ins.ref_index is None

    def test_deletion_is_missing_token_in_pred(self) -> None:
        # ref の 2 が pred に無い → deletion
        ops = align_sequences([1, 3], [1, 2, 3])
        assert _kinds(ops) == ["equal", "deletion", "equal"]
        dele = ops[1]
        assert dele.ref_token == 2
        assert dele.pred_token is None
        assert dele.pred_index is None

    def test_empty_pred_gives_all_deletions(self) -> None:
        ops = align_sequences([], [1, 2, 3])
        assert _kinds(ops) == ["deletion"] * 3

    def test_empty_ref_gives_all_insertions(self) -> None:
        ops = align_sequences([1, 2], [])
        assert _kinds(ops) == ["insertion"] * 2

    def test_both_empty(self) -> None:
        assert align_sequences([], []) == []

    @pytest.mark.parametrize("pred,ref", [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([1, 2, 3], [1, 2]),
        ([1, 2, 3], [1, 2, 3, 4]),
        ([5, 1, 9, 2, 2], [1, 2, 3, 4]),
        ([], [1, 2, 3]),
        ([1, 2], []),
        ("kitten", "sitting"),
    ])
    def test_op_count_matches_levenshtein_distance(self, pred, ref) -> None:
        """非 equal の op 数 == 編集距離 (既存 API との整合)."""
        ops = align_sequences(pred, ref)
        n_errors = sum(1 for op in ops if op.kind != "equal")
        assert n_errors == levenshtein_distance(pred, ref)

    def test_ops_reconstruct_both_sequences(self) -> None:
        """op 列から pred / ref を復元できる (アライメントの健全性)."""
        pred, ref = [5, 1, 9, 2, 2], [1, 2, 3, 4]
        ops = align_sequences(pred, ref)
        rebuilt_pred = [op.pred_token for op in ops if op.pred_index is not None]
        rebuilt_ref = [op.ref_token for op in ops if op.ref_index is not None]
        assert rebuilt_pred == pred
        assert rebuilt_ref == ref

    def test_indices_are_monotonic(self) -> None:
        ops = align_sequences([5, 1, 9, 2, 2], [1, 2, 3, 4])
        ref_idx = [op.ref_index for op in ops if op.ref_index is not None]
        pred_idx = [op.pred_index for op in ops if op.pred_index is not None]
        assert ref_idx == sorted(ref_idx) == list(range(4))
        assert pred_idx == sorted(pred_idx) == list(range(5))


class TestTokenErrorAnalysis:
    def test_counts_substitution_deletion_insertion(self) -> None:
        analysis = TokenErrorAnalysis()
        # ref=[1,2,3,4,5,6,7] pred=[1,9,3,5,6,7,8]
        #   → 2→9:sub, 4:del, 8:ins, 残りは equal (この対応は一意)
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2, 3, 4, 5, 6, 7], pred_tokens=[1, 9, 3, 5, 6, 7, 8],
            ref_text="", pred_text="",
        ))
        assert analysis.counts[2].substituted == 1
        assert analysis.counts[4].deleted == 1
        assert analysis.counts[8].inserted == 1
        for tid in (1, 3, 5, 6, 7):
            assert analysis.counts[tid].correct == 1
        # 9 は ref に存在しない token として誤出力された
        assert analysis.counts[9].wrongly_output == 1

    def test_counts_insertion(self) -> None:
        analysis = TokenErrorAnalysis()
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[1, 7, 2],
            ref_text="", pred_text="",
        ))
        assert analysis.counts[7].inserted == 1
        assert analysis.counts[7].wrongly_output == 1
        assert analysis.counts[1].correct == 1
        assert analysis.counts[2].correct == 1

    def test_ref_count_and_recall(self) -> None:
        analysis = TokenErrorAnalysis()
        # token 1 は 3 回出現し、2 回正解・1 回置換
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 1, 1], pred_tokens=[1, 1, 9],
            ref_text="", pred_text="",
        ))
        c = analysis.counts[1]
        assert c.ref_count == 3
        assert c.recall == pytest.approx(2 / 3)
        assert c.error_count == 1

    def test_accumulates_across_records(self) -> None:
        analysis = TokenErrorAnalysis()
        for _ in range(3):
            analysis.add_record(EvalRecord(
                ref_tokens=[1, 2], pred_tokens=[1, 9],
                ref_text="", pred_text="",
            ))
        assert analysis.counts[2].substituted == 3
        assert analysis.counts[1].correct == 3
        assert analysis.n_samples == 3

    def test_totals(self) -> None:
        analysis = TokenErrorAnalysis()
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2, 3], pred_tokens=[1, 9, 3],     # sub
            ref_text="", pred_text="",
        ))
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2, 3], pred_tokens=[1, 3],        # del
            ref_text="", pred_text="",
        ))
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[1, 7, 2],        # ins
            ref_text="", pred_text="",
        ))
        totals = analysis.totals
        assert totals["substitutions"] == 1
        assert totals["deletions"] == 1
        assert totals["insertions"] == 1
        assert totals["correct"] == 6
        assert totals["ref_tokens"] == 8

    def test_top_errors_sorted_by_error_count(self) -> None:
        analysis = TokenErrorAnalysis()
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 1, 1, 2], pred_tokens=[9, 9, 9, 2],
            ref_text="", pred_text="",
        ))
        top = analysis.top_errors(limit=1)
        assert len(top) == 1
        assert top[0]["token_id"] == 1
        assert top[0]["substituted"] == 3


class TestConfusionMatrix:
    def test_substitution_recorded(self) -> None:
        analysis = TokenErrorAnalysis()
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[1, 9],
            ref_text="", pred_text="",
        ))
        assert analysis.confusion[(2, 9)] == 1
        assert analysis.confusion[(1, 1)] == 1   # 対角 (正解)

    def test_deletion_and_insertion_use_none_sentinel(self) -> None:
        analysis = TokenErrorAnalysis()
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2], pred_tokens=[1, 7, 7],
            ref_text="", pred_text="",
        ))
        # 2→7 の置換 + 7 の挿入
        assert analysis.confusion[(2, 7)] == 1
        assert analysis.confusion[(None, 7)] == 1

    def test_deletion_sentinel(self) -> None:
        analysis = TokenErrorAnalysis()
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2, 3], pred_tokens=[1, 3],
            ref_text="", pred_text="",
        ))
        assert analysis.confusion[(2, None)] == 1

    def test_counts_accumulate(self) -> None:
        analysis = TokenErrorAnalysis()
        for _ in range(4):
            analysis.add_record(EvalRecord(
                ref_tokens=[2], pred_tokens=[9],
                ref_text="", pred_text="",
            ))
        assert analysis.confusion[(2, 9)] == 4

    def test_confusion_to_dict_is_json_serializable(self) -> None:
        analysis = TokenErrorAnalysis()
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2, 3], pred_tokens=[1, 9, 3],     # 2→9 sub
            ref_text="", pred_text="",
        ))
        analysis.add_record(EvalRecord(
            ref_tokens=[1, 2, 3], pred_tokens=[1, 3],        # 2 del
            ref_text="", pred_text="",
        ))
        payload = analysis.confusion_to_dict()
        dumped = json.dumps(payload, ensure_ascii=False)
        assert json.loads(dumped) == payload
        entries = {(e["ref"], e["pred"]): e["count"] for e in payload["entries"]}
        assert entries[("2", "9")] == 1
        assert entries[("2", "<DEL>")] == 1

    def test_confusion_entries_carry_labels(self) -> None:
        analysis = TokenErrorAnalysis()
        e_id = TOKEN_TO_ID["・"]        # 欧文 E / 和文 ヘ
        t_id = TOKEN_TO_ID["-"]         # 欧文 T / 和文 ム
        analysis.add_record(EvalRecord(
            ref_tokens=[e_id], pred_tokens=[t_id],
            ref_text="", pred_text="",
        ))
        payload = analysis.confusion_to_dict()
        entry = payload["entries"][0]
        assert entry["ref_code"] == "・"
        assert entry["pred_code"] == "-"


class TestTokenLabels:
    def test_describe_known_token(self) -> None:
        info = describe_token(TOKEN_TO_ID["・-"])
        assert info["code"] == "・-"
        assert info["european"] == "A"
        assert info["japanese"] == "イ"

    def test_describe_word_break(self) -> None:
        info = describe_token(WORD_BREAK_TOKEN_ID)
        assert info["code"] == "<WORDBREAK>"

    def test_describe_unknown_token_does_not_raise(self) -> None:
        info = describe_token(999999)
        assert info["code"] == "<UNKNOWN>"

    def test_token_label_includes_code_and_display(self) -> None:
        label = token_label(TOKEN_TO_ID["・-"])
        assert "・-" in label
        assert "A" in label


class TestDetailedEvalReport:
    def _report(self) -> DetailedEvalReport:
        detailed = DetailedEvalReport()
        detailed.add(
            EvalRecord(
                ref_tokens=[1, 2, 3], pred_tokens=[1, 9, 3],
                ref_text="ABC", pred_text="AXC",
            ),
            name="sample_a", mode="european",
        )
        detailed.add(
            EvalRecord(
                ref_tokens=[4, 5], pred_tokens=[4, 5],
                ref_text="DE", pred_text="DE",
            ),
            name="sample_b", mode="japanese",
        )
        return detailed

    def test_overall_metrics_match_eval_report(self) -> None:
        detailed = self._report()
        # token errors 1 / 5 ref tokens
        assert detailed.report.overall.ter == pytest.approx(0.2)
        assert detailed.report.overall.cer == pytest.approx(0.2)
        assert detailed.report.overall.n_samples == 2

    def test_per_sample_details(self) -> None:
        detailed = self._report()
        assert [s.name for s in detailed.samples] == ["sample_a", "sample_b"]
        first = detailed.samples[0]
        assert first.mode == "european"
        assert first.ter == pytest.approx(1 / 3)
        assert first.cer == pytest.approx(1 / 3)
        assert first.substitutions == 1
        assert first.insertions == 0
        assert first.deletions == 0

    def test_analysis_is_populated(self) -> None:
        detailed = self._report()
        assert detailed.analysis.counts[2].substituted == 1
        assert detailed.analysis.confusion[(2, 9)] == 1

    def test_to_dict_is_json_serializable(self) -> None:
        detailed = self._report()
        payload = detailed.to_dict()
        dumped = json.dumps(payload, ensure_ascii=False, indent=2)
        restored = json.loads(dumped)
        assert restored["overall"]["ter"] == pytest.approx(0.2)
        assert restored["overall"]["n_samples"] == 2
        assert len(restored["samples"]) == 2
        assert restored["samples"][0]["ref_text"] == "ABC"
        assert restored["samples"][0]["pred_text"] == "AXC"
        assert restored["samples"][0]["ref_tokens"] == [1, 2, 3]
        assert restored["samples"][0]["pred_tokens"] == [1, 9, 3]

    def test_to_dict_contains_token_errors(self) -> None:
        detailed = self._report()
        payload = detailed.to_dict()
        by_id = {t["token_id"]: t for t in payload["token_errors"]}
        assert by_id[2]["substituted"] == 1
        assert "code" in by_id[2]

    def test_to_dict_totals(self) -> None:
        detailed = self._report()
        totals = detailed.to_dict()["totals"]
        assert totals["substitutions"] == 1
        assert totals["insertions"] == 0
        assert totals["deletions"] == 0
        assert totals["ref_tokens"] == 5

    def test_empty_report_is_serializable(self) -> None:
        detailed = DetailedEvalReport()
        payload = detailed.to_dict()
        json.dumps(payload, ensure_ascii=False)
        assert payload["overall"]["n_samples"] == 0
        assert payload["samples"] == []

    def test_summary_lines_include_top_errors(self) -> None:
        detailed = self._report()
        lines = detailed.summary_lines()
        assert any("Overall" in line for line in lines)
        assert any("S=" in line or "sub" in line.lower() for line in lines)
