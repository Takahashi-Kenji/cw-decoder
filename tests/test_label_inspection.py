"""ラベル検品ロジックのテスト."""
from __future__ import annotations

from src.finetune.label_inspection import recall_by_code_length
from src.train.metrics import EvalRecord, TokenErrorAnalysis
from src.tokens.morse_tokens import TOKEN_TO_ID


class TestRecallByCodeLength:
    def test_groups_recall_by_code_length(self) -> None:
        e = TOKEN_TO_ID["・"]        # 長さ 1
        i = TOKEN_TO_ID["・・"]       # 長さ 2
        sk = TOKEN_TO_ID["・・・-・-"]  # 長さ 6
        analysis = TokenErrorAnalysis()
        # ・ 正解、・・ 正解、6要素符号は脱落
        analysis.add_record(EvalRecord(
            ref_tokens=[e, i, sk], pred_tokens=[e, i],
            ref_text="", pred_text="",
        ))
        result = recall_by_code_length(analysis)
        assert result[1] == (1, 100.0)
        assert result[2] == (1, 100.0)
        assert result[6] == (1, 0.0)   # 6要素符号 recall 0%

    def test_empty_analysis(self) -> None:
        assert recall_by_code_length(TokenErrorAnalysis()) == {}
