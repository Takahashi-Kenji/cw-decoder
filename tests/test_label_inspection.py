"""ラベル検品ロジックのテスト."""
from __future__ import annotations

from src.finetune.label_inspection import boundary_extras, recall_by_code_length
from src.train.metrics import EvalRecord, TokenErrorAnalysis
from src.tokens.morse_tokens import HORE_CODE, RATA_CODE, TOKEN_TO_ID


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


class TestBoundaryExtras:
    """音の先頭/末尾に、ラベルに無いホレ・ラタが居ないかを見る.

    **雑音の尖りに邪魔されないこと**が要点。包絡線から組み立てた符号列には
    孤立した ``・`` が混じるので、境界ちょうどだけを見ると本物のホレを見逃す
    (実測: held-out 和文 10 件のうち、境界一致では 2 件しか拾えなかったが
    先頭付近まで見ると 7 件でホレが鳴っていた)。
    """

    _KANA = "-・-・・"          # ラ行でもホレでもない適当な符号

    def test_境界ちょうどのホレを拾う(self) -> None:
        heard = [HORE_CODE, self._KANA]
        assert boundary_extras(heard, [self._KANA]).head == 0

    def test_雑音の尖りを挟んだホレも拾う(self) -> None:
        heard = ["・", "・", "・", HORE_CODE, self._KANA]
        assert boundary_extras(heard, [self._KANA]).head == 3

    def test_ラベルにホレがあるなら報告しない(self) -> None:
        heard = ["・", HORE_CODE, self._KANA]
        assert boundary_extras(heard, [HORE_CODE, self._KANA]).head is None

    def test_末尾のラタを拾う(self) -> None:
        heard = [self._KANA, RATA_CODE, "・", "・"]
        assert boundary_extras(heard, [self._KANA]).tail == 1

    def test_窓の外は見ない(self) -> None:
        """本文の奥にある符号を「先頭のホレ」と言わないこと."""
        heard = [self._KANA] * 12 + [HORE_CODE]
        assert boundary_extras(heard, [self._KANA] * 12).head is None

    def test_何も無ければ報告しない(self) -> None:
        extras = boundary_extras([self._KANA], [self._KANA])
        assert extras.head is None and extras.tail is None

    def test_空の入力(self) -> None:
        extras = boundary_extras([], [])
        assert extras.head is None and extras.tail is None
