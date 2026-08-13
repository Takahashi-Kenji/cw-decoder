"""TokenConverter の時刻情報付き変換 (語間スペース復元) のテスト."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.tokens.converter import TokenConverter
from src.tokens.morse_tokens import TOKEN_TO_ID


@dataclass
class FT:
    """テスト用 FrameToken (TimedToken Protocol 互換)."""

    token_id: int
    confidence: float
    frame_start: int
    frame_end: int


def make_token(code: str, start: int, end: int, conf: float = 0.9) -> FT:
    return FT(token_id=TOKEN_TO_ID[code], confidence=conf, frame_start=start, frame_end=end)


class TestConvertTimedBasic:
    def test_no_gaps_no_spaces(self) -> None:
        # AB BC が連続 (各 token 1 frame、間 1 frame ギャップ — 全部小さい)
        tokens = [
            make_token("・-", 0, 5),       # A
            make_token("-・・・", 6, 12),  # B
            make_token("-・-・", 13, 20),  # C
        ]
        conv = TokenConverter(mode="european")
        result = conv.convert_timed(tokens, gap_threshold_frames=10)
        assert result.text == "ABC"

    def test_single_word_gap_inserts_one_space(self) -> None:
        # A と B の間に大きいギャップ → A B
        tokens = [
            make_token("・-", 0, 5),       # A
            make_token("-・・・", 100, 110),  # B (95 frame gap)
            make_token("-・-・", 113, 120),  # C (3 frame gap = within word)
        ]
        conv = TokenConverter(mode="european")
        result = conv.convert_timed(tokens, gap_threshold_frames=30)
        assert result.text == "A BC"

    def test_two_word_gaps(self) -> None:
        tokens = [
            make_token("・-", 0, 5),
            make_token("-・・・", 100, 110),
            make_token("-・-・", 200, 210),
        ]
        conv = TokenConverter(mode="european")
        result = conv.convert_timed(tokens, gap_threshold_frames=30)
        assert result.text == "A B C"


class TestAutoEstimateThreshold:
    def test_auto_estimate_recognizes_word_gap(self) -> None:
        # 小ギャップ (10) が多く 1 つだけ大ギャップ (50) があるパターン
        # → 自動推定で 50 だけ語間と判定
        tokens = [
            make_token("・-", 0, 5),
            make_token("-・・・", 15, 25),     # gap 10
            make_token("-・-・", 35, 45),     # gap 10
            make_token("・-", 95, 100),       # gap 50 ← 語間
            make_token("-・・・", 110, 120),   # gap 10
        ]
        conv = TokenConverter(mode="european")
        result = conv.convert_timed(tokens)
        # 期待: ABCA B (or ABCAB with space before A)
        assert " " in result.text
        # 語間は 1 つだけ
        assert result.text.count(" ") == 1


class TestJapaneseDakutenWithSpaces:
    def test_no_space_within_dakuten_compose(self) -> None:
        # カ + 濁点 → ガ. その後大ギャップで次の文字
        # ka frame 0-5, dakuten frame 6-10 (gap 1), 大ギャップ後 ロ
        tokens = [
            make_token("・-・・", 0, 5),     # カ
            make_token("・・", 6, 10),       # 濁点
            make_token("・-・-", 100, 110),  # ロ (gap 90)
        ]
        conv = TokenConverter(mode="japanese")
        result = conv.convert_timed(tokens, gap_threshold_frames=30)
        # ガとロの間にスペースが入る、ガの中にスペースは入らない
        assert result.text == "ガ ロ"


class TestEmptyAndEdge:
    def test_empty_list(self) -> None:
        conv = TokenConverter(mode="european")
        result = conv.convert_timed([])
        assert result.text == ""

    def test_single_token_no_space(self) -> None:
        tokens = [make_token("・-", 0, 5)]
        conv = TokenConverter(mode="european")
        result = conv.convert_timed(tokens)
        assert result.text == "A"


class TestRealisticCQPattern:
    """CQ DE JA0XYZ 風: グループ間の大きなギャップにスペース."""

    def test_cq_de_callsign(self) -> None:
        # WPM 20 想定: 1 dot = 6 frames, 文字間 18, 語間 42
        # CQ : C-Q 文字間 ~18
        # CQ と DE の間に語間 ~42
        # DE と JA0 の間に語間 ~42
        def at(code: str, start: int, length: int = 5) -> FT:
            return make_token(code, start, start + length)

        char_gap = 6
        word_gap = 50

        # C Q  D E  J A 0
        codes = ["-・-・", "--・-",      # CQ
                  "-・・", "・",          # DE
                  "・---", "・-", "-----"]  # JA0
        tokens: list[FT] = []
        t = 0
        for i, c in enumerate(codes):
            duration = 5
            tokens.append(at(c, t, duration))
            t += duration
            # 文字 i と次の文字の間のギャップを決める
            if i == 1 or i == 3:    # CQ→DE, DE→JA0 (語間)
                t += word_gap
            elif i < len(codes) - 1:
                t += char_gap

        conv = TokenConverter(mode="european")
        result = conv.convert_timed(tokens, gap_threshold_frames=30)
        assert result.text == "CQ DE JA0"
