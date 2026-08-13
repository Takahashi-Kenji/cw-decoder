"""学習用テキスト生成のテスト."""
from __future__ import annotations

import numpy as np
import pytest

from src.synth.text_generator import (
    DEFAULT_EUROPEAN_WEIGHTS,
    DEFAULT_JAPANESE_WEIGHTS,
    TextGenConfig,
    generate_european_text,
    generate_japanese_text,
    generate_text,
)
from src.tokens.morse_tokens import (
    EUROPEAN_CHAR_TO_CODE,
    JAPANESE_CHAR_TO_CODES,
    text_to_codes,
)


class TestEuropean:
    def test_generates_nonempty_string(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            text = generate_european_text(rng)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_seeded_deterministic(self) -> None:
        text1 = generate_european_text(np.random.default_rng(42))
        text2 = generate_european_text(np.random.default_rng(42))
        assert text1 == text2

    def test_all_chars_synthesizable(self) -> None:
        """生成したテキストが text_to_codes で例外なく符号列に変換できる."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            text = generate_european_text(rng)
            codes = text_to_codes(text, "european")
            assert all(isinstance(c, str) and c for c in codes)


class TestJapanese:
    def test_generates_nonempty_string(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            text = generate_japanese_text(rng)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_seeded_deterministic(self) -> None:
        text1 = generate_japanese_text(np.random.default_rng(42))
        text2 = generate_japanese_text(np.random.default_rng(42))
        assert text1 == text2

    def test_all_chars_synthesizable(self) -> None:
        """生成したテキストが text_to_codes で例外なく符号列に変換できる."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            text = generate_japanese_text(rng)
            codes = text_to_codes(text, "japanese")
            assert all(isinstance(c, str) and c for c in codes)


class TestConfig:
    def test_default_weights_sum_positive(self) -> None:
        assert sum(DEFAULT_EUROPEAN_WEIGHTS.values()) > 0
        assert sum(DEFAULT_JAPANESE_WEIGHTS.values()) > 0

    def test_zero_weight_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            generate_european_text(rng, pattern_weights={"random": 0.0})  # type: ignore[arg-type]


class TestGenerateText:
    def test_european_mode(self) -> None:
        rng = np.random.default_rng(1)
        text = generate_text(rng, "european")
        assert len(text) > 0

    def test_japanese_mode(self) -> None:
        rng = np.random.default_rng(1)
        text = generate_text(rng, "japanese")
        assert len(text) > 0

    def test_unknown_mode(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            generate_text(rng, "klingon")  # type: ignore[arg-type]

    def test_custom_config(self) -> None:
        rng = np.random.default_rng(0)
        cfg = TextGenConfig(european_length_range=(5, 6))
        # 短い random しか出ないように
        cfg.european_pattern_weights = {"random": 1.0}
        text = generate_text(rng, "european", cfg)
        assert 5 <= len(text) <= 6
