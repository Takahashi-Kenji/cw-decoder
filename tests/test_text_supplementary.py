"""補充学習向け: 新トークン (?, 」, ホレ, ラタ, SK) の生成保証テスト."""
from __future__ import annotations

import numpy as np

from src.synth.text_generator import (
    generate_european_text,
    generate_japanese_text,
)
from src.tokens.morse_tokens import (
    SPECIAL_INPUT_MARKERS,
    TOKEN_TO_ID,
    text_to_codes,
)


# 対象トークンの符号
QUESTION_CODE = "・・--・・"
DANRAKU_CODE = "・-・-・・"
HORE_CODE = SPECIAL_INPUT_MARKERS["{HORE}"]
RATA_CODE = SPECIAL_INPUT_MARKERS["{RATA}"]
SK_CODE = SPECIAL_INPUT_MARKERS["{SK}"]


def _european_codes_in_n_samples(n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    all_codes: list[str] = []
    for _ in range(n):
        text = generate_european_text(rng)
        all_codes.extend(text_to_codes(text, "european"))
    return all_codes


def _japanese_codes_in_n_samples(n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    all_codes: list[str] = []
    for _ in range(n):
        text = generate_japanese_text(rng)
        all_codes.extend(text_to_codes(text, "japanese"))
    return all_codes


class TestEuropeanQuestionAndSK:
    def test_question_mark_appears(self) -> None:
        codes = _european_codes_in_n_samples(500)
        assert QUESTION_CODE in codes, "欧文サンプル500件中に ? が現れていない"

    def test_sk_prosign_appears(self) -> None:
        codes = _european_codes_in_n_samples(500)
        assert SK_CODE in codes, "欧文サンプル500件中に [SK] が現れていない"

    def test_question_density_reasonable(self) -> None:
        codes = _european_codes_in_n_samples(500)
        n_question = codes.count(QUESTION_CODE)
        # 500 サンプルに対して 1% 以上は ? を含むことを期待
        assert n_question >= 5, f"? 出現数 {n_question} が少なすぎ"

    def test_sk_density_reasonable(self) -> None:
        codes = _european_codes_in_n_samples(500)
        n_sk = codes.count(SK_CODE)
        # weights で qso_close=0.10 → 50 サンプル前後で SK 含む
        assert n_sk >= 20, f"[SK] 出現数 {n_sk} が少なすぎ"


class TestJapaneseDanrakuHoreRata:
    def test_danraku_appears(self) -> None:
        codes = _japanese_codes_in_n_samples(500)
        assert DANRAKU_CODE in codes, "和文サンプル500件中に 」 が現れていない"

    def test_hore_appears(self) -> None:
        codes = _japanese_codes_in_n_samples(500)
        assert HORE_CODE in codes, "和文サンプル500件中に [ホレ] が現れていない"

    def test_rata_appears(self) -> None:
        codes = _japanese_codes_in_n_samples(500)
        assert RATA_CODE in codes, "和文サンプル500件中に [ラタ] が現れていない"

    def test_hore_rata_density_reasonable(self) -> None:
        codes = _japanese_codes_in_n_samples(500)
        # horerata pattern 20% → 100 サンプル前後で出現
        n_hore = codes.count(HORE_CODE)
        n_rata = codes.count(RATA_CODE)
        assert n_hore >= 40, f"[ホレ] 出現数 {n_hore} が少なすぎ"
        assert n_rata >= 40, f"[ラタ] 出現数 {n_rata} が少なすぎ"


class TestSpecialInputMarkers:
    def test_markers_resolve_to_correct_codes(self) -> None:
        assert text_to_codes("{HORE}", "japanese") == [HORE_CODE]
        assert text_to_codes("{RATA}", "japanese") == [RATA_CODE]
        assert text_to_codes("{SK}", "european") == [SK_CODE]

    def test_marker_with_surrounding_text_european(self) -> None:
        result = text_to_codes("73 TU {SK}", "european")
        assert result[-1] == SK_CODE

    def test_marker_with_surrounding_text_japanese(self) -> None:
        result = text_to_codes("{HORE}イロハ{RATA}", "japanese")
        assert result[0] == HORE_CODE
        assert result[-1] == RATA_CODE

    def test_token_ids_for_new_codes_exist(self) -> None:
        for code in [QUESTION_CODE, DANRAKU_CODE, HORE_CODE, RATA_CODE, SK_CODE]:
            assert code in TOKEN_TO_ID, f"{code!r} がトークン辞書に無い"
