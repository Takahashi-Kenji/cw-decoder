"""手書き打鍵コーパスの設計値を固定するテスト.

コーパスは「6要素符号の出現数を保証する」ために手書きにしている。本文を
編集したときにその保証が崩れないよう、カバレッジと打鍵時間をここで固定する。

出現数は文字ではなく**符号**で数える。濁点・半濁点は独立トークンなので
文字単位では数えられず、NN が学習する単位とも合わないため。
"""
from __future__ import annotations

import pytest

from src.finetune.keying_corpus import (
    ALL_SCRIPTS,
    DURATION_MAX_SEC,
    DURATION_MIN_SEC,
    EUROPEAN_MIN_DIGIT_OCCURRENCES,
    EUROPEAN_MIN_OCCURRENCES,
    EUROPEAN_SCRIPTS,
    JAPANESE_BASE_KANA,
    JAPANESE_MIN_DIGIT_OCCURRENCES,
    JAPANESE_MIN_OCCURRENCES,
    JAPANESE_SCRIPTS,
    KeyingScript,
    code_histogram,
    resolve_code,
)
from src.finetune.keying_scripts import estimate_duration_sec, read_script_text
from src.tokens.morse_tokens import Mode, text_to_codes

from scripts.generate_corpus_scripts import main as generate_main


def _shortfalls(
    scripts: tuple[KeyingScript, ...], mode: Mode, minimums: dict[str, int]
) -> dict[str, tuple[int, int]]:
    counter = code_histogram(scripts)
    return {
        key: (counter[resolve_code(key, mode)], minimum)
        for key, minimum in minimums.items()
        if counter[resolve_code(key, mode)] < minimum
    }


def test_corpus_sizes() -> None:
    assert len(EUROPEAN_SCRIPTS) == 30
    assert len(JAPANESE_SCRIPTS) == 60


def test_names_are_unique_and_train_prefixed() -> None:
    """``t`` 付きの命名は評価用 (script_eu_01..10) との取り違え防止に必須."""
    names = [s.name for s in ALL_SCRIPTS]
    assert len(names) == len(set(names))
    assert all(s.name.split("_")[1].startswith("t") for s in ALL_SCRIPTS)


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda s: s.name)
def test_every_script_is_tokenizable(script: KeyingScript) -> None:
    """FT のラベル検証で落ちる本文が混じっていないこと."""
    assert text_to_codes(script.text, script.mode)


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda s: s.name)
def test_duration_within_range(script: KeyingScript) -> None:
    """CTC の 3〜30 秒制約の内側に収める."""
    dur = estimate_duration_sec(script.text, script.mode, script.wpm)
    assert DURATION_MIN_SEC <= dur <= DURATION_MAX_SEC


@pytest.mark.parametrize(
    "scripts", [EUROPEAN_SCRIPTS, JAPANESE_SCRIPTS], ids=["european", "japanese"]
)
def test_wpm_is_weighted_to_operating_range(scripts: tuple[KeyingScript, ...]) -> None:
    """実運用実測 14.6〜19.4 WPM 寄りにする (24 WPM は補助)."""
    wpms = [s.wpm for s in scripts]
    assert set(wpms) == {16, 20, 24}
    assert sum(1 for w in wpms if w <= 20) > len(wpms) / 2


def test_european_weak_token_coverage() -> None:
    """6要素符号 (recall 26.7%) の出現数が設計値を下回らないこと."""
    shortfalls = _shortfalls(EUROPEAN_SCRIPTS, "european", EUROPEAN_MIN_OCCURRENCES)
    assert not shortfalls, f"欧文の出現数不足: {shortfalls}"


def test_japanese_weak_token_coverage() -> None:
    shortfalls = _shortfalls(JAPANESE_SCRIPTS, "japanese", JAPANESE_MIN_OCCURRENCES)
    assert not shortfalls, f"和文の出現数不足: {shortfalls}"


@pytest.mark.parametrize(
    ("scripts", "mode", "minimum"),
    [
        (EUROPEAN_SCRIPTS, "european", EUROPEAN_MIN_DIGIT_OCCURRENCES),
        (JAPANESE_SCRIPTS, "japanese", JAPANESE_MIN_DIGIT_OCCURRENCES),
    ],
    ids=["european", "japanese"],
)
def test_all_digits_covered(
    scripts: tuple[KeyingScript, ...], mode: Mode, minimum: int
) -> None:
    shortfalls = _shortfalls(scripts, mode, {d: minimum for d in "0123456789"})
    assert not shortfalls, f"数字の出現数不足 ({mode}): {shortfalls}"


def test_all_base_kana_appear() -> None:
    """清音 48 字に 1 度も出ない字を作らない."""
    shortfalls = _shortfalls(
        JAPANESE_SCRIPTS, "japanese", {k: 1 for k in JAPANESE_BASE_KANA}
    )
    assert not shortfalls, f"未出現のカナ: {sorted(shortfalls)}"


def test_all_letters_appear() -> None:
    shortfalls = _shortfalls(
        EUROPEAN_SCRIPTS, "european", {c: 1 for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    )
    assert not shortfalls, f"未出現の英字: {sorted(shortfalls)}"


def test_no_prosign_markers() -> None:
    """プロサインを原稿に入れない.

    収録に使うオートキーヤーが ``{SK}`` ``{HORE}`` ``{RATA}`` を送出できない。
    原稿に残すと音に無いトークンをラベルが主張することになり、既存 keyed_val
    で起きたのと同じラベル汚染を再現してしまう。
    """
    for script in ALL_SCRIPTS:
        for marker in ("{SK}", "{HORE}", "{RATA}"):
            assert marker not in script.text, f"{script.name} に {marker}"


def test_generate_writes_readable_scripts(tmp_path) -> None:
    """書き出した原稿が既存の原稿リーダーで読み戻せること."""
    assert generate_main(["--out-dir", str(tmp_path)]) == 0
    written = sorted(tmp_path.glob("script_*.txt"))
    assert len(written) == len(ALL_SCRIPTS)
    by_name = {s.name: s.text for s in ALL_SCRIPTS}
    for path in written:
        assert read_script_text(path) == by_name[path.stem.removeprefix("script_")]


def test_dry_run_writes_nothing(tmp_path) -> None:
    assert generate_main(["--out-dir", str(tmp_path), "--dry-run"]) == 0
    assert not list(tmp_path.glob("*"))
