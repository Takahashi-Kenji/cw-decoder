"""符号表・時間軸定数の TypeScript 生成の検証."""
from __future__ import annotations

from pathlib import Path

from scripts.export_tokens import OUTPUT_PATH, render_tokens_ts
from src.tokens.morse_tokens import (
    EUROPEAN_TABLE,
    ID_TO_TOKEN,
    JAPANESE_TABLE,
    VOCAB_SIZE,
    WORD_BREAK_TOKEN_ID,
)
from src.train.preprocessing import MelConfig


def test_contains_all_ids() -> None:
    """全トークン ID が ID_TO_CODE に現れること."""
    ts = render_tokens_ts()
    for token in ID_TO_TOKEN.values():
        assert token.code in ts, f"符号 {token.code!r} が生成物に含まれていない"


def test_contains_constants() -> None:
    ts = render_tokens_ts()
    assert f"export const VOCAB_SIZE = {VOCAB_SIZE}" in ts
    assert f"export const WORD_BREAK_TOKEN_ID = {WORD_BREAK_TOKEN_ID}" in ts


def test_contains_time_axis_constants() -> None:
    """時間軸の定数が MelConfig から出力されていること.

    ``FRAME_HOP_SAMPLES`` はフレーム位置 → 絶対サンプル位置の変換係数であり、
    TypeScript 側に手書きすると ``hop_ms`` を変えたときにブラウザ側だけ
    確定境界が黙ってずれる (符号表と同じ「二重定義は必ず事故る」議論)。
    """
    ts = render_tokens_ts()
    mel = MelConfig()
    assert f"export const SAMPLE_RATE = {mel.sample_rate}" in ts
    assert f"export const FRAME_HOP_SAMPLES = {mel.hop_length}" in ts


def test_marks_generated_and_forbids_edit() -> None:
    """手編集を禁じる注意書きが入っていること."""
    ts = render_tokens_ts()
    assert "自動生成" in ts
    assert "scripts/export_tokens.py" in ts


def test_table_entry_counts() -> None:
    """表のエントリ数が Python 側と一致すること."""
    ts = render_tokens_ts()
    # "= {" 以降 (レコード本体) だけを取り出す。素朴に名前直後から "}" までを
    # 取ると型注釈 ": Readonly<Record<string, string>>" のコロンを1個
    # 余分に数えてしまうため、開き波括弧の後ろから数える。
    european_block = ts.split("EUROPEAN_TABLE")[1].split("= {", 1)[1].split("}")[0]
    japanese_block = ts.split("JAPANESE_TABLE")[1].split("= {", 1)[1].split("}")[0]
    assert european_block.count(":") == len(EUROPEAN_TABLE)
    assert japanese_block.count(":") == len(JAPANESE_TABLE)


def test_committed_file_is_up_to_date() -> None:
    """コミット済みの tokens.ts が最新の生成結果と一致すること.

    符号表を変更したのに再生成を忘れると、Python と TypeScript で
    符号が食い違う。それをこのテストで検出する.
    """
    path = Path(OUTPUT_PATH)
    assert path.exists(), f"{path} が無い。scripts/export_tokens.py を実行すること"
    assert path.read_text(encoding="utf-8") == render_tokens_ts(), (
        "tokens.ts が古い。scripts/export_tokens.py を再実行してコミットすること"
    )
