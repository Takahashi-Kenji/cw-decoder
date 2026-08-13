"""符号表と時間軸の定数を TypeScript として生成する.

``src/tokens/morse_tokens.py`` を唯一の真正なソースとする原則を守るため、
TypeScript 側に符号表を手で書かず、ここから生成する。和文符号は実装誤記が
起きやすく、二重定義を作れば必ず事故る。

同じ議論が**時間軸の定数**にもそのまま当てはまるため、``MelConfig`` の
``sample_rate`` と ``hop_length`` もここから出力する。``FRAME_HOP_SAMPLES`` は
CTC のフレーム位置を絶対サンプル位置に直す係数であり、``hop_ms`` を変えたときに
TypeScript 側だけ手書きのままだと確定境界が全部ずれる (しかも黙ってずれる)。

使い方:
    python scripts/export_tokens.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.train.preprocessing import MelConfig
from src.tokens.morse_tokens import (
    BLANK_TOKEN_ID,
    DAKUTEN_CHAR,
    DAKUTEN_COMPOSE,
    EUROPEAN_TABLE,
    HANDAKUTEN_CHAR,
    HANDAKUTEN_COMPOSE,
    ID_TO_TOKEN,
    JAPANESE_TABLE,
    VOCAB_SIZE,
    WORD_BREAK_TOKEN_ID,
)

OUTPUT_PATH = "web/src/generated/tokens.ts"

HEADER = """/**
 * 符号表と時間軸の定数 (自動生成 — 手で編集しないこと)
 *
 * 生成元: src/tokens/morse_tokens.py, src/train/preprocessing.py (MelConfig)
 * 生成方法: python scripts/export_tokens.py
 *
 * 符号定義の唯一の真正なソースは Python 側です。ここを直接書き換えても
 * 次の再生成で失われ、Python 側とずれた状態は tests/test_export_tokens.py
 * が検出します。
 *
 * 符号表記: ドット = ・ (U+30FB 中黒), ダッシュ = - (U+002D ハイフン)
 */
"""


def _js_string(value: str) -> str:
    """JSON 文字列として安全に出力 (中黒などの非 ASCII はそのまま残す)."""
    return json.dumps(value, ensure_ascii=False)


def _render_record(name: str, table: dict[str, str], comment: str) -> str:
    lines = [f"/** {comment} */", f"export const {name}: Readonly<Record<string, string>> = {{"]
    for code, char in table.items():
        lines.append(f"  {_js_string(code)}: {_js_string(char)},")
    lines.append("}")
    return "\n".join(lines)


def render_tokens_ts() -> str:
    """生成する TypeScript の全文を返す."""
    max_id = max(ID_TO_TOKEN)
    id_to_code = [ID_TO_TOKEN[i].code for i in range(max_id + 1)]

    parts: list[str] = [HEADER]
    parts.append(f"export const BLANK_TOKEN_ID = {BLANK_TOKEN_ID}")
    parts.append(f"export const WORD_BREAK_TOKEN_ID = {WORD_BREAK_TOKEN_ID}")
    parts.append(f"export const VOCAB_SIZE = {VOCAB_SIZE}")
    parts.append(f"export const DAKUTEN_CHAR = {_js_string(DAKUTEN_CHAR)}")
    parts.append(f"export const HANDAKUTEN_CHAR = {_js_string(HANDAKUTEN_CHAR)}")
    parts.append("")
    mel = MelConfig()
    parts.append("/** 推論のサンプルレート (Hz)。MelConfig.sample_rate。 */")
    parts.append(f"export const SAMPLE_RATE = {mel.sample_rate}")
    parts.append(
        "/**\n"
        " * mel の hop_length (サンプル)。MelConfig.hop_length。\n"
        " *\n"
        " * CTC のフレーム位置を絶対サンプル位置に直す係数。手書きしてはいけない\n"
        " * (Python 側の hop_ms を変えるとブラウザ側だけ確定境界が黙ってずれる)。\n"
        " */"
    )
    parts.append(f"export const FRAME_HOP_SAMPLES = {mel.hop_length}")
    parts.append("")
    parts.append("/** 添字が token id。0 は CTC blank、末尾は語間 WORDBREAK。 */")
    parts.append("export const ID_TO_CODE: readonly string[] = [")
    for code in id_to_code:
        parts.append(f"  {_js_string(code)},")
    parts.append("]")
    parts.append("")
    parts.append(_render_record("EUROPEAN_TABLE", EUROPEAN_TABLE, "欧文符号表 (符号 → 表示文字)"))
    parts.append("")
    parts.append(_render_record("JAPANESE_TABLE", JAPANESE_TABLE, "和文符号表 (符号 → 表示文字)"))
    parts.append("")
    parts.append(_render_record("DAKUTEN_COMPOSE", DAKUTEN_COMPOSE, "プレーンカナ + 濁点 → 合成カナ"))
    parts.append("")
    parts.append(_render_record("HANDAKUTEN_COMPOSE", HANDAKUTEN_COMPOSE, "プレーンカナ + 半濁点 → 合成カナ"))
    parts.append("")
    return "\n".join(parts)


def main() -> None:
    path = Path(OUTPUT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 改行は LF 固定 (newline="" で Python 側の変換を抑止)
    path.write_text(render_tokens_ts(), encoding="utf-8", newline="")
    print(f"書き出し: {path}")


if __name__ == "__main__":
    main()
