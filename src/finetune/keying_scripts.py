"""自己打鍵データ収集用の打鍵原稿生成 (Phase 4).

実交信 50 件の収集は負荷が大きいため、**既知の原稿をユーザー自身が打鍵して
録音**することで「完全ラベル + 本物の手送りの癖」を持つ実データを低コストで
作る。本モジュールはその原稿を生成する。

- 原稿は ``text_to_codes`` で必ずトークン化できることを保証する
  (漢字等の混入なし → FT 時のラベル検証で落ちない)
- 1 件の打鍵時間は WPM 24 で 3〜30 秒 (CTC 範囲) に収まるよう構成する
- すべて ``np.random.Generator`` を受け取り再現可能
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.finetune.dataset import _infer_mode_from_filename, _parse_txt
from src.finetune.label_markers import normalize_label_markers
from src.synth.text_generator import (
    generate_european_text,
    generate_japanese_text,
)
from src.tokens.morse_tokens import WORD_BREAK_CODE, Mode, text_to_codes

# PARIS 標準: 1 unit = 1.2 / WPM 秒
_UNIT_SEC_FACTOR = 1.2

# 原稿 1 件の目標打鍵時間 (秒, @wpm)。CTC 範囲 3〜30 秒に余裕を持って収める
_TARGET_MIN_SEC = 8.0
_TARGET_MAX_SEC = 20.0

# 打鍵原稿向けパターン重み: 手送りしにくい完全ランダム文字列を抑え、
# 実運用文 (CQ・RST・ラグチュー) を中心にする
_EU_SCRIPT_WEIGHTS = {
    "random": 0.05,
    "callsign": 0.15,
    "rst": 0.10,
    "qcode": 0.10,
    "qcode_q": 0.10,
    "cq": 0.20,
    "qso_stamp": 0.20,
    "qso_close": 0.10,
}
_JA_SCRIPT_WEIGHTS = {
    "random": 0.10,
    "greeting": 0.20,
    "weather": 0.25,
    "rig": 0.25,
    "lagchew": 0.20,
    "horerata": 0.0,   # ホレ/ラタは原稿全体を包む形で別途付与する
}


def estimate_duration_sec(text: str, mode: Mode, wpm: float = 24.0) -> float:
    """PARIS 標準タイミングでの打鍵時間を見積もる (秒)."""
    unit_sec = _UNIT_SEC_FACTOR / wpm
    units = 0.0
    for code in text_to_codes(text, mode):
        if code == WORD_BREAK_CODE:
            units += 4.0  # 文字間 3 → 語間 7 への差分
            continue
        # 要素長 (・=1, -=3) + 要素間ギャップ (len-1) + 文字間ギャップ 3
        units += sum(1.0 if c == "・" else 3.0 for c in code)
        units += (len(code) - 1) + 3.0
    return units * unit_sec


def _gen_part(rng: np.random.Generator, mode: Mode) -> str:
    if mode == "european":
        return generate_european_text(
            rng, length_range=(5, 15), pattern_weights=_EU_SCRIPT_WEIGHTS  # type: ignore[arg-type]
        )
    return generate_japanese_text(
        rng, length_range=(4, 10), pattern_weights=_JA_SCRIPT_WEIGHTS  # type: ignore[arg-type]
    )


def generate_keying_script(
    rng: np.random.Generator,
    mode: Mode,
    wpm: float = 24.0,
) -> str:
    """打鍵原稿 1 件を生成 (WPM ``wpm`` で 3〜30 秒に収まる)."""
    target_sec = float(rng.uniform(_TARGET_MIN_SEC, _TARGET_MAX_SEC))
    sep = "、" if mode == "japanese" else " "
    parts: list[str] = [_gen_part(rng, mode)]
    while estimate_duration_sec(sep.join(parts), mode, wpm) < target_sec:
        part = _gen_part(rng, mode)
        if estimate_duration_sec(sep.join([*parts, part]), mode, wpm) > _TARGET_MAX_SEC:
            break
        parts.append(part)
    text = sep.join(parts)
    # 和文は運用通り 6 割をホレ〜ラタで包む (時間に余裕がある場合のみ)
    if mode == "japanese" and rng.random() < 0.6:
        wrapped = f"{{HORE}}{text}{{RATA}}"
        if estimate_duration_sec(wrapped, mode, wpm) <= _TARGET_MAX_SEC + 5.0:
            text = wrapped
    return text


def write_script_files(
    out_dir: Path | str,
    mode: Mode,
    count: int,
    seed: int,
    wpm: float = 24.0,
) -> list[Path]:
    """打鍵原稿を連番ファイルに書き出す (決定的).

    ファイル形式は実信号 TXT と同じ ``ヘッダ --- 本文`` 形式。録音後は
    ``apply_script_label`` で録音 TXT の本文に転記する。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "ja" if mode == "japanese" else "eu"
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for i in range(1, count + 1):
        text = generate_keying_script(rng, mode, wpm)
        dur = estimate_duration_sec(text, mode, wpm)
        path = out_dir / f"script_{prefix}_{i:02d}.txt"
        path.write_text(
            f"mode: {mode}\n"
            f"estimated_duration_s: {dur:.1f}\n"
            f"wpm: {wpm:.0f}\n"
            "---\n"
            f"{text}\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def read_script_text(path: Path | str) -> str:
    """原稿ファイルから本文テキストを読み出す."""
    _, body = _parse_txt(Path(path))
    return body


def apply_script_label(txt_path: Path | str, text: str) -> None:
    """録音 TXT の本文を原稿テキストで置き換える (ヘッダは保持).

    テキストがトークン化できない場合 (漢字混入・空) は ``ValueError``.
    """
    txt_path = Path(txt_path)
    header, _ = _parse_txt(txt_path)
    mode_str = header.get("mode", "").lower()
    if mode_str in ("european", "japanese"):
        mode: Mode = mode_str  # type: ignore[assignment]
    else:
        inferred = _infer_mode_from_filename(txt_path.stem)
        if inferred is None:
            raise ValueError(f"mode を判定できません: {txt_path}")
        mode = inferred
    try:
        codes = text_to_codes(text, mode)
    except KeyError as exc:
        raise ValueError(f"トークン化できない文字が含まれます: {exc}") from exc
    if not codes:
        raise ValueError("本文が空です")

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    header_lines: list[str] = []
    for line in lines:
        if line.strip() == "---":
            break
        header_lines.append(line)
    if not header_lines:
        header_lines = [f"mode: {mode}"]
    txt_path.write_text(
        "\n".join([*header_lines, "---", text]) + "\n", encoding="utf-8"
    )


__all__ = [
    "apply_script_label",
    "estimate_duration_sec",
    "generate_keying_script",
    "normalize_label_markers",
    "read_script_text",
    "write_script_files",
]
