"""実録音のラベルが**音と合っているか**を、NN を使わずに検査する.

なぜ要るか
----------
**間違った物差しで測っている限り、どの学習が良かったのかも分からない。**

2026-08-03 に「ラベルに音の無いトークンが 19 個ある」として ``{HORE}`` 7 個・
``{RATA}`` 7 個・``{SK}`` 5 個をラベルから外した。根拠は「ラベルにホレが無い
録音 3 件すべてでモデルが冒頭にホレを出す ＝ **和文冒頭で無条件にホレを吐く癖**
とラベルがたまたま一致していただけ」という推論だった。

**2026-08-12 にこれが覆った。** 独立した別のデコーダが同じ位置で
ホレ・ラタを出し、**波形の要素列も一致した**:

    script_ja_03 の先頭 6 要素:  - ・ ・ - - -    (ホレ = -・・---)
    script_ja_05 の末尾 5 要素:  ・ ・ ・ - ・     (ラタ = ・・・-・)

癖ではなく正しい認識であり、**外したラベルの方が誤りだった**。

やり方
------
**NN を通さない。** 包絡線から ON/OFF の並びを取り出し、短点の長さを基準に
要素を ``・`` / ``-`` に振り分け、文字間の無音で区切って符号に組み立てる。
モデルの癖に影響されない独立した物差しにするためである。

使い方::

    .venv/Scripts/python.exe scripts/audit_labels.py
    .venv/Scripts/python.exe scripts/audit_labels.py --dirs data/real/train

限界
----
実録音なので完全な復元はできない。**この道具が答えられるのは
「先頭/末尾にラベルに無い符号があるか」まで**である。本文の 1 文字ずつを
突き合わせるものではない (そこは人間が波形を見る)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.finetune.dataset import discover_real_samples  # noqa: E402
from src.infer.wpm import element_runs, split_dot_dash  # noqa: E402
from src.tokens.morse_tokens import HORE_CODE, RATA_CODE, text_to_codes  # noqa: E402

# 要素の振り分けと文字の区切りに使う閾値 (短点の何倍か)。
# 教科書は 長音 3 / 文字間 3 / 語間 7。手打ちは大きく揺れるので緩く取る。
DASH_MIN_UNITS = 2.0
CHAR_GAP_MIN_UNITS = 2.0

# これ以下の ON は符号ではなく雑音の尖りとみなす (秒)。
# ``envelope_on_off`` が ON を絞るのに使っているのと同じ値。
MIN_ON_SEC = 0.020


def _merge_noise_spikes(
    runs: tuple[tuple[bool, float], ...]
) -> list[tuple[bool, float]]:
    """短すぎる ON を前後の無音に併合する.

    **これをやらないと単位長の推定が壊れる。** 実測 (`script_ja_03`) では
    20 ms 以下の尖りが 97 個あり、短点の平均が 40.2 ms から **13.6 ms** に
    引き下げられていた。その結果、文字間の判定閾値が本物の要素間スペースより
    短くなり、**全要素が 1 文字ずつに刻まれていた** (ラベル 7 文字に対し 46)。
    """
    merged: list[tuple[bool, float]] = []
    for is_on, sec in runs:
        if is_on and sec <= MIN_ON_SEC:
            is_on = False                      # 雑音の尖り = 無音として扱う
        if merged and merged[-1][0] == is_on:
            merged[-1] = (is_on, merged[-1][1] + sec)
        else:
            merged.append((is_on, sec))
    return merged


def codes_from_audio(wave: np.ndarray, sample_rate: int) -> list[str]:
    """波形から符号の並びを組み立てる (``・`` と ``-`` の文字列のリスト).

    **NN を使わない。** 包絡線の ON/OFF だけで決める。
    """
    runs = _merge_noise_spikes(element_runs(wave, sample_rate))
    ons = np.array([sec for is_on, sec in runs if is_on], dtype=np.float64)
    if ons.size < 3:
        return []
    dot, _dash = split_dot_dash(ons)
    if dot.size == 0:
        return []
    unit = float(dot.mean())
    if unit <= 0.0:
        return []

    codes: list[str] = []
    current: list[str] = []
    for is_on, sec in runs:
        if is_on:
            current.append("-" if sec >= unit * DASH_MIN_UNITS else "・")
        elif current and sec >= unit * CHAR_GAP_MIN_UNITS:
            codes.append("".join(current))
            current = []
    if current:
        codes.append("".join(current))
    return codes


def audit(sample, *, verbose: bool = False) -> dict:
    """1 件を検査して結果を返す."""
    wave, sample_rate = sf.read(sample.wav_path, dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave[:, 0]
    heard = codes_from_audio(wave, sample_rate)
    labelled = [c for c in text_to_codes(sample.text, sample.mode) if c.startswith(("・", "-"))]

    result = {
        "name": sample.wav_path.stem,
        "mode": sample.mode,
        "n_label": len(labelled),
        "n_heard": len(heard),
        "head_extra": "",
        "tail_extra": "",
    }
    # **先頭と末尾だけを見る** (本文の逐一照合はこの道具の守備範囲外)
    if heard and heard[0] == HORE_CODE and (not labelled or labelled[0] != HORE_CODE):
        result["head_extra"] = "ホレ"
    if heard and heard[-1] == RATA_CODE and (not labelled or labelled[-1] != RATA_CODE):
        result["tail_extra"] = "ラタ"
    if verbose:
        result["heard"] = heard
        result["labelled"] = labelled
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="実録音のラベルを音と突き合わせる")
    parser.add_argument("--dirs", nargs="+",
                        default=["data/keying_scripts", "data/keyed_extra"])
    parser.add_argument("--mode", choices=["european", "japanese"], default=None)
    parser.add_argument("--verbose", action="store_true", help="符号の並びも出す")
    args = parser.parse_args()

    samples = [s for d in args.dirs for s in discover_real_samples(d)]
    if args.mode:
        samples = [s for s in samples if s.mode == args.mode]
    if not samples:
        raise SystemExit("ラベル付きの録音が見つかりません")

    rows = [audit(s, verbose=args.verbose) for s in samples]

    print(f"{'ファイル':<24}{'モード':>6}{'ラベル':>7}{'音':>5}   ラベルに無いもの")
    for r in rows:
        extra = "、".join(x for x in (r["head_extra"], r["tail_extra"]) if x)
        mark = f"  ← {extra}" if extra else ""
        print(f"{r['name']:<26}{r['mode'][:2]:>4}{r['n_label']:>7}{r['n_heard']:>5}{mark}")
        if args.verbose:
            print(f"    ラベル: {' '.join(r['labelled'])}")
            print(f"    音    : {' '.join(r['heard'])}")

    head = sum(1 for r in rows if r["head_extra"])
    tail = sum(1 for r in rows if r["tail_extra"])
    print()
    print(f"検査 {len(rows)} 件")
    print(f"  **音にホレがあるのにラベルに無い**: {head} 件")
    print(f"  **音にラタがあるのにラベルに無い**: {tail} 件")
    if head or tail:
        print()
        print("  ラベルを直さないと、正しく読めた分が誤りとして数えられます。")


if __name__ == "__main__":
    main()
