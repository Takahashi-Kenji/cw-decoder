"""ゴールデンテスト用の見本音声を**合成音で**作る.

なぜ合成音なのか
----------------
以前は実際に受信した録音を使っていたため、``web/tests/fixtures/`` に
**交信相手のコールサインが残っていた** (golden.json の本文と .f32 の音声の
両方に)。公開リポジトリに第三者のコールサインを載せるのは避けたい
(運用者の判断、2026-08-15)。

**ゴールデンテストの目的は「Python と TypeScript が同じ結果を出すこと」**の
検証であって、音の中身が実受信である必要はない。合成音で置き換えても
テストの意味は変わらない。

使うコールサインは運用者自身の ``JH0ILL`` だけにする。

使い方
------
    python scripts/generate_golden_samples.py
    python scripts/export_golden.py          # これで fixtures を作り直す
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.synth.keying import KeyingParams                             # noqa: E402
from src.synth.synthesizer import SynthConfig, synthesize_from_text   # noqa: E402

OUT_DIR = _PROJECT_ROOT / "sample_wav"
SAMPLE_RATE = 8000
# 書き出す振幅のピーク。元の実録音 (0.44) に合わせてある。
PEAK_LEVEL = 0.44

# **コールサインは運用者自身のものだけ。** 相手局は書かない
# (CQ 形式にすれば相手を名指しせずに済む)。
#
# **``export_golden.py`` は先頭 30 秒しか使う。** それを超えると文の途中で
# 切れて、見本として読みにくくなる。25 秒前後に収まる長さにしてある。
SAMPLES: tuple[tuple[str, str, str, float], ...] = (
    (
        "oubun",
        "european",
        "CQ CQ DE JH0ILL K UR RST 599 QTH TOKYO NAME TARO HW?",
        20.0,
    ),
    (
        "wabun",
        "japanese",
        "コンニチハ、テンキ ハ ハレ デス。アンテナ ハ ダイポール デス。",
        20.0,
    ),
)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260815)
    for name, mode, text, wpm in SAMPLES:
        keying = KeyingParams(
            wpm=wpm,
            dash_dot_ratio=3.0,
            element_jitter_sigma_ratio=0.08,   # 教科書に近いが機械的すぎない程度
            tone_freq_hz=600.0,
            rise_fall_ms=5.0,
            pre_silence_sec=0.3,
            post_silence_sec=0.3,
        )
        config = SynthConfig(
            mode=mode,                          # type: ignore[arg-type]
            keying=keying,
            # **実録音と同じくらいの雑音を入れる (実測 13 dB)。**
            # きれいすぎるとエネルギーが数ビンに集中し、メル出力の一致を見る
            # テストが数値差で落ちる (純トーンだと 1.4e-4 で閾値 1e-4 を超えた)。
            # 雑音があるとスペクトルが広がって実録音に近くなる。
            # 読みやすさは保てる (20 WPM なら 15 dB でも完全に読める)。
            snr_db=15.0,
            filter_center_hz=600.0,
            filter_bandwidth_hz=400.0,
        )
        result = synthesize_from_text(text, config, rng, sample_rate=SAMPLE_RATE)
        # **振幅を下げてから書く。** 合成音はピークが 1.0 をわずかに超えることが
        # あり (実測 1.045)、PCM_16 に落とすとクリップして波形が方形に潰れる。
        # 潰れた音は符号として読めなくなる — 実際に 52 文字が 3 トークンに
        # なった (2026-08-15)。メモリ上では完璧に読めていたので、
        # **ファイルに落とす一手だけが原因**だった。
        #
        # **振幅は元の実録音 (ピーク 0.44) に合わせる。** メル出力の一致を見る
        # テスト (test_mel_matches_reference_on_real_audio) は**絶対値**の差を
        # 1e-4 未満で見ており、振幅が大きいほど誤差も大きくなる。0.7 にした
        # ところ 1.67e-4 で落ちた。
        samples = result.samples
        peak = float(np.abs(samples).max())
        if peak > 0:
            samples = (samples / peak * PEAK_LEVEL).astype(np.float32)
        path = OUT_DIR / f"{name}.wav"
        sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
        seconds = result.samples.size / SAMPLE_RATE
        print(f"{path.name}: {seconds:5.1f} 秒  {wpm:.0f} WPM  {len(text)} 文字")
        print(f"  {text}")
    print("\n次に scripts/export_golden.py を実行してください")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
