# 語間スペース復元の現状と限界

> **注記 (2026-07-31)**: 本文書の「案A: 語間トークンを語彙に追加」は既に実装済みです
> (`src/tokens/morse_tokens.py` の `WORD_BREAK_TOKEN_ID`)。以下の「現在の実装」節は
> 語彙追加前の音声エンベロープ方式の記述であり、`src/eval/harness.py` の評価経路は
> これを使いません。最新の状況は `docs/word_break_threshold_result.md` を参照。

CW モールス符号には「スペース文字」が存在せず、語間は **連続無音時間**
(7 dot 相当) で表現される. CTC モデルの出力はトークン列のみで時間情報を
失うため、デコード後の語間スペース復元は別途仕組みが必要.

## 現在の実装

### 推奨経路

```python
from src.infer.engine import InferenceEngine
from src.infer.word_breaks import detect_word_breaks_from_audio
from src.tokens.converter import TokenConverter

engine = InferenceEngine.from_checkpoint("models/full/best.pt")
wave = ...   # 8 kHz, float32

tokens = engine.decode_chunk(wave)
word_breaks = detect_word_breaks_from_audio(
    wave, tokens,
    sample_rate=8000,
    hop_samples=engine.frame_hop_samples,
)
converter = TokenConverter(mode="european")
text = converter.convert_timed(tokens, word_break_flags=word_breaks).text
# → "CQ DE JA0XYZ K" のようにスペース入り
```

### 内部処理

1. ``compute_envelope``: 音声 |x| の短窓 (~10ms) 移動平均
2. ``detect_silence_mask``: エンベロープのピーク比閾値 (~12%) で
   無音マスクを作る + モルフォロジカル closing でノイズ穴埋め
3. ``estimate_dot_samples``: OFF ラン長中央値 → 1 dot 推定
4. ``detect_word_breaks_from_audio``: 各トークン間ウィンドウの **合計無音**
   が ``word_gap_dots × 1 dot`` (デフォルト 6 dot) を超えれば語間

## 既知の制限

| 条件 | 動作 |
|---|---|
| クリーン (SNR > +15 dB) | おおむね正確、稀に false positive |
| 中程度 (SNR +5〜+15 dB) | 過半数は正解、文字間と語間の境界が紛らわしい |
| 低 SNR (SNR < 0 dB) | ノイズで無音マスクが断裂、false positive 増加傾向 |
| 高速 (WPM > 35) | 1 dot が短くなり時間分解能が苦しい |
| 低速 (WPM < 10) | 1 dot 長期間の探索で false positive 増加 |

### 根本原因

CTC のフレーム位置は **音声の物理的なタイミングではなく、モデルの確信度
ピーク位置** を表す. そのため、トークン間フレームギャップは実際の無音
時間と相関が弱い. 補完手段として音声エンベロープを使うが、ノイズに
弱く、各 SNR/WPM で最適パラメータが異なる.

## より正確な解 (将来作業)

### 案 A: 語間トークンを語彙に追加 (推奨)

語彙に ``WORD_BREAK`` トークンを 1 個追加し、合成器が語間を含むテキスト
("CQ DE JA0XYZ") を符号化する際に、語間 7 dot OFF の直後に ``WORD_BREAK``
ラベルを挿入. これにより:

- モデルは「7 dot OFF を見たら ``WORD_BREAK`` を出す」と学習
- 推論時は ``WORD_BREAK`` トークンを文字 " " に変換するだけ
- 全 SNR/WPM 域でロバスト

実装コスト: 中 (トークン追加 + データセット改造 + 再学習)

### 案 B: 教師あり強制アラインメント

CTC posteriors を CTC alignment で hard-align し、各 token の正確な発火
時刻を取り出す. ライブラリ (``torchaudio.functional.forced_align``) で
実装可能だが、CTC blank の扱いに注意が必要.

実装コスト: 中 (forced align + 後処理)

### 案 C: バンドパスフィルタで信号分離

検波周波数を知っているなら、信号トーン (例 600Hz ± 100Hz) のみを抽出
してエンベロープ計算すれば、低 SNR でも信号と無音を区別しやすくなる.
ただし周波数を自動推定する必要あり.

実装コスト: 小 (前処理追加のみ)

## 推奨

実運用では、デコード結果に語間スペースが入らないことを **既知の挙動** と
して受け入れ、必要なら案 A の再学習をエンハンスメントとして検討する.

現状の``convert_timed`` は **オプション機能** として位置付け、UI 表示は
デフォルト OFF にすることも検討の余地あり.
