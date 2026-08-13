# B1: 実効SNR学習範囲の拡張 設計書

作成日: 2026-07-19
対象: `cw-decorder` / Phase B レバー B1
前提: B0 評価ハーネス (`eval_model.py`, `effective_snr_db`, synth_val/keyed_val) は
main にマージ済み (`266c46f`)。

## 1. 目的

実信号「50% 頭打ち」の正体は、B0 スモークで実測した **実効SNR 0dB 付近の崖**
(synth_val TER: +10dB 1.1% / +5dB 9.3% / **0dB 51.9%** / −5dB 91.9%)。
原因は、学習の SNR 指定が **BPF 前の nominal 値** (`uniform(-10,20)`) で、受信機 BPF の
下駄 (帯域500で +9.5dB、帯域可変250-500で +9.5〜+12.5dB) により、モデルが
**実効 −0.5dB より下を一度も学習していない**こと (Phase A §2.1/2.2)。

B1 は学習の SNR を **実効(帯域内)SNR で指定**できるようにし、実効 −8dB まで
学習範囲を広げて崖を下げる。効果は B0 の `eval_model.py` で前後比較して測る。

**モデル構造・特徴量・decode・推論は変更しない。** 変更は合成器の SNR 付加のみ。

## 2. 実効SNR指定の仕組み

目標実効SNRを受け取り、信号とノイズをそれぞれ受信機 BPF に通した後のパワー比が
目標になるよう、広帯域 AWGN をスケールして **BPF 前に加算**する。その後パイプライン
通り BPF が適用され、BPF 後の実効SNRが目標に一致する。ノイズは BPF 前で加わる
(物理的に正しい: HF ノイズは受信機フィルタの前段から入る) ままで、**指定だけが実効値**。

試作で **帯域 {250, 375, 500} すべてで目標 {−5, 0, 10, 20}dB を誤差 ±0.00dB で命中**
することを確認済み。BPF 帯域が可変でも実効SNRの下限が確定する。

## 3. コンポーネント

変更は狭く、既定オフで後方互換。

### 3.1 `src/synth/noise.py`

追加:
```
add_awgn_effective(
    sig: np.ndarray, target_snr_db: float,
    center_hz: float, bandwidth_hz: float,
    sample_rate: int, rng: np.random.Generator,
) -> np.ndarray
```
- 単位分散の広帯域ノイズを生成、`apply_cw_filter` で信号・ノイズそれぞれの BPF 後
  パワーを求め、`scale = sqrt(P_sig_bpf / (target_lin * P_noise_bpf))` を計算、
  `sig + scale * unit_noise` を返す (BPF 前の状態)。
- `signal_power` が 0 (無音信号 or ゼロ長) の場合は `sig.copy()` を返す (例外を出さない)。
- 既存 `add_awgn` / `add_real_noise` / `apply_cw_filter` / `effective_snr_db` は不変。

### 3.2 `src/synth/synthesizer.py`

- `SynthConfig` に `snr_is_effective: bool = False` を追加 (既定 False = 現状維持)。
- `synthesize_from_text` の step5 (AWGN) を分岐:
  - `config.snr_is_effective` が True なら
    `add_awgn_effective(samples, config.snr_db, config.filter_center_hz,
    config.filter_bandwidth_hz, sample_rate, rng)`。
  - False なら現状の `add_awgn(samples, config.snr_db, rng)`。
- step6 (BPF) 以降は不変。`apply_receiver_filter=False` の場合の実効モードは
  BPF が無いので実効≒nominal になるが、実運用では常に BPF 有効なので許容
  (設計上は BPF 有効前提)。

### 3.3 `src/synth/dataset.py`

- `DefaultConfigSampler.__init__` に `effective_snr_range: tuple[float, float] | None = None`
  を追加 (既定 None = 現行 nominal `uniform(-10,20)`)。
- `__call__`: `effective_snr_range` が指定されていれば
  `snr_db=uniform(*effective_snr_range)` かつ `snr_is_effective=True` の
  `SynthConfig` を返す。None なら現行どおり `snr_db=uniform(-10,20)`,
  `snr_is_effective=False`。

### 3.4 `scripts/train.py`

- CLI `--eff-snr-min` / `--eff-snr-max` (float, 既定 None) を追加。
- 両方指定時: `DefaultConfigSampler(mode, effective_snr_range=(min,max))` を
  `MorseSynthDataset(config_sampler=...)` に渡す。
- 片方のみ指定 → エラー (両方必須)。min >= max → エラー。
- 未指定 → 現行動作 (nominal サンプラ)。

## 4. 学習範囲の既定と実験手順

**実効SNR学習範囲の既定: `[-8, +25] dB`**
- 下限 −8: eval グリッド最低ビン −5dB を分布の内側に置き、−5dB での頑健性を上げる。
  −10 まで下げないのは、そこは人間可読域を超え容量を浪費しかねないため。
- 上限 +25: クリーンのヘッドルーム。
- CLI 可変なので結果を見て `[-6,...]`/`[-10,...]` へスイープ可能。

**実験手順 (測定込み)**:
```
1. baseline: python scripts/eval_model.py --ckpt models/full/best.pt \
     --noise-dir data/keying_scripts --keyed-dir data/keying_scripts \
     --out models/eval/b1_baseline.json
2. 再学習: python scripts/train.py --eff-snr-min -8 --eff-snr-max 25 \
     (他は現行と同条件) → models/full_b1/best.pt
3. 改善測定: python scripts/eval_model.py --ckpt models/full_b1/best.pt \
     --noise-dir data/keying_scripts --keyed-dir data/keying_scripts \
     --out models/eval/b1.json --baseline models/eval/b1_baseline.json
```

**成功条件**:
- synth_val の 0dB / −5dB ビンの TER が baseline より明確に低下 (例 0dB 51.9%→30%台)。
- keyed_val (実打鍵、合成ノイズ非使用で漏洩なし) の TER も低下すれば実信号転移を確認。
- +10dB 等の高SNRビンが悪化しない (分布拡張で easy ケースを落としていない)。

再学習ランは**ユーザーが手元 GPU で実行**する (本 B1 はコード・CLI 整備まで)。
測定は eval_model.py で即実行可能。

## 5. テスト方針

- `add_awgn_effective` 実効SNR命中: 目標 {−5, 0, 10}dB を渡し `effective_snr_db` で
  測って ±1dB 以内。帯域 {250, 500} 両方で命中。
- 決定性: 同 Generator シード → 同一出力。
- 退化ケース: 無音信号で例外を出さず `sig.copy()` を返す。
- `SynthConfig.snr_is_effective` 既定 False で既存合成経路が不変。
- `synthesize_from_text(snr_is_effective=True)` で目標実効SNRが結果に反映
  (BPF 後 `effective_snr_db` で ±1dB 程度)。
- `DefaultConfigSampler(effective_snr_range=(a,b))` の config が
  `snr_is_effective=True` かつ snr_db が範囲内。None で現行 nominal のまま。
- train.py CLI: `--eff-snr-min/max` 両指定で実効モード、片方のみ/逆転でエラー、
  未指定で現行動作。
- 既存 573 テストを壊さない。`add_awgn`/`make_fixed_eval_set`/既存 SynthConfig
  利用箇所は不変。e-Gov 符号照合も不変。

## 6. エラー処理

- `--eff-snr-min` のみ or `--eff-snr-max` のみ → エラー (両方必須の明示メッセージ)。
- `--eff-snr-min >= --eff-snr-max` → エラー。
- `add_awgn_effective` の無音/ゼロ長信号 → `sig.copy()` (例外なし)。

## 7. 性能の注意

`add_awgn_effective` はスケール計算に信号・ノイズの BPF 通過 (sosfiltfilt 2回/サンプル)
を要する。オンザフライ合成で GPU が待たないこと (CLAUDE.md 目標 GPU 80%) を確認。
ボトルネックなら後で高速化 (帯域ごとのノイズ等価帯域の近似で filter 省略) するが、
まず正確さを優先する (再学習は一度のため correctness > speed)。

## 8. スコープ外

- 実ノイズ混合を base 学習に入れること (別レバー、今回不採用)。
- トーン中心ピーク追従 (B2)、WORD_BREAK (B3)、AGC (B4)、プロサイン (B5)。
- 実際の再学習ランの実行 (コード整備まで)。
- `add_awgn_effective` の性能最適化 (ボトルネック時に対応)。

## 9. 成果物

- 学習SNRを実効(帯域内)値で指定できる (`--eff-snr-min/max`)。
- 学習グリッドと eval_model の synth_val グリッドが同じ実効SNR単位になる。
- 帯域可変下でも実効SNR下限が確定 (±0dB 実証)。
- 実効 −8dB まで学習を広げ、崖を下げる再学習を回せる状態。
- 改善は eval_model.py で実効SNR別・モード別に前後比較できる。
