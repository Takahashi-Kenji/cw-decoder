# B0: 評価ハーネス配線 設計書

作成日: 2026-07-18
対象: `cw-decorder` / Phase B の基盤サブプロジェクト (B0)
前提: Phase A 評価基盤 (実効SNR・実ノイズ固定評価セット・token別エラー分析) は
main にマージ済み (`3a9cd9d`)。本 B0 はそれらを実利用に繋ぐ。

## 1. 目的

Phase A で作った評価部品 (`make_fixed_real_noise_eval_set`, `EvalRecord.eff_snr_db`,
`EvalReport.by_eff_snr`, `DetailedEvalReport`) は現状 dormant (単体テストでしか
使われていない)。B0 はこれらを **独立した評価スクリプト** に配線し、任意の
チェックポイントに対して実信号相当の精度を測定・保存・比較できるようにする。

これは Phase B の全レバー (B1: 実効SNR学習範囲拡張、B2: トーン追従、B3: WORD_BREAK、
B4: AGC、B5: プロサイン) の効果を測る共通の物差しになる。B0 が無ければ改善を
検証できない。

**モデル・特徴量・decode アルゴリズム・学習ロジックは変更しない。** B0 は測定の
配線のみ。

## 2. 成果物

`scripts/eval_model.py`:

```bash
python scripts/eval_model.py --ckpt models/full/best.pt \
    --noise-dir data/keying_scripts --keyed-dir data/keying_scripts \
    --out models/eval/baseline.json

# FT 後にもう一度、baseline と比較
python scripts/eval_model.py --ckpt models/ft/best.pt \
    --noise-dir data/keying_scripts --keyed-dir data/real/val \
    --out models/eval/ft.json --baseline models/eval/baseline.json
```

## 3. 評価セット

性質の違う 2 セットを別々に評価・報告する。

| セット | 中身 | 供給元 |
|---|---|---|
| `synth_val` | 合成キーイング + 実ノイズ、実効SNR×WPM グリッド、seed 固定 | `--noise-dir` のノイズ WAV |
| `keyed_val` | 打鍵録音 (実受信経路) | `--keyed-dir` の WAV+TXT |

`keyed_val` は `--keyed-dir` 未指定ならスキップ (synth_val のみ評価)。

### 3.1 synth_val の既定値 (Phase A 実測、CLI で上書き可)

| パラメータ | 既定 | 根拠 (Phase A 設計書) |
|---|---|---|
| トーン中心 | 494.0 Hz | §2.6 実測 (20件安定) |
| BPF 帯域 | 300.0 Hz | §2.6 ノイズ帯域 300–550 Hz |
| WPM グリッド | [17.0, 25.0] | ラグチュー / コンテスト |
| SNR グリッド | [10.0, 5.0, 0.0, -5.0] | §2.2 崖の周辺を厚く |
| samples/cell | 25 | 分散と実行時間の折衷 |
| モード | european, japanese 両方 | モード別に集計 |
| seed | 20260718 | 決定的生成 |

SNR グリッド値は `make_fixed_real_noise_eval_set` の `snr_grid` (= `add_real_noise`
の目標 SNR) として渡す。実録音ノイズは帯域内なので実効 SNR ≒ 目標だが、各サンプルの
`eff_snr_db` は `effective_snr_db` で実測して記録し、集計は実効SNRビンで行う。

`data/keying_scripts/noise_sample.wav` はラベル無しなので `discover_real_samples`
から自動除外され、`--keyed-dir data/keying_scripts` は打鍵20件のみを拾う。

## 4. コンポーネント

### 4.1 `src/eval/harness.py` (新規、再利用可能な評価ロジック)

- `decode_wave(model, mel_extractor, wave, device) -> DecodeOutput`
  - 波形 (1D tensor) を受けて `mel → model → log_softmax → compute_input_lengths →
    ctc_greedy_decode` を実行し、予測 token 列を返す純粋な推論ラッパ。
  - `DecodeOutput` は `token_ids: list[int]` を持つ (confidence は将来拡張余地)。
- `evaluate_real_dataset(model, mel_extractor, dataset, device) -> DetailedEvalReport`
  - keyed_val 用。`RealSignalDataset` を走査し、各サンプルを `decode_wave` で
    デコード、`TokenConverter(mode=meta.mode)` で表示テキスト化、
    `EvalRecord` を作り `DetailedEvalReport.add(..., name=meta.stem, mode=meta.mode)`。
  - 現 `scripts/finetune.py::evaluate_real` と同一ロジック。finetune はこれに委譲する。
- `evaluate_synth_noise(model, mel_extractor, samples, device) -> DetailedEvalReport`
  - synth_val 用。`list[RealNoiseEvalSample]` を走査、各 `samples` (波形) を
    `decode_wave`、`EvalRecord(eff_snr_db=s.eff_snr_db)` を作り、
    `DetailedEvalReport.add(..., mode=s.mode, eff_snr_bin=bin_snr(s.eff_snr_db))`。
  - `bin_snr` は既存 (metrics.py)。

### 4.2 `scripts/eval_model.py` (新規 CLI)

引数: `--ckpt` (必須), `--noise-dir` (synth_val 用), `--keyed-dir` (keyed_val 用、任意),
`--out` (JSON 出力先、既定 `models/eval/eval.json`), `--baseline` (比較元 JSON、任意),
`--device` (既定 cuda→cpu フォールバック), `--seed`, synth グリッド上書き用の
`--tone-center` / `--bpf-bandwidth` / `--wpm` / `--snr` / `--samples-per-cell`。

処理: ckpt ロード → (noise-dir あれば) synth_val 評価 → (keyed-dir あれば) keyed_val 評価
→ report dict 構築 → `--out` へ JSON 保存 → summary を stdout → (`--baseline` あれば)
`compare_reports` の結果を表示。

### 4.3 `DetailedEvalReport.to_dict()` / `summary_lines()` の拡張 (metrics.py)

- `to_dict()` に `by_eff_snr` と `by_mode` の集計を追加する。
  - 既存キー (`overall`, `totals`, `token_errors`, `samples`) は壊さない (後方互換)。
  - `by_eff_snr`: `{ "<bin>": {n_samples, ter, cer} }`。
  - `by_mode`: `{ "european": {...}, "japanese": {...} }`。`DetailedEvalReport` が
    サンプルの mode を保持する必要があるため、`SampleEval.mode` (既存) を集計する。
- `summary_lines()` に "By EffSNR" 節を追加 (by_eff_snr が空でなければ)。
  - Phase A の deferred Minor (#3) をここで解消。
- confusion は `analysis.confusion_to_dict()` を JSON に含める。

注意: `EvalReport.by_eff_snr` は Task 3 で追加済みだが `DetailedEvalReport` 側からは
未surface。`DetailedEvalReport.add` は既に `eff_snr_bin` を forward する
(Task 3 で実装済み) ので、`evaluate_synth_noise` が `eff_snr_bin` を渡せば集計される。

### 4.4 `scripts/finetune.py` の委譲 (重複解消)

`evaluate_real` の本体を `harness.evaluate_real_dataset` の呼び出しに置換する。
CLI インターフェース・出力・保存フローは不変 (`evaluate_real` の呼び出し側は変えない、
関数を薄いラッパにするか import して差し替え)。final review が指摘した
inspect_real_labels ↔ finetune の decode ループ重複も、harness 経由に寄せて解消する
(inspect_real_labels も `decode_wave` を使う形に更新)。

### 4.5 `compare_reports(baseline: dict, current: dict) -> dict` (harness.py、純関数)

2 つの report dict を受け、各セクション (synth_val / keyed_val) について:
- overall TER/CER の delta (current - baseline)
- by_eff_snr の各ビンの TER delta (どちらかに欠損するビンは "n/a")
- by_mode の各モードの TER delta
- token別 recall の改善上位 (baseline に対する recall 変化が大きい token)

を返す。表示は eval_model.py 側。TER は減少が改善なので、符号をそのまま delta として
出し (負が改善)、ラベルで「(改善)」を示す。

## 5. 出力 JSON

```json
{
  "ckpt": "models/full/best.pt",
  "seed": 20260718,
  "synth_val": {
    "config": {"tone_center_hz":494.0, "bpf_bandwidth_hz":300.0,
               "wpm_grid":[17.0,25.0], "snr_grid":[10.0,5.0,0.0,-5.0],
               "samples_per_cell":25},
    "overall": {"n_samples":..., "ter":..., "cer":...},
    "by_eff_snr": {"-5.0":{"n_samples":..,"ter":..,"cer":..}, ...},
    "by_mode": {"european":{...}, "japanese":{...}},
    "totals": {"substitutions":.., "deletions":.., "insertions":..},
    "token_errors": [...],
    "confusion": {...}
  },
  "keyed_val": {
    "overall": {...}, "by_mode": {...}, "totals": {...},
    "token_errors": [...], "confusion": {...}, "samples": [...]
  }
}
```

`synth_val` はサンプル数が多くなり得るので `samples` (サンプル別詳細) は含めない
(集計のみ)。`keyed_val` は件数が少ないので `samples` を含める (誤ラベル追跡用)。

## 6. テスト方針

torch 非依存の純ロジックを単体テスト:

- `by_mode` 集計: 欧文・和文混在の `DetailedEvalReport` で正しくモード別分離。
- `to_dict()` 拡張: `by_eff_snr` / `by_mode` が JSON 化でき往復一致。既存キー不変。
- `summary_lines()`: by_eff_snr がある時 "By EffSNR" 節が出る、無い時は出ない。
- `compare_reports`: TER delta の符号、欠損ビンの "n/a"、token recall 改善抽出。
  決定的な dict 入力で検証。
- CLI smoke (実データがある場合のみ): `models/full/best.pt` + `noise_sample.wav`
  + 打鍵20件で走らせ、JSON が出て synth_val の 0dB 付近で TER が急増する
    (崖が再現する) ことを確認。無い場合はその旨報告。

既存 562 テストを壊さない。`evaluate_real` 委譲後も finetune のテストが通ること。

## 7. エラー処理

- `--noise-dir` にノイズ WAV が無い → `RealNoisePool.from_dir` が ValueError。
  明確なメッセージで exit 2。
- `--keyed-dir` 未指定 or 空 → keyed_val をスキップ、synth_val のみ評価 (警告表示)。
- synth_val も keyed_val も評価できない → exit 2。
- `--baseline` の JSON が壊れ/キー欠損 → 比較をスキップして単体評価は続行
  (測定を止めない)。警告表示。
- `models/eval/` 出力先が無ければ mkdir。

## 8. スコープ外

- モデル・特徴量・decode・学習ロジックの変更。
- B1 (実効SNR学習範囲拡張) 以降の全レバー。
- keyed_val の train/val 予約分割 (B1 で FT する際に対応)。B0 は `--keyed-dir` を
  受けるだけ。
- confidence を使った低信頼度評価 (decode_wave は token_ids のみ)。

## 9. 成果物まとめ

- 任意 ckpt に対し synth_val (実効SNR別・モード別) と keyed_val を測定できる。
- baseline JSON と比較して「どの実効SNR・モード・token が何%改善したか」を出せる。
- Phase A の dormant だった `by_eff_snr` / `make_fixed_real_noise_eval_set` が
  実利用される。
- decode ループ重複 (finetune / inspect_real_labels) が harness に集約される。
- Phase B の各レバーの効果を測る土台が整う。
