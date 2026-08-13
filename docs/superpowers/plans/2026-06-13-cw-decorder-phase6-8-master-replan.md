# cw-decorder Phase 6〜8 マスター再計画

> 前提文書: `docs/archive/cw-decorder-phase6-8-instructions.md` (改修指示書 v1.0, 2026-06-13)、`docs/design.md` (Phase 1〜5)
> 本書は指示書を「実行可能な計画」に落とし込んだ上位計画。各 Phase の詳細 TDD 計画は別ファイルに分割する。

**ゴール:** リアルタイム精度のオフライン同等化 (Phase 6) と、実音声ギャップの構造的解消 (Phase 7→8) を、リグレッション基準を保ったまま段階的に達成する。

---

## 0. 現状コードとの突合せ (計画前の事実確認)

指示書を実装に落とす前に、実コードを確認して判明した重要な事実。**指示書の前提と差分があるものは方針を調整した。**

| 指示書の前提 | 実コードの現状 | 計画への反映 |
|---|---|---|
| §3.4 「特徴量正規化を学習/推論共通で新規実装」 | `src/train/preprocessing.py:73-78` で `MelExtractor` が既にチャンク内 z-score 正規化済み。学習・推論とも同一 `MelExtractor` を使用 | **WP7-3 は「新規実装」ではなく「① 時間領域 AGC をデコード経路から排除」＋「② 正規化を `src/dsp/features.py` へ切り出して明示的に共有」に再定義。** 正規化アルゴリズム自体は流用 (ε は既に `clamp_min(1e-5)` あり) |
| §2.2 「既存オーバーラップマージを置換」 | `StreamingDecoder.push()` はライブ `_tick` 経路で**実際には未使用**。ライブは蓄積バッファ + 無音自動チャンク (`decode_and_reset`)。`StreamingDecoder` は `stop()` の `flush()` でのみ作用 | Phase 6 の「ライブ (連続)」は**ワーカーへの新規追加**。旧 `StreamingDecoder` は新 `SlidingWindowDecoder` に置換し、テストを新仕様へ書き換える |
| §3.3 BPF 経路統一 | 推論側 BPF = `src/app/workers.py` 内の `_StreamingBPF` (バターワース4次SOS)。合成側 BPF = `src/synth/noise.py` (別実装) | WP7-2 で `src/dsp/bandpass.py` に一本化。推論側実装 (`_StreamingBPF`) を正とする |
| 既存テスト 365+ | 実カウント `def test` = **248 件** (365 は assert 数か旧集計)。`StreamingDecoder` テストは `tests/test_inference_engine.py::TestStreamingDecoder` | 「既存テストを壊さない」基準は **248 件 + 各 Phase 追加分**。`TestStreamingDecoder` は新仕様で同数以上に書き換え |
| `models/full/best.pt` を基準保持 | 同上 (存在) | 削除・上書き禁止。Phase 7 は `models/v2/`、Phase 8 は `models/ft/` へ保存 |

---

## 1. Phase 分割と依存関係

```
Phase 6  ストリーミング推論再設計 + settings 移行   [モデル無改修・即実行可]
  │   detail plan: docs/superpowers/plans/2026-06-13-phase6-streaming-redesign.md
  │   ← 完了後、新方式 (ライブ連続) で実録音を継続収集 → Phase 8 のデータが貯まる
  ▼
Phase 7  合成器刷新 + 特徴量経路統一 + ゼロ再学習   [一晩学習・GPU 必須]
  │   detail plan: 着手時に作成 (本書 §3 に WP 分解と検収を記載)
  │   依存: Phase 6 完了 (ライブ経路が SlidingWindowDecoder 化されていること)
  │         ※ src/dsp/features.py への特徴量切出しは WP7-3 自身の作業であり、
  │           Phase 6 は特徴量経路に触れない (Phase 6 はモデル無改修)
  │   外部依存: WP7-1 はユーザーの実バンドノイズ録音待ち (無ければ WP7-2〜4 で先行再学習可)
  ▼
Phase 8  実信号ファインチューニング                 [実録音 20 件以上で実施]
      detail plan: 着手時に作成 (本書 §4 に仕様と検収を記載)
      依存: Phase 7 完了 (models/v2/best.pt) + 実録音 ≥ 20 件
```

**重要な切替作業 (指示書 §6 注意):** Phase 7 完了後、Phase 6 のライブ連続デコードを `models/v2/best.pt` に差し替えて再検証する。`InferenceEngine` の前処理が `src/dsp/features.py` 経由になっている必要がある (WP7-3 のチェックリスト項目)。

---

## 2. ブランチ戦略

現状 `feature/word-spacing` に Phase 5 の改修群が**未マージ蓄積** + 未コミットの変更・ログファイルあり。
共通ルール (CLAUDE.md): main 直コミット禁止、`feature/説明` で作業、論理単位で細かくコミット、PR 経由でマージ。

推奨手順:

1. **先に現ブランチを整理** — 不要ログ (`decode_result*.txt`, `qso_decode.txt`, `*_decode.txt`, `app_record_decode.txt`) を `.gitignore` 追加 or 削除。`docs/design.md` と本指示書をコミット。`feature/word-spacing` を PR → main マージ (Phase 5 を確定)。
2. Phase 6: `feature/phase6-streaming` を main から切る。
3. Phase 7: `feature/phase7-resynth` (Phase 6 マージ後)。
4. Phase 8: `feature/phase8-finetune` (Phase 7 マージ後)。

> Phase 6 を現 `feature/word-spacing` の続きで進める選択肢もあるが、指示書が「リグレッション基準を保つ」明確な区切りを求めているため、**Phase 5 を一度 main へ確定**してから Phase 6 を切るのを推奨。

---

## 3. Phase 7 ワークパッケージ分解 (詳細 TDD は着手時に作成)

実装順 (指示書 §3.7): WP7-2 → WP7-3 → WP7-4 → WP7-1 → WP7-5。

| WP | 内容 | 主な新規/変更ファイル | 受け入れ確認 |
|---|---|---|---|
| **WP7-2** フィルタ経路統一 | BPF を `src/dsp/bandpass.py` に一本化。推論・合成が import | `src/dsp/bandpass.py` (新), `src/infer/audio.py`/`src/app/workers.py`/`src/synth/noise.py` (import 化) | 推論側 BPF と合成側 BPF が同一実装・同一パラメータ分布。二重フィルタ回避テスト |
| **WP7-3** AGC 排除 + 正規化共有 | 時間領域 AGC をデコード経路から除去 (表示用に残す)。log-mel 正規化を `src/dsp/features.py` へ切出し、学習/推論が共有。学習にランダムゲイン拡張 (-30〜0 dBFS) | `src/dsp/features.py` (新, MelExtractor 移設), `src/infer/engine.py`/`src/train/` (import 化), `src/app/workers.py` (AGC をメータ専用に) | 無音窓で分散ゼロ割りしない (ε)。学習/推論で同一特徴量。メタデータ検証 |
| **WP7-4** キーイング写実化 | 語間 4〜15 dot 分布 (中央値 7)、文字間 2〜6 dot、送信者単位の非対称打鍵癖、簡易 AGC 圧縮模擬 30% | `src/synth/keying.py`, `src/synth/text_generator.py` | 語間分布テスト。WORD_BREAK 過剰検出の根本対策 |
| **WP7-1** 実バンドノイズ重畳 | `RealNoiseBank` で `data/noise/*.wav` をランダム切出し重畳。`real_noise_prob=0.7`、空なら AWGN 100% フォールバック + 警告 | `src/synth/noise.py` (RealNoiseBank 追加), `data/noise/` (新規・ユーザー録音) | フォールバック動作。SNR スケーリング。QRM 重畳継続 |
| **WP7-5** ゼロ再学習 + 検収 | 新評価セット `data/eval/v2/`、ゼロから ~100k step、ckpt に `meta` (合成器版/特徴量設定/git hash/語彙版) 埋込。`models/v2/best.pt` | `scripts/train.py`/`src/train/checkpoint.py`, `scripts/compare_models.py` (新) | 下表 |

**WP7-5 検収基準 (指示書 §3.6):**

| 項目 | 基準 |
|---|---|
| 合成評価 v2 | TER ≤ 1% / SNR≥0dB、TER ≤ 15% / SNR -5dB |
| 旧評価セット | TER ≤ 5% (大幅劣化なし) |
| 実音声リグレッション | `data/real/` 4 件で旧モデル比トークン一致率改善。`docs/v2_regression.md` に新旧並記 |
| WORD_BREAK | 実音声 4 件で過剰検出が旧比減少 |
| 学習時間 | RTX 5060 Ti で ≤ 10 時間 |

> WP7-1 の録音が揃わない場合、WP7-2〜4 のみで一度再学習して効果確認 → 実ノイズ後追加で再々学習 (指示書 §3.7)。

---

## 4. Phase 8 仕様 (詳細 TDD は着手時に作成)

既存 `src/finetune/` 基盤を強化。詳細は指示書 §4。

| 項目 | 仕様 |
|---|---|
| データ | `data/real/*.wav + .txt`、目標 20〜30 件 (10 件未満は警告)。**現在 4 件 → 不足** |
| 混合 | `MixedRealSynthDataset` 実データ比率 20% (10〜30%)、残りは新合成器オンザフライ |
| 学習率 | ベースの 1/10 (3e-5) |
| 凍結 | CNN 凍結、BiLSTM + Linear のみ更新 (`--freeze-cnn` デフォルト ON) |
| 忘却監視 | 合成評価 v2 で定期評価、TER 2 倍悪化で早期停止 |
| 保存 | `models/ft/` (ベース meta 引継ぎ) |
| 検収 | ホールドアウト録音 ≥3 件でベース比改善、合成 v2 劣化 ≤2 倍 |

---

## 5. 運用者のアクション項目 (Claude Code 作業外)

指示書が明記する、コード実装と並行して必要な人手作業:

1. **[Phase 6 と並行] 実 QSO 録音の継続収集** — Phase 6 のライブ連続モードで運用しつつ `data/real/` を 20 件以上へ。各 `.txt` の正解を人手確認・修正。
2. **[WP7-1 前提] 実バンドノイズ録音** — 実運用と同じ受信機設定 (ナローフィルタ・AGC 込み) で無信号周波数を数時間録音し `data/noise/` に配置 (8kHz)。複数時間帯・コンディションが望ましい。README に手順を追記する (本計画の Phase 7 ドキュメント要件)。

---

## 6. スコープ外 (指示書 §5 — 実装しない)

beam search + 言語モデル / ホレ・ラタ自動切替 / 単方向 LSTM 化 / ONNX・PyInstaller・RasPi。効果測定後に別途判断。

---

## 7. 次アクション

1. 本マスター再計画をユーザー確認。
2. Phase 6 詳細計画 (`2026-06-13-phase6-streaming-redesign.md`) に従い実装着手。実行方式は subagent-driven / inline をユーザー選択。
3. Phase 6 完了・マージ後に Phase 7 詳細計画を作成。
