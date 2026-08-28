# モデル学習計画 v4

作成日: 2026-08-24
対象: `cw-decorder` (公開名 `cw-decoder`)
前提文書: [認識精度向上計画書 v3](./accuracy_improvement_plan_v3.md) の**学習側**を実行計画に落としたもの。
推論側 (LM 融合・WORD_BREAK 適応閾値) は範囲外。
進捗と結果は [実験台帳](./experiments/ledger.md) に記録する。

## 運用者の指示 (2026-08-27)

**用語**: 「手打ち」という呼び方は使わない。**誤解を生む。** 実測すると
`data/ft/real_*` の 90 件も L4 もエレキー打鍵で、要素長のばらつきは
むしろ L4 の方が大きい (長音σ 0.509 対 0.153)。違うのは動作点である。

| 呼び方 | 実体 | WPM | 文字間 | コントラスト |
|---|---|---|---|---|
| 旧録音 (90 件) | `data/ft/real_train` 72 + `real_val` 18 | 20.1 | 3.13 | 9.0 dB |
| L4 (2,400 件) | `data/l4/2026082{4,5}_*` | 24.3 | 3.24 | 22.5 dB |
| L4 最新 (1,200 件) | `data/l4/20260825_50w_18m` | 24.3 | 3.24 | 22.5 dB |
| held-out (20 件) | `data/keying_scripts` | 28.8 | 4.38 | 28.4 dB |

**学習データ**: 以後の再学習は **L4 2,400 件、または L4 最新 1,200 件**を使う。
旧録音 90 件は学習に使わない。

**根拠 (運用者)**: ばらつきがある場合は悪化することが分かった。
**本来のコードに近いほど認識率が上がる**。

## Context

`docs/accuracy_improvement_plan_v3.md` は held-out 実録音で **欧文 15% / 和文 15%** を掲げた。
現在は欧文 **13.06%** / 和文 **27.15%** (2026-08-24 にラベルを是正して測り直した値。
下記 P0 の結果)。**欧文は既に目標を満たしていた** — 満たしていないように見えていたのは
物差しの誤りだった。実体は**和文を 12.2pt 下げる計画**である。

v3 は推論側 (LM 融合・WORD_BREAK 適応閾値) と学習側 (素材移植合成・FT・フル再学習) の
両方を含む。**本計画は学習側だけを扱う** — データをどう作り、何を学習し、何で判定するか。
LM 融合は別計画とし、ここでは「LM が乗る前の素の読み取り精度」を上げる。

着手前に片付けなければならないことがある。**物差しが壊れている。**

| 欠陥 | 本日の実測 |
|---|---|
| held-out ラベルが音と食い違う | `scripts/audit_labels.py` 実行: **ホレ 2 件・ラタ 1 件が「音にあるのにラベルに無い」** (`script_ja_03`/`ja_08` 冒頭のホレ、`ja_05` 末尾のラタ)。和文は 208 トークンしかないので **3 個 ≈ 1.4pt** |
| `best.pt` が飽和した指標で選ばれる | `scripts/train.py` の学習中評価は `make_fixed_eval_set(mode="european")` のみ。`models/full/eval.csv` は step 100k/101k/102k が**すべて TER 0.00%・CER 2.48% の同点**。`new_best = ter < best_ter` は strict なので、**同点が続く区間ではどの重みが残るかは実質的に運** |
| 主要 KPI が CLI に出ない | `scripts/eval_model.py:96-109` はモード別 TER を JSON に書くだけで印字しない。`src/eval/compare.py:70` は計算済みなのに `_print_comparison` が出さない |
| 実データ用の val が無い | `data/real/` にあるのは `train` だけ。実信号で best を選ぶ土台がない |

2026-08-03 に「打鍵されていない」として除去したホレ/ラタは、**2026-08-12 に独立デコーダ
別実装のデコーダ と波形の要素列で「除去の方が誤りだった」と判明している** (`scripts/audit_labels.py`
の docstring に記録)。にもかかわらずラベルは直っていない。
一方**学習データ 90 件は同じ監査で 0 件**。壊れているのは物差しだけである。

GPU は夜間に回し放題という条件を得た。だからこそ順序が要る。**壊れた物差しのまま夜間
学習を始めると、10 時間の実験を何本回しても何が効いたか永久に分からない。** 過去 8 本の
学習のうち 7 本が不採用に終わった歴史は、モデルの問題である以上に評価と実験順序の問題だった。

---

## 目標 (段階評価)

| 指標 | 現在 | P0 後 | P3 後 | P4-A 後 | P4-B 後 |
|---|---:|---:|---:|---:|---:|
| 和文 TER | 31.7% | (ラベル是正で見かけ ~30.3%) | **24.0%** | **19.0%** | **18% 前後** |
| 欧文 TER | 15.9% | — | 15.0% | 13.5% | **12〜13%** |
| synth_val TER | 45.2% | 悪化 3pt 以内を維持 | ← | ← | ← |

**和文 15% は LM 融合 (別計画) と合わせて到達する数字**とし、学習側単独では 18% を射程とする。
根拠: 同じ録音を熟練運用者が耳で読んで CER 28.57%、モデル 34.92%、**両者は同じ箇所を
読み違える**。素の読み取りで 15% は人間の耳を 13pt 超える領域である。
各フェーズ末に未達なら**次へ進まず予算配分を見直す** (§8)。

---

## P0 — 物差しを直す 【完了 2026-08-24 / ブランチ `fix/eval-yardstick`】

**これが終わるまで夜間学習を始めない。** 判定基準を「−2pt で採用」に置くのに、
ラベル誤差が 1.4pt ある状態では、判定の 7 割をラベルバグが決めてしまう。

| # | やること | 触るファイル | 判定 |
|---|---|---|---|
| 1 | `{HORE}`/`{RATA}` を復元。`label_note` は**除去の履歴を消さず**「2026-08-03 の除去は誤り、2026-08-12 に波形と 別実装のデコーダ で確認し復元」に書き換え | `data/keying_scripts/script_ja_03.txt` / `ja_05.txt` / `ja_08.txt` | `audit_labels.py` 警告 0 件 |
| 2 | 21 件中 12 件に 2026-08-03 の編集跡。`--verbose` で全件の符号列を目視照合 | 同上 | ホレ/ラタ以外の編集も音と一致 |
| 3 | **ベースラインを打ち直す**。以後の全比較の原点を `models/eval/baseline_v4.json` に固定 | — | 新基準値を `docs/training_conditions_and_results.md` に記録 |
| 4 | by_mode を印字。`DetailedEvalReport.summary_lines()` に 1 節足すと **`eval_model.py` と `finetune.py` の両方に出る** | `src/train/metrics.py:576`、`scripts/eval_model.py:142` | CLI にモード別 TER が出る |
| 5 | held-out を名前付き複数セットに。`--keyed-dir` → `--keyed-set NAME=PATH` (append)、旧形式は互換で受ける | `scripts/eval_model.py:46,100-111` | 1 コマンドで v1/v2 両方が出る |
| 6 | 学習中評価に和文合成を追加 (`make_fixed_eval_set(mode="japanese")` は既存)。`--keyed-dir` + `evaluate_real_dataset` で実録音 val、`--best-metric {synth_eu,synth_ja,keyed}` (既定 keyed) | `scripts/train.py` | 500 step スモークで **best が 2 回以上更新される** (飽和していない) |
| 7 | `--resume` が不在パスで黙ってスクラッチ開始するのを `return 2` に | `scripts/train.py:205` | タイポで 10 時間を溶かさない |
| 8 | FT に `--synth-val-noise-dir` (既存 `evaluate_synth_noise` を呼ぶ) と `--subsample-n/--subsample-seed` | `scripts/finetune.py` | 採否 3 条件が 1 コマンドで判定できる / 学習曲線がディレクトリ分割なしで引ける |
| 9 | `save_checkpoint(extra=…)` (既にあるのに未使用) に argv/git_rev/seed/データ件数。同じ dict を `<ckpt-dir>/meta.json` にも | `src/train/checkpoint.py` 呼出側 | ckpt を開かずに台帳へ転記できる |
| 10 | 黙って捨てている件数を印字 (モード不明・空ラベル)、`--strict` で 1 件でも落ちたらエラー | `src/finetune/dataset.py:87,136` | 新データ源で何件落ちたか分かる |
| 11 | `MixedRealSynthDataset` が全 worker で同一 seed を使う不具合を修正 | `src/finetune/pipeline.py` | フル学習で workers を上げても実サンプルが重複しない |

**実データ val の作り方**: `data/real/val` は存在しない。P0 のスモークは合成和文で足りる。
実信号 val は **P1 の新規データから作る** (既存 90 件を削ると学習側が痩せる)。
held-out v1/v2 は val に**絶対に使わない** (物差しの汚染)。

---

## P0.5 — 先頭幻覚 (0.5 日) 【ブランチ `fix/lead-in-hallucination`】

> **2026-08-26 追記: この節の仮説は実測で覆った。`docs/head_hallucination_cause.md` を見ること。**
>
> - 「ゼロ → ノイズ床の段差」が原因ではない。**段差を作ると悪化する**
>   (ノイズ床 2.0 秒で TER 19.74% → 24.03%)。ゼロパッドを剥がしても
>   **数値が 1 ビットも動かない**
> - 真の原因は**学習分布の偏り**。合成は符号が中央 **0.14 秒**で始まるのに
>   実信号は中央 **1.38 秒**待つ。先頭を 0.14 秒に切り詰めると
>   **学習なしで TER −1.29pt / 先頭誤り 24 → 11**
> - 影響の見積り「~0.4pt」も誤り。**実測 5.15pt** (誤りの 26%) で 1 桁違う
> - 対策は「合成の先頭長をばらつかせる」。下記の A/B/C はいずれも的を外している

`docs/deepcw_comparison.md`: held-out 欧文**全 10 件**で先頭に余計な文字が出る (別実装のデコーダ は 0/10)。
和文では「冒頭で無条件にホレを吐く癖」として現れる。推論側で無音を足しても 0.4pt しか動かない。

**確認した非対称** (コードを読んだ事実):

| 経路 | 先頭の並び |
|---|---|
| 合成 (学習の大多数) | `synthesize_from_text` はステップ 5 で **AWGN を無音込みの全体に加算** → **サンプル 0 からノイズが乗る**。段差が無い |
| 実録音 FT データ (90 件) | `import_keying_recordings.py` の `_pad` が BPF **後**にゼロ埋め → **ゼロ → ノイズ床の立ち上がり → 符号** |
| 推論 | `sliding_window.py:53` `DEFAULT_LEAD_IN_S=0.3` がゼロを前置 → 実音声 → 符号。**実録音データと同じ形** |

つまり「ゼロからノイズ床が立ち上がる段差」は**合成側にだけ無い**。露出は合成が圧倒的に
多いので、この段差をキーダウンと取り違えている可能性がある。**仮説であって確定ではない。**

安い順に 3 本:

| # | 内容 | 所要 | 判定 |
|---|---|---|---|
| A | **推論の lead-in をゼロ→入力先頭のノイズ床コピーに変える** (`src/infer/sliding_window.py`) | 30 分 | 先頭余剰 10/10 → 3/10 以下なら**学習不要で解決** |
| B | 合成で再現するか (`ablate_keying_factors.py` に `zero_lead_in` arm) | 1 時間 | 再現すれば原因確定。**A で消えても B は回す** (P4 のデータ生成に反映が要るため) |
| C | FT データ側を直す: `import_keying_recordings.py` に `--pad-mode {zero,noise}` → 再インポート → 1,000 step FT | 1.5 時間 | 欧文 −0.3pt 以上、和文の冒頭ホレが減る |

**期待値を正しく置く**: 1 トークン/ターンなので held-out 全体では ~0.4pt。
**単独では目標に届かない。安いから先にやるだけである。** C を採用するなら
P1 以降の全データを `--pad-mode noise` で作り直す。

---

## P1 — データを作る (4 日、うち夜間 3)

### P1-A: L4 実受信機チェーンの閉ループ 【最優先・新規は接着 1 本だけ】

| 役割 | 既存の部品 |
|---|---|
| 打鍵 (無線機 PC) | `scripts/cw_key_server.py` (`--check-lines` / `--dry-run` / `--text`)。**2026-08-12 に実機で送信確認済み** |
| 送信指示 (GPU PC) | `src/tx/net_key.py` (`check` → `send`) |
| 音声の返送 | `scripts/audio_send.py` → `src/infer/net_audio.py` `NetworkAudioCapture` |
| 原稿 | `src/synth/text_generator.generate_text` / `src/finetune/keying_scripts.py` |
| 保存形式 | `scripts/import_keying_recordings.py` と同じ WAV+TXT |

**新規は `scripts/collect_l4.py` 1 本**: 原稿を 1 件ずつ送信 → 送信前から録音 → 完了 +0.5 秒で停止 →
`data/l4/<session>/<ts>_<mode>.wav|txt` に保存。**送ったテキストがそのままラベル**。
引数: `--count --mode --wpm-list --session-id --dry-run`。
TXT ヘッダにセッション条件 (wpm/filter_bw/agc/tone/band/antenna) を残す。

**セッション設計 (1 セッション = 1 設定、混ぜない)**: WPM 15/18/22/26/30 × 受信帯域 500/250 Hz ×
AGC fast/slow/off × トーン 500/600/700 Hz × 強/弱 (AGC が動く領域) × 夜/深夜。
1 件 20 秒 + 間 10 秒 → **一晩 8 時間で約 900 件**。3 晩で 2,700 件 (既存 90 件 32 分から桁が変わる)。

**安全 (この順を飛ばさない)**: `--check-lines` (無線機の電源を切って結線・極性確認) →
`--dry-run` → `--text "V V V"` 単発 → 本番。原則**ダミーロード・低出力**。
アンテナに出す場合はバンドプランと免許の範囲内。各セッション先頭に自局コールで識別。
送信 20 秒 / 休止 10 秒のデューティは終段の熱に注意し、セッション間に冷却休止。
**運用者の確認が取れるまで工場は動かさない。**

**限界を先に書いておく**: L4 で得られるのは**オートキーヤーの打鍵**。人間の手打ちの癖は入らない。
L4 の役割は**受信機チェーン (AGC ポンピング・フィルタリンギング・実ノイズ床・伝搬)** であり、
打鍵の癖は L2 タイミング移植と P1-C が担当する。**この分担を混同すると
「L4 を大量に入れたのに手打鍵 TER が下がらない」で誤結論を出す。**

**汎化の保険**: KiwiSDR で自局の送信を受信できれば**別受信機のラベル付きデータ**になる
(テキストは既知)。これを held-out v2 のサブセット `data/heldout/v2/kiwi/` に入れ、
**L4 が自局受信機に特化していないか**を測る。特化が起きればこのサブセットだけ悪化する。

### P1-B: KiwiSDR / WebSDR — **素材専用、ラベルを作らない**

用途は 2 つだけ: ① 背景ベッド (L1 素材) ② 実 QRM 素材 (`add_qrm` の合成局は理想波形)。
**教師データにしない** (推測ラベルは指標を壊す。2026-08-03 の前例)。
現在の実ノイズは train 1 件 1,360 秒のみ = 事実上 1 種類 → **30 分以上・10 ファイル以上・
バンドと時間帯を分散**させて `data/noise/train/kiwi/<band>_<time>/` へ (`from_dir` は再帰読込)。
切り出しは `scripts/clip_onair.py` を流用。

メモリ注意: `RealNoisePool` は全波形をメモリ常駐で **num_workers 倍にコピー**される
(30 分 ≒ 57MB/worker)。**今回は 30 分〜1 時間で止める。** これ以上増やすなら memmap 化が要る。

### P1-C: 自局の追加打鍵 (和文重点)

`scripts/generate_keying_scripts.py --japanese 60 --european 20 --out-dir data/keying_scripts_v2`

**評価用を打鍵前に取り分ける**:

| 用途 | 和文 | 欧文 | 配置 |
|---|---|---|---|
| held-out v2 (凍結) | 20 | 8 | `data/heldout/v2/` |
| train 追加 | 40 | 12 | `data/real/train/` |

held-out 和文 208 トークン → **v1+v2 で 800〜1,000 トークン**。1 トークン ≈ 0.1pt になり、
今回のようなラベル誤差が致命傷にならない。**取り分けは原稿生成直後に確定させ、
打鍵後に「こっちの方が綺麗に打てたから train に回す」を絶対にやらない** (選択バイアス)。
原稿には `デ`/`テ`/`。` の混同を狙った語を足す (符号 1 要素差、既知の弱点)。
取り込み後に必ず `scripts/audit_labels.py`。実信号 val はここから 12 件ほど取る。

---

## P2 — 素材移植合成 (4 日) 【ブランチ `feat/material-transplant-synth`】

### 共通土台 (先に 1 回だけ)
`SynthConfig` に素材フィールド (`noise_bed` / `timing_override` / `ramp_template`) を足し、
`synthesize_from_text` に分岐を入れる。**既存の `snr_is_effective` と同じパターン**。
現在**実素材の混合は `MorseSynthDataset.__iter__` にしか無く**、`ablate_keying_factors.py` や
`make_fixed_eval_set` から使えない。**この土台が無いと検収そのものが回せない。**

### L1 背景ベッド (0.5 日)
ステップ 5 で `add_awgn` の代わりに `RealNoisePool.sample_segment` + `add_real_noise` (実装済み)。
BPF 済み素材は `apply_receiver_filter=False` で二重フィルタを回避 (既存フィールド)。新規コードほぼゼロ。

### L2 打鍵タイミング移植 (2 日、本命)
- **抽出** (新規 `src/synth/timing_pool.py`): `src/infer/wpm.py` の `envelope_on_off` / `element_runs`
  で ON/OFF 秒の時系列を得る。要素→符号の組み立ては `scripts/audit_labels.py` が雛形。
  `estimate_wpm` で dot 長を出し dot 単位に正規化して `(kind, units)` を**系列順のまま**保存。
- **ランプ補正を必ず入れる**: raised-cosine が要素の内側に食い込むため、抽出 ON は真値より
  `rise_fall_ms` 短く OFF は同じだけ長い (`scripts/analyze_keying.py` docstring に定量記述)。
  **補正しないと往復で長短比が歪み、「実測どおりに合成したのに実信号と違う」の再演になる。**
- **適用**: `keying.codes_to_waveform` に `timing_override` を追加。
  **`build_element_sequence` のシグネチャは触らない** (`src/tx/encoder.py` の実送信が共有)。
- **系列は連続して読む。ランダム抽選にしない。** 抽選すると iid ジッタに戻り、
  「徐々に間延びする」「特定の文字だけ癖がある」という相関構造が消える。**ここが L2 の全部。**
- **回帰**: `tests/test_synth_keying.py::test_matches_pre_change_implementation_bitwise` は
  rng の描画順に依存。`timing_override is None` のとき乱数の消費順を一切変えないこと。

### L3 包絡線テンプレート (1 日、優先度低)
`OnOff` に hilbert 包絡線を返すフィールドを追加 (計算済みで捨てている)。
`_apply_raised_cosine_ramp` を差し替え可能に。**L1+L2 で 25% に届くならスキップして P3 へ。**

### 検収 — 難しさ再現チェック (学習を回す前に必ず)
生成データを**実信号と同じ WAV+TXT 形式**で書けば `eval_model.py --keyed-set` がそのまま
検収に使える (`discover_real_samples` が読む)。専用の評価コードを書かない。

| 現行モデルでの TER | 判定 |
|---|---|
| **25〜35%** | 合格。実録音と同じ難しさを持つ → 学習へ |
| **< 10%** | その層は難しさを運んでいない → **学習を回さず棄却** |
| **> 45%** | 実装バグ (ラベルずれ・タイミング破綻) を疑う → 修正して再検収 |

誤り内訳も見る (実録音は 挿入 48% / 置換 31% / 脱落 21%)。

**Ablation**: `ablate_keying_factors.py` に `baseline / l1 / l2 / l1+l2 / l1+l2+l3` の arm。
各 5 分。**整合性チェック: `l1` 単独で 30% が出たら実装を疑う** — 過去 3 本の失敗
(低SNR拡張・実ノイズ混合 FT・手打ち分布フル学習) の説明が付かなくなるからである。

---

## P3 — FT 実験マトリクス (4 晩) 【ブランチ `feat/ft-matrix-v4`】

### 固定条件 (掟)
起点は**常に** `models/full/best.pt` (FT-on-FT 禁止) / `--num-workers 0` /
`--steps` は絶対値 / 実行コマンドを結果文書にそのまま残す。

### まず対照群 3 seed で σ を測る
```
scripts/finetune.py --data-dir data/real/train --resume models/full/best.pt \
  --ckpt-dir models/ft/ctrl_s{42,43,44} --steps 1000 --real-ratio 0.7 --lr 1e-4 \
  --num-workers 0 --seed {42,43,44} --synth-val-noise-dir data/noise/val
```
**σ の 2 倍を超えない改善は「効いた」と言わない。** これを決めずに 20 本回すと、
seed のばらつきを施策の効果と誤読する。

### 夜間の順序 (1 本 ≒ 25 分。評価が支配的。一晩 8h で 19 本)

| 晩 | 内容 | 本数 | 目的 |
|---|---|---|---|
| 1 | 対照 3 seed + データ構成 5 種 (実録音のみ / +L1 / +L1L2 / +L1L2+L4 / L4 のみ) | 8 | σ を測り勝ち構成を 1 つ選ぶ |
| 2 | 勝ち構成 × `--real-ratio` 0.5/0.7/0.85 × `--steps` 1000/2000/4000 | 9 | 混合比とステップ数 |
| 3 | 勝ち設定 × lr 5e-5/1e-4 × `--mode-filter japanese` 有無 × seed 2 | 8 | 微調整と再現性 |
| 4 | 学習曲線 (`--subsample-n` 20/40/60/90/all) + 予備 | 6 | **傾きが立っていれば L4 増産へ、寝ていれば合成側へ投資** |

### 採否 (3 つ全部を満たすこと)
1. held-out **和文 TER** が対照群平均 −2σ より良い
2. held-out **欧文 TER** の悪化が +0.5pt 未満
3. **synth_val TER** の悪化が +3pt 未満 (自動判定)

1 つでも外れたら不採用。台帳に理由を書いて閉じる。**過去 8 本中採用 1 本という打率を
前提に、不採用を素早く確定させるのがこの表の役割。**

**Gate: 和文 24.0% 以下。** 未達なら P4 へ進まず §8 の撤退判断へ。

---

## P4 — フル再学習 (夜間 10h × 2) 【ブランチ `feat/full-retrain-v4` / `feat/vocab-brackets`】

### Gate (全部満たすまで 10 時間を賭けない)
1. P2 の難しさ再現チェック合格
2. P3 で和文 24.0% 以下 = **移植合成が実際に効いた証拠がある**
3. 物差し確定 (v1 凍結 + v2 で和文 800+ トークン)
4. `train.py` に**実データを混ぜる口がある** — 現状 `MorseSynthDataset` だけで**実データ経路が
   無い** (実ノイズ混合はある)。`src/finetune/pipeline.py` の `MixedRealSynthDataset` を
   `--real-dir` / `--real-ratio` として配線する (FT 側で実績のあるクラスを再利用)
5. 500 step スモークで `best.pt` が 2 回以上更新される (飽和指標で選んでいない)

### 2 本立てにする (帰属不能を避ける)

| run | 変更点 | 所要 |
|---|---|---|
| **P4-A** | 現語彙 (73) のまま、学習データを「移植合成 + L4 + 実録音」に置換。混合比は P3 の勝ち値 | 10h |
| **P4-B** | A の勝ちに **語彙 73→75 (「」)** を追加 | 10h |

**回し放題でも 2 本立てを崩さない。** A で 19% を割れなければ B に進まず原因究明に戻る。

### 語彙追加 (73 → 75)
- 実測: 現在 `VOCAB_SIZE = 73` (docs の「72」は誤り)。追加で 75。
  **ID 1〜63 は不変、末尾 9 個がずれる** (ホレ 64→65、`。` 68→69、`?` 69→71、
  [SK] 70→72、WORDBREAK 72→74)。高影響トークンが全部動くので既存 ckpt は実質使えない。
- `train.py --resume` の語彙拡張は `new_w[:copy_n] = old_w[:copy_n]` の**先頭からの単純コピー**で
  末尾追加しか想定していない。中間挿入では誤った行に乗る。
  → **warm start しない。スクラッチで 102k step を回す。** 索引写像の実装と検証より
  夜間 10 時間の方が安い (GPU 回し放題が効く場面)。
- 下流の再生成を同一 PR で: `export_tokens.py` (`web/src/generated/tokens.ts`) /
  `export_onnx.py` / `export_golden.py`、`src/train/model.py:24` の既定値 72、
  `tests/test_tokens.py::TestDanrakuAndBrackets` (「語彙に無い」歯止め →
  **削除ではなく「語彙にある」への書き換え**)、`tests/test_tx_only_chars.py` (VOCAB_SIZE と
  全 ID のハッシュ)、`tests/test_export_onnx.py` (shape)、docs の「72」表記。
- **LM 融合との順序**: 語彙は LM のトークン集合を変える。**P4-B (語彙確定) を
  LM 融合の実装着手より前に置く。**

### メル分解能 (25ms/10ms → 15ms/5ms) は**保留**
18.2 WPM で短点に 6.6 フレームあり分解能は足りている。理屈上は 40 WPM 超で効くが
**その速度の実録音を 1 件も持っていない**。入手して誤りがそこに集中することを確認してから。

### 容量増 (hidden 256→384) は L4 が量産できてから
90 件のままで容量を増やすのは過学習にしかならない。L4 が数時間規模になった後、対照群つきで 1 本。

### 実験台帳 `docs/experiments/ledger.md` (追記のみ・削除禁止)
run_id / **仮説 (1 行)** / 出発点 ckpt + sha256 / **コマンド全文** / データ件数 + seed /
held-out 欧文・和文 TER + synth_val / 採否 / 理由。
**仮説の列を書けない run は回さない。** これが「何が効いたか分からなくなる」への唯一の歯止め。
`save_checkpoint(extra=…)` と `meta.json` (P0-9) が同じ内容を機械可読で残すので転記は自動化できる。

---

## リスクと撤退基準

過去 3 回の失敗 (B1 低SNR拡張 +19.7pt / 実ノイズ混合 FT +2.8pt / 手打ち分布フル学習
25.95→52.35%) の共通点は「**合成分布を広げただけで実信号の難所を入れていない**」+
「**難しさ再現チェックを事前にやっていない**」。全施策に安いテストを前置する。

| リスク | 安いテスト | 所要 | 撤退基準 |
|---|---|---|---|
| L1 がまた効かない | ablation `l1` arm | 5 分 | < 10% なら L1 単独を棄却し L2 に集中 |
| L2 が難しさを運ばない | ablation `l2` arm | 5 分 | < 10% なら棄却、予算を L4 増産へ |
| L2 の実装バグ (ランプ補正忘れ) | `analyze_keying.py` で合成と実録音の長短比を比較 | 15 分 | > 45% ならバグ確定、学習を回さない |
| 移植合成が実信号を悪化させる (過去の再演) | FT 1,000 step で held-out 和文が対照群 +2σ 悪化 | 25 分 | 即不採用。**長い step で挽回する期待を持たない** |
| L4 が自局受信機に特化 | held-out v2 の KiwiSDR サブセットだけ悪化するか | FT に同梱 | +2pt 悪化なら混合比を下げ受信機設定を多様化 |
| L4 に手打ちの癖が入らない | L4 のみ FT (arm e) で手打鍵 held-out が改善するか | 25 分 | 改善しないのは**想定内**。役割を受信機チェーンに限定し、打鍵の癖は L2 と P1-C に帰属 |
| ラベル誤差が判定を汚す | 打鍵直後に `audit_labels.py` | 5 分 | 警告が出たら**評価から外す前にラベルを直す** (2026-08-03 の再演を防ぐ) |
| データを増やしても頭打ち | P3 晩 4 の学習曲線 | 一晩 | 傾きが寝たら L4 増産を止め、合成品質 / 容量へ予算移動 |
| 和文 15% が原理的に届かない | — | — | **19% で頭打ちしたら学習側の投資を止める。** 残りは (a) LM 融合 (b) held-out ラベルの再検証 (c) 目標値の再定義 でしか埋まらない |

**LM 融合との二重計上に注意 (範囲外だが重要)**: 誤りの内訳は挿入 48% (うち **WORD_BREAK 30%**)。
WORD_BREAK 挿入は LM / 適応閾値側で取る分が大きい。**学習側で減らすと LM の効き代がその分減る。**
→ **学習側の評価は常に LM オフ・WORD_BREAK 既定閾値で行う。** 合算するときは同一条件で測り直す。

## 実験の原則 (踏んだ穴から)
1. **対照群を置く** — 対照群の欠落で「分布のせい」と誤結論した (実験 F)
2. **held-out に正解ラベルで当てる** — 代理指標 (一致率) の掃引は結論が逆だった
3. **合成で先に否定する** — 10 時間を賭ける前に数分で潰す
4. **原稿はラベルではない** — 取り込み後に必ず `audit_labels.py`
5. **実行コマンドを残す** — 現行モデルの学習コマンドが正確に残っていない

## 検証方法
- `python scripts/audit_labels.py --dirs data/keying_scripts data/keyed_extra` → 警告 0 件
- `python scripts/eval_model.py --ckpt <新> --keyed-set v1=data/keying_scripts --keyed-set v2=data/heldout/v2 --noise-dir data/noise/val --out models/eval/<run_id>.json --baseline models/eval/baseline_v4.json`
  → **モード別 TER が印字され**、和文・欧文・synth_val の 3 条件で採否を判定
- `pytest` 全件 (テストファイル 91 本)。合成器を触ったら
  `tests/test_synth_keying.py::test_matches_pre_change_implementation_bitwise` を特に確認
- 語彙・モデルを変えたら `web/` のゴールデンテストまで通す

## 作業手順
`main` 直コミット禁止。Phase ごとにブランチと PR を分ける (§各節に記載)。
コミットは `fix:` (物差し・バグ) / `feat:` (収集・合成・学習) / `docs:` (台帳・結果)。
**`docs/accuracy_improvement_plan_v3.md` と `docs/training_conditions_and_results.md` は
未コミット**なので、最初の `docs/` ブランチで一緒に入れる。

## 直近 3 週間
```
週1: P0 (物差し) → ベースライン打ち直し → P0.5 (先頭幻覚) → P1-C 原稿生成と取り分け
     並行: L4 の配線確認 (--check-lines → --dry-run → 単発送信)
週2: 夜 = L4 収集 ×3 晩 / 昼 = P1-C 打鍵 + P1-B 素材抽出
     並行: P2 の共通土台 + L1 → ablation
週3: P2 L2 → 難しさ再現チェック → P3 FT マトリクス ×4 晩 → Gate 判定
週4: Gate 通過なら P4-A (10h) → P4-B (10h)
```
