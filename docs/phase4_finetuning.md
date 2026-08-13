# Phase 4: 実信号ファインチューニング 運用ノート

要件 §3.3.4 / §6 Phase 4: 実信号 (WebSDR・自局録音) を用いた追加学習による運用適応。

合成データで学習したベースモデル (Phase 2) を、現実の HF CW 環境固有の特性 (受信機の周波数特性、伝搬条件、操作者の癖等) に適応させる。

---

## ワークフロー

> 実交信を大量に集めなくても FT を始められる低コスト収集戦略
> (実ノイズ録音・自己打鍵録音・擬似ラベル訂正) は
> [phase4_data_collection.md](phase4_data_collection.md) を参照。

### 1. 実信号の収集

Phase 3 のデコーダアプリを使用:

```powershell
python scripts/run_app.py --ckpt models/full/best.pt
```

1. アプリでマイク / ラインを開いて受信音を流す
2. 録音ボタン (●) を ON にして区切りまで録音
3. 停止 (■) で WAV + TXT が `data/real/` に保存される

保存形式:
```
data/real/<timestamp>_<mode>.wav   # 8 kHz / 16-bit PCM
data/real/<timestamp>_<mode>.txt   # メタ + デコード結果 (初期値)
```

`.txt` の構造:
```
mode: european
sample_rate: 8000
duration_s: 4.523
timestamp: 20260612_184500
---
<モデルがデコードした結果が初期保存される>
```

### 2. 正解テキストの確認・修正

`.txt` の区切り行 `---` の後ろを **人手で確認し、正解に修正**する。
これが FT の教師信号となる。

ヒント:
- ファイル名規則 `<timestamp>_<mode>` は維持
- ノイズで判別不能な部分は `?` で残す (FT 側で扱われる)
- 完全に意味不明なサンプルは `.txt` 本文を空にしておけば、`discover_real_samples` が除外する

### 3. データの目安

| 項目 | 推奨 |
|---|---|
| サンプル数 | 50〜200 件 (少なくとも 20 件以上) |
| 1 サンプル長 | 3〜30 秒 (CTC が扱える時間範囲) |
| モード分布 | 学習させたい方向に偏らせて OK (例: 和文専用なら和文のみ) |
| SNR レンジ | できれば良好〜厳しい条件まで散らす |

### 4. ファインチューニング実行

```powershell
# 実信号のみで FT
python scripts/finetune.py --data-dir data/real --resume models/full/best.pt `
    --ckpt-dir models/ft --steps 2000 --lr 1e-4

# 実信号 + 合成混合 (カタストロフィックフォゲッティング抑制)
python scripts/finetune.py --data-dir data/real --resume models/full/best.pt `
    --ckpt-dir models/ft --steps 2000 --lr 1e-4 --mix-synth --real-ratio 0.7

# 和文だけで FT
python scripts/finetune.py --data-dir data/real --resume models/full/best.pt `
    --ckpt-dir models/ft --steps 1500 --lr 5e-5 --mode-filter japanese
```

主要オプション:

| オプション | 既定 | 説明 |
|---|---|---|
| `--data-dir` | `data/real` | WAV+TXT ペアを置いたディレクトリ |
| `--resume` | (必須) | 出発点チェックポイント (通常 `models/full/best.pt`) |
| `--ckpt-dir` | `models/ft` | FT 結果保存先 |
| `--steps` | 2000 | FT ステップ数 |
| `--batch-size` | 4 | FT は小バッチで OK |
| `--lr` | 1e-4 | フル学習の 1/3 程度を推奨 |
| `--eval-interval` | 200 | 評価頻度 |
| `--eval-ratio` | 0.2 | hold-out 検証セットの割合 |
| `--mode-filter` | (なし) | european / japanese で絞る |
| `--mix-synth` | OFF | 合成データを混合する |
| `--real-ratio` | 0.7 | 混合時の実データ比率 |
| `--eval-details-out` | `<ckpt-dir>/ft_eval_details.json` | 詳細評価 JSON の出力先 |
| `--confusion-out` | `<ckpt-dir>/ft_confusion.json` | confusion matrix JSON の出力先 |
| `--eval-top-n` | 10 | 評価ログに出す「誤りの多い token」の件数 |

### 5. 評価結果の読み方

評価のたびに `--ckpt-dir` 配下へ次が保存されます。

| ファイル | 内容 |
|---|---|
| `ft_eval.csv` | step ごとの TER / CER / サンプル数 (従来通り) |
| `ft_eval_details.json` | **最新**評価の詳細 (全体・サンプル別・token 別) |
| `ft_confusion.json` | **最新**評価の confusion matrix |
| `ft_eval_details_step0.json` | FT 前 (step 0) の詳細。改善前後比較の基準 |
| `ft_confusion_step0.json` | FT 前の confusion matrix |

`ft_eval_details.json` の構造:

- `overall`: 全体 TER / CER、ref token 数・文字数、誤り数
- `totals`: 置換 (`substitutions`) / 脱落 (`deletions`) / 挿入 (`insertions`) の内訳
- `token_errors[]`: token 別の `correct` / `substituted` / `deleted` / `inserted` と recall・precision
- `samples[]`: サンプル別の TER / CER、予測・正解の token 列と符号列とテキスト

`ft_confusion.json` の `entries[]` は 1 行が「正解 token → 予測 token」の組で、
`kind` が `equal` / `substitution` / `deletion` / `insertion`。脱落は `pred` が
`<DEL>`、挿入は `ref` が `<INS>` になります。

用語はすべて **正解 (ref) 基準**です。

- substitution: 正解の token を別の token として出力した
- deletion: 正解の token を出力しなかった (脱落)
- insertion: 正解に無い token を余分に出力した

FT 前後の比較は `ft_eval_details_step0.json` と `ft_eval_details.json` の
`totals` / `token_errors` を突き合わせます。「TER が何 % 動いたか」だけでなく
「どの符号の脱落が減ったか」まで確認できます。

学習ログにも上位の誤り token が出ます。

```text
[eval     0] Overall  n=    4  TER=  2.56%  CER=  2.56%
[eval     0] Edits    S=    0  D=    1  I=    0  (ref tokens=39)
[eval     0] Top 1 error tokens:
[eval     0]   ・・--・・(?)        ref=   1  S=  0 D=  1 I=  0  recall=  0.0%
```

### 6. FT モデルでアプリ起動

```powershell
python scripts/run_app.py --ckpt models/ft/best.pt
```

実信号評価セットでの TER と、運用時の体感を比較しながら反復。

---

## 設計ポイント

### カタストロフィックフォゲッティング対策

実信号サンプルが少ない (数十件) と、FT 中に既存の合成データ性能が劣化する場合がある。
`--mix-synth --real-ratio 0.7` で合成 30% を混ぜると、Phase 2 で獲得した能力を保持しながら実信号適応できる。

### 学習率

- FT では Phase 2 (3e-4) より低い `1e-4` 〜 `5e-5` を推奨
- 不安定なら `5e-5` まで下げる
- データが多い (>200 件) なら `2e-4` まで上げてもよい

### 検証セットの偏り

`discover_real_samples` → `split_train_validation` のランダム分割は再現可能だが、
サンプル数が少ないと評価の分散が大きい。実用上は **アプリで実機試聴** が最終判断。

### Phase 1〜3 との接続

- 共通: トークン語彙、`MelExtractor`、`CWModel`、CTC、`TokenConverter`
- データのみ実信号に差し替える形

### 既知の制限

- `RealSignalDataset` は全波形をメモリ常駐 (合計 < 数 GB が前提)。大量データは別途ストリーミング Dataset が必要
- 学習中に `data/real/` の内容が変わると挙動が変わる。再現性が必要な実験では事前に固定する
- 録音 UI からの初期保存テキストはモデルの推定結果なので、必ず人手で確認すること

---

## 既知のバグと対処 (補充学習で発見)

### `last.pt` の `best_metric` が古い値で保存される問題

`scripts/train.py` の eval 部分で、当該 eval の TER が `best_ter` 変数に反映される前に
`last.pt` が保存される実装になっていました。これにより:

- 補充学習を `--resume models/full/last.pt` で実行
- `[resume]` 行で表示される `best_ter` が古い値 (例: 0.9477) になる
- 結果として `best.pt` の更新条件 (`ter < best_ter`) が誤動作する可能性

**対処** (本 PR で修正):
1. eval 結果で `best_ter` を先に更新
2. 更新後の `best_ter` で `last.pt` に書き込む
3. `best.pt` 更新条件もそれに合わせる

`scripts/finetune.py` は最初から正しい順序で実装してあります。
