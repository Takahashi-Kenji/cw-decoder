# 実ノイズ混合ファインチューニング 設計書

作成日: 2026-07-21
対象: `cw-decorder` / Phase B レバー (実ノイズ混合)
前提: B0 評価基盤・B1 実効SNR拡張は main にマージ済み。B1 の結果 (synth 0dB
−25改善だが keyed +19.7悪化) を受け、AWGN≠実受信ノイズのドメインギャップを埋める。

## 1. 目的

B1 (合成AWGNの実効SNR範囲拡張) は synth_val の 0dB を大きく改善したが、実打鍵
録音 (keyed_val) では TER が 23.4%→43.1% と悪化した。主因候補は **AWGNと実受信
ノイズのドメインギャップ**。本レバーは実録音ノイズ (`RealNoisePool`) を合成に混ぜて
学習し、実録音に近い音でモデルを適応させる。

**良いベースモデル `models/full/best.pt` を短時間 FT** する方式を採る (ゼロからの
28時間再学習ではなく、~1時間の継続学習)。B1 のゼロ学習で起きたクリーン域劣化も
避けやすい。モデル構造・特徴量・decode・推論は変更しない。

## 2. アプローチ

- `train.py` に `--noise-dir` と混合パラメータを追加し、`MorseSynthDataset` の
  既存の実ノイズ対応 (`noise_pool`/`noise_prob`/`noise_snr_range`) に配線する
  (B1 で `--eff-snr` を足したのと同じ CLI 配線パターン)。
- FT: `train.py --resume models/full/best.pt --noise-dir data/noise ...`。
- 実験の変数を1つに絞るため、B1 の実効SNRモードとは併用しない (実ノイズは元々
  帯域内=実効なので不要)。

**落とし穴**: `RealNoisePool.from_dir` はディレクトリ内の全 wav をノイズとして
読む。`data/keying_scripts/` には打鍵録音20件 (信号) も入るため、そこを
`--noise-dir` に指すと録音をノイズ扱いしてしまう。したがって **`data/noise/` を
新規作成し `noise_sample.wav` だけを置く**。

## 3. コンポーネント

### 3.1 `data/noise/` (新規ディレクトリ)

- `noise_sample.wav` (現 `data/keying_scripts/noise_sample.wav`) をコピー配置。
- `.gitignore` に `data/noise/` を追加 (大容量wav)。

### 3.2 `scripts/train.py` (変更)

- CLI 追加:
  - `--noise-dir` (Path, 既定 None): 指定時、実ノイズ混合を有効化。
  - `--noise-prob` (float, 既定 0.8): 合成サンプルへの実ノイズ適用率
    (finetune.py と同名・同義)。
  - `--noise-snr-min` / `--noise-snr-max` (float, 既定 -5.0 / 15.0):
    実ノイズ混合時の SNR 範囲 (finetune.py と同名・同義)。
- 検証付き純関数 `resolve_noise_pool(noise_dir) -> RealNoisePool | None`:
  None なら None を返し現行動作。指定時 `RealNoisePool.from_dir` を返す
  (ディレクトリ不在/wav無しは `RealNoisePool.from_dir` が ValueError)。
- `main()`: `noise_pool` と noise パラメータを `MorseSynthDataset(...)` に渡す。
  `--eff-snr` (B1) と `--noise-dir` の同時指定時の扱い: 実ノイズが主目的なので
  両立可能だが、本実験では併用しない (スコープ外)。実装上は両方渡せる形で良い。

### 3.3 `MorseSynthDataset` (変更なし)

既に `noise_pool`/`noise_prob`/`noise_snr_range`/`noise_clean_snr_db` に対応済み。
`__iter__` は確率 `noise_prob` で「クリーン合成 (snr=40) → 実ノイズを
`noise_snr_range` で加算」する。変更不要。

## 4. FT のハイパーパラメータ (提案・CLI可変)

| パラメータ | 提案値 | 理由 |
|---|---|---|
| `--resume` | `models/full/best.pt` | 良いベースから継続 |
| `--noise-dir` | `data/noise` | noise_sample.wav のみ |
| `--noise-prob` | 0.5 | 半分に実ノイズ、半分は通常合成。over-noising 回避 |
| `--noise-snr-min/max` | 0 / 20 | 実効SNR。keyed は比較的クリーンなのでその域中心 |
| `--steps` | 3000 | 短時間FT (num_workers=0 で ~1時間) |
| `--lr` | 5e-5 | ベースを壊さない低lr |
| `--num-workers` | 0 | 孤児プロセス回避 (B1の教訓) |

FT が安いので、keyed_val が下がらなければ noise_prob / snr_range をスイープ
(各 ~1時間)。

## 5. 実験手順と成功条件

```
1. data/noise/ を作り noise_sample.wav を配置。
2. FT: python scripts/train.py --resume models/full/best.pt \
     --noise-dir data/noise --num-workers 0 --steps 3000 --lr 5e-5 \
     --noise-prob 0.5 --noise-snr-min 0 --noise-snr-max 20 \
     --ckpt-dir models/ft_noise --eval-interval 500 --log-interval 100
3. 測定: python scripts/eval_model.py --ckpt models/ft_noise/best.pt \
     --noise-dir data/noise --keyed-dir data/keying_scripts \
     --out models/eval/ft_noise.json --baseline models/eval/b1_baseline.json
```

**成功条件 (keyed_val が真の指標)**:
- keyed_val TER が baseline 23.4% を下回る (本命)。少なくとも B1 の 43.1% より
  明確に良い。
- keyed の WORD_BREAK recall (baseline 80%) を維持/改善。
- synth_val は **noise_sample.wav を学習と評価の両方に使うため漏洩する**。
  良く出ても割り引く。0dB が極端に悪化していないかは見る。

## 6. テスト方針

- `resolve_noise_pool`: None→None、有効ディレクトリ→RealNoisePool、wav無し
  ディレクトリ→ValueError (from_dir の挙動) をテスト (小さな一時 wav を作る)。
- `train.py` CLI: `--noise-dir` 指定で MorseSynthDataset に noise_pool が渡ること
  (main の配線を薄い純関数に切り出してテスト、または import 確認)。
- noise パラメータの受け渡し (prob/snr) が MorseSynthDataset に反映されること。
- 既存 594 テストを壊さない。`MorseSynthDataset` は変更しないので既存の合成
  テストは不変。
- CPU 極小スモーク: `train.py --resume <既存ckpt> --noise-dir <tmp> --steps 2
  --num-workers 0 --device cpu` が完走すること。

## 7. エラー処理

- `--noise-dir` が存在しない / wav 無し → `RealNoisePool.from_dir` の ValueError
  を明確なメッセージで表示して exit。
- `--noise-snr-min >= --noise-snr-max` → エラー。
- `--noise-prob` が [0,1] 外 → エラー。

## 8. スコープ外

- ノイズ録音の追加収集 (1本で開始)。
- B1 実効SNRモードとの併用。
- ゼロからの再学習。
- パラメータの本格スイープ (初回FTの結果を見てから)。
- 実際のFTラン実行 (コード整備まで。起動・監視は可能)。

## 9. リスクと成果物

**リスク**: ノイズ録音が1本のみ (`noise_sample.wav`) なので、モデルがその固有の
スペクトル特性に過適合する恐れ。keyed_val (別物の実録音) が悪化しないかで過適合を
検出する。

**成果物**:
- `train.py` が実ノイズ混合 FT に対応 (`--noise-dir`)。
- 良いベースを実録音ノイズで短時間 FT し、keyed_val で AWGN→実ノイズのドメイン
  ギャップが埋まるかを ~1時間で検証できる。
- 効けば運用モデル更新、効かなければ次のレバー (B3 WORD_BREAK 等) の判断材料。
