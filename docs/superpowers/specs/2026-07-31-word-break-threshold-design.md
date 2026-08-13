# WORD_BREAK 抑制パラメータの掃引 設計書

作成日: 2026-07-31

## 1. 背景と問題

`models/full/best.pt` の keyed_val（実打鍵20件）TER は **23.35%** で、
誤りの内訳は 置換 S=34 / 脱落 D=23 / 挿入 I=53（合計110、参照471トークン）。
**挿入53件のうち33件が WORD_BREAK の過剰出力**であり、全誤りの約3割を1トークン種が
占めている。

この事実は 2026-07-18 の Phase A 分析で既に判明していたが、その後に実施した学習2本
（B1 実効SNR拡張・実ノイズ混合FT）はいずれもノイズ耐性のレバーであり、どちらも
keyed_val を悪化させて不採用となった。語間スペースはタイミングの問題であって
ノイズ耐性の問題ではないため、原理的に噛み合っていなかった。
経緯は `docs/retrospective-accuracy-improvement.md` を参照。

本設計は**再学習を伴わず**、デコード側のパラメータだけでこの過剰出力を抑える。

### 前提の訂正（重要）

`docs/retrospective-accuracy-improvement.md` および `docs/word_spacing.md` は
「`detect_word_breaks_from_audio` の `word_gap_dots`（現在6）を振れば測定できる」と
書いているが、**これは誤り**である。

- WORD_BREAK は既に語彙トークンとして存在する（`src/tokens/morse_tokens.py:215`,
  `WORD_BREAK_TOKEN_ID = 72`, `VOCAB_SIZE = 73`）。`word_spacing.md` が「将来作業・案A」
  として挙げている語彙追加は実装済みで、同文書はその前の状態を記述したまま陳腐化している。
- keyed_val の評価経路（`src/eval/harness.py:22` `decode_wave`）は音声エンベロープを
  一切使わず、CTC greedy のトークン列をそのまま TER に掛ける。
  `detect_word_breaks_from_audio` はこの経路を通らないため、`word_gap_dots` を振っても
  keyed_val TER は 1 ビットも動かない。

したがって実際に効くレバーは **CTC デコード側の WORD_BREAK 抑制**である。

## 2. ゴールと非ゴール

**ゴール**: β・τ を掃引して keyed_val TER を測り、採否基準（§6）を満たす設定が
見つかった場合のみ、デコード経路に恒久的なパラメータとして実装する。

**非ゴール**:
- 再学習・ファインチューニング（本設計は学習を一切行わない）
- 案C（フレームラン長による抑制）の実装。診断（§5 Step 0）で有望度だけ測り、
  実装は別設計とする
- `detect_word_breaks_from_audio` 系（音声エンベロープ経路）の改修
- アプリ（`run_app.py`）のリアルタイム経路への反映

## 3. 手法

### 案B: ロジットバイアス β（主軸）

argmax の**前**に `log_probs[..., WORD_BREAK] += β`（β<0 で抑制）し、再正規化する。
抑制されたフレームでは2位のトークン（多くは blank）が採用される。

案A の効果を含んだうえで、WORD_BREAK に押しのけられていた正解トークンが浮上しうる
ため、挿入だけでなく置換・脱落にも効く可能性がある。CTC の事前確率補正として標準的で、
パラメータは1個。

### 案A: 確信度閾値 τ（比較対象）

argmax の**後**、デコード済みトークン列から `conf < τ` の WORD_BREAK だけを除去する。
`ctc_greedy_decode` は既に `confidences` を返している（`src/train/decode.py:24`）ため
追加情報を必要としない。argmax を変えないので押しのけ救済はできない。

両者は log_probs をキャッシュすれば同じ掃引ループで同時に測れるため、
「確信度で切る」と「事前確率で押し下げる」のどちらが効くかを1回の実験で比較する。

### 退化解が成立しないことの確認

keyed_val の参照471トークンのうち **81個（17.2%）が WORD_BREAK** である。
WORD_BREAK を全て抑制すると挿入33件が消える代わりに脱落が最大81件増えるため、
「全部消せば TER が下がる」という退化解は成立しない。
**TER をそのまま目的関数にして安全**であり、別途 F1 等の指標を用意する必要はない。

## 4. 構成要素

### 4.1 新規: `src/infer/word_break_policy.py`

純関数のみ。依存は torch と `src.tokens.morse_tokens` のみ。

```python
@dataclass(frozen=True)
class WordBreakPolicy:
    logit_bias: float = 0.0       # β: WORD_BREAK ロジットへの加算 (β<0 で抑制)
    conf_threshold: float = 0.0   # τ: この確信度未満の WORD_BREAK を除去

    @property
    def is_identity(self) -> bool: ...   # β=0 かつ τ=0
```

- `apply_logit_bias(log_probs, bias) -> Tensor`
  WORD_BREAK 列に β を加え、`log_softmax` で再正規化して返す。
- `filter_word_breaks(result, threshold) -> CTCDecodeResult`
  `conf < τ` の WORD_BREAK **だけ**を除去する。他トークンには一切触らない。

トークン ID は `morse_tokens.WORD_BREAK_TOKEN_ID` を import して使い、
`72` をハードコードしない。

### 4.2 既存への変更: `src/eval/harness.py`（採否基準を満たした場合のみ）

`decode_wave` に `policy: WordBreakPolicy | None = None` を追加する。
`None` および `is_identity` のときは現行と完全に同じ経路を通す。
`evaluate_real_dataset` / `evaluate_synth_noise` にも同じ引数を通す。

これが本設計における既存コードへの唯一の変更点である。
掃引スクリプトはキャッシュした log_probs に対して `ctc_greedy_decode` を直接呼ぶため
`decode_wave` を必要としない。したがってこの変更は**採否が決まってから**行う。

### 4.3 新規: `scripts/sweep_word_break.py`

掃引専用 CLI。推論を1回だけ行い、以降は CPU 上の純後処理でグリッドを回す。
Step 0 の診断も同スクリプトが担う。予測 WORD_BREAK を正解／偽陽性に分類するには
既存の `src/train/metrics.py:191` `align_sequences(pred, ref)` を再利用する。
返る `EditOp` は `kind` と `pred_index` を持つので、`kind == "equal"` の位置に来た
予測 WORD_BREAK を正解、`insertion` / `substitution` の位置に来たものを偽陽性と
判定できる。

## 5. データフロー

```
wav → MelExtractor → CWModel → log_softmax ─┬─→ [log_probs キャッシュ] (T, 73)
                                            └─→ [input_lengths キャッシュ]
                    ┌───────────────────────────┘
                    ▼
        apply_logit_bias(β)        ← 案B: argmax の前
                    ▼
        ctc_greedy_decode          → token_ids + confidences
                    ▼
        filter_word_breaks(τ)      ← 案A: argmax の後
                    ▼
        TokenConverter → DetailedEvalReport → TER
```

推論はグリッド全体で1サンプルにつき1回だけ実行する。

**キャッシュ量**: `hop_length=80` @8kHz なので15秒の録音で T≈1500 フレーム、V=73。
1サンプル約 440 KB、keyed_val 20件で **約9 MB**、synth_val 200件でも約90 MB。
どちらもメモリに載る。GPU メモリを占有しないよう、キャッシュは CPU 上に置く。

### Step 0: 診断（掃引の前に1回）

baseline（β=0, τ=0）で予測された WORD_BREAK を参照とアラインし、
**正解WB と偽陽性WB** に分けて確信度分布とフレームラン長を比較する。
掃引本体より安く、先に結果が出る。

- 確信度分布が**分離していれば** τ に見込みあり
- **完全に重なっていれば** τ は原理的に効かないと即断でき、β の効果も限定的と予測できる。
  その場合は案C（ラン長）に切り替える判断材料になる
- ラン長分布を同時に出すことで、案Cの有望度を追加コストほぼゼロで測れる

### Step 1: 2次元グリッド掃引

- **β**: `0, -0.5, -1, -1.5, -2, -3, -4, -6, -8, -12`（log 確率への加算。単位は nat）
- **τ**: `0.0, 0.1, ..., 0.9, 0.95, 0.99`

全120点 × 20サンプル = 2400 デコード。`ctc_greedy_decode` は T≈1500 の Python ループ
なので数分の見込み。実測が5分を超えた場合は β を `0, -1, -2, -4, -8` の5点に、
τ を 0.1 刻みの11点に粗くして再実行する。

## 6. 採否判定

**理論上限**: 偽陽性 WB 33件を全て消し、正解 WB を1つも失わない理想ケースで
TER = (110−33)/471 = **16.35%（−7.0pt）**。これがこのレバーの天井である。

採用は以下を**すべて**満たす場合のみ。

| 条件 | 内容 |
|---|---|
| 改善幅 | keyed_val TER が baseline 23.35% から **2pt 以上**改善（1トークン=0.21pt なので約10トークン分。これ未満は誤差として不採用） |
| プラトー | 最適点から β 方向・τ 方向に1点ずつ隣接する格子点（格子内に存在するもの、最大4点）の**すべて**で baseline より改善している（単一スパイクは過適合とみなし不採用） |
| synth_val | 同じ (β,τ) で synth_val overall が baseline 48.06% から **+1pt を超えて悪化しない** |

補助情報として欧文/和文別の内訳と S/D/I の変化も出力する。挿入が減った分だけ脱落が
増えていないか（＝正解WBを巻き添えにしていないか）を直接確認するため。

**過適合対策**: keyed_val 20件には hold-out がない。全20件で閾値を選ぶが、
上記の「プラトー」条件と「synth_val 非悪化」条件の2つを健全性チェックとして課す。
パラメータが2個（実質1個ずつ独立）と少ないため過適合の余地は小さいと判断する。

**基準を満たさなかった場合**: 実装はせず、掃引結果と Step 0 の診断だけを文書に残す。
その場合、Step 0 のラン長分布が案Cの有望度を示していれば次の設計対象とする。

## 7. エラー処理

| 状況 | 挙動 |
|---|---|
| τ が [0,1] の外 | `ValueError`（`src/tokens/converter.py:79` の既存流儀に合わせる） |
| β が NaN / ±inf | `ValueError` |
| β > 0 | **許容する**。WORD_BREAK を増やす方向だが、掃引の対称性確認に使えるため値域制限しない |
| `log_probs` が3次元でない | `ValueError`（`src/train/decode.py:43` の既存流儀） |
| ckpt の語彙サイズが 73 でない | `ValueError` で明示的に落とす |

最後の1点が重要である。WORD_BREAK 追加前の古いチェックポイントを渡されると
`id=72` は別の意味を持つか存在しないため、**黙って無関係なトークンにバイアスをかける**
という静かな失敗になる。語彙サイズを検証して明示的に落とす。

## 8. テスト

TDD で実装前に書く。

**`apply_logit_bias`**
- β=0 で入力と数値的に一致する（再正規化しても不変）
- β<0 で WORD_BREAK の確率が下がり、**それ以外のトークン同士の確率比が保存される**
  （バイアスが WORD_BREAK 以外を歪めていないことの確認）
- 出力が正規化されている（`exp` の和が1）

**`filter_word_breaks`**
- τ=0 で入力そのまま
- `conf < τ` の WORD_BREAK **だけ**除去され、同じく低確信度の他トークンは残る
- `token_ids` と `confidences` の長さが一致し続ける

**回帰（最重要・`harness.py` を変更する場合のみ）**
- `decode_wave(policy=None)` と `decode_wave(policy=WordBreakPolicy())` が
  同一のトークン列を返す（既存挙動を1ビットも変えない保証）

**スモーク**
- 掃引スクリプトが小さな合成ケースでグリッドを回し JSON を書ける

既存602テストが全て通ることも合格条件に含める。
`pytest` 全実行は全パス後に `0xC0000409` で落ちサマリ行が出ない既知の環境問題があるため、
合否は `FAILED` / `ERROR` の有無で判定する。

## 9. 成果物

掃引の実行に必要なので**採否に関わらず作る**もの:

- `src/infer/word_break_policy.py`（新規）
- `tests/test_word_break_policy.py`（新規）
- `scripts/sweep_word_break.py`（新規）
- `models/eval/word_break_sweep.json`（掃引結果）
- 掃引結果と Step 0 診断のまとめ（採否の判断根拠として文書化）

**採否基準（§6）を満たした場合のみ**作るもの:

- `src/eval/harness.py` の `policy` 引数追加と、その回帰テスト

基準を満たさなかった場合は `harness.py` に手を触れず、掃引スクリプトと結果・診断だけを
残す。その場合、Step 0 のラン長分布が案Cの有望度を示していれば次の設計対象とする。
