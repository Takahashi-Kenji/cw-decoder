# ブラウザ版 CW デコーダ 設計書

作成日: 2026-08-05

## 1. 目的と背景

学習済みの CW デコーダモデル (CNN + BiLSTM + CTC) を、ブラウザ上の JavaScript で
リアルタイム動作させる。PySide デスクトップアプリを立ち上げなくても、ブラウザを
開くだけで受信中の CW をデコードできる状態を目指す。

きっかけは同種の公開アプリの存在。`onnxruntime-web` (WASM) + AudioWorklet という
構成に実績があることが分かったため、本設計もこの標準的な構成に沿う。

### スコープ

初版に**含む**もの:

- マイク / ライン入力からのリアルタイム受信・連続デコード
- 欧文表示 / 欧文＋和文の併記表示 (切替)
- 確定 (committed) / 暫定 (provisional) の 2 色表示
- 信号モニタ (入力レベルメータ + スペクトル表示)
- WAV ファイル入力 (§7 のゴールデンテスト用。開発者向けの入口として実装)

初版に**含まない**もの:

- LAN 経由の音声入力 (`src/infer/net_audio.py`)。ブラウザは生の TCP を扱えず、
  WebSocket ブリッジという別部品が必要になるため見送る
- ホレ / ラタによる欧文・和文の自動モード切替。デスクトップ版でも実信号で
  取れておらず未解決 (`docs/superpowers/plans/` の auto-mode 系を参照)
- LLM による後処理 (`src/llm/`)
- 公開ホスティング。静的ファイルのみを出力する構成にしておき、公開するか否かは
  動くものを見てから判断する

## 2. 実現方式の選定

### 採用: 「波形 in → logits out」の単一 ONNX を ORT Web (WASM) で実行

メルスペクトログラム変換をモデルと同じ ONNX グラフに焼き込む。JS 側に残るのは
音声取得・CTC greedy デコード・符号変換表のみ。

**採否の理由**: この規模のモデル移植で失敗する要因はほぼすべて「前処理の再現ミス」に
集約される。`AmplitudeToDB(top_db=80)` の全体最大値依存クランプ、`center=True` の
reflect padding、z-score 正規化を JS で書き直すとズレる余地が大きく、しかもズレても
「なんとなく精度が落ちた」としか観測できない。グラフに焼き込めばこの失敗モード自体が
消え、Python と JS の数値一致を直接検証できる。

### 不採用: メル変換を JS で実装し、ONNX はモデル本体のみ

エクスポートは `torch.onnx.export` 一発で済むが、上記の再現リスクを丸ごと抱える。
FFT の実装も別途必要になる。

### 不採用: WebGPU による高速化

ORT Web の WebGPU 実行系は LSTM を実装しておらず CPU にフォールバックする。現行の
CNN + BiLSTM 構成では効果が出ない。モデル側を Conv / Transformer に置き換える将来の
検討とセットの項目。

## 3. 性能の実測 (設計の前提)

`models/full/best_infer.pt` (4,268,265 パラメータ / 17,083,893 バイト) を CPU に載せ、
`InferenceEngine.decode_chunk` で 8.5 秒ぶんの音声をデコードした所要時間:

| 条件 | 所要時間 | RTF |
|---|---|---|
| PyTorch CPU 4 スレッド | 25.6 ms | 0.003 |
| PyTorch CPU 1 スレッド | 72.3 ms | 0.009 |

8.5 秒は `settings.py` の既定値から算出したデコード区間の上限
(`decode_left_context_s` 5.0 + `commit_lag_s` 2.5 + `hop_s` 1.0)。

ORT Web の WASM(SIMD) はネイティブ 1 スレッドに対しておよそ 3〜6 倍遅い (SIMD 幅が
128 bit で AVX2 の半分、加えて WASM 自体のオーバーヘッド)。したがって**シングル
スレッドでも 1 回のデコードは 0.2〜0.45 秒**と見積もる。

この見積もりから導かれる設計上の判断:

- **COOP/COEP (cross-origin isolation) は必須ではない**。ORT Web のマルチスレッドが
  無効でも `hop_s` 1.0 秒に対して十分な余裕がある。ホスティング先を選ばない
- **`hop_s` を 0.5 秒に短縮する余地がある** (§6)
- **int8 量子化は行わない**。速度が足りているのに精度リスクを取る理由がない。将来
  必要になった場合の手段としてのみ残す

## 4. 全体構成

```
[マイク / ライン入力]
   │ getUserMedia({ echoCancellation:false,
   │                noiseSuppression:false,
   │                autoGainControl:false })
   ▼
[AudioContext @ 8000 Hz] ─── AnalyserNode ──▶ レベルメータ / スペクトル表示
   │
   ▼
[AudioWorkletProcessor]  128 サンプル単位で受け、0.05 秒ぶんに束ねて postMessage
   ▼
[Decoder Worker]
   │  リングバッファ (30 秒)
   │  hop ごとに 1 回:
   │    デコード区間を切り出す → ORT Web 推論 → CTC greedy (フレーム位置付き)
   │    → 中点ウォーターマークで 確定 / 暫定 を判定
   ▼  postMessage(DecodeView)
[UI (メインスレッド)]  確定 = 通常色 / 暫定 = グレー、欧文列・和文列を併記
```

### 各部の責務

| 部品 | 責務 | 依存 |
|---|---|---|
| `audio/capture.ts` | getUserMedia、AudioContext 構築、Worklet 登録 | Web Audio API |
| `audio/resample-worklet.ts` | 8 kHz 化 (フォールバック時)、ブロック化 | なし |
| `worker/decoder.worker.ts` | リングバッファ、推論、確定判定 | ORT Web, 下記 3 つ |
| `decode/ctc.ts` | CTC greedy decode (フレーム位置付き) | なし |
| `decode/sliding-window.ts` | 確定 / 暫定の判定 | `ctc.ts` |
| `tokens/converter.ts` | トークン列 → 欧文 / 和文テキスト | `generated/tokens.ts` |
| `generated/tokens.ts` | 符号表 (Python から自動生成、手編集禁止) | なし |
| `ui/` | 表示 | 上記すべて |

音声の受け渡しは **`postMessage` のみ**とし、SharedArrayBuffer は使わない。
8 kHz モノラルは 32 KB/s しかなく `postMessage` で十分であり、SAB を使うと
COOP/COEP が必須になってホスティング条件が制約されるため。

### サンプルレート変換

`src/infer/audio.py` に記録されている通り、**ステートレスなリサンプルはブロック境界に
歪みを生じ、CW の dot/dash を破壊する**。デスクトップ版は `soxr.ResampleStream` の
ステートフル版を使っている。ブラウザでは同じ罠を避けるため:

1. **主系**: `new AudioContext({ sampleRate: 8000 })` を構築し、
   `MediaStreamAudioSourceNode` を接続する。ブラウザ内蔵のリサンプラはグラフ内で
   状態を保持するため境界歪みが生じない
2. **フォールバック**: `ctx.sampleRate` が 8000 にならない環境では、ネイティブ
   レートで AudioContext を作り、AudioWorklet 内に**状態を保持する FIR
   デシメータ**を置く。48000 / 8000 = 6 のような整数比が大半なので、
   ローパス FIR + 間引きで足りる

起動時に `ctx.sampleRate` を検査してどちらの経路かを確定し、UI に表示する。

### getUserMedia の制約指定

`echoCancellation` / `noiseSuppression` / `autoGainControl` を**すべて false** にする。
既定では有効であり、いずれも CW のトーンを破壊する。デスクトップ版でも AGC の
除去が必要だった経緯がある。

## 5. ONNX エクスポート

新規 `scripts/export_onnx.py` が、メル変換とモデルを 1 つのグラフに束ねて出力する。

### メル変換のグラフ化

`torch.stft` (ONNX の STFT op) は使わず、畳み込みとして展開する。

- Hann 窓を掛けた DFT 基底を重みとする `Conv1d` (カーネル長 = `n_fft` = 256)。
  出力 258 ch = 実部 129 + 虚部 129 (`n_fft` 256 の片側スペクトル)、`stride` = 80。
  **`win_length` (200) は `n_fft` (256) より短いため、Hann 窓を中央寄せで 256 に
  ゼロ詰めしてからカーネルに掛ける** (torchaudio の内部挙動と一致させる。
  ここを取り違えると窓が 28 サンプルずれる)
- `center=True` 相当の reflect padding (`n_fft // 2` = 128) を前段に明示的に置く
- パワースペクトル = 実部^2 + 虚部^2
- メルフィルタバンクは **`torchaudio.functional.melscale_fbanks` から係数を取り出して
  定数化**する。自前で式を書き直さない
- `AmplitudeToDB(stype="power", top_db=80)` 相当:
  `10 * log10(clamp(x, min=1e-10))` の後、`max(x_db, x_db.max() - 80)`。
  バッチサイズ 1 前提なので `max()` はテンソル全体の最大値
- z-score 正規化 (平均・標準偏差は周波数軸と時間軸の両方にわたって取り、
  標準偏差は `1e-5` で下限クランプ)

### モデル本体

- `build_model_from_checkpoint` で読み込み、**`model.eval()` を必ず通す**
  (BatchNorm を推論モードにし、Dropout を無効化する)
- グラフ末尾に `log_softmax` を入れる。JS 側は argmax と exp だけで済む
- opset 17。入力は `wave (1, T_wave)`、出力は `log_probs (1, T_frames, 73)` とし、
  **`T_wave` と `T_frames` を動的軸に指定する** (バッチは 1 固定)

### エクスポート時の検証 (必須)

エクスポート直後に同スクリプト内で検証を走らせ、失敗したら出力を破棄する。

- `sample_wav/` の実音声を `MelExtractor` と ONNX グラフの両方に通し、
  メル出力の**最大絶対誤差が 1e-4 未満**であること
- 同じ音声で `InferenceEngine.decode_chunk` と ONNX 推論 + CTC の
  **トークン ID 列が完全一致**すること

この検証を通せば「前処理のズレ」という失敗モードは構造的に消える。

### 配布

fp32 のまま約 17 MB。初回のみダウンロードし、**Cache API に保存**して 2 回目以降は
即起動する。読み込み中は UI に進捗を出す。

## 6. 確定の高速化

現状の確定までの遅延の内訳:

| 要素 | 既定値 | 内容 |
|---|---|---|
| `commit_lag_s` | 2.5 s | 右文脈が足りるまで確定を保留 |
| `hop_s` 待ち | 0〜1.0 s | 次の再デコードまでの待ち |
| 推論時間 | 0.2〜0.45 s | ORT Web での 1 回のデコード (§3 の見積もり) |

平均で約 3 秒。以下の 3 つで詰める。

### 6.1 暫定表示 (体感遅延の切り離し)

`DecodeView.provisional` は既に実装済み。これを薄いグレーで即座に表示することで、
**体感遅延は `commit_lag` から切り離され `hop` + 推論時間になる**。文字はまず
グレーで現れ、あとから通常色に固まる。

### 6.2 `commit_lag_s` の実測による短縮

2.5 秒という値は余裕を見た数字で、実測の裏付けがない。20 WPM なら語間ギャップは
0.42 秒であり、BiLSTM が実際に必要とする右文脈はもっと短いと考えられる。

新規 `scripts/sweep_commit_lag.py` で掃引する:

- `keyed_val` の各音声について `SlidingWindowDecoder` を `hop` ずつ回す
- `commit_lag_s ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 2.5}` の各値について確定列を得る
- `finalize()` の結果 (右文脈が最大限ある状態) を基準とし、確定した**トークン ID 列**の
  一致率と CER を算出する (文字に変換する前の段階で測る。変換表の影響を混ぜない)
- 劣化が始まる手前の値を採用する

**この作業は Python 側で完結するのでブラウザ実装と並行でき、デスクトップ版も
同時に速くなる**。結果は `docs/` に残す (`word_break_threshold_result.md` と同じ形式。
下げられないという結論になった場合もその旨を記録する)。

### 6.3 `hop_s` の短縮

§3 の実測から 0.5 秒への短縮に計算量の余裕がある。ただし体感遅延と CPU 負荷の
トレードオフなので、実機で ORT Web の実測を取ってから確定する。

### 採らない選択肢: 安定性ベースの早期確定

連続する再デコードで同じトークンが同じ位置に出続けたら `commit_lag` 到達前に確定する
案は、効果は大きいが、**確定トークンは不変 (immutable) 設計のため早すぎる確定は
永久に消えない誤りとして残る**。6.1 と 6.2 で目標に届く見込みがあるため、初版では
採らない。

## 7. 符号表の単一ソース管理

`.claude/CLAUDE.md` の「`src/tokens/morse_tokens.py` を唯一の真正なソースとする」
原則をブラウザ版でも守る。TypeScript 側に符号表を**手で書き写さない**。

- 新規 `scripts/export_tokens.py` が `morse_tokens.py` から
  `web/src/generated/tokens.ts` を生成する
- 生成物である旨と手編集禁止をファイル冒頭に明記する
- 和文符号は実装誤記が起きやすく、二重定義を作れば必ず事故る

`converter.py` の変換ロジック (濁点・半濁点の合成、`?` の TABLE_MISS /
LOW_CONFIDENCE の区別) は TypeScript へ移植し、**Python の既存ユニットテストと
同じケース**を TS 側でも回す。

## 8. 移植対象

| 移植する | 移植元 | 行数目安 |
|---|---|---|
| `ctc_greedy_decode_with_frames` | `src/infer/engine.py` | 約 40 行 |
| `SlidingWindowDecoder` | `src/infer/sliding_window.py` | 約 120 行 |
| 符号変換ロジック | `src/tokens/converter.py` | 約 200 行 |
| 符号表 | `src/tokens/morse_tokens.py` | 自動生成 |
| 入力バンドパスフィルタ (BPF) | `src/app/workers.py` の `_StreamingBPF` | 約 30 行 |
| 時間軸の定数 (`sample_rate` / `hop_length`) | `src/train/preprocessing.py` の `MelConfig` | 自動生成 |

移植しないもの: `src/infer/net_audio.py`、`src/infer/settings.py` (Qt 依存)、
`src/llm/`、`src/app/` (PySide)。

### 入力バンドパスフィルタ (BPF) について

`src/app/` は PySide 依存として一括除外したが、`_StreamingBPF` は Qt に依存しない
**信号処理**であり、除外に巻き込まれていた。デスクトップ版はこれを**既定で有効**に
して全音声ブロックにかけてから `SlidingWindowDecoder.push` に渡しており、実受信
精度に直結するため、最終レビューで追加移植した (`web/src/audio/capture.ts`)。

| | デスクトップ版 | ブラウザ版 |
|---|---|---|
| 実装 | scipy `butter(4, ..., btype="bandpass", output="sos")` + `sosfilt` | `BiquadFilterNode` (`type = 'bandpass'`) を 2 段直列 |
| 特性 | bandpass 指定で次数が 2 倍になるため SOS 4 セクション = 実効 8 次 | RBJ Audio EQ Cookbook のバイカッド × 2 段 = 4 次相当 |
| 中心 / 帯域 | 600 Hz / 400 Hz | 600 Hz / 400 Hz (`Q = 600/400`) |
| 既定 | 有効 | 有効 |
| 切替 | UI のチェックボックス | UI のチェックボックス (受信中も可) |

**両者の周波数特性は厳密には一致しない。** デスクトップ版は実効 8 次、ブラウザ版は
4 次相当と段数が半分であり、**ロールオフ (肩の落ち方) はブラウザ版のほうが約半分
緩い。** 通過帯域の平坦さ・群遅延も異なる。Web Audio API には SOS バターワースを
直接構成する手段が無く (`IIRFilterNode` で係数を手書きすれば可能だが、状態管理を
自前で持つことになる)、**同等の目的 (帯域外のノイズを落として推論に渡す) を果たす
近似**として BiquadFilterNode を選んだ。そのぶん、同一 WAV に対するデスクトップ版
とブラウザ版の出力は BPF の差だけでもずれうる。BPF オン/オフの比較で差が小さくても、
この段数差があるため「BPF が無意味」とは即断できない。

設計上の注意:

- **BPF はデコード経路にだけ入れ、`AnalyserNode` は生の `source` に繋ぐ。**
  スペクトル表示は同調 (トーンがどの周波数に来ているか) の確認に使うので、
  通過帯域の外も見えなければならない。
- フィルタはグラフ内にあるのでブラウザが状態を保持する。**自前の状態管理は不要**
  (ステートレスなリサンプルで踏んだ轍を避けるため、あえてグラフに置いている)。
- WAV ファイル入力 (`decode-file.ts`) には BPF をかけていない。Python 側の
  `scripts/decode_user_wav.py` も BPF を通さないため、**同一 WAV での
  デスクトップ版との突き合わせ (実信号確認シート項目 5) を汚さないようにするため**。

## 9. 表示仕様

### 表示モード

| モード | 内容 |
|---|---|
| 欧文 | 欧文表による解釈のみを 1 列で表示 |
| 欧文 + 和文 | 同じトークン列を欧文表と和文表の両方に通し、2 列で併記 |

NN は統合符号トークン ID を出力するだけで、欧文表・和文表は決定論的な変換表なので、
**同じトークン列を 2 つの表に通すだけ**で併記できる。追加の推論コストはない。

### 確定 / 暫定

- 確定 (committed): 通常色。以後変化しない
- 暫定 (provisional): グレー。次の再デコードで書き換わる可能性がある

### `?` の 2 分類

`.claude/CLAUDE.md` の原則通り、TABLE_MISS (変換表に該当なし) と LOW_CONFIDENCE
(CTC 事後確率が閾値未満) を区別する。表示上はどちらも `?` だが、
ホバー時のツールチップと開発者向けログでは種別を区別する。

### 信号モニタ

`AnalyserNode` を 8 kHz の AudioContext から直接引く。

- 入力レベルメータ (RMS dBFS)
- スペクトル表示。CW のトーン周波数が見えるので同調確認に使える

## 10. 検証方針

移植で最大のリスクは「動くが微妙に精度が落ちている」ことなので、これを
一致 / 不一致の二値で判定できる形にする。

### ゴールデンテスト

1. `sample_wav/` の数本について、Python の `InferenceEngine.decode_chunk` が出した
   トークン ID 列を JSON に固める (`scripts/export_golden.py`)
2. ブラウザ側の WAV ファイル入力から同じ音声を流し、**トークン ID 列が完全一致する
   こと**を確認する

### その他

- メル一致テスト: §5 のエクスポート時検証
- `converter.ts` のユニットテスト: Python 側の濁点合成テストと同じケースを移植
- `sliding-window.ts` のユニットテスト: 合成トークン列を入力し、確定 / 暫定の
  境界判定と中点ウォーターマークの挙動を検証する (音声もモデルも要らない)

## 11. 配置とビルド

`cw-decorder/web/` に Vite + TypeScript で構築する。

```
cw-decorder/
├── scripts/
│   ├── export_onnx.py        # 新規: 波形 in → logits out の ONNX を出力
│   ├── export_tokens.py      # 新規: morse_tokens.py → tokens.ts
│   ├── export_golden.py      # 新規: ゴールデンテスト用の期待値を出力
│   └── sweep_commit_lag.py   # 新規: commit_lag の掃引
└── web/
    ├── public/
    │   └── model/cw.onnx     # エクスポート成果物 (git 管理外)
    ├── src/
    │   ├── audio/
    │   ├── worker/
    │   ├── decode/
    │   ├── tokens/
    │   ├── generated/        # 自動生成 (手編集禁止)
    │   └── ui/
    └── tests/
```

静的ファイルのみを出力するため、ローカル起動でも将来の公開でもそのまま使える。
公開する判断をした場合は `web/` だけを切り出せる形にしておく。

`cw.onnx` は 17 MB あるため git 管理外とし、`scripts/export_onnx.py` で再生成する
手順を `web/README.md` に書く。

## 12. 実装順序

リスクの高い順に並べる。

**Step 0 — スパイク (本命のリスク)**
`scripts/export_onnx.py` で ONNX を出力し、ORT Web で 1 回推論して RTF を実測する。
確認すること: (a) ORT Web の WASM で LSTM が動くか、(b) §3 の見積もり
(0.2〜0.45 秒) が実機で成立するか、(c) メル一致テストが通るか。
ここが赤なら、左文脈の短縮・`hop` の延長・int8 量子化のどれで吸収するかを
実測に基づいて決め直す。

**Step 1 — `commit_lag` の掃引** (Step 0 と並行可能、Python のみ)

**Step 2 — 音声経路**
getUserMedia → AudioContext 8 kHz → Worklet → Worker までを繋ぎ、
レベルメータで音が届いていることを確認する。

**Step 3 — デコーダ移植**
`ctc.ts` / `sliding-window.ts` / `converter.ts` / `tokens.ts` 生成。
ユニットテストとゴールデンテストを通す。

**Step 4 — UI**
確定 / 暫定の 2 色表示、欧文・和文の併記、スペクトル表示。

**Step 5 — 実信号での確認**
実際の受信音でデコードし、デスクトップ版と結果を突き合わせる。

## 13. 未解決事項

- ORT Web における LSTM の実行速度は §3 の外挿値であり、Step 0 で実測して確定する
- `commit_lag_s` の採用値は Step 1 の掃引結果で決まる
- `hop_s` を 0.5 秒にするかは Step 0 の実測後に判断する
- ホスティング先 (ローカル / 公開) は動くものを見てから判断する。設計は
  どちらでも成立する
