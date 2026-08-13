# cw-decoder-web

CW (モールス信号) デコーダのブラウザ版。学習済みモデルを ONNX 化し、ブラウザ上で
マイク / ライン入力からリアルタイムに CW をデコードする。ONNX Runtime Web
(wasm バックエンド) でブラウザ内推論する。`/` がデコーダ本体の画面、
`/bench.html` が ORT Web の実測ベンチ。

設計書: `../docs/superpowers/specs/2026-08-05-browser-cw-decoder-design.md`
(`../docs/design.md` はデスクトップ版の設計書なので混同しないこと)

## セットアップ

```bash
cd web
npm install
```

### モデル・符号表・fixture の生成 (clone 直後は必ず実行)

モデルと符号表と golden fixture は Python 側から生成する。生成物は 3 つとも
git 管理外なので、**clone 直後や pull で `morse_tokens.py` / モデルが変わった後は
必ず再生成すること**。

```bash
# リポジトリルート (cw-decoder/) で実行。web/ ではない点に注意
.venv/Scripts/python.exe scripts/export_onnx.py      # → web/public/model/cw.onnx (約17MB)
.venv/Scripts/python.exe scripts/export_tokens.py    # → web/src/generated/tokens.ts
.venv/Scripts/python.exe scripts/export_golden.py    # → web/tests/fixtures/
```

- `export_onnx.py`: 学習済みモデル (`models/full/best_infer.pt` 等) を ONNX 化する。
  自己検証つき (PyTorch と ONNX Runtime の出力を比較してから書き出す)。
- `export_tokens.py`: `src/tokens/morse_tokens.py` (符号定義の唯一の真正なソース) から
  `web/src/generated/tokens.ts` を生成する。**`web/src/generated/tokens.ts` は自動生成物
  なので手で編集しないこと。** 符号を直すときは必ず `morse_tokens.py` を直してから
  このスクリプトを再実行する。
- `export_golden.py`: Python 推論エンジンの出力 (波形・トークン ID 列・確信度・
  デコード結果) を `web/tests/fixtures/golden.json` 等に書き出す。ブラウザ側の
  ゴールデンテスト (`web/tests/golden.test.ts`) が Python との完全一致を確認するのに使う。

### ★★ `npm test` の前に必ず `export_onnx.py` を実行すること (最重要)

`web/public/model/cw.onnx` は約 17 MB あるため git 管理外 (`.gitignore` 済み)。
これが無い状態で `npm test` を実行すると、**`golden.test.ts` の全テストが
黙ってスキップされ、それでも終了コード 0 で「全部緑」に見える。**
`golden.test.ts` は Python の推論結果とブラウザ (TypeScript) 側のトークン ID 列が
完全一致することを確認する、この移植プロジェクトで最も重要なテストである。
モデル未生成のままテストを緑と見なして進めると、移植の正しさを一切検証しないまま
作業を進めてしまうことになるので注意すること
(スキップ時はテスト側にも `console.warn` で警告が出るが、見落としやすいのでここにも明記する)。

## 起動

```bash
cd web
npm run dev
```

表示された URL (既定 `http://localhost:5173/`) を開くとデコーダ本体の画面が出る。

> **`localhost` か `127.0.0.1` で開くこと。** マイク取得 (`getUserMedia`) は
> セキュアコンテキストでしか動かない。`localhost` は例外扱いされるが、
> LAN IP (`http://192.168.x.x:5173/`) や `file://` ではマイクがブロックされる。
>
> `npm run dev` は環境によって IPv6 (`::1`) にしか bind せず、
> `http://127.0.0.1:5173/` で繋がらないことがある。その場合は `localhost` で開くか、
> `npm run dev -- --host 127.0.0.1` を使う。

### 無音で改行する (`?linebreak=`)

送信のターンの切れ目で行を分ける。**この長さ以上の無音が空いたら改行**する
(既定 3.0 秒)。`?linebreak=5` のように秒で上書きでき、`?linebreak=0` で無効になる。

判定はトークンの時刻差だけで行い、音声は見ない。確定テキストは毎回トークン列全体から
作り直されるので、判定がトークン列だけから決まることが重要 (何度作り直しても同じ結果に
なり、「確定した文字は書き換わらない」保証を壊さない)。

**既定 3.0 秒の限界**: held-out 実録音 21 件で「1 回の送信の中に現れる無音」を測ると
中央値 650 ms・99%tile 2.0 秒・**最大 4.4 秒**だった。つまり 3.0 秒では**送信の途中でも
まれに改行が入る**。余分な改行が入っても文字は失われないので許容している。
気になる場合は `?linebreak=5` を試す。

### モデルを切り替える (`?model=`)

`?model=<名前>` を付けると `public/model/<名前>.onnx` を読む。既定は `cw` (配布モデル)。
**モデルを比較したいときに `public/model/cw.onnx` を上書きせずに済ませるための口。**

```bash
# 別のチェックポイントを別名で書き出す (リポジトリルートで実行)
.venv/Scripts/python.exe scripts/export_onnx.py \
  --checkpoint models/full_hand/best.pt --out web/public/model/cw_hand.onnx
```

```
http://localhost:5173/              ← 配布モデル
http://localhost:5173/?model=cw_hand ← 書き出したモデル
```

**2 つのタブを同時に開いて並べられる。** 実測で、同時に走らせても推論時間は
単独実行時と変わらなかった (53 ms / 54 ms)。どちらのモデルで出た結果かは
画面下の診断行 (`… / モデル cw_hand`) に出る。

### 別の PC のブラウザから使う (そのPCのマイクで受信する)

サーバ役の PC で配信し、無線機を繋いだ**別の PC のブラウザ**で開いて、
**そのPCのマイクから**デコードできる。デコードは全部ブラウザ内で完結するので、
サーバ側は静的ファイルを配るだけで、音声はネットワークに流れない。

ただし前述のとおり LAN IP + HTTP ではマイクが使えないので、**HTTPS で配信する**。
認証局の証明書は要らない (自己署名で十分)。

```bash
# 1. サーバ役の PC の LAN IP を調べる
powershell -c "Get-NetIPAddress -AddressFamily IPv4 | Select IPAddress,InterfaceAlias"

# 2. その IP を入れた自己署名証明書を作る (web/certs/ に出る。git 管理外)
cd web
bash scripts/make-lan-cert.sh 192.168.0.20

# 3. HTTPS + LAN 公開で起動する
npm run dev:lan
```

**HTTPS になるのは `--host` を付けたとき (= `dev:lan` / `preview:lan`) だけ。**
`web/certs/lan.{key,crt}` があってもローカル起動 (`npm run dev`) は HTTP のままにしてある
(`vite.config.ts`)。`localhost` はセキュアコンテキストの例外なので HTTP でマイクが使え、
そこで HTTPS にすると自己署名の警告を毎回通す手間だけが増えるため。
ビルド済みを配る場合は `npm run build` → `npm run preview:lan`。

**ローカルと LAN を同時に立てられる** (ポートが別なら共存する)。例:

```bash
npm run dev                      # http://localhost:5173/    このマシン用
npm run dev:lan -- --port 5443   # https://192.168.x.x:5443/ 別 PC 用
```

別の PC のブラウザで `https://192.168.0.20:5173/` を開く。自己署名なので初回だけ
**「この接続ではプライバシーが保護されません」の警告**が出る →「詳細設定」→
「(安全ではないページ) にアクセスする」で通す。以降はマイクが使える
(警告を通したあとの HTTPS オリジンはセキュアコンテキストとして扱われる)。

補足:

- モデル (17 MB) と WASM が相手のブラウザにダウンロードされる。2 回目以降は
  Cache API から読むので再ダウンロードしない。
- Windows ファイアウォールで受信がブロックされる場合がある。ネットワークの
  プロファイル (Public / Private) と node.exe の受信規則を確認すること。
- 相手 PC のブラウザ側でマイクの入力デバイス選択と入力レベル調整を行う。

## テスト

```bash
cd web
npx vitest run
npx tsc --noEmit
```

`tests/golden.test.ts` は Python の推論結果との完全一致を確認する。
モデル・fixture が未生成の場合は自動でスキップされる (前述の「★★」参照)。

## 使い方・注意事項

- **マイク入力**: 「受信開始」を押すとマイクへのアクセス許可を求められる。
  `getUserMedia` の `echoCancellation` / `noiseSuppression` / `autoGainControl`
  は**すべて明示的に無効化している** (`src/audio/capture.ts`)。これらのブラウザ標準の
  音声処理は人の声向けに作られており、CW のトーン (単一周波数の断続音) にかけると
  波形が歪んだり音量が急に絞られたりして、デコード精度を大きく落とすため。
- **BPF (バンドパスフィルタ)**: デコード経路には中心 600 Hz / 帯域 400 Hz の
  バンドパスフィルタが入っており、**既定で有効**。デスクトップ版
  (`src/app/workers.py` の `_StreamingBPF`) と揃えてある。画面の「BPF」
  チェックボックスで受信中でもオン/オフでき、実信号で効果を比較できる
  (実信号確認シート `../docs/browser_decoder_field_test.md` の項目 6)。
  - **実装は `BiquadFilterNode` (`type = 'bandpass'`) の 2 段直列 (4 次相当) で
    あり、デスクトップ版の scipy `butter(4, ..., btype='bandpass')` (bandpass
    指定で次数が 2 倍になるため SOS 4 セクション = 実効 8 次) とは段数が半分
    で、周波数特性は厳密には一致しない。** `BiquadFilterNode` の 'bandpass' は
    RBJ Audio EQ Cookbook のバイカッドで、通過帯域の平坦さ・肩の落ち方・群遅延
    が異なり、**ロールオフ (肩の落ち方) はブラウザ版のほうが約半分緩い。**
    **同等の目的 (帯域外のノイズを落として推論に渡す) を果たすための近似で
    あって、同一のフィルタではない。** 同一 WAV に対するデスクトップ版との
    出力差には、この差も寄与しうる。BPF オン/オフの比較で差が小さくても、
    この段数差があるため「BPF が無意味」とは即断できない。
  - **BPF はデコード経路にだけ入っており、スペクトル表示 (`AnalyserNode`) は
    生の入力に繋がったまま。** 同調確認のために通過帯域の外も見える必要があるため。
    BPF をオン/オフしてもスペクトル表示は変わらない。
  - フィルタは AudioContext のグラフ内にあるのでブラウザが状態を保持する
    (自前の状態管理は無い。ステートレスなリサンプルがブロック境界に歪みを生む
    のと同じ罠を避けるため、あえてグラフに置いている)。
  - **WAV ファイル入力には BPF がかからない。** Python 側の
    `scripts/decode_user_wav.py` も BPF を通さないため、同一 WAV での突き合わせを
    汚さないようにしてある。
- **WAV ファイル入力**: 「WAV を読む」から録音済みファイルを選ぶとその場でデコードする。
  マイクを使わずに動作確認できるほか、同じ音声ファイルを Python のデスクトップ版
  (`scripts/decode_user_wav.py` 等) でもデコードすれば、両者の出力を突き合わせて
  移植の正しさを検証できる。
- **「クリア」ボタン**: 画面の確定テキスト・暫定テキストと、Worker 側の確定状態
  (`SlidingWindowDecoder` の `committed`) の両方をクリアする。表示だけ消しても
  次の再デコードで戻ってくるため、両方を消す必要がある。
- **確定遅延と hop の関係 (重要)**: 効くのは `commit_lag_s` 単独ではなく
  **`commit_lag_s + hop_s / 2`** (= 実効右文脈)。確定境界は `totalConsumed - commitLag`
  なので、トークンが確定されるときに実際に使われた右文脈は平均
  `commit_lag_s + hop_s/2` になる。**目標値は 2.25 秒**で、ブラウザ版は
  `HOP_S = 0.5` / `COMMIT_LAG_S = 2.0`、デスクトップ版も同じ値。
  **hop を変えるときはこの和を保つこと** (`../docs/commit_lag_sweep_result.md` §8)。
- **確信度不足の文字は「?」ではなく最尤文字を薄字で出す**。確定テキスト自体は
  変えず、表示層だけで置き換えている (CER は悪化しない)。マウスカーソルを乗せると
  確信度と符号がツールチップに出る。
  - 濃いオレンジの `?` が残るのは `TABLE_MISS` (変換表に該当なし。例: 欧文モード中に
    和文符号が来た場合)。**表に文字が無いので出しようがない**。
  - この 2 分類の区別は `CLAUDE.md` の原則に対応する。
  - **閾値を下げて `?` を消してはいけない。** held-out 実録音で閾値 0.5 → 0.0 に
    すると CER が 19.74% → 23.64% と悪化した。`?` の裏に正解が隠れているのではなく、
    ほとんど間違った文字が出てくるだけだった。
- **ベンチページ**: `/bench.html` (下記「ORT Web 実測ベンチ」参照)。`?threads=N` で
  スレッド数を指定する。

## ORT Web 実測ベンチ

設計書 §12 Step 0 (計画全体で最もリスクの高い判定ポイント) の確認用。
ONNX Runtime Web の wasm バックエンドで LSTM を含むモデルが動くか、
実用速度に収まるかを計測する。

### ブラウザで計測する

```bash
npm run dev
```

表示された URL の `/bench.html` に `?threads=1` を付けて開き、ページ本文と
devtools コンソールに出る数値を読む。4 スレッドの数値も欲しい場合は、別タブで
同じ URL に `?threads=4` を付けて開き直す (`src/bench/bench.ts`)。同一ページで
両方測ろうとしてはいけない理由は下記「計測時に見つかった落とし穴」を参照。

### Node で計測する

```bash
npm run bench:node
```

`onnxruntime-web` は Node.js から import すると `package.json` の `"node"`
export 条件により `dist/ort.node.min.mjs` が解決される。これはネイティブ
アドオン (`onnxruntime-node`) ではなく、ブラウザ版と同じ
`ort-wasm-simd-threaded.wasm` バイナリを Node 上で動かすビルドなので、
ブラウザを操作できない環境でもシングルスレッド性能の代理として意味のある
数値が得られる (`scripts/bench-node.mjs`)。

実測結果は `../docs/browser_ort_bench.md` に記録している。

### bench.ts と bench-node.mjs の重複について

トーンバースト波形の生成式 (`Math.sin(2π·600·t)` × `Math.sin(2π·4·t) > 0`
のエンベロープ) は `src/bench/bench.ts` (ブラウザ用 TypeScript) と
`scripts/bench-node.mjs` (Node 用プレーン ESM) の両方に書かれており、
コードとしては重複している。

意図的にモジュール共有を避けた。理由:
- Node はビルドステップなしに `.ts` を直接 `import` できない。
  `scripts/` 側を `.mjs` にすることで `node scripts/bench-node.mjs` を
  そのまま実行できるようにしたかった (Vite や tsx 等の追加ツールを
  このタスクのために導入したくなかった)。
- 波形生成のロジックは 10 行に満たない小さな純関数で、今後仕様が変わる
  可能性も低い (決定的なテスト用ダミー波形であり、実プロダクトの信号処理
  ロジックではない)。共有モジュールを切り出すコストの方が、2 箇所を
  目で見比べて同期を保つコストより高いと判断した。
- 将来 `src/` 側に `.js` (拡張子なしで Node からも読める形) の共有モジュール
  を切り出す価値が出てきたら (例: ベンチ波形を本体のテストでも使うように
  なったら) そのとき切り出せばよい。

### 計測時に見つかった落とし穴 (重要)

ORT Web の wasm スレッドプールは、**プロセス内で最初に
`InferenceSession.create()` を呼んだ時点の `numThreads` に固定される**。
同一プロセス内で `ort.env.wasm.numThreads` を後から変更しても、既に
初期化済みの wasm ランタイムには反映されない。

ブリーフの元の `bench.ts` はページ内で `benchmark(1)` の後に `benchmark(4)`
を呼ぶ構造で、この落とし穴を踏んでいた (4 スレッド側の数値が実際には
1 スレッド相当のまま計測される)。そのため **`bench.ts` をブリーフから変更**し、
URL クエリパラメータ `?threads=N` で指定した 1 つのスレッド数だけをそのページ
読み込みで計測する構成にした。1 スレッドと 4 スレッドを両方測るには、
`http://localhost:5173/bench.html?threads=1` と
`http://localhost:5173/bench.html?threads=4`
のように別タブ (別ページ読み込み) でそれぞれ開くこと
(Task 13 でベンチを `bench.html` に分離し、`/` はデコーダ本体の画面になった)。

`bench-node.mjs` も同じ理由で、スレッド数ごとに `node scripts/bench-node.mjs
--threads=N` を**別プロセス**として起動する構成にした。詳細は
`../docs/browser_ort_bench.md` を参照。
