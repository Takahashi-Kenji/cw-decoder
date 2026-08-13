/**
 * ブラウザ版 CW デコーダの画面.
 *
 * 表示は 2 モード:
 *   欧文        — 欧文表による解釈のみ
 *   欧文 + 和文 — 同じトークン列を両方の表に通して併記する
 *
 * 確定 (committed) は通常色、暫定 (provisional) はグレーで表示する。
 * 暫定を出すことで体感遅延が commit_lag から切り離される (設計書 §6.1)。
 */
import { AudioCapture } from '../audio/capture'
import { DecoderClient } from '../worker/decoder-client'
import { SignalMonitor } from './monitor'
import { SAMPLE_RATE, EUROPEAN_TABLE, JAPANESE_TABLE } from '../generated/tokens'
import { DEFAULT_LINE_BREAK_GAP_S } from '../worker/render-mode'
import type { FallbackEvent, Mode } from '../tokens/converter'
import type { WorkerResponse } from '../worker/protocol'

type TextResponse = Extract<WorkerResponse, { type: 'text' }>

/**
 * 使うモデル。既定は配布モデル (`/model/cw.onnx`)。
 *
 * `?model=<名前>` を付けると `/model/<名前>.onnx` に切り替わる。モデルを差し替えて
 * 比較したいときに、配布モデルのファイルを上書きせずに済ませるための開発用の口。
 * 例: `http://localhost:5173/?model=cw_hand`
 *
 * 値は `[A-Za-z0-9_-]` に限る。ユーザー入力をそのままパスに入れないため
 * (この画面はローカル開発用だが、URL は他人に渡りうる)。
 */
function resolveModelUrl(): string {
  const name = new URLSearchParams(location.search).get('model')
  if (name && /^[A-Za-z0-9_-]+$/.test(name)) return `/model/${name}.onnx`
  return '/model/cw.onnx'
}

const MODEL_URL = resolveModelUrl()

// hop 間隔。Task 3 の実測 (8.5 秒の音声のデコードが 1 スレッド 155.5 ms /
// 4 スレッド 58.3 ms) で hop 0.5 秒に対して 3 倍以上の余裕があると確認できたため
// 既定の 1.0 から 0.5 に変更した。暫定文字が出るまでの体感遅延が半減する。
const HOP_S = 0.5

// 確定までの遅延。
//
// ★ 効くのは `commit_lag_s` 単独ではなく **`commit_lag_s + hop_s / 2`** である。
// 確定境界は `totalConsumed - commitLag` なので、終端が時刻 t のトークンが確定
// されるときに実際に使われた右文脈は `[commitLag, commitLag + hop)`、平均
// `commitLag + hop/2` になる。この関係は変わらない。
//
// **2026-08-07 に 2.75 から 2.0 に下げた (実効 3.0 → 2.25 秒)。**
// held-out の実録音 21 件を正解ラベルと突き合わせた CER の実測:
//
//   lag 0.75 (実効 1.00s) 25.45%
//   lag 1.00 (実効 1.25s) 23.38%
//   lag 1.25 (実効 1.50s) 21.56%
//   lag 1.50 (実効 1.75s) 21.82%
//   lag 2.00 (実効 2.25s) 19.74%  ← 最良
//   lag 2.75 (実効 3.00s) 20.78%  ← 旧設定
//
// **2.0 は 2.75 より速く、かつ精度も良い。** 旧設定の根拠だった Task 4 の掃引は
// 「学習データ」に対して「オフライン一括デコードとの一致率」という代理指標で
// 測ったものだった。今回は held-out の実録音を正解ラベルで測っており、
// 測っている対象が違う (docs/commit_lag_sweep_result.md §8)。
//
// **これ以上下げるとはっきり悪くなる。** 1.25 まで下げると +1.8pt、
// 0.75 では +5.7pt。速さが要るときの選択肢ではあるが、ただではない。
const COMMIT_LAG_S = 2.0

const DECODE_LEFT_CONTEXT_S = 5.0

/**
 * この長さ以上の無音が空いたら改行する (送信のターンの切れ目で行を分ける)。
 * `?linebreak=<秒>` で上書きできる。0 を渡すと改行しない。
 *
 * 既定 3.0 秒の根拠と限界は render-mode.ts の DEFAULT_LINE_BREAK_GAP_S を参照。
 * 実運用で合わなければここを動かす前に URL で試すのが速い。
 */
function resolveLineBreakGapS(): number {
  const raw = new URLSearchParams(location.search).get('linebreak')
  if (raw === null) return DEFAULT_LINE_BREAK_GAP_S
  const v = Number(raw)
  return Number.isFinite(v) && v >= 0 ? v : DEFAULT_LINE_BREAK_GAP_S
}

const LINE_BREAK_GAP_S = resolveLineBreakGapS()

/**
 * WORD_BREAK のロジットに足す下駄。`?wb=<数値>` で指定。既定 0 (従来どおり)。
 *
 * 正にすると語間が出やすくなる。モデルの WORD_BREAK は recall が 57〜69% しか無く、
 * 実運用で語がくっついて読みにくいという報告があったための調整口。
 * 副作用としてモデルが余計に出すプロサイン ([SK] 等) も押さえられる。
 *
 * **既定値を 0 のままにしてあるのは、手元の held-out 録音では逆にスペースが
 * 出すぎており (101 個 / 正解 77 個)、実運用の「足りない」と向きが逆だったため。**
 * どの値が良いかはこのデータからは決められない。実信号で試して決めること。
 */
function resolveWordBreakBias(): number {
  const raw = new URLSearchParams(location.search).get('wb')
  if (raw === null) return 0
  const v = Number(raw)
  return Number.isFinite(v) ? v : 0
}

const WORD_BREAK_BIAS = resolveWordBreakBias()

/**
 * スキッシュ閾値 (dBFS)。既定 **-25**。**画面のスライダで受信中でも変えられる**
 * (ノイズは状況で変わるため)。初期値は `?squelch=<dBFS>` でも指定でき、
 * それ以外は前回スライダで決めた値を localStorage から復元する。
 *
 * **BPF 通過後**のレベルがこの値を下回る間、デコーダに無音を流し込む
 * (ブロックを捨てるのではなく 0 で置き換える)。捨てると時間軸が止まり、
 * 確定タイミングと改行の間隔判定が壊れるため。
 *
 * なぜ必要か: メル特徴抽出はサンプル内で z-score 正規化するので**音量の情報が
 * 捨てられる**。そのため -80 dBFS のノイズでも増幅されて信号のように見え、
 * 30 秒で 38 個ものトークン (ほとんど [SN]/[SK]) が出る。しかも確信度は 0.6〜0.7 と
 * 高く、確信度閾値では防げない。**デコーダの手前で止めるしかない。**
 *
 * なぜ BPF 後で測るか: 生の入力レベルでは無信号時 -28 dBFS、信号時 -40〜-10 dBFS と
 * 範囲が重なり分離できなかった。ノイズの大半は帯域外なので、BPF 後なら差が開く
 * (実測で BPF はノイズ由来のトークンを 25 個 → 2 個に減らす)。
 *
 * 既定 -25 は運用者の実信号で有効と確認した値。
 * **これより弱い信号は丸ごと落ちる**ので、弱い局を追うときは下げること。
 */
const DEFAULT_SQUELCH_DB = -25

/** スライダの下端。ここまで下げるとスキッシュを切る (デスクトップ版と同じ範囲)。 */
const SQUELCH_OFF_DB = -60
const SQUELCH_STORAGE_KEY = 'cw-decoder.squelch-db'

/** スライダの生の値 (dB) を実効閾値に変換する。下端と -100 以下は「切」。 */
function squelchValueToThreshold(v: number): number | null {
  return v <= SQUELCH_OFF_DB ? null : v
}

function resolveSquelchDb(): number {
  const raw = new URLSearchParams(location.search).get('squelch')
  if (raw !== null) {
    const v = Number(raw)
    // -100 以下 (Python 側 DISABLED_BELOW_DB) の指定は「切」= スライダ下端に丸める
    if (Number.isFinite(v)) return Math.max(SQUELCH_OFF_DB, Math.min(0, v))
  }
  const stored = Number(localStorage.getItem(SQUELCH_STORAGE_KEY))
  if (localStorage.getItem(SQUELCH_STORAGE_KEY) !== null && Number.isFinite(stored)) {
    return Math.max(SQUELCH_OFF_DB, Math.min(0, stored))
  }
  return DEFAULT_SQUELCH_DB
}

/** 現在の実効閾値 (null = 切)。スライダで受信中でも書き換わる。 */
let squelchDb: number | null = squelchValueToThreshold(resolveSquelchDb())

/** スキッシュが閉じるまでの保持時間 (秒)。符号の頭を削らないための余裕。 */
const SQUELCH_HOLD_S = 1.0

/** ブロックの RMS を dBFS で返す。 */
function blockDb(block: Float32Array): number {
  let sum = 0
  for (let i = 0; i < block.length; i++) sum += block[i] * block[i]
  const rms = Math.sqrt(sum / Math.max(block.length, 1))
  return rms < 1e-6 ? -120 : 20 * Math.log10(rms)
}

/**
 * スキッシュを適用したブロックを返す (閉じている間は同じ長さの無音)。
 *
 * ブロックを捨てずに 0 で置き換えるのは、捨てると時間軸が止まって確定タイミングと
 * 改行の間隔判定が壊れるため。`elapsedS` は音声時間 (WAV 入力でも実時間に依存しない)。
 */
function createSquelch(): (block: Float32Array, elapsedS: number) => Float32Array {
  let lastOpenAt = -Infinity
  return (block, elapsedS) => {
    // 毎ブロック読み直す。スライダで受信中に変えられるようにするため
    const threshold = squelchDb
    if (threshold === null) return block
    if (blockDb(block) >= threshold) lastOpenAt = elapsedS
    return elapsedS - lastOpenAt < SQUELCH_HOLD_S ? block : new Float32Array(block.length)
  }
}

const TEMPLATE = `
<h1>CW デコーダ</h1>
<div class="controls">
  <button id="toggle">受信開始</button>
  <button id="clear">クリア</button>
  <label><input type="checkbox" id="show-japanese" /> 和文を併記する</label>
  <label title="デコード経路のバンドパスフィルタ (中心 600 Hz / 帯域 400 Hz)。スペクトル表示には影響しません">
    <input type="checkbox" id="bpf" checked /> BPF (600±200 Hz)
  </label>
  <label title="BPF 後のレベルがこの値を下回る間、デコードを止めます。ノイズを文字として拾うのを防ぎます。下げすぎると弱い信号まで落ちます">
    スキッシュ
    <input type="range" id="squelch" min="-60" max="0" step="1" />
    <span id="squelch-value" class="squelch-value"></span>
  </label>
  <label class="file">WAV を読む<input type="file" id="wav" accept=".wav,audio/*" /></label>
  <span id="status">停止中</span>
</div>
<div class="monitor">
  <canvas id="spectrum" width="640" height="120"></canvas>
  <div class="level">入力レベル: <span id="level">-120.0</span> dBFS</div>
</div>
<section class="output" id="out-european">
  <h2>欧文</h2>
  <p class="text"><span class="committed"></span><span class="provisional"></span></p>
</section>
<section class="output" id="out-japanese" hidden>
  <h2>和文</h2>
  <p class="text"><span class="committed"></span><span class="provisional"></span></p>
</section>
<p class="diag" id="diag"></p>
`

/**
 * WASM のスレッド数を決める。
 *
 * crossOriginIsolated (COOP/COEP) でない環境ではマルチスレッドが使えないため 1。
 * 設計書 §3 の実測により、1 スレッドでも hop に対して余裕がある。
 */
function resolveThreadCount(): number {
  return globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1
}

/**
 * テキストを span に描画し、「?」の扱いを種別で分ける。
 *
 * 「?」には 2 種類ある (CLAUDE.md の原則):
 *   TABLE_MISS     — 変換表に該当なし (欧文モード中の和文符号など)
 *   LOW_CONFIDENCE — CTC 事後確率が閾値未満
 *
 * **LOW_CONFIDENCE は「?」ではなく最尤の文字を薄字で出す。**
 * 実測 (held-out 実録音 21 件) では、閾値を下げて「?」を消すと CER が
 * 19.74% → 23.64% と 3.9pt 悪化した。「?」の裏に正解が隠れているのではなく、
 * ほとんど間違った文字が出てくるだけだった。一方、読み手にとっては「?」より
 * 「怪しいがこう読めた」のほうが文脈で補える。そこで**確定テキスト自体は
 * 変えずに (CER は悪化しない)、表示だけ最尤文字にして薄く出す**。
 * 怪しいという情報は色で残るので、読み手が判断できる。
 *
 * TABLE_MISS は表に文字が無いので「?」のまま。
 */
function renderInto(
  target: HTMLElement,
  text: string,
  fallbacks: FallbackEvent[],
  mode: Mode,
): void {
  target.replaceChildren()
  const table = mode === 'japanese' ? JAPANESE_TABLE : EUROPEAN_TABLE
  const byPosition = new Map(fallbacks.map((event) => [event.position, event]))
  let plain = ''

  const flush = (): void => {
    if (plain) {
      target.appendChild(document.createTextNode(plain))
      plain = ''
    }
  }

  for (let i = 0; i < text.length; i++) {
    const event = byPosition.get(i)
    if (event === undefined) {
      plain += text[i]
      continue
    }
    flush()
    const span = document.createElement('span')
    // 最尤の文字が引けるのは LOW_CONFIDENCE のときだけ (TABLE_MISS は定義上引けない)
    const guess = event.kind === 'LOW_CONFIDENCE' ? table[event.code] : undefined
    if (guess !== undefined) {
      span.className = 'uncertain'
      span.textContent = guess
      span.title = `確信度不足 (${event.confidence.toFixed(2)}) 符号 ${event.code}`
    } else {
      span.className = 'fallback'
      span.textContent = text[i]
      span.title =
        event.kind === 'LOW_CONFIDENCE'
          ? `確信度不足 (${event.confidence.toFixed(2)}) 符号 ${event.code}`
          : `変換表に該当なし 符号 ${event.code}`
    }
    target.appendChild(span)
  }
  flush()
}

export function mountApp(root: HTMLElement): void {
  root.innerHTML = TEMPLATE

  const toggle = root.querySelector<HTMLButtonElement>('#toggle')!
  const clearButton = root.querySelector<HTMLButtonElement>('#clear')!
  const showJapanese = root.querySelector<HTMLInputElement>('#show-japanese')!
  const bpfCheck = root.querySelector<HTMLInputElement>('#bpf')!
  const status = root.querySelector<HTMLSpanElement>('#status')!
  const level = root.querySelector<HTMLSpanElement>('#level')!
  const diag = root.querySelector<HTMLParagraphElement>('#diag')!
  const canvas = root.querySelector<HTMLCanvasElement>('#spectrum')!
  const squelchSlider = root.querySelector<HTMLInputElement>('#squelch')!
  const squelchValue = root.querySelector<HTMLSpanElement>('#squelch-value')!
  const japaneseSection = root.querySelector<HTMLElement>('#out-japanese')!

  const european = {
    committed: root.querySelector<HTMLSpanElement>('#out-european .committed')!,
    provisional: root.querySelector<HTMLSpanElement>('#out-european .provisional')!,
  }
  const japanese = {
    committed: root.querySelector<HTMLSpanElement>('#out-japanese .committed')!,
    provisional: root.querySelector<HTMLSpanElement>('#out-japanese .provisional')!,
  }

  // スキッシュのスライダ。受信中でも動かせる (ノイズは状況で変わるため)。
  // 下端 (-60) は「切」。値は localStorage に残して次回起動時に復元する。
  squelchSlider.value = String(resolveSquelchDb())
  const applySquelch = (): void => {
    const raw = Number(squelchSlider.value)
    squelchDb = squelchValueToThreshold(raw)
    squelchValue.textContent = squelchDb === null ? '切' : `${raw} dB`
    localStorage.setItem(SQUELCH_STORAGE_KEY, String(raw))
  }
  applySquelch()
  squelchSlider.addEventListener('input', applySquelch)

  let capture: AudioCapture | null = null
  let client: DecoderClient | null = null
  let monitor: SignalMonitor | null = null
  let numThreads = 1
  /** WAV デコード中なら true。2 個目の wav が同時に走るのを防ぐ。 */
  let wavBusy = false

  /** Worker からの結果を画面に反映する (ライブ受信とファイル入力の共通経路)。 */
  const showResult = (response: TextResponse): void => {
    renderInto(european.committed, response.european.committed, response.european.committedFallbacks, 'european')
    renderInto(european.provisional, response.european.provisional, response.european.provisionalFallbacks, 'european')
    renderInto(japanese.committed, response.japanese.committed, response.japanese.committedFallbacks, 'japanese')
    renderInto(japanese.provisional, response.japanese.provisional, response.japanese.provisionalFallbacks, 'japanese')
    diag.textContent =
      `推論 ${response.decodeMs.toFixed(0)} ms / hop ${HOP_S}s / ` +
      `確定遅延 ${COMMIT_LAG_S}s / 改行 ${LINE_BREAK_GAP_S}s / wb ${WORD_BREAK_BIAS} / ` +
      `スレッド ${numThreads} / ` +
      // どのモデルで出た結果かが画面に残らないと、モデルを切り替えて比較したときに
      // 取り違える。診断行に必ず出す
      `モデル ${MODEL_URL.replace('/model/', '').replace('.onnx', '')}`
  }

  showJapanese.addEventListener('change', () => {
    japaneseSection.hidden = !showJapanese.checked
  })

  // BPF は受信中でも切り替えられる (グラフの配線を付け替えるだけ)。実信号で
  // オン/オフの効果を比較するための機能。停止中はチェック状態だけ覚えておき、
  // 次の start() で AudioCapture に渡す。
  bpfCheck.addEventListener('change', () => {
    capture?.setBandpassEnabled(bpfCheck.checked)
  })

  /**
   * 画面のテキストと Worker 側の確定状態をクリアする。
   *
   * 表示だけ消して Worker の `committed` を残すと、次の redecode で消したはずの
   * テキストが丸ごと戻ってくるため、両方をクリアする。
   */
  const clearOutput = (): void => {
    client?.reset()
    for (const span of [
      european.committed,
      european.provisional,
      japanese.committed,
      japanese.provisional,
    ]) {
      span.replaceChildren()
    }
  }

  clearButton.addEventListener('click', clearOutput)

  const stop = async (): Promise<void> => {
    // start() と同様に無効化する。連打で stop() に再入すると capture.stop()
    // (AudioContext.close()) が二重に呼ばれて reject し、void (...) の中で
    // 未処理の rejection になる。
    toggle.disabled = true
    try {
      // finalize の完了を待ってから terminate すること。待たずに terminate すると
      // commitLagS の圏内にある末尾の暫定トークンが確定されないまま Worker が
      // 強制終了され、黙って消える (Worker.terminate() は処理中のメッセージを
      // 即座に破棄する仕様のため)。
      await client?.finalize()
      monitor?.stop()
      await capture?.stop()
    } finally {
      // 途中で例外が起きても Worker を孤児にしないよう、後始末は必ず行う。
      client?.terminate()
      capture = null
      client = null
      monitor = null
      toggle.textContent = '受信開始'
      status.textContent = '停止中'
      toggle.disabled = false
    }
  }

  const start = async (): Promise<void> => {
    toggle.disabled = true
    status.textContent = 'モデル読込中…'
    try {
      numThreads = resolveThreadCount()

      client = await DecoderClient.create({
        modelUrl: MODEL_URL,
        numThreads,
        hopS: HOP_S,
        commitLagS: COMMIT_LAG_S,
        decodeLeftContextS: DECODE_LEFT_CONTEXT_S,
        lineBreakGapS: LINE_BREAK_GAP_S,
        wordBreakBias: WORD_BREAK_BIAS,
        onText: showResult,
        onError: (message) => {
          status.textContent = `エラー: ${message}`
        },
      })

      status.textContent = 'マイク準備中…'
      // BPF 通過後のレベル (デコーダに実際に入る音)。生の入力レベルとは別物で、
      // スキッシュの判定にはこちらを使う。
      let filteredDb = -120
      const squelch = createSquelch()
      let elapsedS = 0
      capture = await AudioCapture.start({
        onAudio: (block) => {
          filteredDb = blockDb(block)
          elapsedS += block.length / SAMPLE_RATE
          client?.pushAudio(squelch(block, elapsedS))
        },
        bandpassEnabled: bpfCheck.checked,
      })

      monitor = new SignalMonitor(canvas, capture.analyser, (dbfs) => {
        const sq = squelchDb === null ? '' : ` / スキッシュ ${squelchDb}`
        level.textContent = `${dbfs.toFixed(1)} (BPF後 ${filteredDb.toFixed(1)}${sq})`
      })
      monitor.start()

      toggle.textContent = '受信停止'
      status.textContent = capture.usedFallbackResampler
        ? `受信中 (${capture.sampleRate} Hz → 8000 Hz 変換)`
        : '受信中 (8000 Hz)'
    } catch (error) {
      status.textContent = `エラー: ${error instanceof Error ? error.message : String(error)}`
      await stop()
    } finally {
      toggle.disabled = false
    }
  }

  const wavInput = root.querySelector<HTMLInputElement>('#wav')!

  /** Worker へ 1 回に送るブロック長 (50 ms 相当)。ライブ受信のブロック長に近い値。 */
  const WAV_BLOCK_SAMPLES = 400
  /**
   * このブロック数ごとに Worker の応答を待つ (= バックプレッシャの単位)。
   *
   * Worker は hop 分の音声が溜まったときだけ redecode して 'text' を返すので、
   * ちょうど 1 hop 分を送ってから 1 回待てば「押し込み速度 <= 推論速度」が
   * 保証される。ブロック単位で待つと最初の 9 ブロックでは 'text' が返らず
   * 止まってしまうため、hop 単位で待つ必要がある。
   *
   * 端数は必ず切り上げる (`Math.ceil`)。`Math.round` だと `HOP_S` の値次第で
   * 切り捨て側に転び、送った量が 1 hop 分に届かないまま待ってしまう
   * (Worker から 'text' が返らず 30 秒タイムアウトで打ち切りになる)。
   * `Math.ceil` なら常に 1 hop 以上を送ってから待つことが構造的に保証される。
   * 現行の `HOP_S = 0.5` では 0.5 * 8000 / 400 = 10 と割り切れるため今は
   * 挙動に差が出ないが、将来 `HOP_S` を変えたときの保険。
   */
  const WAV_BLOCKS_PER_HOP = Math.max(
    1,
    Math.ceil((HOP_S * SAMPLE_RATE) / WAV_BLOCK_SAMPLES),
  )
  /**
   * 1 hop 分の応答を待つ上限。
   *
   * 1 回のフル推論は Task 3 の実測で 1 スレッド 155.5 ms だったので、遅い端末を
   * 大きく見積もってもこの値には届かない。超えたら「Worker が応答しない」と
   * みなして中断する (待ちを打ち切って押し込みを続けると、まさに防ぎたかった
   * 「未デコードの音声が黙って落ちる」状態に戻ってしまうため、続行しない)。
   */
  const WAV_BATCH_TIMEOUT_MS = 30000

  wavInput.addEventListener('change', () => {
    const file = wavInput.files?.[0]
    // input.value は処理後に必ずクリアする。クリアしないと同じファイルを 2 回
    // 続けて選んでも change イベントが発火しない (値が変わらないため)。
    wavInput.value = ''
    if (!file) return
    // デコード中に 2 個目の wav を選ぶと Worker が二重に立ち上がり、両方が同じ
    // showResult を叩いて DOM に交互出力する。
    if (wavBusy) {
      status.textContent = 'WAV をデコード中です。終わるまでお待ちください'
      return
    }
    wavBusy = true
    void (async () => {
      // ライブ受信中なら止める。ライブ用の client / capture とは別の client を
      // 使うので状態は混ざらないが、マイクと WAV を同時に処理させても意味が無い。
      if (capture) await stop()
      status.textContent = `${file.name} を読み込み中…`

      // ライブ受信用の `client` とは別変数にする。使い回すと WAV デコード後に
      // `client` が残ってしまい、その後「受信開始」を押したときの挙動が
      // 不明瞭になる (ライブ用の client は start()/stop() だけが管理する)。
      let wavClient: DecoderClient | null = null
      // 1 hop 分の応答を待っている間だけ非 null。onText / onError が解決する。
      let settleBatch: ((error?: Error) => void) | null = null
      try {
        const { fileToWave8k, planChunks } = await import('../audio/decode-file')
        const wave = await fileToWave8k(file)

        numThreads = resolveThreadCount()
        wavClient = await DecoderClient.create({
          modelUrl: MODEL_URL,
          numThreads,
          hopS: HOP_S,
          commitLagS: COMMIT_LAG_S,
          decodeLeftContextS: DECODE_LEFT_CONTEXT_S,
          lineBreakGapS: LINE_BREAK_GAP_S,
          wordBreakBias: WORD_BREAK_BIAS,
          // ライブ受信と同じ表示経路を通す (二重実装しない)
          onText: (response) => {
            showResult(response)
            settleBatch?.()
          },
          onError: (message) => {
            status.textContent = `エラー: ${message}`
            settleBatch?.(new Error(message))
          },
        })

        /** 直前に送った hop 分が処理されるまで待つ。 */
        const awaitBatch = (): Promise<void> =>
          new Promise<void>((resolve, reject) => {
            const timer = setTimeout(() => {
              settleBatch = null
              reject(new Error('Worker が応答しません (デコードを中断しました)'))
            }, WAV_BATCH_TIMEOUT_MS)
            settleBatch = (error?: Error) => {
              clearTimeout(timer)
              settleBatch = null
              if (error) reject(error)
              else resolve()
            }
          })

        // ライブ受信と同じ経路を通す (hop ごとに push して再デコードさせる)。
        //
        // ★ 押し込みは必ず Worker の応答で駆動すること。以前は setTimeout(0) で
        // 一方的に押し込んでいたが、受け側は redecode 実行中の 'audio' を捨て
        // (decoder-core.ts の `if (this.inFlight) return`)、SlidingWindowDecoder は
        // リング 30 秒を超えた古い音声を無言で切り捨てる。押し込み速度が推論
        // スループットを超えると未デコードの音声が黙って落ちていた。
        let pushed = 0
        // WAV 入力にもスキッシュを適用する (録音ファイルで閾値を試せるように)。
        // **ただし WAV 経路には BPF がかからない**ので、ここで測るレベルは
        // ライブ受信の「BPF 後」ではなく生のレベルである。閾値を移すときは注意。
        const wavSquelch = createSquelch()
        let wavElapsedS = 0
        for (const chunk of planChunks(wave.length, WAV_BLOCK_SAMPLES, 0)) {
          const block = wave.slice(chunk.start, chunk.end)
          wavElapsedS += block.length / SAMPLE_RATE
          wavClient.pushAudio(wavSquelch(block, wavElapsedS))
          pushed++
          // 端数の最終バッチは redecode を起こさない (= 'text' が返らない) ので
          // 待たない。取りこぼしはこの後の finalize() が拾う。
          if (pushed % WAV_BLOCKS_PER_HOP === 0 && chunk.end < wave.length) {
            await awaitBatch()
          }
        }
        // finalize の完了を待ってから terminate すること (stop() と同じ理由:
        // 待たずに terminate すると commitLagS 圏内の末尾トークンが確定されずに消える)。
        await wavClient.finalize()
        const seconds = (wave.length / SAMPLE_RATE).toFixed(1)
        status.textContent = `${file.name} をデコードしました (${seconds} 秒)`
      } catch (error) {
        status.textContent = `エラー: ${error instanceof Error ? error.message : String(error)}`
      } finally {
        settleBatch = null
        // 成功時・失敗時のどちらでも WAV 用の Worker を孤児にしない。
        wavClient?.terminate()
        wavBusy = false
      }
    })()
  })

  toggle.addEventListener('click', () => {
    // WAV デコード中にライブ受信を始めると、2 つの client が同じ showResult を
    // 叩いて DOM に交互出力する (wavInput 側と同じ理由の排他)。
    if (wavBusy) {
      status.textContent = 'WAV をデコード中です。終わるまでお待ちください'
      return
    }
    void (capture ? stop() : start())
  })
}
