/**
 * ORT Web の実測ベンチ (設計書 §12 Step 0).
 *
 * 確認すること:
 *   (a) ORT Web の WASM で LSTM が動くか
 *   (b) 1 回のデコードが 0.2〜0.45 秒に収まるか (設計書 §3 の見積もり)
 *   (c) スレッド数による差
 *
 * Node 版のベンチは scripts/bench-node.mjs を参照。波形生成の式はそちらと
 * 揃えてある (重複の扱いは web/README.md に記載)。
 *
 * 【ブリーフからの変更点】ブリーフの元コードは同一ページ内で benchmark(1) の
 * 直後に benchmark(4) を呼ぶ構造だったが、ORT Web の wasm スレッドプールは
 * プロセス (= ページ) 内で最初の InferenceSession.create() 時点の numThreads
 * に固定され、後から ort.env.wasm.numThreads を変えても反映されないことを
 * Node 版ベンチの計測中に確認した (docs/browser_ort_bench.md 参照)。そのため
 * 1 回のページ読み込みで 1 スレッド数だけを計測する構成に変更し、
 * URL クエリパラメータ `?threads=N` でどのスレッド数を計測するか選べるようにした。
 * 1 スレッドと 4 スレッドを両方測るには、ページを 2 回 (別タブ/別読み込みで)
 * 開き直すこと (例: /index.html?threads=1 と /index.html?threads=4)。
 */
import * as ort from 'onnxruntime-web'

const SAMPLE_RATE = 8000

/** CW に近いトーンバースト波形を作る (決定的). */
function toneWave(seconds: number): Float32Array {
  const n = Math.floor(SAMPLE_RATE * seconds)
  const wave = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE
    const envelope = Math.sin(2 * Math.PI * 4.0 * t) > 0 ? 1 : 0
    wave[i] = Math.sin(2 * Math.PI * 600.0 * t) * envelope
  }
  return wave
}

async function benchmark(threads: number): Promise<void> {
  ort.env.wasm.numThreads = threads
  ort.env.wasm.simd = true

  const t0 = performance.now()
  const session = await ort.InferenceSession.create('/model/cw.onnx', {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  log(`threads=${threads} セッション構築: ${(performance.now() - t0).toFixed(0)} ms`)

  for (const seconds of [5.0, 8.5]) {
    const wave = toneWave(seconds)
    const feed = { wave: new ort.Tensor('float32', wave, [1, wave.length]) }

    // ウォームアップ
    await session.run(feed)

    const runs = 5
    const start = performance.now()
    let frames = 0
    for (let i = 0; i < runs; i++) {
      const out = await session.run(feed)
      frames = out.log_probs.dims[1]
    }
    const ms = (performance.now() - start) / runs
    log(
      `threads=${threads} 音声 ${seconds}s → ${ms.toFixed(1)} ms ` +
        `(RTF=${(ms / 1000 / seconds).toFixed(3)}, frames=${frames})`
    )
  }
  await session.release()
}

function log(message: string): void {
  console.log(message)
  const el = document.getElementById('app')
  if (el) el.innerHTML += `<pre>${message}</pre>`
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search)
  const threads = Number(params.get('threads') ?? '1') || 1

  log(`crossOriginIsolated = ${globalThis.crossOriginIsolated}`)
  log(`hardwareConcurrency = ${navigator.hardwareConcurrency}`)
  log(`threads クエリパラメータ = ${threads}`)

  if (threads > 1 && !globalThis.crossOriginIsolated) {
    log('crossOriginIsolated=false のためマルチスレッドは計測できません (?threads=1 で再読み込みしてください)')
    return
  }

  await benchmark(threads)

  log(
    threads === 1
      ? '4 スレッドも計測するには、このページを ?threads=4 を付けて開き直してください (同一ページ内での連続計測は wasm スレッドプールの制約により無効になります)'
      : '他のスレッド数も計測するには、このページを ?threads=1 等を付けて開き直してください'
  )
}

void main()
