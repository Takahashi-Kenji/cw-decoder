/**
 * ORT Web (wasm バックエンド) の Node 版実測ベンチ (設計書 §12 Step 0)。
 *
 * ブラウザは操作できない環境で計測するための代替手段。`onnxruntime-web` は
 * Node.js から import すると package.json の "node" 条件により
 * `dist/ort.node.min.mjs` を解決するが、これはブラウザ版と同じ
 * `ort-wasm-simd-threaded.wasm` バイナリを Node 上で動かすビルドであり、
 * ネイティブアドオン (onnxruntime-node) ではない。したがってここで測る数値は
 * シングルスレッド性能の代理として意味がある。
 *
 * 波形生成の式は web/src/bench.ts (ブラウザ版) と同一にしてあるが、Node から
 * TypeScript を直接 import できないため、ビルド不要なプレーン ESM として
 * 意図的に重複させている (経緯は web/README.md 参照)。
 *
 * 【重要】計測時に見つかった落とし穴:
 *   ORT Web の wasm スレッドプールはプロセス内で最初に `InferenceSession.create()`
 *   を呼んだ時点のスレッド数で固定される。同一プロセス内で
 *   `ort.env.wasm.numThreads` を後から変更しても、既に初期化済みの wasm
 *   ランタイムには反映されない (実測: 同一プロセスで 1→4 の順に計測すると
 *   4 スレッド側も 1 スレッド相当の速度のまま。プロセスを分けて 4 スレッドを
 *   最初から設定すると約 2.7 倍速くなることを確認した)。
 *   ブリーフの `bench.ts` (ブラウザ版) の元コードは同一ページ内で `benchmark(1)` →
 *   `benchmark(4)` の順に呼ぶ構造で同じ落とし穴を踏んでいたため、`bench.ts` 側も
 *   URL クエリパラメータでページ読み込みごとに 1 スレッド数だけ計測する構成に
 *   変更した (詳細は bench.ts 冒頭のコメント参照)。
 *   このスクリプトはスレッド数ごとに **別プロセス** を起動することでこれを回避する。
 *
 * 実行方法:
 *   cd web && npm run bench:node
 *   (内部で `node scripts/bench-node.mjs --threads=1` と `--threads=4` を
 *   それぞれ独立した子プロセスとして実行する)
 */
import * as ort from 'onnxruntime-web'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { spawn } from 'node:child_process'

const SAMPLE_RATE = 8000

/** CW に近いトーンバースト波形を作る (決定的)。web/src/bench.ts の toneWave と同一の式。 */
function toneWave(seconds) {
  const n = Math.floor(SAMPLE_RATE * seconds)
  const wave = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const t = i / SAMPLE_RATE
    const envelope = Math.sin(2 * Math.PI * 4.0 * t) > 0 ? 1 : 0
    wave[i] = Math.sin(2 * Math.PI * 600.0 * t) * envelope
  }
  return wave
}

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const MODEL_PATH = path.join(__dirname, '..', 'public', 'model', 'cw.onnx')
const ORT_WEB_VERSION = JSON.parse(
  readFileSync(path.join(__dirname, '..', 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8')
).version

/** 指定スレッド数で 1 回分の計測を行い、結果を標準出力にログする (このプロセス内で完結させる)。 */
async function benchmark(threads) {
  ort.env.wasm.numThreads = threads
  ort.env.wasm.simd = true

  const t0 = performance.now()
  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  const sessionMs = performance.now() - t0
  console.log(`threads=${threads} セッション構築: ${sessionMs.toFixed(0)} ms`)

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
    console.log(
      `threads=${threads} 音声 ${seconds}s → ${ms.toFixed(1)} ms ` +
        `(RTF=${(ms / 1000 / seconds).toFixed(3)}, frames=${frames})`
    )
  }
  await session.release()
}

/** 子プロセスとして自分自身を `--threads=N` 付きで起動し、完了を待つ。 */
function runInChildProcess(threads) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [__dirnameSelfPath(), `--threads=${threads}`], {
      stdio: 'inherit',
    })
    child.on('exit', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`子プロセス (threads=${threads}) が exit code ${code} で終了しました`))
    })
    child.on('error', reject)
  })
}

function __dirnameSelfPath() {
  return fileURLToPath(import.meta.url)
}

async function main() {
  const arg = process.argv.find((a) => a.startsWith('--threads='))

  if (arg) {
    // 子プロセスとして呼ばれた場合: 単一スレッド数だけ計測してすぐ終了する
    const threads = Number(arg.slice('--threads='.length))
    console.log(`node = ${process.version}`)
    console.log(`onnxruntime-web = ${ORT_WEB_VERSION}`)
    await benchmark(threads)
    return
  }

  // 親プロセス: スレッド数ごとに別プロセスで実行する
  // (同一プロセス内だと wasm ランタイムのスレッドプールが最初の numThreads に
  //  固定され、2 回目以降の numThreads 変更が反映されないため)
  for (const threads of [1, 4]) {
    await runInChildProcess(threads)
  }
}

await main()
