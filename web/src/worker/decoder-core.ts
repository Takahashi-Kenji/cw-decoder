/**
 * デコーダ Worker のメッセージ処理ロジック本体.
 *
 * `decoder.worker.ts` は `self.onmessage = ...` というトップレベル副作用を
 * 持つため vitest (node 環境) から直接 import できない (`self` が無い)。
 * ここに状態機械としてのロジックを切り出し、`post` (メッセージ送信) と
 * `createEngine` (推論エンジン生成) を差し替え可能にすることで、実 Worker
 * を起動せずにテストできるようにしてある。
 *
 * ★ 重要: redecode (audio 由来) / finalize / reset は同じ SlidingWindowDecoder の
 * 可変状態 (committed / ring / lastCommitEnd) を触る。これらが同時に走ると
 * 状態が壊れる (確定列の重複・消失) ため、`inFlight` で直列化している。
 * - audio: 実行中なら見送る (drop)。キューに溜めると遅延が積み上がるため。
 * - finalize / reset: 実行中なら待つ (await)。捨てると末尾の取りこぼしや
 *   古い状態の混入につながるため。
 */
import { SlidingWindowDecoder, type DecodeEngine, type DecodeView } from '../decode/sliding-window'
import { renderMode } from './render-mode'
// サンプルレートは手書きしない (MelConfig.sample_rate から生成)
import { SAMPLE_RATE } from '../generated/tokens'
import type { WorkerRequest, WorkerResponse } from './protocol'

/** モデル URL とスレッド数から推論エンジンを作る。実 Worker では OnnxDecodeEngine.create、テストではフェイクを渡す。 */
export type EngineFactory = (
  modelUrl: string,
  numThreads: number,
  wordBreakBias: number,
) => Promise<DecodeEngine>

export class DecoderWorkerCore {
  private decoder: SlidingWindowDecoder | null = null
  private hopSamples = SAMPLE_RATE
  /** 改行を入れる無音の長さ (秒)。init で受け取る。 */
  private lineBreakGapS = 0
  private samplesSinceRedecode = 0
  /** 実行中の redecode/finalize 処理。非 null の間は他の処理が同じ decoder を触ってはいけない。 */
  private inFlight: Promise<void> | null = null

  constructor(
    private readonly post: (message: WorkerResponse) => void,
    private readonly createEngine: EngineFactory,
  ) {}

  async handle(request: WorkerRequest): Promise<void> {
    try {
      switch (request.type) {
        case 'init': {
          const engine = await this.createEngine(
            request.modelUrl, request.numThreads, request.wordBreakBias,
          )
          this.decoder = new SlidingWindowDecoder(engine, {
            hopS: request.hopS,
            commitLagS: request.commitLagS,
            decodeLeftContextS: request.decodeLeftContextS,
            sampleRate: SAMPLE_RATE,
          })
          this.lineBreakGapS = request.lineBreakGapS
          this.hopSamples = Math.floor(request.hopS * SAMPLE_RATE)
          this.samplesSinceRedecode = 0
          this.post({ type: 'ready' })
          break
        }
        case 'audio': {
          if (!this.decoder) return
          this.decoder.push(request.block)
          this.samplesSinceRedecode += request.block.length
          if (this.samplesSinceRedecode < this.hopSamples) return
          // 実行中 (redecode/finalize/reset) なら今回は見送る (キューを溜めない)
          if (this.inFlight) return
          this.samplesSinceRedecode = 0
          await this.runDecode(() => this.decoder!.redecode())
          break
        }
        case 'finalize': {
          // decoder が無くても 'finalized' は返す。メインスレッド側は応答を
          // 待って terminate() するため、返さないと待ちが timeout 頼みになる。
          if (!this.decoder) {
            this.post({ type: 'finalized' })
            return
          }
          // 実行中の redecode を待ってから確定する。捨てると末尾の取りこぼしになる。
          // (エラーはここでは無視する。元の処理側の catch が既に error を報告している)
          if (this.inFlight) await this.inFlight.catch(() => {})
          await this.runDecode(() => this.decoder!.finalize())
          // 完了をメインスレッドへ知らせる (DecoderClient.finalize が待っている)。
          this.post({ type: 'finalized' })
          break
        }
        case 'reset': {
          // 実行中の redecode/finalize の書き込みが終わるまで待ってからクリアする。
          // 待たずに reset すると、後から届く書き込みが reset 直後の状態を汚す。
          if (this.inFlight) await this.inFlight.catch(() => {})
          this.decoder?.reset()
          this.samplesSinceRedecode = 0
          break
        }
        case 'setMode':
          // 表示モードはメインスレッド側で選ぶ (両方を毎回返しているため何もしない)
          break
      }
    } catch (error) {
      this.post({ type: 'error', message: error instanceof Error ? error.message : String(error) })
    }
  }

  /**
   * decoder.redecode()/finalize() を実行し、結果を post する。
   * 実行中は `inFlight` に登録し、他の呼び出しがブロックできるようにする。
   */
  private async runDecode(task: () => Promise<DecodeView>): Promise<void> {
    const promise = (async () => {
      const start = performance.now()
      const view = await task()
      this.emit(view, performance.now() - start)
    })()
    this.inFlight = promise
    try {
      await promise
    } finally {
      if (this.inFlight === promise) this.inFlight = null
    }
  }

  private emit(view: DecodeView, decodeMs: number): void {
    this.post({
      type: 'text',
      european: renderMode(view, 'european', this.lineBreakGapS),
      japanese: renderMode(view, 'japanese', this.lineBreakGapS),
      decodeMs,
    })
  }
}
