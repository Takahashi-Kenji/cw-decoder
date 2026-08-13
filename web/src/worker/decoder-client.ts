/** Decoder Worker をメインスレッドから扱うためのラッパ. */
import type { WorkerRequest, WorkerResponse } from './protocol'

type TextResponse = Extract<WorkerResponse, { type: 'text' }>

export interface DecoderClientOptions {
  modelUrl: string
  numThreads: number
  commitLagS: number
  hopS: number
  decodeLeftContextS: number
  /** 改行を入れる無音の長さ (秒)。0 以下なら改行しない。 */
  lineBreakGapS: number
  /** WORD_BREAK のロジットに足す下駄。0 なら従来どおり。 */
  wordBreakBias: number
  onText: (response: TextResponse) => void
  onError: (message: string) => void
}

// finalize() が Worker からの 'finalized' 応答を待つ上限。
// Task 3 の実測で 1 スレッドでも 1 回のフル推論は最大 155.5 ms 程度だったため、
// 端末差やモデル読込み直後の揺らぎを見込んでも十分すぎる余裕を持たせた。
// これを超えたら「Worker が応答しない」とみなし、末尾の確定を諦めて
// (止まり続けるよりまし) 先に進む。
const FINALIZE_TIMEOUT_MS = 3000

export class DecoderClient {
  /** 実行中の finalize() 呼び出しを解決する関数。同時に 1 個しか無い前提。 */
  private pendingFinalize: (() => void) | null = null

  private constructor(private readonly worker: Worker) {}

  static create(options: DecoderClientOptions): Promise<DecoderClient> {
    const worker = new Worker(new URL('./decoder.worker.ts', import.meta.url), {
      type: 'module',
    })
    return new Promise((resolve, reject) => {
      // init 完了 (= 'ready') より前に失敗したら Worker を孤児にしないよう
      // terminate してから reject する。'ready' 後の失敗は呼び出し元
      // (app.ts の stop()) が client.terminate() で後始末する。
      let client: DecoderClient | undefined

      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const response = event.data
        switch (response.type) {
          case 'ready':
            client = new DecoderClient(worker)
            resolve(client)
            break
          case 'text':
            options.onText(response)
            break
          case 'finalized':
            client?.settleFinalize()
            break
          case 'error':
            options.onError(response.message)
            if (!client) worker.terminate()
            reject(new Error(response.message))
            break
        }
      }
      worker.onerror = (event) => {
        options.onError(event.message)
        if (!client) worker.terminate()
        reject(new Error(event.message))
      }
      const init: WorkerRequest = {
        type: 'init',
        modelUrl: options.modelUrl,
        numThreads: options.numThreads,
        commitLagS: options.commitLagS,
        hopS: options.hopS,
        decodeLeftContextS: options.decodeLeftContextS,
        lineBreakGapS: options.lineBreakGapS,
        wordBreakBias: options.wordBreakBias,
      }
      worker.postMessage(init)
    })
  }

  pushAudio(block: Float32Array): void {
    const message: WorkerRequest = { type: 'audio', block }
    // 転送してコピーを避ける
    this.worker.postMessage(message, [block.buffer])
  }

  /**
   * Worker 側の finalize (実行中の redecode 待ち + フル再推論) の完了を待つ。
   *
   * terminate() の前に必ずこれを await すること。Worker.terminate() は仕様上
   * 処理中のメッセージを即座に破棄するため、finalize の推論が終わる前に
   * terminate すると commitLagS の圏内にあった末尾の暫定トークンが確定される
   * ことなく黙って消える。
   */
  finalize(): Promise<void> {
    this.worker.postMessage({ type: 'finalize' } satisfies WorkerRequest)
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        this.pendingFinalize = null
        resolve()
      }, FINALIZE_TIMEOUT_MS)
      this.pendingFinalize = () => {
        clearTimeout(timer)
        this.pendingFinalize = null
        resolve()
      }
    })
  }

  /** 'finalized' 応答を受けて finalize() の Promise を解決する。 */
  private settleFinalize(): void {
    this.pendingFinalize?.()
  }

  reset(): void {
    this.worker.postMessage({ type: 'reset' } satisfies WorkerRequest)
  }

  terminate(): void {
    this.worker.terminate()
  }
}
