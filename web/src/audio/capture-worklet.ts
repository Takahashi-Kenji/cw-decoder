/**
 * 音声取得用の AudioWorkletProcessor.
 *
 * 128 サンプル単位で呼ばれるので、BLOCK_SAMPLES ぶん溜めてから
 * メインスレッドへ postMessage する (メッセージ数を抑える)。
 *
 * このファイルは AudioWorkletGlobalScope で動くため、import は使えない。
 * capture.ts から Blob URL 経由で読み込む。
 */
const WORKLET_SOURCE = `
const BLOCK_SAMPLES = 400   // 8 kHz で 0.05 秒

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.buffer = new Float32Array(BLOCK_SAMPLES)
    this.filled = 0
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || input.length === 0) return true
    const channel = input[0]
    if (!channel) return true

    for (let i = 0; i < channel.length; i++) {
      this.buffer[this.filled++] = channel[i]
      if (this.filled === BLOCK_SAMPLES) {
        const out = this.buffer.slice()
        this.port.postMessage(out, [out.buffer])
        this.buffer = new Float32Array(BLOCK_SAMPLES)
        this.filled = 0
      }
    }
    return true
  }
}

registerProcessor('cw-capture', CaptureProcessor)
`

/** Worklet のソースを Blob URL にして返す (別ファイル配信を不要にする)。 */
export function captureWorkletUrl(): string {
  return URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'application/javascript' }))
}
