/**
 * 信号モニタ: スペクトル表示と入力レベルメータ.
 *
 * CW のトーン周波数が見えるので同調確認に使える。実運用では
 * 「音が届いているか」「周波数が合っているか」の切り分けにこれが要る。
 */
export class SignalMonitor {
  // TS 5.9 で TypedArray が ArrayBuffer 種別についてジェネリック化された影響で、
  // 素の `Uint8Array` 型注釈は `Uint8Array<ArrayBufferLike>` に解決されてしまい、
  // AnalyserNode の各メソッドが要求する `Uint8Array<ArrayBuffer>` と食い違う。
  // コンストラクタの戻り値の型に合わせて明示する。
  private readonly bins: Uint8Array<ArrayBuffer>
  private readonly timeDomain: Float32Array<ArrayBuffer>
  private rafId: number | null = null

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly analyser: AnalyserNode,
    private readonly onLevel: (dbfs: number) => void,
  ) {
    this.bins = new Uint8Array(analyser.frequencyBinCount)
    this.timeDomain = new Float32Array(analyser.fftSize)
  }

  start(): void {
    if (this.rafId !== null) return
    const draw = (): void => {
      this.render()
      this.rafId = requestAnimationFrame(draw)
    }
    this.rafId = requestAnimationFrame(draw)
  }

  stop(): void {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId)
    this.rafId = null
  }

  private render(): void {
    const ctx = this.canvas.getContext('2d')
    if (!ctx) return

    this.analyser.getByteFrequencyData(this.bins)
    this.analyser.getFloatTimeDomainData(this.timeDomain)

    let sumSquares = 0
    for (const v of this.timeDomain) sumSquares += v * v
    const rms = Math.sqrt(sumSquares / this.timeDomain.length)
    this.onLevel(rms < 1e-6 ? -120 : 20 * Math.log10(rms))

    const { width, height } = this.canvas
    ctx.clearRect(0, 0, width, height)

    // 8 kHz の AudioContext なので bin 全体が 0〜4 kHz に対応する
    const barWidth = width / this.bins.length
    ctx.fillStyle = '#2f9e6f'
    for (let i = 0; i < this.bins.length; i++) {
      const barHeight = (this.bins[i] / 255) * height
      ctx.fillRect(i * barWidth, height - barHeight, Math.max(1, barWidth), barHeight)
    }

    // 1 kHz ごとの目盛り
    ctx.fillStyle = '#888'
    ctx.font = '10px sans-serif'
    for (let khz = 1; khz < 4; khz++) {
      const x = (khz / 4) * width
      ctx.fillRect(x, 0, 1, height)
      ctx.fillText(`${khz}k`, x + 2, 10)
    }
  }
}
