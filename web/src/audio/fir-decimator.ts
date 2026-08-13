/**
 * ステートフルな FIR デシメータ (整数比専用).
 *
 * AudioContext を 8 kHz で作れない環境向けのフォールバック。
 *
 * **状態を保持することが必須**。ステートレスなリサンプルはブロック境界に歪みを
 * 生じ、CW の dot/dash を破壊する (src/infer/audio.py の記録を参照)。
 * 直前ブロックの末尾 numTaps-1 サンプルを保持して連続性を保つ。
 */

/** ローパスの遮断周波数 (目標レートのナイキストの 95%)。 */
const CUTOFF_RATIO = 0.95
/** FIR のタップ数 (奇数。線形位相)。 */
const NUM_TAPS = 121

export class FirDecimator {
  readonly factor: number
  private readonly taps: Float32Array
  /**
   * まだ出力に使い切っていない過去サンプル。
   * 次回の畳み込み窓はこの配列の先頭から始まる (だから phase は不要)。
   */
  private history: Float32Array

  constructor(sourceRate: number, targetRate: number) {
    if (!FirDecimator.supports(sourceRate, targetRate)) {
      throw new Error(`整数比でないレート変換には対応していません: ${sourceRate} → ${targetRate}`)
    }
    this.factor = sourceRate / targetRate
    this.taps = FirDecimator.designLowpass(
      (targetRate / 2) * CUTOFF_RATIO / sourceRate,
      NUM_TAPS,
    )
    this.history = new Float32Array(NUM_TAPS - 1)
  }

  static supports(sourceRate: number, targetRate: number): boolean {
    return Number.isInteger(sourceRate / targetRate) && sourceRate >= targetRate
  }

  /**
   * 窓関数法によるローパス FIR。
   *
   * @param normalizedCutoff 遮断周波数 / サンプリング周波数 (0 < f < 0.5)
   */
  private static designLowpass(normalizedCutoff: number, numTaps: number): Float32Array {
    const taps = new Float32Array(numTaps)
    const center = (numTaps - 1) / 2
    let sum = 0
    for (let i = 0; i < numTaps; i++) {
      const n = i - center
      // sinc
      const sinc =
        n === 0 ? 2 * normalizedCutoff : Math.sin(2 * Math.PI * normalizedCutoff * n) / (Math.PI * n)
      // Hamming 窓
      const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (numTaps - 1))
      taps[i] = sinc * window
      sum += taps[i]
    }
    // 直流利得を 1 に正規化
    for (let i = 0; i < numTaps; i++) taps[i] /= sum
    return taps
  }

  /**
   * 入力ブロックを処理し、間引き済みのサンプルを返す。
   *
   * 「窓が収まる位置まで出力し、残りを丸ごと history に持ち越す」だけで
   * 連続性が保たれる。持ち越し量は numTaps + factor 未満に収まるので増え続けない。
   */
  process(input: Float32Array): Float32Array {
    const taps = this.taps
    const numTaps = taps.length

    // 持ち越し + 今回の入力
    const buffer = new Float32Array(this.history.length + input.length)
    buffer.set(this.history, 0)
    buffer.set(input, this.history.length)

    // 出力できるのは、畳み込み窓が buffer に収まる位置まで
    const nOut = Math.max(0, Math.floor((buffer.length - numTaps) / this.factor) + 1)
    const out = new Float32Array(nOut)
    for (let n = 0; n < nOut; n++) {
      const i = n * this.factor
      let acc = 0
      for (let k = 0; k < numTaps; k++) acc += buffer[i + k] * taps[k]
      out[n] = acc
    }

    // 窓が収まらなかった位置以降を丸ごと持ち越す (次回はここが先頭になる)
    this.history = buffer.slice(nOut * this.factor)
    return out
  }
}
