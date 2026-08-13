import { describe, expect, it } from 'vitest'
import { FirDecimator } from '../src/audio/fir-decimator'

/** 指定周波数の正弦波を作る。 */
function sine(freq: number, rate: number, seconds: number, offset = 0): Float32Array {
  const n = Math.floor(rate * seconds)
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = Math.sin((2 * Math.PI * freq * (i + offset)) / rate)
  return out
}

function rms(x: Float32Array): number {
  let sum = 0
  for (const v of x) sum += v * v
  return Math.sqrt(sum / Math.max(1, x.length))
}

describe('FirDecimator', () => {
  it('整数比のみ対応する', () => {
    expect(FirDecimator.supports(48000, 8000)).toBe(true)
    expect(FirDecimator.supports(16000, 8000)).toBe(true)
    expect(FirDecimator.supports(44100, 8000)).toBe(false)
  })

  it('出力長が入力長の 1/factor 前後になる', () => {
    const dec = new FirDecimator(48000, 8000)
    expect(dec.factor).toBe(6)
    const out = dec.process(sine(600, 48000, 1.0))
    expect(out.length).toBeGreaterThan(8000 * 0.95)
    expect(out.length).toBeLessThanOrEqual(8000)
  })

  it('通過帯域の信号は振幅をほぼ保つ', () => {
    const dec = new FirDecimator(48000, 8000)
    const out = dec.process(sine(600, 48000, 1.0))
    // 先頭は FIR の立ち上がりなので後半で評価する
    const tail = out.subarray(Math.floor(out.length / 2))
    expect(rms(tail)).toBeGreaterThan(0.6)
  })

  it('ナイキストを超える信号を減衰させる', () => {
    const dec = new FirDecimator(48000, 8000)
    // 8 kHz 側のナイキストは 4 kHz。9 kHz は折り返す前に落ちるべき
    const out = dec.process(sine(9000, 48000, 1.0))
    const tail = out.subarray(Math.floor(out.length / 2))
    expect(rms(tail)).toBeLessThan(0.05)
  })

  it('ブロック分割しても連続処理と一致する (状態を保持する)', () => {
    const full = sine(600, 48000, 0.5)

    const whole = new FirDecimator(48000, 8000).process(full)

    const streamed = new FirDecimator(48000, 8000)
    const chunks: Float32Array[] = []
    const blockSize = 480
    for (let i = 0; i < full.length; i += blockSize) {
      chunks.push(streamed.process(full.subarray(i, i + blockSize)))
    }
    const joined = new Float32Array(chunks.reduce((n, c) => n + c.length, 0))
    let offset = 0
    for (const c of chunks) {
      joined.set(c, offset)
      offset += c.length
    }

    expect(joined.length).toBe(whole.length)
    for (let i = 0; i < whole.length; i++) {
      expect(joined[i]).toBeCloseTo(whole[i], 5)
    }
  })

  it('端数の出るブロック分割でも連続処理と一致する (状態を保持する)', () => {
    // 480 は 24000 (0.5 秒 * 48000) をちょうど割り切ってしまい、末尾に端数ブロックが
    // 来る場合の継続性を検証できない。AudioWorklet は 128 サンプル単位で呼ばれ、
    // それを 400 サンプルに束ねるため、実際には割り切れないブロック境界も起こり得る。
    // ここでは意図的に 24000 を割り切らないブロックサイズを使う。
    const full = sine(600, 48000, 0.5)
    expect(full.length % 377).not.toBe(0)

    const whole = new FirDecimator(48000, 8000).process(full)

    const streamed = new FirDecimator(48000, 8000)
    const chunks: Float32Array[] = []
    const blockSize = 377
    for (let i = 0; i < full.length; i += blockSize) {
      chunks.push(streamed.process(full.subarray(i, i + blockSize)))
    }
    const joined = new Float32Array(chunks.reduce((n, c) => n + c.length, 0))
    let offset = 0
    for (const c of chunks) {
      joined.set(c, offset)
      offset += c.length
    }

    expect(joined.length).toBe(whole.length)
    for (let i = 0; i < whole.length; i++) {
      expect(joined[i]).toBeCloseTo(whole[i], 5)
    }
  })
})
