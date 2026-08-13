import { describe, expect, it } from 'vitest'
import { SlidingWindowDecoder, type DecodeEngine } from '../src/decode/sliding-window'
import type { FrameToken } from '../src/decode/ctc'

const SAMPLE_RATE = 8000
const HOP = 80 // 1 フレーム = 80 サンプル = 10 ms

/**
 * 投入された区間の長さに応じて 0.5 秒おきにトークンを返すフェイク。
 * decodeChunk に渡された波形の先頭を時刻 0 とみなす。
 */
class FakeEngine implements DecodeEngine {
  frameHopSamples = HOP
  lastChunkLength = 0

  async decodeChunk(wave: Float32Array): Promise<FrameToken[]> {
    this.lastChunkLength = wave.length
    const nFrames = Math.floor(wave.length / HOP)
    const out: FrameToken[] = []
    for (let start = 0; start + 5 < nFrames; start += 50) {
      out.push({
        tokenId: 1 + ((start / 50) % 10),
        confidence: 0.9,
        frameStart: start,
        frameEnd: start + 4,
      })
    }
    return out
  }
}

function silence(seconds: number): Float32Array {
  return new Float32Array(Math.floor(seconds * SAMPLE_RATE))
}

/**
 * 呼び出しごとに固定のトークン列を順番に返すフェイク。中点ウォーターマークの
 * 境界 (jitterMarginSamples の内側/外側、Math.floor の端数切り捨て) を
 * 1 サンプル単位で狙い撃ちするため、frameHopSamples を 1 にして
 * frameStart/frameEnd をそのまま絶対フレーム位置として扱えるようにしている。
 * decodeStartAbs が 0 のまま推移するよう、テスト側で push 量を
 * leftContextSamples (既定 5 秒 = 40000 サンプル) 未満に抑えている。
 */
class ScriptedEngine implements DecodeEngine {
  frameHopSamples = 1
  private callIndex = 0

  constructor(private readonly script: FrameToken[][]) {}

  async decodeChunk(_wave: Float32Array): Promise<FrameToken[]> {
    const tokens = this.script[Math.min(this.callIndex, this.script.length - 1)]
    this.callIndex++
    return tokens
  }
}

function zeros(n: number): Float32Array {
  return new Float32Array(n)
}

describe('SlidingWindowDecoder', () => {
  it('音声が無ければ空のビューを返す', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    const view = await decoder.redecode()
    expect(view.committed).toEqual([])
    expect(view.provisional).toEqual([])
  })

  it('commit_lag 圏内のトークンは暫定になる', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine(), { commitLagS: 2.5 })
    decoder.push(silence(6))
    const view = await decoder.redecode()
    expect(view.provisional.length).toBeGreaterThan(0)
    // 暫定は必ず「今 - commit_lag」より後ろで終わる
    const limit = 6 * SAMPLE_RATE - 2.5 * SAMPLE_RATE
    for (const token of view.provisional) {
      expect(token.absoluteSampleEnd).toBeGreaterThanOrEqual(limit)
    }
  })

  it('確定済みトークンは後から変化しない', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    decoder.push(silence(6))
    const first = await decoder.redecode()
    const snapshot = JSON.stringify(first.committed)

    decoder.push(silence(2))
    const second = await decoder.redecode()
    expect(JSON.stringify(second.committed.slice(0, first.committed.length))).toBe(snapshot)
  })

  it('同じトークンを二重に確定しない', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    for (let i = 0; i < 8; i++) {
      decoder.push(silence(1))
      await decoder.redecode()
    }
    const view = await decoder.redecode()
    const ends = view.committed.map((t) => t.absoluteSampleEnd)
    expect(new Set(ends).size).toBe(ends.length)
    // 絶対位置は単調増加
    for (let i = 1; i < ends.length; i++) {
      expect(ends[i]).toBeGreaterThan(ends[i - 1])
    }
  })

  it('finalize は残った暫定をすべて確定する', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    decoder.push(silence(6))
    const before = await decoder.redecode()
    const view = await decoder.finalize()
    expect(view.provisional).toEqual([])
    expect(view.committed.length).toBeGreaterThan(before.committed.length)
  })

  it('reset で状態が消える', async () => {
    const decoder = new SlidingWindowDecoder(new FakeEngine())
    decoder.push(silence(6))
    await decoder.redecode()
    decoder.reset()
    const view = await decoder.redecode()
    expect(view.committed).toEqual([])
  })

  it('リングバッファは window_s を超えない', async () => {
    const engine = new FakeEngine()
    const decoder = new SlidingWindowDecoder(engine, { windowS: 5, decodeLeftContextS: 5 })
    decoder.push(silence(20))
    await decoder.redecode()
    expect(engine.lastChunkLength).toBeLessThanOrEqual(5 * SAMPLE_RATE)
  })
})

// 中点ウォーターマークの境界 (jitterMarginSamples = 既定 0.02s * 8000Hz = 160 サンプル) を
// 1 サンプル単位で検証する。既定オプションのまま (commitLagSamples=20000,
// leftContextSamples=40000, headGuardSamples=8000) decodeStartAbs が終始 0 になるよう
// push 量を調整しているので、frameStart/frameEnd (hop=1) がそのまま絶対サンプル位置になる。
describe('SlidingWindowDecoder の中点ウォーターマーク境界', () => {
  it('ジッタ幅内で再出現したトークンは二重確定しない (テストA)', async () => {
    const engine = new ScriptedEngine([
      // 1 回目: 最初の確定トークン (absStart=1000, absEnd=1100) → lastCommitEnd=1100
      [{ tokenId: 1, confidence: 0.9, frameStart: 1000, frameEnd: 1100 }],
      // 2 回目: 左文脈が少しずれて再出現した想定 (absStart=1150, absEnd=1250)。
      // 中点=1200 は lastEnd(1100) + jitterMargin(160) = 1260 以内 → 二重確定してはいけない。
      [{ tokenId: 1, confidence: 0.9, frameStart: 1150, frameEnd: 1250 }],
    ])
    const decoder = new SlidingWindowDecoder(engine)

    decoder.push(zeros(25000))
    const first = await decoder.redecode()
    expect(first.committed.length).toBe(1)

    decoder.push(zeros(5000))
    const second = await decoder.redecode()
    expect(second.newlyCommitted.length).toBe(0)
    expect(second.committed.length).toBe(1)
  })

  it('ジッタ幅ぎりぎり外側の新規トークンは脱落しない (テストB)', async () => {
    const engine = new ScriptedEngine([
      // 1 回目: 最初の確定トークン (absStart=1000, absEnd=1100) → lastCommitEnd=1100
      [{ tokenId: 1, confidence: 0.9, frameStart: 1000, frameEnd: 1100 }],
      // 2 回目: 中点=1261 は lastEnd(1100) + jitterMargin(160) + 1 = 1261。
      // 境界のちょうど外側なので新規確定されなければならない。
      [{ tokenId: 2, confidence: 0.9, frameStart: 1211, frameEnd: 1311 }],
    ])
    const decoder = new SlidingWindowDecoder(engine)

    decoder.push(zeros(25000))
    await decoder.redecode()

    decoder.push(zeros(5000))
    const second = await decoder.redecode()
    expect(second.newlyCommitted.length).toBe(1)
    expect(second.committed.length).toBe(2)
  })

  it('中点は端数を切り捨てる (Math.floor がないと誤って確定される) (おまけ)', async () => {
    const engine = new ScriptedEngine([
      // 1 回目: 最初の確定トークン (absStart=790, absEnd=890) → lastCommitEnd=890
      [{ tokenId: 1, confidence: 0.9, frameStart: 790, frameEnd: 890 }],
      // 2 回目: absStart+absEnd=2101 (奇数) → 真の中点は 1050.5。
      // floor(1050.5)=1050 は lastEnd(890)+jitterMargin(160)=1050 以内なので二重確定してはいけない。
      // floor を外すと 1050.5 > 1050 になり誤って新規確定されてしまう。
      [{ tokenId: 1, confidence: 0.9, frameStart: 1000, frameEnd: 1101 }],
    ])
    const decoder = new SlidingWindowDecoder(engine)

    decoder.push(zeros(25000))
    await decoder.redecode()

    decoder.push(zeros(5000))
    const second = await decoder.redecode()
    expect(second.newlyCommitted.length).toBe(0)
    expect(second.committed.length).toBe(1)
  })
})
