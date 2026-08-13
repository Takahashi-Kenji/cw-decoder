import { describe, expect, it } from 'vitest'
import { planChunks } from '../src/audio/decode-file'

describe('planChunks', () => {
  it('短い入力は 1 チャンク', () => {
    expect(planChunks(1000, 8000, 800)).toEqual([{ start: 0, end: 1000 }])
  })

  it('全域を覆う', () => {
    const chunks = planChunks(20000, 8000, 800)
    expect(chunks[0].start).toBe(0)
    expect(chunks[chunks.length - 1].end).toBe(20000)
    // 隣接チャンクに隙間が無い
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i].start).toBeLessThan(chunks[i - 1].end)
    }
  })

  it('チャンクが overlap ぶん重なる', () => {
    const chunks = planChunks(20000, 8000, 800)
    expect(chunks[1].start).toBe(8000 - 800)
  })

  it('長さ 0 なら空', () => {
    expect(planChunks(0, 8000, 800)).toEqual([])
  })
})
