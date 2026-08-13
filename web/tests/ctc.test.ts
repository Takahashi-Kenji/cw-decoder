import { describe, expect, it } from 'vitest'
import { ctcGreedyDecodeWithFrames } from '../src/decode/ctc'
import { BLANK_TOKEN_ID, VOCAB_SIZE, WORD_BREAK_TOKEN_ID } from '../src/generated/tokens'

const VOCAB = 4
const BLANK = 0

/** フレームごとの「最大確率のトークン」から log_probs を組み立てる補助. */
function buildLogProbs(frames: Array<[number, number]>): Float32Array {
  const out = new Float32Array(frames.length * VOCAB).fill(Math.log(0.01))
  frames.forEach(([tokenId, prob], t) => {
    out[t * VOCAB + tokenId] = Math.log(prob)
  })
  return out
}

describe('ctcGreedyDecodeWithFrames', () => {
  it('blank だけなら何も出さない', () => {
    const logProbs = buildLogProbs([[BLANK, 0.9], [BLANK, 0.9]])
    expect(ctcGreedyDecodeWithFrames(logProbs, 2, VOCAB)).toEqual([])
  })

  it('連続する同一トークンを 1 つに畳む', () => {
    const logProbs = buildLogProbs([[1, 0.8], [1, 0.9], [1, 0.7]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 3, VOCAB)
    expect(out).toHaveLength(1)
    expect(out[0].tokenId).toBe(1)
    expect(out[0].frameStart).toBe(0)
    expect(out[0].frameEnd).toBe(2)
    // ラン内の最大確率を confidence とする
    expect(out[0].confidence).toBeCloseTo(0.9, 5)
  })

  it('blank を挟んだ同一トークンは 2 つに分かれる', () => {
    const logProbs = buildLogProbs([[1, 0.8], [BLANK, 0.9], [1, 0.7]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 3, VOCAB)
    expect(out.map((t) => t.tokenId)).toEqual([1, 1])
    expect(out[0].frameEnd).toBe(0)
    expect(out[1].frameStart).toBe(2)
  })

  it('末尾のトークンも取りこぼさない', () => {
    const logProbs = buildLogProbs([[BLANK, 0.9], [2, 0.8]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 2, VOCAB)
    expect(out.map((t) => t.tokenId)).toEqual([2])
    expect(out[0].frameEnd).toBe(1)
  })

  it('異なるトークンが並ぶ場合は境界を正しく取る', () => {
    const logProbs = buildLogProbs([[1, 0.8], [2, 0.9], [3, 0.7]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 3, VOCAB)
    expect(out.map((t) => t.tokenId)).toEqual([1, 2, 3])
    expect(out.map((t) => t.frameStart)).toEqual([0, 1, 2])
    expect(out.map((t) => t.frameEnd)).toEqual([0, 1, 2])
  })

  it('フレーム 0 件なら空', () => {
    expect(ctcGreedyDecodeWithFrames(new Float32Array(0), 0, VOCAB)).toEqual([])
  })

  it('同値トークンの場合は小さいインデックスを選ぶ (Python版との同値性確認)', () => {
    // token 1 と token 2 が同じ log 確率を持つフレーム。
    // 動作: value > bestLogProb の厳密な > により、
    // token 1 で best=1, bestLogProb=log(0.5) となり、
    // token 2 でも value=log(0.5) だが > ではなく = なので更新されず。
    // 結果: token 1 が選ばれる (Python の torch.argmax と同じ)
    const logProbs = new Float32Array([
      Math.log(0.01), // token 0
      Math.log(0.5),  // token 1 (最初の最大)
      Math.log(0.5),  // token 2 (同値、選ばれない)
      Math.log(0.01), // token 3
    ])
    const out = ctcGreedyDecodeWithFrames(logProbs, 1, 4)
    expect(out).toHaveLength(1)
    expect(out[0].tokenId).toBe(1) // token 2 ではなく token 1
  })

  it('blankId を上書きしたとき、指定した ID が blank として扱われる', () => {
    // VOCAB=4, blankId=2 を指定。
    // フレーム: token 1 (0.8), blank 2 (0.9), token 1 (0.7)
    // blankId=0 (既定) なら token 2 は出力されるが、
    // blankId=2 なら token 2 は blank として扱われ、
    // token 1 が blank で分割される (2 つの別ラン)。
    const logProbs = buildLogProbs([[1, 0.8], [2, 0.9], [1, 0.7]])
    const out = ctcGreedyDecodeWithFrames(logProbs, 3, VOCAB, 2)
    expect(out).toHaveLength(2)
    expect(out.map((t) => t.tokenId)).toEqual([1, 1])
    expect(out[0].frameStart).toBe(0)
    expect(out[0].frameEnd).toBe(0)
    expect(out[1].frameStart).toBe(2)
    expect(out[1].frameEnd).toBe(2)
  })

  it('logProbs が nFrames * vocabSize より短ければ例外', () => {
    // 検査が無いと `logProbs[base + v]` が undefined になり、比較が全部 false に
    // なって「全フレームが blank」= 空配列が返る。無音と区別が付かず、
    // 呼び出し側は静かに全文を失う (Python 版は ValueError を投げる)。
    const logProbs = new Float32Array(VOCAB * 2) // 2 フレーム分しかない
    expect(() => ctcGreedyDecodeWithFrames(logProbs, 3, VOCAB)).toThrow()
    // ぴったり足りていれば通ること (境界)
    expect(() => ctcGreedyDecodeWithFrames(logProbs, 2, VOCAB)).not.toThrow()
  })

  it('vocabSize が 0 以下なら例外', () => {
    expect(() => ctcGreedyDecodeWithFrames(new Float32Array(4), 1, 0)).toThrow()
  })
})

// ============================================================
// WORD_BREAK の下駄 (wordBreakBias)
// ============================================================
describe('wordBreakBias', () => {
  const V = VOCAB_SIZE

  /** 指定フレームの指定トークンだけを高くした log_probs を作る。 */
  function frames(winners: readonly number[], margin = 1.0): Float32Array {
    const out = new Float32Array(winners.length * V).fill(-10)
    winners.forEach((w, t) => {
      out[t * V + w] = -1.0
      // WORD_BREAK は僅差の 2 位にしておく (下駄で逆転しうる位置)
      if (w !== WORD_BREAK_TOKEN_ID) out[t * V + WORD_BREAK_TOKEN_ID] = -1.0 - margin
    })
    return out
  }

  it('既定 (0) では従来と同じ結果になる', () => {
    const lp = frames([5, BLANK_TOKEN_ID, 6])
    const a = ctcGreedyDecodeWithFrames(lp, 3, V)
    const b = ctcGreedyDecodeWithFrames(lp, 3, V, undefined, 0)
    expect(b).toEqual(a)
  })

  it('十分大きい下駄で WORD_BREAK が勝つ', () => {
    // 2 位との差は margin=1.0。下駄 2.0 なら逆転する
    const lp = frames([5, BLANK_TOKEN_ID, 6], 1.0)
    const out = ctcGreedyDecodeWithFrames(lp, 3, V, undefined, 2.0)
    expect(out.some((t) => t.tokenId === WORD_BREAK_TOKEN_ID)).toBe(true)
  })

  it('小さすぎる下駄では逆転しない', () => {
    const lp = frames([5, BLANK_TOKEN_ID, 6], 1.0)
    const out = ctcGreedyDecodeWithFrames(lp, 3, V, undefined, 0.5)
    expect(out.map((t) => t.tokenId)).toEqual([5, 6])
  })

  it('負の下駄で WORD_BREAK を抑えられる', () => {
    const lp = frames([WORD_BREAK_TOKEN_ID, BLANK_TOKEN_ID, 6])
    // WORD_BREAK が 1 位のフレームで、2 位 (-10) より下がるまで下げる
    const out = ctcGreedyDecodeWithFrames(lp, 3, V, undefined, -20.0)
    expect(out.some((t) => t.tokenId === WORD_BREAK_TOKEN_ID)).toBe(false)
  })
})
