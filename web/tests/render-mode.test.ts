import { describe, expect, it } from 'vitest'
import { renderMode } from '../src/worker/render-mode'
import { FALLBACK_CHAR, convert, type Mode } from '../src/tokens/converter'
import type { CommittedToken, DecodeView } from '../src/decode/sliding-window'
import { ID_TO_CODE, WORD_BREAK_TOKEN_ID } from '../src/generated/tokens'

/** 符号から token id を引く (テスト用の逆引き。converter.test.ts と同じ)。 */
function idOf(code: string): number {
  const id = ID_TO_CODE.indexOf(code)
  if (id < 0) throw new Error(`符号が見つからない: ${code}`)
  return id
}

/** tokenId 列から (renderMode が要求する) CommittedToken 列を作る。位置情報は使われないのでダミー。 */
function toTokens(tokenIds: readonly number[]): CommittedToken[] {
  return tokenIds.map((tokenId, i) => ({
    tokenId,
    confidence: 1.0,
    absoluteSampleStart: i,
    absoluteSampleEnd: i + 1,
  }))
}

/**
 * ★ 語間スペース消失バグの回帰テスト (2026-08-04 にデスクトップ版で実際に発生)。
 *
 * 確定列と暫定列を別々に変換して連結すると、分割位置がちょうど WORD_BREAK に
 * 落ちたときスペースが消える ('GL 73 CQ' → 'GL 73CQ')。renderMode の
 * keepLeadingSpace 計算がこれを防いでいることを、あり得る**すべての分割位置**
 * について確認する (Python 側 test_split_at_word_break_concatenates_correctly 相当)。
 */
function assertAllSplitPositionsConcatenateCorrectly(tokenIds: readonly number[], mode: Mode): void {
  const expected = convert(tokenIds, null, { mode }).text

  for (let i = 0; i <= tokenIds.length; i++) {
    const view: DecodeView = {
      committed: toTokens(tokenIds.slice(0, i)),
      newlyCommitted: [],
      provisional: toTokens(tokenIds.slice(i)),
    }
    const rendered = renderMode(view, mode)
    const actual = rendered.committed + rendered.provisional
    expect(actual, `分割位置 i=${i} (mode=${mode})`).toBe(expected)
  }
}

describe('renderMode (語間スペース消失バグの回帰テスト)', () => {
  it('欧文: "GL 73 CQ" 相当のトークン列。全 9 通りの分割位置を網羅する', () => {
    // G, L, WORD_BREAK, 7, 3, WORD_BREAK, C, Q → 8 トークン、分割位置は 0..8 の 9 通り
    const tokenIds = [
      idOf('--・'), // G
      idOf('・-・・'), // L
      WORD_BREAK_TOKEN_ID,
      idOf('--・・・'), // 7
      idOf('・・・--'), // 3
      WORD_BREAK_TOKEN_ID,
      idOf('-・-・'), // C
      idOf('--・-'), // Q
    ]
    expect(convert(tokenIds, null, { mode: 'european' }).text).toBe('GL 73 CQ')
    assertAllSplitPositionsConcatenateCorrectly(tokenIds, 'european')
  })

  it('欧文: 連続する WORD_BREAK を含む列。全分割位置を網羅する', () => {
    // E, WORD_BREAK, WORD_BREAK, T → 4 トークン、分割位置は 0..4 の 5 通り。
    // 分割位置がちょうど連続 WORD_BREAK の間に落ちるケース (committed が既に
    // 空白で終わっている状態) を含む。
    const tokenIds = [idOf('・'), WORD_BREAK_TOKEN_ID, WORD_BREAK_TOKEN_ID, idOf('-')]
    expect(convert(tokenIds, null, { mode: 'european' }).text).toBe('E T')
    assertAllSplitPositionsConcatenateCorrectly(tokenIds, 'european')
  })

  it('和文: "カ タ ワ" 相当のトークン列。全 6 通りの分割位置を網羅する', () => {
    // カ, WORD_BREAK, タ, WORD_BREAK, ワ → 5 トークン、分割位置は 0..5 の 6 通り
    const tokenIds = [
      idOf('・-・・'), // カ
      WORD_BREAK_TOKEN_ID,
      idOf('-・'), // タ
      WORD_BREAK_TOKEN_ID,
      idOf('-・-'), // ワ
    ]
    expect(convert(tokenIds, null, { mode: 'japanese' }).text).toBe('カ タ ワ')
    assertAllSplitPositionsConcatenateCorrectly(tokenIds, 'japanese')
  })
})

// ============================================================
// 無音による改行 (送信のターンの切れ目で行を分ける)
// ============================================================
describe('無音による改行', () => {
  const SR = 8000

  /** 指定した無音 (秒) を挟んだトークン列を作る。各トークンは 0.1 秒とする。 */
  function tokensWithGaps(tokenIds: readonly number[], gapsS: readonly number[]): CommittedToken[] {
    const dur = Math.round(0.1 * SR)
    const out: CommittedToken[] = []
    let cursor = 0
    tokenIds.forEach((tokenId, i) => {
      if (i > 0) cursor += Math.round((gapsS[i - 1] ?? 0) * SR)
      out.push({
        tokenId,
        confidence: 1.0,
        absoluteSampleStart: cursor,
        absoluteSampleEnd: cursor + dur,
      })
      cursor += dur
    })
    return out
  }

  const view = (committed: CommittedToken[]): DecodeView => ({
    committed,
    newlyCommitted: [],
    provisional: [],
  })

  it('閾値未満の無音では改行しない', () => {
    const ids = [idOf('・-'), idOf('-・・・'), idOf('-・-・')]
    const r = renderMode(view(tokensWithGaps(ids, [2.9, 0.5])), 'european', 3.0)
    expect(r.committed).not.toContain('\n')
  })

  it('閾値以上の無音で改行する', () => {
    const ids = [idOf('・-'), idOf('-・・・'), idOf('-・-・')]
    const r = renderMode(view(tokensWithGaps(ids, [3.0, 0.5])), 'european', 3.0)
    const lines = r.committed.split('\n')
    expect(lines).toHaveLength(2)
    expect(lines[0]).toBe('A')
    expect(lines[1]).toBe('BC')
  })

  it('改行が複数回入る', () => {
    const ids = [idOf('・-'), idOf('-・・・'), idOf('-・-・')]
    const r = renderMode(view(tokensWithGaps(ids, [5.0, 5.0])), 'european', 3.0)
    expect(r.committed.split('\n')).toEqual(['A', 'B', 'C'])
  })

  it('同じ入力を 2 回変換すると同じ結果になる (作り直しても確定が動かない)', () => {
    const ids = [idOf('・-'), idOf('-・・・'), idOf('-・-・')]
    const tokens = tokensWithGaps(ids, [4.0, 0.5])
    const a = renderMode(view(tokens), 'european', 3.0)
    const b = renderMode(view(tokens), 'european', 3.0)
    expect(a.committed).toBe(b.committed)
    expect(a.committedFallbacks).toEqual(b.committedFallbacks)
  })

  it('トークンが増えても既存の行は変わらない (追記のみ)', () => {
    const ids = [idOf('・-'), idOf('-・・・')]
    const tokens = tokensWithGaps(ids, [4.0])
    const before = renderMode(view(tokens), 'european', 3.0).committed
    const grown = [...tokens]
    const last = tokens[tokens.length - 1]
    grown.push({
      tokenId: idOf('-・-・'),
      confidence: 1.0,
      absoluteSampleStart: last.absoluteSampleEnd + Math.round(0.5 * SR),
      absoluteSampleEnd: last.absoluteSampleEnd + Math.round(0.6 * SR),
    })
    const after = renderMode(view(grown), 'european', 3.0).committed
    expect(after.startsWith(before)).toBe(true)
  })

  it('閾値 0 なら改行しない (機能を切れる)', () => {
    const ids = [idOf('・-'), idOf('-・・・')]
    const r = renderMode(view(tokensWithGaps(ids, [10.0])), 'european', 0)
    expect(r.committed).not.toContain('\n')
  })

  it('fallback の位置が改行後もずれない', () => {
    // 欧文表に無い和文専用符号を混ぜて TABLE_MISS を作る (ホレ = -・・---)
    const ids = [idOf('・-'), idOf('-・・---'), idOf('-・・・')]
    const r = renderMode(view(tokensWithGaps(ids, [4.0, 0.5])), 'european', 3.0)
    expect(r.committedFallbacks.length).toBeGreaterThan(0)
    for (const ev of r.committedFallbacks) {
      // position は最終テキストの文字列インデックス。UI はここを直接添字参照する
      expect(r.committed[ev.position]).toBe(FALLBACK_CHAR)
    }
  })
})
