import { describe, expect, it } from 'vitest'
import { FALLBACK_CHAR, convert } from '../src/tokens/converter'
import {
  EUROPEAN_TABLE,
  JAPANESE_TABLE,
  WORD_BREAK_TOKEN_ID,
  ID_TO_CODE,
  BLANK_TOKEN_ID,
} from '../src/generated/tokens'

/** 符号から token id を引く (テスト用の逆引き). */
function idOf(code: string): number {
  const id = ID_TO_CODE.indexOf(code)
  if (id < 0) throw new Error(`符号が見つからない: ${code}`)
  return id
}

describe('convert (欧文)', () => {
  it('基本的な符号を文字にする', () => {
    const ids = [idOf('・'), idOf('-'), idOf('・-')]
    const result = convert(ids, null, { mode: 'european' })
    expect(result.text).toBe('ETA')
  })

  it('WORD_BREAK を空白にする', () => {
    const ids = [idOf('・'), WORD_BREAK_TOKEN_ID, idOf('-')]
    expect(convert(ids, null, { mode: 'european' }).text).toBe('E T')
  })

  it('連続する WORD_BREAK は 1 つにまとめる', () => {
    const ids = [idOf('・'), WORD_BREAK_TOKEN_ID, WORD_BREAK_TOKEN_ID, idOf('-')]
    expect(convert(ids, null, { mode: 'european' }).text).toBe('E T')
  })

  it('先頭の WORD_BREAK は既定で捨てる', () => {
    const ids = [WORD_BREAK_TOKEN_ID, idOf('・')]
    expect(convert(ids, null, { mode: 'european' }).text).toBe('E')
  })

  it('keepLeadingSpace で先頭の空白を残す', () => {
    const ids = [WORD_BREAK_TOKEN_ID, idOf('・')]
    const result = convert(ids, null, { mode: 'european', keepLeadingSpace: true })
    expect(result.text).toBe(' E')
  })

  it('blank は無視する', () => {
    const ids = [BLANK_TOKEN_ID, idOf('・'), BLANK_TOKEN_ID]
    expect(convert(ids, null, { mode: 'european' }).text).toBe('E')
  })

  it('確信度が閾値未満なら ? にして LOW_CONFIDENCE を記録する', () => {
    const ids = [idOf('・'), idOf('-')]
    const result = convert(ids, [0.9, 0.2], { mode: 'european', confidenceThreshold: 0.5 })
    expect(result.text).toBe('E' + FALLBACK_CHAR)
    expect(result.fallbackLog).toHaveLength(1)
    expect(result.fallbackLog[0].kind).toBe('LOW_CONFIDENCE')
    expect(result.fallbackLog[0].position).toBe(1)
  })

  it('欧文表に無い和文符号は ? にして TABLE_MISS を記録する', () => {
    // コ (----) は和文表のみ
    const ids = [idOf('----')]
    expect(EUROPEAN_TABLE['----']).toBeUndefined()
    const result = convert(ids, null, { mode: 'european' })
    expect(result.text).toBe(FALLBACK_CHAR)
    expect(result.fallbackLog[0].kind).toBe('TABLE_MISS')
  })
})

describe('convert (和文)', () => {
  it('カナに変換する', () => {
    const ids = [idOf('・-'), idOf('・-・-')]
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('イロ')
  })

  it('濁点を直前のカナと合成する', () => {
    // カ (・-・・) + 濁点 (・・) → ガ
    const ids = [idOf('・-・・'), idOf('・・')]
    expect(JAPANESE_TABLE['・・']).toBe('゛')
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('ガ')
  })

  it('半濁点を直前のカナと合成する', () => {
    // ハ (-・・・) + 半濁点 (・・--・) → パ
    const ids = [idOf('-・・・'), idOf('・・--・')]
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('パ')
  })

  it('合成できない位置の濁点は ? にする', () => {
    // イ は濁点と合成できない
    const ids = [idOf('・-'), idOf('・・')]
    const result = convert(ids, null, { mode: 'japanese' })
    expect(result.text).toBe('イ' + FALLBACK_CHAR)
    expect(result.fallbackLog[0].kind).toBe('TABLE_MISS')
  })

  it('先頭の濁点は ? にする', () => {
    const result = convert([idOf('・・')], null, { mode: 'japanese' })
    expect(result.text).toBe(FALLBACK_CHAR)
  })

  it('濁点を 2 つ続けても 2 個目は合成しない', () => {
    const ids = [idOf('・-・・'), idOf('・・'), idOf('・・')]
    const result = convert(ids, null, { mode: 'japanese' })
    expect(result.text).toBe('ガ' + FALLBACK_CHAR)
  })

  it('空白のあとの濁点は合成しない', () => {
    const ids = [idOf('・-・・'), WORD_BREAK_TOKEN_ID, idOf('・・')]
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('カ ' + FALLBACK_CHAR)
  })

  it('別のカナを挟んでからの濁点は直近のカナに合成する', () => {
    // カ (・-・・) + ハ (-・・・) + 濁点 (・・) → カバ (ハ に合成、カ ではない)
    // composableAt が新しいカナごとに上書きされることの確認 (Python 版
    // test_two_kana_then_dakuten_composes_with_last と同じケース)
    const ids = [idOf('・-・・'), idOf('-・・・'), idOf('・・')]
    expect(convert(ids, null, { mode: 'japanese' }).text).toBe('カバ')
  })
})

describe('convert (共通)', () => {
  it('空入力は空文字', () => {
    expect(convert([], null, { mode: 'european' }).text).toBe('')
  })

  it('confidences の長さが違えば例外', () => {
    expect(() => convert([1, 2], [0.5], { mode: 'european' })).toThrow()
  })

  it('未知の token id は ? にする', () => {
    const result = convert([9999], null, { mode: 'european' })
    expect(result.text).toBe(FALLBACK_CHAR)
    expect(result.fallbackLog[0].code).toBe('<UNKNOWN>')
  })

  it('confidenceThreshold が範囲外なら例外', () => {
    expect(() =>
      convert([idOf('・')], null, { mode: 'european', confidenceThreshold: 50 }),
    ).toThrow()
    expect(() =>
      convert([idOf('・')], null, { mode: 'european', confidenceThreshold: -0.1 }),
    ).toThrow()
  })

  it('confidenceThreshold が NaN なら例外', () => {
    // `t < 0 || t > 1` という書き方だと NaN が素通りし、以降 `conf < NaN` が
    // 常に false になって LOW_CONFIDENCE 判定が丸ごと無効化される。
    expect(() =>
      convert([idOf('・')], null, { mode: 'european', confidenceThreshold: NaN }),
    ).toThrow()
  })
})

describe('FallbackEvent.position は文字列インデックス', () => {
  /** log の全 position が本当に text 上の FALLBACK_CHAR を指していることを確認する。 */
  const expectAllPositionsPointAtFallback = (result: {
    text: string
    fallbackLog: readonly { position: number }[]
  }): void => {
    for (const event of result.fallbackLog) {
      expect(result.text[event.position]).toBe(FALLBACK_CHAR)
    }
  }

  it('複数文字の欧文プロサイン ([SK]) の後の読めなかった印を正しく指す', () => {
    // [SK] は表示が 4 文字。position を「出力スロット番号」にしていると 1 を
    // 返し、UI が text[1] = 'S' を装飾してしまう (本物の ? は素通し)。
    expect(EUROPEAN_TABLE['・・・-・-']).toBe('[SK]')
    const ids = [idOf('・・・-・-'), idOf('・')]
    const result = convert(ids, [0.9, 0.1], { mode: 'european', confidenceThreshold: 0.5 })
    expect(result.text).toBe('[SK]' + FALLBACK_CHAR)
    expect(result.fallbackLog).toHaveLength(1)
    expect(result.fallbackLog[0].position).toBe(4)
    expectAllPositionsPointAtFallback(result)
  })

  it('複数文字の和文専用記号 ([ホレ]) の後の読めなかった印を正しく指す', () => {
    expect(JAPANESE_TABLE['-・・---']).toBe('[ホレ]')
    const ids = [idOf('-・・---'), idOf('・-')]
    const result = convert(ids, [0.9, 0.1], { mode: 'japanese', confidenceThreshold: 0.5 })
    expect(result.text).toBe('[ホレ]' + FALLBACK_CHAR)
    expect(result.fallbackLog[0].position).toBe(4)
    expectAllPositionsPointAtFallback(result)
  })

  it('複数文字表示・空白・濁点合成が混ざっても全 position が読めなかった印を指す', () => {
    // [SN] (2 文字ではなく 4 文字) + 空白 + 未知 id + カ + 濁点(合成) + 低確信度
    const ids = [
      idOf('・・・-・'),
      WORD_BREAK_TOKEN_ID,
      9999,
      idOf('・-・・'),
      idOf('・・'),
      idOf('・-'),
    ]
    const confs = [0.9, 1.0, 0.9, 0.9, 0.9, 0.1]
    const result = convert(ids, confs, { mode: 'japanese', confidenceThreshold: 0.5 })
    // [ラタ] + ' ' + FALLBACK_CHAR + 'ガ' + FALLBACK_CHAR
    expect(result.text).toBe('[ラタ] ' + FALLBACK_CHAR + 'ガ' + FALLBACK_CHAR)
    expect(result.fallbackLog).toHaveLength(2)
    expect(result.fallbackLog[0].position).toBe(5)
    expect(result.fallbackLog[1].position).toBe(7)
    expectAllPositionsPointAtFallback(result)
  })
})
