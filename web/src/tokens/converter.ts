/**
 * 符号トークン列 → 表示テキスト変換器.
 *
 * 移植元: src/tokens/converter.py の TokenConverter.convert
 *
 * 初版は固定モード (欧文 / 和文) のみを扱う。ホレ/ラタによる自動モード切替は
 * デスクトップ版でも実信号で未解決のためスコープ外 (設計書 §1)。
 *
 * 「?」は 2 種類を区別してログに残す:
 *   TABLE_MISS     — 変換表に該当なし (欧文モード中の和文符号など)
 *   LOW_CONFIDENCE — CTC 事後確率が閾値未満
 */
import {
  DAKUTEN_CHAR,
  DAKUTEN_COMPOSE,
  EUROPEAN_TABLE,
  HANDAKUTEN_CHAR,
  HANDAKUTEN_COMPOSE,
  ID_TO_CODE,
  JAPANESE_TABLE,
  BLANK_TOKEN_ID,
  WORD_BREAK_TOKEN_ID,
} from '../generated/tokens'

export const FALLBACK_CHAR = '?'

export type Mode = 'european' | 'japanese'
export type FallbackKind = 'TABLE_MISS' | 'LOW_CONFIDENCE'

/** 「?」を出したイベント (デバッグ用ログ). */
export interface FallbackEvent {
  /**
   * 最終テキスト `text` における ? の**文字列インデックス** (`text[position] === '?'`)。
   *
   * トークンの出力スロット番号ではない点に注意。表示値には `[SK]` `[ホレ]` のような
   * 複数文字のものが実在するため、スロット番号と文字列位置は一致しない
   * (`"[SK]?"` の ? はスロット 1 だが文字列位置は 4)。UI 側 (ui/app.ts の
   * renderInto) はこの値で `text` を直接添字参照して装飾するので、
   * ここが文字列インデックスでないと別の文字にツールチップが付く。
   */
  position: number
  /** 入力 tokenIds のインデックス */
  inputIndex: number
  tokenId: number
  /** 符号、または未知 id のとき '<UNKNOWN>' */
  code: string
  kind: FallbackKind
  confidence: number
}

export interface ConvertResult {
  text: string
  fallbackLog: FallbackEvent[]
}

export interface ConvertOptions {
  mode: Mode
  /** この値**未満**の確信度は ? に置換する。既定 0.5。 */
  confidenceThreshold?: number
  /**
   * 先頭の WORD_BREAK を空白として残すか。既定 false。
   *
   * 確定列と暫定列を別々に変換して連結すると、境界が語間に落ちたとき空白が
   * 消える ('GL 73 CQ' → 'GL 73CQ')。暫定列の変換で true を渡すと防げる。
   * **確定列が既に空白で終わっている場合は false を渡すこと** (二重空白になる)。
   */
  keepLeadingSpace?: boolean
}

export function convert(
  tokenIds: readonly number[],
  confidences: readonly number[] | null,
  options: ConvertOptions,
): ConvertResult {
  if (confidences !== null && confidences.length !== tokenIds.length) {
    throw new Error(
      `confidences length ${confidences.length} != tokenIds length ${tokenIds.length}`,
    )
  }
  const threshold = options.confidenceThreshold ?? 0.5
  // Python 版 (TokenConverter.__init__) と同様に範囲を検証する。
  // 呼び出し側が「0.5 のつもり」で「50」を渡すような単位の取り違えは、
  // 検査が無いと「全トークンが LOW_CONFIDENCE になる」形で静かに壊れる。
  // NaN を弾くため否定形で書く。`threshold < 0 || threshold > 1` は NaN を
  // 通してしまい、通ると `conf < NaN` が常に false になって LOW_CONFIDENCE の
  // 判定が丸ごと無効化される (まさにこの検査が防ぎたい「静かに壊れる」形)。
  if (!(threshold >= 0 && threshold <= 1)) {
    throw new Error(`confidenceThreshold must be in [0, 1], got ${threshold}`)
  }
  const keepLeadingSpace = options.keepLeadingSpace ?? false
  const japanese = options.mode === 'japanese'
  const table = japanese ? JAPANESE_TABLE : EUROPEAN_TABLE

  const outChars: string[] = []
  const log: FallbackEvent[] = []
  /** 直前の出力が「濁点/半濁点と合成可能なカナ」だった outChars 上の位置 */
  let composableAt: number | null = null
  /**
   * ここまでに出力した文字数 (= outChars.join('').length)。
   *
   * FallbackEvent.position は「出力スロット番号」ではなく「最終テキストの
   * 文字列インデックス」でなければならない (`[SK]` のような複数文字の表示値が
   * あるため両者は一致しない)。毎回 join して数え直すと O(n^2) になるので
   * 累積長をここで持つ。
   */
  let textLength = 0

  /** outChars に 1 スロット追加し、累積文字数を更新する。 */
  const pushOut = (value: string): void => {
    outChars.push(value)
    textLength += value.length
  }

  const emitFallback = (
    inputIndex: number,
    tokenId: number,
    code: string,
    kind: FallbackKind,
    confidence: number,
  ): void => {
    log.push({ position: textLength, inputIndex, tokenId, code, kind, confidence })
    pushOut(FALLBACK_CHAR)
  }

  for (let i = 0; i < tokenIds.length; i++) {
    const tid = tokenIds[i]
    if (tid === BLANK_TOKEN_ID) continue

    if (tid === WORD_BREAK_TOKEN_ID) {
      if (outChars.length > 0) {
        if (outChars[outChars.length - 1] !== ' ') pushOut(' ')
      } else if (keepLeadingSpace) {
        pushOut(' ')
      }
      composableAt = null
      continue
    }

    const conf = confidences !== null ? confidences[i] : 1.0
    const code = tid >= 0 && tid < ID_TO_CODE.length ? ID_TO_CODE[tid] : undefined

    if (code === undefined) {
      emitFallback(i, tid, '<UNKNOWN>', 'TABLE_MISS', conf)
      composableAt = null
      continue
    }
    if (conf < threshold) {
      emitFallback(i, tid, code, 'LOW_CONFIDENCE', conf)
      composableAt = null
      continue
    }

    const display = table[code]
    if (display === undefined) {
      emitFallback(i, tid, code, 'TABLE_MISS', conf)
      composableAt = null
      continue
    }

    if (japanese && (display === DAKUTEN_CHAR || display === HANDAKUTEN_CHAR)) {
      const composeMap = display === DAKUTEN_CHAR ? DAKUTEN_COMPOSE : HANDAKUTEN_COMPOSE
      if (composableAt !== null) {
        const composed = composeMap[outChars[composableAt]]
        if (composed !== undefined) {
          // 合成カナは 1 文字だが、表の値が将来変わっても累積長がずれないよう
          // 差分で更新する
          textLength += composed.length - outChars[composableAt].length
          outChars[composableAt] = composed
          composableAt = null
          continue
        }
      }
      // 直前カナが無い、または合成対象外 → 単独の濁点/半濁点は意味を成さない
      emitFallback(i, tid, code, 'TABLE_MISS', conf)
      composableAt = null
      continue
    }

    pushOut(display)
    // Object.hasOwn を使い、プロトタイプチェーン経由の誤検出を避ける
    // (`in` 演算子は継承プロパティも見るため Python の dict `in` とは意味が違う)
    composableAt =
      japanese &&
      (Object.hasOwn(DAKUTEN_COMPOSE, display) || Object.hasOwn(HANDAKUTEN_COMPOSE, display))
        ? outChars.length - 1
        : null
  }

  return { text: outChars.join(''), fallbackLog: log }
}
