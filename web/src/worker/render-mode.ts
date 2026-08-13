/**
 * 確定/暫定トークン列を表示テキストに変換する純粋関数.
 *
 * `decoder.worker.ts` はトップレベルで `self.onmessage = ...` を設定するため
 * (Worker グローバルの `self` に依存する)、vitest の node 環境からそのまま
 * import すると実行時エラーになりテストできない。ここに切り出すことで
 * Worker の起動なしにテストできるようにする (Task 12 申し送り事項)。
 */
import { SAMPLE_RATE } from '../generated/tokens'
import type { CommittedToken, DecodeView } from '../decode/sliding-window'
import { convert, type Mode } from '../tokens/converter'
import type { RenderedText } from './protocol'

/**
 * 改行を入れる無音の長さ (秒)。0 以下なら改行しない。
 *
 * 送信のターンの切れ目で行を分けるための機能。判定は**トークンの時刻差だけ**で
 * 行い、音声は見ない。確定テキストは毎回トークン列全体から作り直されるので、
 * 判定がトークン列だけから決まることが重要 (何度作り直しても同じ結果になり、
 * 「確定した文字は書き換わらない」という保証を壊さない)。
 *
 * トーンを使わないのはこのため。リングバッファは 30 秒しか保持しないので、
 * 古いトークンのトーンは作り直し時にはもう取れず、状態を持つ必要が出る。
 *
 * **判定はトークン間の「ピーク間距離」であって純粋な無音長ではない。**
 * CTC の貪欲デコードが返す位置はピークでほぼ幅ゼロ (0〜20 ms) であり、符号の実際の
 * 長さを持たない。したがってここで測る間隔には文字自身の長さが含まれる。文字は長くても
 * 0.5 秒程度なので、閾値 3 秒なら実際の無音は 2.5 秒以上あり実用上問題ない。
 * **ただし語間 (7 dot ≒ 0.4 秒) の判定にはこの値は使えない** (文字の長さに埋もれる)。
 *
 * 既定 3.0 秒の根拠: held-out 実録音 21 件で「1 回の送信の中に現れる無音」を
 * 測ると中央値 650 ms・99%tile 2.0 秒・最大 4.4 秒だった。3.0 秒だと送信の
 * 途中でも 1% 程度は改行が入るが、**余分な改行が入っても文字は失われない**ので
 * 許容している。実運用で合わなければこの値を動かす。
 */
export const DEFAULT_LINE_BREAK_GAP_S = 3.0

/** 確定トークン列を、閾値以上の無音で区切る。 */
function splitAtGaps(
  tokens: readonly CommittedToken[],
  gapSamples: number,
): CommittedToken[][] {
  if (tokens.length === 0) return []
  if (gapSamples <= 0) return [[...tokens]]
  const segments: CommittedToken[][] = [[tokens[0]]]
  for (let i = 1; i < tokens.length; i++) {
    const gap = tokens[i].absoluteSampleStart - tokens[i - 1].absoluteSampleEnd
    if (gap >= gapSamples) segments.push([tokens[i]])
    else segments[segments.length - 1].push(tokens[i])
  }
  return segments
}

/** 確定列と暫定列を指定モードで変換する (workers.py の _emit_texts 相当)。 */
export function renderMode(
  view: DecodeView,
  mode: Mode,
  lineBreakGapS: number = DEFAULT_LINE_BREAK_GAP_S,
): RenderedText {
  const gapSamples = Math.round(lineBreakGapS * SAMPLE_RATE)
  const segments = splitAtGaps(view.committed, gapSamples)

  // 区間ごとに変換して改行で連結する。
  //
  // **fallback の位置をずらすこと。** position は最終テキストの文字列インデックスで、
  // UI (renderInto) はこれで直接添字参照して装飾する。区間ごとの位置をそのまま
  // 使うと別の文字に装飾が付く。
  //
  // Python 版と違いモードの引き継ぎは不要。ブラウザ版の Mode は
  // 'european' | 'japanese' の 2 値で自動モードが無く、区間をまたいで
  // 引き継ぐ状態が存在しない (デスクトップ版に同じ機能を入れるときは必要になる)。
  let committedText = ''
  const committedFallbacks: RenderedText['committedFallbacks'] = []
  for (const segment of segments) {
    const res = convert(
      segment.map((t) => t.tokenId),
      segment.map((t) => t.confidence),
      { mode },
    )
    const offset = committedText.length + (committedText ? 1 : 0)   // +1 は改行ぶん
    if (committedText) committedText += '\n'
    committedText += res.text
    for (const ev of res.fallbackLog) {
      committedFallbacks.push({ ...ev, position: ev.position + offset })
    }
  }

  // 確定列と暫定列の境界が語間に落ちると空白が消える ('GL 73 CQ' → 'GL 73CQ')。
  // 確定列が既に空白で終わっていない場合だけ暫定列の先頭空白を残す。
  const keepLeadingSpace = committedText.length > 0 && !committedText.endsWith(' ')
  // 暫定列は区切らない。末尾の数秒しかなく区切りが入る余地がほとんど無いうえ、
  // 確定列との連結が複雑になるだけで得るものが無い。
  const provisional = convert(
    view.provisional.map((t) => t.tokenId),
    view.provisional.map((t) => t.confidence),
    { mode, keepLeadingSpace },
  )
  return {
    committed: committedText,
    provisional: provisional.text,
    committedFallbacks,
    provisionalFallbacks: provisional.fallbackLog,
  }
}
