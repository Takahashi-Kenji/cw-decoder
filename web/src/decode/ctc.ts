/**
 * CTC greedy デコード (フレーム位置付き).
 *
 * 移植元: src/infer/engine.py の ctc_greedy_decode_with_frames
 *
 * フレーム位置を残すのは、スライディングウィンドウで重複区間を取り除き、
 * 確定/暫定の境界を絶対時刻で判定するため。
 */
import { BLANK_TOKEN_ID, WORD_BREAK_TOKEN_ID } from '../generated/tokens'

/** 1 つのデコード済みトークン。frameStart〜frameEnd は inclusive。 */
export interface FrameToken {
  tokenId: number
  confidence: number
  frameStart: number
  frameEnd: number
}

/**
 * log_softmax 済みの `(nFrames, vocabSize)` を平坦化した配列からトークン列を作る。
 *
 * @param logProbs 長さ nFrames * vocabSize の平坦配列 (行優先)
 * @param nFrames 時間フレーム数
 * @param vocabSize 語彙サイズ
 * @param blankId CTC blank の token id
 */
export function ctcGreedyDecodeWithFrames(
  logProbs: Float32Array,
  nFrames: number,
  vocabSize: number,
  blankId: number = BLANK_TOKEN_ID,
  wordBreakBias: number = 0,
): FrameToken[] {
  // 長さが足りないと `logProbs[base + v]` が undefined になり、比較が全部 false に
  // なって「全フレームが blank」と解釈される。例外ではなく空配列が返るため、
  // 呼び出し側からは「無音だった」と区別が付かない (Python 版は ValueError)。
  if (nFrames < 0 || vocabSize <= 0) {
    throw new Error(`invalid shape: nFrames=${nFrames}, vocabSize=${vocabSize}`)
  }
  if (logProbs.length < nFrames * vocabSize) {
    throw new Error(
      `logProbs length ${logProbs.length} < nFrames * vocabSize (${nFrames} * ${vocabSize} = ${nFrames * vocabSize})`,
    )
  }
  const out: FrameToken[] = []
  let prev = -1
  let runStart = 0
  let runMax = 0

  for (let t = 0; t < nFrames; t++) {
    const base = t * vocabSize
    let best = 0
    let bestLogProb = logProbs[base]
    for (let v = 1; v < vocabSize; v++) {
      // WORD_BREAK にだけ下駄を履かせる (0 なら何もしない = 既定)。
      // 語間が出やすくなるほか、モデルが余計に出すプロサイン ([SK] 等) を
      // 押さえる効果が実測で確認されている (docs/word_break_bias.md)。
      const value =
        v === WORD_BREAK_TOKEN_ID ? logProbs[base + v] + wordBreakBias : logProbs[base + v]
      if (value > bestLogProb) {
        bestLogProb = value
        best = v
      }
    }
    const conf = Math.exp(bestLogProb)

    if (best === prev) {
      if (best !== blankId && conf > runMax) runMax = conf
      continue
    }
    // 切り替わり: 直前のランを確定する
    if (prev !== -1 && prev !== blankId) {
      out.push({ tokenId: prev, confidence: runMax, frameStart: runStart, frameEnd: t - 1 })
    }
    prev = best
    runStart = t
    runMax = conf
  }

  if (prev !== -1 && prev !== blankId) {
    out.push({ tokenId: prev, confidence: runMax, frameStart: runStart, frameEnd: nFrames - 1 })
  }
  return out
}
