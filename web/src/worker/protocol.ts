/** メインスレッドと Decoder Worker の間でやり取りするメッセージ定義. */
import type { FallbackEvent, Mode } from '../tokens/converter'

/** 1 つの表示モード (欧文 or 和文) のレンダリング結果. */
export interface RenderedText {
  committed: string
  provisional: string
  /** 確定テキスト中の「?」の由来 (TABLE_MISS / LOW_CONFIDENCE の区別) */
  committedFallbacks: FallbackEvent[]
  provisionalFallbacks: FallbackEvent[]
}

export type WorkerRequest =
  | {
      type: 'init'
      modelUrl: string
      numThreads: number
      commitLagS: number
      hopS: number
      decodeLeftContextS: number
      /** 改行を入れる無音の長さ (秒)。0 以下なら改行しない。 */
      lineBreakGapS: number
      /** WORD_BREAK のロジットに足す下駄。0 なら従来どおり。 */
      wordBreakBias: number
    }
  | { type: 'audio'; block: Float32Array }
  | { type: 'setMode'; mode: Mode }
  | { type: 'finalize' }
  | { type: 'reset' }

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'error'; message: string }
  /**
   * finalize リクエストの処理が完了したことを表す (実行中の redecode 待ち +
   * フル再推論の完了後に 1 回だけ送られる)。
   *
   * メインスレッド側はこれを見てから Worker を terminate() すること。
   * 待たずに terminate すると、finalize が確定させるはずだった末尾のトークンが
   * 推論の途中で消える (DecoderClient.finalize 参照)。
   *
   * `text` で代用しない理由: 通常のデコードでも `text` は届くため、
   * どの finalize 呼び出しに対応する応答かを区別できない。
   */
  | { type: 'finalized' }
  | {
      type: 'text'
      /** 同じトークン列を両方の表に通した結果 (追加の推論コストは無い) */
      european: RenderedText
      japanese: RenderedText
      decodeMs: number
    }
