/**
 * スライディングウィンドウ再デコード + prefix commit.
 *
 * 移植元: src/infer/sliding_window.py
 *
 * ライブ音声をリングバッファに保持し、hop ごとに窓全体を再デコードする。
 * [max(headGuard, lastCommit), now - commitLag) に入るトークンだけを
 * **不変 (immutable)** に確定する。確定済みは後から変化しないため表示がちらつかない。
 */
import type { FrameToken } from './ctc'

/** 波形チャンクを 1 回デコードするエンジン。 */
export interface DecodeEngine {
  /** 1 フレームあたりのサンプル数 (= mel の hop_length)。 */
  frameHopSamples: number
  decodeChunk(wave: Float32Array): Promise<FrameToken[]>
}

/**
 * 確定 (immutable) トークン。絶対サンプル位置付き。
 *
 * Python 版は `@dataclass(frozen=True)` で不変性を保証する。TypeScript には
 * dataclass 相当が無いため `readonly` フィールドで同等の保証をコンパイル時に
 * 与える (`view.committed[0].absoluteSampleEnd = 999` のような書き換えは
 * 型エラーになる)。プロジェクト内の既存の不変データ (`FallbackEvent` 等) も
 * 同じ `readonly` の慣習に従っている。
 */
export interface CommittedToken {
  readonly tokenId: number
  readonly confidence: number
  readonly absoluteSampleStart: number
  readonly absoluteSampleEnd: number
}

/**
 * 1 回の再デコード結果のスナップショット。
 *
 * 配列自体も `readonly` にしている。`committed` は内部状態の浅いコピーなので
 * `push`/`splice` 等で書き換えても内部の `this.committed` は破壊されないが、
 * 呼び出し側 (Task 12 Worker 等) に「このビューは書き換えて使うものではない」
 * ことを型で伝えるため、配列・要素の双方を不変にした。map/filter など
 * 非破壊メソッドの利用は妨げない。
 */
export interface DecodeView {
  /** 確定済み全体 */
  committed: readonly CommittedToken[]
  /** 今回新規に確定したもの */
  newlyCommitted: readonly CommittedToken[]
  /** 暫定 (グレー表示) */
  provisional: readonly CommittedToken[]
}

export interface SlidingWindowOptions {
  windowS?: number
  /** 呼び出し側が redecode() をこの間隔でスケジューリングする想定の値。クラス内では参照しない (Python 版の hop_samples_audio と同様)。 */
  hopS?: number
  commitLagS?: number
  headGuardS?: number
  decodeLeftContextS?: number
  commitJitterMarginS?: number
  sampleRate?: number
}

export class SlidingWindowDecoder {
  private readonly engine: DecodeEngine
  /** サンプルレート (Hz)。Python 版の self.sample_rate と同じく公開属性として保持する。 */
  readonly sampleRate: number
  private readonly windowSamples: number
  private readonly commitLagSamples: number
  private readonly headGuardSamples: number
  private readonly leftContextSamples: number
  private readonly jitterMarginSamples: number

  private ring = new Float32Array(0)
  /** 累積投入サンプル数 (= 現在時刻) */
  private totalConsumed = 0
  private committed: CommittedToken[] = []
  /** 確定済み末尾の絶対サンプル位置 (中点ウォーターマーク基準) */
  private lastCommitEnd: number | null = null

  constructor(engine: DecodeEngine, options: SlidingWindowOptions = {}) {
    const sampleRate = options.sampleRate ?? 8000
    this.engine = engine
    this.sampleRate = sampleRate
    this.windowSamples = Math.floor((options.windowS ?? 30.0) * sampleRate)
    this.commitLagSamples = Math.floor((options.commitLagS ?? 2.5) * sampleRate)
    this.headGuardSamples = Math.floor((options.headGuardS ?? 1.0) * sampleRate)
    this.leftContextSamples = Math.floor((options.decodeLeftContextS ?? 5.0) * sampleRate)
    this.jitterMarginSamples = Math.floor((options.commitJitterMarginS ?? 0.02) * sampleRate)
  }

  reset(): void {
    this.ring = new Float32Array(0)
    this.totalConsumed = 0
    this.committed = []
    this.lastCommitEnd = null
  }

  /** 音声を追加する (デコードはしない)。窓長を超えた古い分は捨てる。 */
  push(audio: Float32Array): void {
    this.totalConsumed += audio.length
    const merged = new Float32Array(this.ring.length + audio.length)
    merged.set(this.ring, 0)
    merged.set(audio, this.ring.length)
    this.ring =
      merged.length > this.windowSamples ? merged.subarray(merged.length - this.windowSamples) : merged
  }

  /** 確定済み末尾 - leftContext 以降を再デコードし、確定/暫定を更新する。 */
  async redecode(): Promise<DecodeView> {
    return this.decode(this.totalConsumed - this.commitLagSamples)
  }

  /**
   * 送信終了時の最終確定。commitLag を無視して残り全部を確定する。
   *
   * ストリーム終端では右文脈がもう増えないため、暫定として保留していた
   * 末尾トークンをここで確定させる。
   */
  async finalize(): Promise<DecodeView> {
    return this.decode(this.totalConsumed + 1)
  }

  private async decode(commitLimitAbs: number): Promise<DecodeView> {
    if (this.ring.length === 0) {
      return { committed: [...this.committed], newlyCommitted: [], provisional: [] }
    }

    const hop = this.engine.frameHopSamples
    const ringStartAbs = this.totalConsumed - this.ring.length

    // デコード区間の動的短縮 (計算量削減)
    let lastEnd = this.lastCommitEnd
    const anchor = lastEnd ?? ringStartAbs
    const decodeStartAbs = Math.max(ringStartAbs, anchor - this.leftContextSamples)
    const sub = this.ring.subarray(decodeStartAbs - ringStartAbs)
    const frameTokens = await this.engine.decodeChunk(sub)

    // head guard: デコード区間先頭の不採用区間。
    // ただし区間がストリーム先頭のときは先頭信号を捨てない。
    const headCutAbs = decodeStartAbs > 0 ? decodeStartAbs + this.headGuardSamples : 0

    const newly: CommittedToken[] = []
    const provisional: CommittedToken[] = []

    for (const tok of frameTokens) {
      const absStart = decodeStartAbs + tok.frameStart * hop
      const absEnd = decodeStartAbs + tok.frameEnd * hop
      const ct: CommittedToken = {
        tokenId: tok.tokenId,
        confidence: tok.confidence,
        absoluteSampleStart: absStart,
        absoluteSampleEnd: absEnd,
      }
      // 右文脈不足 (commit 境界を終了がまたぐ) → 暫定
      if (absEnd >= commitLimitAbs) {
        provisional.push(ct)
        continue
      }
      // 左文脈なし → 不採用
      if (absStart < headCutAbs) continue
      // 中点ウォーターマーク: 確定済み末尾を中点が超えるものだけ新規確定する。
      // 既確定トークンの再出現は確実にスキップされ、文字間ギャップが小さくても
      // 新規トークンは脱落しない。
      const midpoint = Math.floor((absStart + absEnd) / 2)
      if (lastEnd !== null && midpoint <= lastEnd + this.jitterMarginSamples) continue

      this.committed.push(ct)
      newly.push(ct)
      lastEnd = absEnd
      this.lastCommitEnd = absEnd
    }

    return { committed: [...this.committed], newlyCommitted: newly, provisional }
  }
}
