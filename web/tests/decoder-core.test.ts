import { describe, expect, it } from 'vitest'
import { DecoderWorkerCore } from '../src/worker/decoder-core'
import type { DecodeEngine } from '../src/decode/sliding-window'
import type { FrameToken } from '../src/decode/ctc'
import type { WorkerRequest, WorkerResponse } from '../src/worker/protocol'
import { ID_TO_CODE } from '../src/generated/tokens'

/** 符号から token id を引く (テスト用の逆引き。converter.test.ts と同じ)。 */
function idOf(code: string): number {
  const id = ID_TO_CODE.indexOf(code)
  if (id < 0) throw new Error(`符号が見つからない: ${code}`)
  return id
}

/**
 * decodeChunk() を呼ぶたびに calls をカウントし、resolveNext() を呼ぶまで
 * 解決しないフェイクエンジン。「推論に時間がかかっている最中」を模擬する。
 * すでに解決待ちの呼び出しがある状態でさらに呼ばれたら overlapped を立てる
 * (= 2 つの decodeChunk 呼び出しが同時に in-flight だったことを示す)。
 */
class SlowEngine implements DecodeEngine {
  frameHopSamples = 1
  calls = 0
  overlapped = false
  private busy = false
  private pendingResolvers: Array<() => void> = []

  async decodeChunk(_wave: Float32Array): Promise<FrameToken[]> {
    this.calls++
    if (this.busy) this.overlapped = true
    this.busy = true
    await new Promise<void>((resolve) => this.pendingResolvers.push(resolve))
    this.busy = false
    return []
  }

  /** 一番古い decodeChunk 呼び出しを完了させる。 */
  resolveNext(): void {
    const resolve = this.pendingResolvers.shift()
    if (!resolve) throw new Error('resolve すべき decodeChunk 呼び出しが無い')
    resolve()
  }
}

/**
 * SlowEngine と同様に pending だが、解決時に確定トークン「E」を 1 個返す。
 * reset の競合テストで「reset 後に確定列が復活していないか」を確認するのに使う。
 */
class TokenResolvingEngine implements DecodeEngine {
  frameHopSamples = 1
  calls = 0
  private pendingResolvers: Array<() => void> = []

  async decodeChunk(_wave: Float32Array): Promise<FrameToken[]> {
    this.calls++
    await new Promise<void>((resolve) => this.pendingResolvers.push(resolve))
    return [{ tokenId: idOf('・'), confidence: 0.9, frameStart: 0, frameEnd: 1 }]
  }

  resolveNext(): void {
    const resolve = this.pendingResolvers.shift()
    if (!resolve) throw new Error('resolve すべき decodeChunk 呼び出しが無い')
    resolve()
  }
}

const INIT: WorkerRequest = {
  type: 'init',
  modelUrl: 'dummy',
  numThreads: 1,
  commitLagS: 2.5,
  hopS: 1, // 1 秒 = 8000 サンプルで redecode がトリガーされる
  decodeLeftContextS: 5,
  lineBreakGapS: 0, // このテストでは改行を無効にする (確定テキストの比較を単純に保つ)
  wordBreakBias: 0, // 既定 (従来どおりの経路)
}

/** commitLagS=0 にして、確定トークンがすぐ committed に載るようにした init。reset 競合テスト用。 */
const INIT_IMMEDIATE_COMMIT: WorkerRequest = { ...INIT, commitLagS: 0 }

function isTextResponse(m: WorkerResponse): m is Extract<WorkerResponse, { type: 'text' }> {
  return m.type === 'text'
}

/** マイクロタスクキューを指定回数分だけ流す (保留中の then/await 継続を進める)。 */
async function flushMicrotasks(times = 5): Promise<void> {
  for (let i = 0; i < times; i++) await Promise.resolve()
}

describe('DecoderWorkerCore の inFlight 直列化 (finalize/reset と redecode の競合防止)', () => {
  it('finalize は実行中の audio 由来 redecode の decodeChunk 呼び出しと同時に走らない', async () => {
    const engine = new SlowEngine()
    const posted: WorkerResponse[] = []
    const core = new DecoderWorkerCore(
      (m) => posted.push(m),
      async () => engine,
    )
    await core.handle(INIT)

    // hopS=1 → 8000 サンプルの投入で audio が redecode をトリガーする。
    // decodeChunk は resolveNext() を呼ぶまで解決しない (推論中を模擬)。
    const audioPromise = core.handle({ type: 'audio', block: new Float32Array(8000) })

    // audio 由来の decodeChunk がまだ pending のうちに finalize を送る (待たずに fire)。
    const finalizePromise = core.handle({ type: 'finalize' })

    // マイクロタスクを流しても、finalize は inFlight を待っているはずなので
    // まだ 2 回目の decodeChunk 呼び出しは起きていない。
    await flushMicrotasks()
    expect(engine.calls).toBe(1)

    engine.resolveNext() // audio 由来の redecode を完了させる
    await audioPromise

    // audio 側が完全に終わった後で、ようやく finalize の decodeChunk が呼ばれる。
    await flushMicrotasks()
    expect(engine.calls).toBe(2)
    engine.resolveNext()
    await finalizePromise

    expect(engine.overlapped).toBe(false)
    expect(posted.filter(isTextResponse)).toHaveLength(2)
  })

  it('reset は実行中の audio 由来 redecode の書き込みが終わってから実行される (取りこぼした確定トークンの復活を防ぐ)', async () => {
    // reset 自体は decodeChunk を呼ばないので、SlowEngine のような呼び出し回数の
    // 比較では reset の競合を検出できない。代わりに「reset を待たずに実行すると
    // 何が起きるか」を直接再現する: commitLagS=0 で確定トークンを 1 個返す
    // decodeChunk が pending のうちに reset を送り、両方が解決した後で改めて
    // finalize を送って確定列を覗く。
    //
    // reset が redecode の完了を待たずに実行されると (バグ)、reset 後に
    // decode() の続き (this.committed.push(...)) が走り、reset で消したはずの
    // トークンが確定列に復活する。fix 済みなら reset は redecode の完了を待って
    // から走るので、確定列は空のままになる。
    const engine = new TokenResolvingEngine()
    const posted: WorkerResponse[] = []
    const core = new DecoderWorkerCore(
      (m) => posted.push(m),
      async () => engine,
    )
    await core.handle(INIT_IMMEDIATE_COMMIT)

    const audioPromise = core.handle({ type: 'audio', block: new Float32Array(8000) })
    // decodeChunk が pending のうちに reset を送る (待たずに fire)
    const resetPromise = core.handle({ type: 'reset' })

    engine.resolveNext() // audio 由来の decodeChunk (トークン「E」を確定させる) を完了させる
    await audioPromise
    await resetPromise

    // reset 後の状態を finalize で覗く。ring が空になっているので
    // finalize は decodeChunk を新たに呼ばずに即座に現在の確定列を返す。
    posted.length = 0
    await core.handle({ type: 'finalize' })
    const textMessages = posted.filter(isTextResponse)
    expect(textMessages).toHaveLength(1)
    expect(textMessages[0].european.committed).toBe('')
    expect(engine.calls).toBe(1)
  })
})

describe("finalize の完了通知 ('finalized')", () => {
  // DecoderClient.finalize() は Worker からの 'finalized' 応答を待ってから
  // terminate() する設計 (末尾の確定トークンが推論完了前に消えるのを防ぐため)。
  // その前提となる「finalize のデコードが完了してから 'finalized' が届く」
  // ことを ここで直接検証する。DecoderClient 自体は実 Worker (postMessage/
  // setTimeout タイマー) に依存するため node 環境の vitest では検証できないが、
  // Worker 側がいつ 'finalized' を投げるかは DecoderWorkerCore の責務であり、
  // ここは実 Worker なしでテストできる。
  it('finalize のデコードが完了した後にのみ finalized が届く (推論中に先走らない)', async () => {
    const engine = new SlowEngine()
    const posted: WorkerResponse[] = []
    const core = new DecoderWorkerCore(
      (m) => posted.push(m),
      async () => engine,
    )
    await core.handle(INIT)

    // 実行中の audio 由来 redecode (SlowEngine が resolveNext() まで解決しない) を作る。
    const audioPromise = core.handle({ type: 'audio', block: new Float32Array(8000) })
    const finalizePromise = core.handle({ type: 'finalize' })

    // audio 側の decodeChunk がまだ pending の間は finalize 自身の decodeChunk も
    // 走っていないはずなので、'finalized' もまだ届かない。
    await flushMicrotasks()
    expect(posted.some((m) => m.type === 'finalized')).toBe(false)

    engine.resolveNext() // audio 由来の redecode を完了させる
    await audioPromise

    // finalize 自身の decodeChunk (2 回目の calls) がまだ pending の間も、
    // 'finalized' はまだ届かない。
    await flushMicrotasks()
    expect(engine.calls).toBe(2)
    expect(posted.some((m) => m.type === 'finalized')).toBe(false)

    engine.resolveNext() // finalize 由来の decodeChunk を完了させる
    await finalizePromise

    // 完了後、最後に投げられたメッセージが 'finalized' であること
    // (その直前に finalize 自身の 'text' が 1 回投げられているはず)。
    expect(posted.at(-1)?.type).toBe('finalized')
    expect(posted.filter((m) => m.type === 'finalized')).toHaveLength(1)
    const textCount = posted.filter(isTextResponse).length
    expect(textCount).toBe(2) // audio 由来 + finalize 由来
  })

  it('init 前 (decoder が無い) に finalize しても finalized は届く (待ち続けない)', async () => {
    const posted: WorkerResponse[] = []
    const core = new DecoderWorkerCore(
      (m) => posted.push(m),
      async () => {
        throw new Error('使われないはず')
      },
    )

    await core.handle({ type: 'finalize' })

    expect(posted).toEqual([{ type: 'finalized' }])
  })
})
