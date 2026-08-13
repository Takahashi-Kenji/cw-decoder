/**
 * ONNX Runtime Web による推論エンジン.
 *
 * モデルは「波形 in → log_probs out」の単一グラフ (scripts/export_onnx.py が生成)。
 * メル変換がグラフに焼き込まれているため、ここで前処理は一切行わない。
 */
import * as ort from 'onnxruntime-web'
import { ctcGreedyDecodeWithFrames, type FrameToken } from './ctc'
import type { DecodeEngine } from './sliding-window'
// FRAME_HOP_SAMPLES (= mel の hop_length) は手書きしない。フレーム位置 → 絶対
// サンプル位置の変換係数なので、Python 側の hop_ms を変えたときにここが古いままだと
// ブラウザ側の確定境界が全部ずれる。符号表と同じく scripts/export_tokens.py が
// MelConfig から生成し、tests/test_export_tokens.py がドリフトを検出する。
import { FRAME_HOP_SAMPLES, VOCAB_SIZE } from '../generated/tokens'

/** モデル (17 MB) を保存する Cache API のキー。 */
const MODEL_CACHE_NAME = 'cw-decoder-model-v1'

export interface OnnxEngineOptions {
  /** WASM スレッド数。crossOriginIsolated でない環境では 1 になる。 */
  numThreads?: number
  /**
   * WORD_BREAK のロジットに足す下駄。既定 0 (何もしない)。
   *
   * 正にすると語間が出やすくなる。モデルの WORD_BREAK は recall が 57〜69% しか
   * 無く、実運用で語がくっついて読みにくいという報告があったため調整口を設けた。
   * 副作用として、モデルが余計に出すプロサイン ([SK] 等) も押さえられる。
   * **0 のときは従来と完全に同じ経路** (ゴールデンテストが通る)。
   */
  wordBreakBias?: number
}

/**
 * モデルを取得する。2 回目以降は Cache API から読むので再ダウンロードしない。
 *
 * Cache API が無い環境 (vitest の Node 実行など) では URL をそのまま返し、
 * ORT にファイルを読ませる。
 */
async function loadModel(modelUrl: string): Promise<string | ArrayBuffer> {
  if (typeof caches === 'undefined') return modelUrl
  try {
    const cache = await caches.open(MODEL_CACHE_NAME)
    let response = await cache.match(modelUrl)
    if (!response) {
      await cache.add(modelUrl)
      response = await cache.match(modelUrl)
    }
    if (response) return await response.arrayBuffer()
  } catch {
    // キャッシュが使えなくても動作は続ける (毎回ダウンロードになるだけ)
  }
  return modelUrl
}

export class OnnxDecodeEngine implements DecodeEngine {
  readonly frameHopSamples = FRAME_HOP_SAMPLES

  private constructor(
    private readonly session: ort.InferenceSession,
    private readonly wordBreakBias: number,
  ) {}

  static async create(
    modelUrl: string,
    options: OnnxEngineOptions = {},
  ): Promise<OnnxDecodeEngine> {
    ort.env.wasm.simd = true
    ort.env.wasm.numThreads = options.numThreads ?? 1
    const source = await loadModel(modelUrl)
    // InferenceSession.create は string 用と ArrayBuffer 用の別オーバーロードなので、
    // source (string | ArrayBuffer) をそのまま渡すことはできない。`as string` で
    // 型検査に嘘をつく代わりに typeof で分岐し、各分岐で正しいオーバーロードへ
    // 静的に解決させる (実行時の分岐は ORT 内部でも行われているが、ここでは
    // TypeScript の型検査自体を正直に保つのが目的)。
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }
    const session =
      typeof source === 'string'
        ? await ort.InferenceSession.create(source, sessionOptions)
        : await ort.InferenceSession.create(source, sessionOptions)
    return new OnnxDecodeEngine(session, options.wordBreakBias ?? 0)
  }

  async decodeChunk(wave: Float32Array): Promise<FrameToken[]> {
    if (wave.length === 0) return []
    // subarray はビューなので、ORT に渡す前に連続領域へコピーする
    const input = new Float32Array(wave)
    const feeds = { wave: new ort.Tensor('float32', input, [1, input.length]) }
    const output = await this.session.run(feeds)
    const logProbs = output.log_probs
    // cw.onnx (export_onnx.py) と tokens.ts (export_tokens.py) は別スクリプトで
    // 生成されるため、片方だけ再生成した状態を誰も検出できない。log_probs は
    // 平坦配列として読むので、語彙サイズが食い違うと誤ったストライドで読み出して
    // 「例外は出ないが全部文字化けする」形で静かに壊れる。ここで弾く。
    if (logProbs.dims[2] !== VOCAB_SIZE) {
      throw new Error(
        `モデルの語彙サイズ ${logProbs.dims[2]} が tokens.ts の VOCAB_SIZE ${VOCAB_SIZE} と違う。` +
          `export_onnx.py と export_tokens.py の両方を再実行すること`,
      )
    }
    const nFrames = logProbs.dims[1]
    return ctcGreedyDecodeWithFrames(
      logProbs.data as Float32Array,
      nFrames,
      VOCAB_SIZE,
      undefined,
      this.wordBreakBias,
    )
  }

  async release(): Promise<void> {
    await this.session.release()
  }
}
