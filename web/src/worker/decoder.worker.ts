/**
 * デコーダ Worker.
 *
 * 音声ブロックを受け取り、hop ごとに再デコードして確定/暫定テキストを返す。
 * 推論はここで行うのでメインスレッドは固まらない。
 *
 * 欧文列と和文列は同じトークン列を 2 つの変換表に通すだけなので、
 * 追加の推論コストは無い (設計書 §9)。
 *
 * メッセージ処理の状態機械本体は `decoder-core.ts` の `DecoderWorkerCore` に
 * 切り出してある。このファイルはそれを実 Worker グローバル (`self`) と
 * `OnnxDecodeEngine` に配線するだけの薄いアダプタ。
 * (`self.onmessage = ...` はトップレベル副作用のため vitest の node 環境から
 * このファイル自体を import してテストすることはできない — テストは
 * `DecoderWorkerCore` に対して行う。)
 */
import { OnnxDecodeEngine } from '../decode/onnx-engine'
import { DecoderWorkerCore } from './decoder-core'
import type { WorkerRequest } from './protocol'

const core = new DecoderWorkerCore(
  (message) => self.postMessage(message),
  (modelUrl, numThreads, wordBreakBias) =>
    OnnxDecodeEngine.create(modelUrl, { numThreads, wordBreakBias }),
)

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  // メッセージごとに非同期で処理する (await しない = 前のメッセージの完了を
  // 待たずに次のメッセージを受け付ける)。直列化が必要な処理は
  // DecoderWorkerCore 内部の `inFlight` で行っている。
  void core.handle(event.data)
}
