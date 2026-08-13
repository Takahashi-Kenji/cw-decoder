/**
 * WAV ファイルを 8 kHz mono float32 に変換する (開発者向けの入口).
 *
 * リサンプルは OfflineAudioContext に任せる。ブラウザ内蔵のリサンプラは
 * グラフ内で状態を保持するため、ブロック境界の歪みが生じない。
 */
// サンプルレートは手書きしない (MelConfig.sample_rate から生成)
import { SAMPLE_RATE as TARGET_SAMPLE_RATE } from '../generated/tokens'

/**
 * 長い音声を推論しやすい長さに分割する計画を返す。
 *
 * BiLSTM は双方向なので、チャンク境界の前後は文脈が欠ける。
 * overlap ぶん重ねて、呼び出し側が重複区間を捨てられるようにする。
 */
export function planChunks(
  totalSamples: number,
  chunkSamples: number,
  overlapSamples: number,
): Array<{ start: number; end: number }> {
  if (totalSamples <= 0) return []
  if (totalSamples <= chunkSamples) return [{ start: 0, end: totalSamples }]

  const stride = chunkSamples - overlapSamples
  const chunks: Array<{ start: number; end: number }> = []
  for (let start = 0; start < totalSamples; start += stride) {
    const end = Math.min(start + chunkSamples, totalSamples)
    chunks.push({ start, end })
    if (end === totalSamples) break
  }
  return chunks
}

/** ファイルを 8 kHz mono float32 の波形にする。 */
export async function fileToWave8k(file: File): Promise<Float32Array> {
  const bytes = await file.arrayBuffer()

  // decodeAudioData はファイル本来のレートでデコードする
  const decodeContext = new AudioContext()
  let decoded: AudioBuffer
  try {
    decoded = await decodeContext.decodeAudioData(bytes)
  } finally {
    await decodeContext.close()
  }

  // 8 kHz の OfflineAudioContext を通してリサンプル + モノラル化する
  const length = Math.max(1, Math.ceil(decoded.duration * TARGET_SAMPLE_RATE))
  const offline = new OfflineAudioContext(1, length, TARGET_SAMPLE_RATE)
  const source = offline.createBufferSource()
  source.buffer = decoded
  source.connect(offline.destination)
  source.start()
  const rendered = await offline.startRendering()

  return rendered.getChannelData(0).slice()
}
