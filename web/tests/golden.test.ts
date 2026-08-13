/**
 * ゴールデンテスト: Python の推論結果と完全一致することを確認する.
 *
 * fixture は scripts/export_golden.py が出力した 8 kHz mono float32 の波形と
 * 期待トークン列。リサンプルは検証対象外 (Python 側で済ませてある)。
 *
 * 前提: python scripts/export_onnx.py で web/public/model/cw.onnx が生成済み。
 */
import { readFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it, beforeAll, afterAll } from 'vitest'
import { OnnxDecodeEngine } from '../src/decode/onnx-engine'
import { convert } from '../src/tokens/converter'

const MODEL_PATH = resolve(__dirname, '../public/model/cw.onnx')
const FIXTURE_DIR = resolve(__dirname, 'fixtures')

interface GoldenEntry {
  name: string
  waveFile: string
  nSamples: number
  tokenIds: number[]
  confidences: number[]
  textEuropean: string
  textJapanese: string
}

const modelExists = existsSync(MODEL_PATH)
const fixtureExists = existsSync(resolve(FIXTURE_DIR, 'golden.json'))
const hasAssets = modelExists && fixtureExists

// `web/public/model/cw.onnx` は git 管理外 (17MB) なので、clone 直後や CI では必ず不在になる。
// describe.skipIf の中に警告を書いても実行されない (describe 自体がスキップされるため) ので、
// ここ (トップレベル、判定した直後) で console.warn する。`npm test` は 0 件検証でも
// 終了コード 0 で緑になってしまうため、この警告が「気づく」唯一の手段になる。
if (!hasAssets) {
  const missing: string[] = []
  if (!modelExists) missing.push(`モデル (${MODEL_PATH})`)
  if (!fixtureExists) missing.push(`fixture (${resolve(FIXTURE_DIR, 'golden.json')})`)
  console.warn(
    '\n' +
      '========================================================================\n' +
      '[golden.test.ts] ゴールデンテストをスキップしました。\n' +
      `不足: ${missing.join(', ')}\n` +
      'このテストはトークン ID 列が Python の推論結果と完全一致することを検証する、\n' +
      '移植プロジェクト最重要のテストです。0 件のまま「全部緑」に見えるので注意してください。\n' +
      '直し方: リポジトリルートで `python scripts/export_onnx.py` を実行し、\n' +
      'web/public/model/cw.onnx を生成してから再実行してください。\n' +
      '========================================================================\n',
  )
}

/**
 * 確信度の許容誤差 (絶対値)。厳密一致 (toBeCloseTo) ではなくこの値にした理由:
 *
 * 1. なぜ厳密一致を要求しないか:
 *    確信度は float32 の exp() 演算の結果であり、PyTorch (Python 側の fixture 生成元)
 *    と ONNX Runtime Web はカーネル実装が異なるため、バックエンド間で微差が出るのは
 *    避けられない。Python 側の ONNX 変換検証 (tests/test_export_onnx.py) も確信度の
 *    近さは検査しておらず、トークン ID 列の一致のみを見ている。つまりプロジェクトの
 *    設計として、確信度の float 値そのものに厳密一致を要求していない。
 * 2. 実測値: 2026-08-05 に oubun/wabun 全トークンで比較した結果、絶対差の最大は
 *    0.0088 (oubun[12], tokenId=72)。0.02 はその約 2 倍の余裕を持たせた値。
 *    ORT / torch のバージョンが変わっても偽陽性になりにくく、かつ「桁が変わるような
 *    ズレ」は検出できる。
 * 3. 振る舞い上の帰結の担保: 確信度は convert() の confidenceThreshold (既定 0.5) を
 *    跨ぐと表示が LOW_CONFIDENCE ('?') に変わるため、差が大きければ最終テキストに
 *    影響しうる。**その帰結は、このテストではなく下の「欧文・和文テキストが Python
 *    と一致する」テストが担保している。** テキストが完全一致している限り、閾値を
 *    跨ぐような確信度の差は実際には発生していない。このテストはあくまで
 *    「異常に大きくズレていないこと」を見る健全性チェックであり、
 *    テキスト一致検査の代わりではない。
 */
const CONFIDENCE_ABS_TOLERANCE = 0.02

describe.skipIf(!hasAssets)('ゴールデンテスト', () => {
  let engine: OnnxDecodeEngine
  let entries: GoldenEntry[]

  beforeAll(async () => {
    entries = JSON.parse(readFileSync(resolve(FIXTURE_DIR, 'golden.json'), 'utf-8'))
    engine = await OnnxDecodeEngine.create(MODEL_PATH, { numThreads: 1 })
  })

  afterAll(async () => {
    await engine?.release()
  })

  it('fixture が読み込めている', () => {
    expect(entries.length).toBeGreaterThan(0)
  })

  it('トークン ID 列が Python と完全一致する', async () => {
    for (const entry of entries) {
      const raw = readFileSync(resolve(FIXTURE_DIR, entry.waveFile))
      const wave = new Float32Array(raw.buffer, raw.byteOffset, entry.nSamples)
      const tokens = await engine.decodeChunk(wave)
      expect(tokens.map((t) => t.tokenId), `${entry.name} のトークン列`).toEqual(entry.tokenIds)
    }
  })

  it('確信度が Python と大きくズレていない (健全性チェック。振る舞い上の帰結はテキスト一致検査が担保)', async () => {
    for (const entry of entries) {
      const raw = readFileSync(resolve(FIXTURE_DIR, entry.waveFile))
      const wave = new Float32Array(raw.buffer, raw.byteOffset, entry.nSamples)
      const tokens = await engine.decodeChunk(wave)
      tokens.forEach((token, i) => {
        const diff = Math.abs(token.confidence - entry.confidences[i])
        expect(
          diff,
          `${entry.name}[${i}] (tokenId=${token.tokenId}) の確信度差: ` +
            `js=${token.confidence} py=${entry.confidences[i]} diff=${diff}`,
        ).toBeLessThan(CONFIDENCE_ABS_TOLERANCE)
      })
    }
  })

  it('欧文・和文テキストが Python と一致する', async () => {
    for (const entry of entries) {
      const raw = readFileSync(resolve(FIXTURE_DIR, entry.waveFile))
      const wave = new Float32Array(raw.buffer, raw.byteOffset, entry.nSamples)
      const tokens = await engine.decodeChunk(wave)
      const ids = tokens.map((t) => t.tokenId)
      const confs = tokens.map((t) => t.confidence)
      expect(convert(ids, confs, { mode: 'european' }).text).toBe(entry.textEuropean)
      expect(convert(ids, confs, { mode: 'japanese' }).text).toBe(entry.textJapanese)
    }
  })
})
