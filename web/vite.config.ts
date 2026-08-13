import { defineConfig } from 'vite'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'

/**
 * LAN 配信用の自己署名証明書 (任意)。
 *
 * `web/certs/lan.{key,crt}` が置かれていればそれを使って HTTPS で listen する。
 * 無ければ従来どおり HTTP。**証明書は git 管理外** (`scripts/make_lan_cert.sh` で生成)。
 *
 * なぜ必要か: `getUserMedia` (マイク) はセキュアコンテキストでしか使えない。
 * `localhost` は例外扱いされるので同じ PC なら HTTP で足りるが、**別 PC の
 * ブラウザから LAN IP で開くとマイクがブロックされる**。HTTPS ならブロックされない
 * (自己署名なので初回は警告画面が出る。「詳細設定」→「アクセスする」で通す)。
 */
const CERT_KEY = resolve(__dirname, 'certs/lan.key')
const CERT_CRT = resolve(__dirname, 'certs/lan.crt')

/**
 * LAN 公開時 (`--host` 付き) だけ HTTPS にする。
 *
 * `localhost` はセキュアコンテキストの例外なので HTTP でマイクが使える。にもかかわらず
 * 証明書があるだけで常に HTTPS にすると、自分のマシンで開くたびに自己署名の警告を
 * 通す羽目になる (それだけの手間で、得るものは無い)。
 * LAN IP で開くときは HTTPS でないとマイクがブロックされるので、そのときだけ有効にする。
 */
const isLanServe = process.argv.includes('--host')
const https =
  isLanServe && existsSync(CERT_KEY) && existsSync(CERT_CRT)
    ? { key: readFileSync(CERT_KEY), cert: readFileSync(CERT_CRT) }
    : undefined

export default defineConfig({
  // onnxruntime-web の .wasm を事前バンドルの対象から外す
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    rollupOptions: {
      // index.html (デコーダ本体) と bench.html (ORT Web ベンチ) の 2 ページ構成。
      // 既定では index.html のみがビルド対象になるため、bench.html を明示する。
      input: {
        main: resolve(__dirname, 'index.html'),
        bench: resolve(__dirname, 'bench.html'),
      },
    },
  },
  server: {
    https,
    headers: {
      // ORT Web のマルチスレッドを有効にするためのヘッダ。
      // 無くても動く (シングルスレッド) が、あれば速くなる。
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    https,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
