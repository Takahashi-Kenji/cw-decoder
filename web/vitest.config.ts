import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    // ONNX の読み込みと推論に時間がかかるため長めに取る
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
})
