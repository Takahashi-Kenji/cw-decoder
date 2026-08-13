# LAN 経由の音声入力 — マージ後に残る保留事項 (2026-08-04)

最終レビューで「後回しで良い」と判定された項目。実装は正しく、いずれも実害は無いか軽微。

## 実運用で気になりうるもの
- M3: `is_connected` / `last_error` を読む本番コードが無く、**運用中に LAN が切れても UI は最後の表示のまま**。設計書 §5.1 が「ワーカーは今は使わない」と明記しているので設計どおりだが、切断が利用者に伝わらない
- M2: 起動時の `capture started ... (LAN host:port)` が数秒で診断メッセージに上書きされる。恒久表示にすると意図どおりになる
- M1: `_on_stop()` が UI スレッドから `stop()` を呼ぶため、再接続待ち中に「停止」を押すと GUI が最大 2 秒固まる
- M4: `parse_endpoint` がポート範囲 (1-65535) を検証しない。`host:99999` は OS 側で折り返し、5 秒待たされてから「接続できません」になる

## 小さいもの
- M5: 正常な `stop()` でも `last_error` に「転送元が切断しました」が残る (診断値のノイズ)
- M6: 送信側は `AF_INET` 固定で IPv6 非対応。受信側の `parse_endpoint` は IPv6 を扱えるので非対称
- M7: `audio_send.py` の `accept()` が `TimeoutError` 以外の `OSError` を捕まえない
- M8: `INSTALL.md` に `--port` / `--bind` が未文書化 (docstring にはある)
- `encode_samples` は 0 方向切り捨て (voice_to_string 互換のため踏襲、誤差 1 LSB)
- `parse_endpoint` は `"host:"` (末尾コロン) を暗黙に既定ポートへフォールバック
- `drain()` は `stop()` 時に `resample_chunk(last=True)` でフラッシュしない (既存 `AudioCapture` と同挙動)
- `_make_resampler()` は `_connect()` 内で `_running` 再チェックより前に呼ばれるため、`stop()` 後の遅延接続成立で `_resampler` が上書きされ得る (fd リークではなく無害)
- `start()` 失敗時に `_capture` に失敗インスタンスが残る (本ブランチ以前からの挙動)
- `--device` に存在しない番号を渡したときのエラーに「`--list` で確認」の誘導が無い
- `local_addresses()` は `127.` のみ除外。リンクローカル 169.254.x.x を接続先候補に出しうる

## テストが無い領域
- `scripts/audio_send.py` に自動テストが 1 行も無い。`build_parser()` と `client_is_gone()` は実ネットワーク無しで書ける
- 送信 → 受信のラウンドトリップ (`audio_send.serve()` が吐く実バイト列の検証)
- `stop()` の常用経路での終了保証 (現在は「遅延ヘッダ」という例外的経路のみ検証)
