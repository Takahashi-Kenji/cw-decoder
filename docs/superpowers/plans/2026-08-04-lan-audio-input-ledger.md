# SDD ledger — plan: docs/superpowers/plans/2026-08-04-lan-audio-input.md
Task 1: minor (deferred): tests/test_net_audio.py の docstring が「ダミーサーバを立てる」と書くが Task 1 のテストはソケット未使用 (Task 2 で実態が追いつく)
Task 1: minor (deferred): encode_samples は astype("<i2") で 0 方向切り捨て。四捨五入ではない (voice_to_string と同じ挙動なので互換のため踏襲)
Task 1: minor (deferred): parse_endpoint は "host:" (末尾コロン) を暗黙に既定ポートへフォールバックする。テスト未定義のケース
Task 1: complete (commits d4be32d..afac55c, review clean)
Task 2: minor (deferred): _connect 内 self._sock = self._connect() 相当の書き込みがロック外。is_connected はロック越しに読む。Task 3/4 で _reader_loop が _sock を書き換えるため踏襲注意
Task 2: minor (deferred): tests の import (contextlib/socket/threading/time) がファイル中盤にある (ブリーフ由来。PEP8 的には先頭が望ましい)
Task 2: minor (deferred): _FakeSender.close() は join(timeout=2.0) だが accept は settimeout(5.0)。デーモンスレッドが最大5秒残る可能性 (実害は無いと判断)
Task 2: fix round 1/5 (1 addressed, 0 open — _make_resampler のソケットリーク; commits 9085fe9..e355248)
Task 2: complete (commits afac55c..e355248, review clean)
Task 3: complete (commits e355248..808df12, review clean)
Task 3: 申し送り -> Task 4: start() の self._sock = self._connect() がロック外。Task 4 の要求に組み込む
Task 3: 申し送り -> Task 4: _reader_loop は _reconnect() の戻り値を self._sock へ書き戻さない。_reconnect() 自身がロック内で self._sock を設定する契約 (_note_disconnect と対)
Task 3: minor (deferred): drain() は stop() 時に resample_chunk(last=True) でフラッシュしない (既存 AudioCapture も同じ挙動)
Task 3: minor (deferred): test_drain_downsamples_16k_to_8k はサンプル数と振幅下限のみ検証。周波数成分は見ていない
Task 3: minor (deferred): 奇数バイトが跨いだ場合を直接検証するテストが無い
Task 4: fix round 1/5 (2 addressed + 回帰テスト追加, 0 open — stop() 後のソケットリーク / 例外捕捉の拡大; commits 7a77312..6a51618)
Task 4: minor (deferred): 回帰テストの最終確認が固定 sleep + 単発 assert (条件ポーリングでない)。flaky にはなりにくいと再レビューが判断
Task 4: minor (deferred): _make_resampler() は _connect() 内で _running 再チェックより前に呼ばれるため、stop() 後の遅延接続成立で self._resampler が上書きされ得る (fd リークではなく無害。次の start() で作り直される)
Task 4: complete (commits 808df12..6a51618, review clean)
Task 5: minor (deferred): ローカル経路のステータス文言が "→ 8000 Hz" ハードコードから "→ {self.sample_rate} Hz" に変化 (ブリーフ記載どおり。既定値 8000 なので通常は同一表示)
Task 5: minor (deferred): start() 失敗時に self._capture に失敗インスタンスが残る (Task 5 以前からの既存挙動。新規持ち込みではない)
Task 5: complete (commits 6a51618..62ec128, review clean)
Task 6: fix round 1/5 (1 addressed + テスト3件追加, 0 open — 不正な --net-source でアプリ無反応; commits 0ef5471..f031165)
Task 6: complete (commits 62ec128..f031165, review clean)
Task 7: minor (deferred): --device に存在しないデバイス番号を渡したときのエラーに「--list で確認」の誘導が無い
Task 7: minor (deferred): local_addresses() は 127. のみ除外。リンクローカル 169.254.x.x を接続先候補に出しうる
Task 7: complete (commits f031165..0f49708, review clean)
最終レビュー: Important-1 接続失敗メッセージが _on_stop() の「停止しました」で上書きされ画面に残らない (設計書 §9-3 未達)
最終レビュー: Important-2 soxr 不在時 _resample_to_8k が 8000 固定のため audio_send が 8kHz を 16000 と申告して送る (無言で全崩壊)
最終レビュー: minor (deferred) M1 _on_stop が UI スレッドから stop() を呼ぶため再接続待ち中は GUI が最大2秒固まる
最終レビュー: minor (deferred) M2 capture started の LAN 表示が数秒で診断メッセージに上書きされる
最終レビュー: minor (deferred) M3 is_connected/last_error を読む本番コードが無く、運用中の切断が UI に伝わらない
最終レビュー: minor (deferred) M4 parse_endpoint がポート範囲 (1-65535) を検証しない
最終レビュー: minor (deferred) M5 正常な stop() でも last_error に「転送元が切断しました」が残る
最終レビュー: minor (deferred) M6 送信側は AF_INET 固定で IPv6 非対応 (受信側は対応) の非対称
最終レビュー: minor (deferred) M7 audio_send の accept() が TimeoutError 以外の OSError を捕まえない
最終レビュー: minor (deferred) M8 INSTALL.md に --port / --bind が未文書化
