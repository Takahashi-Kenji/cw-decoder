# LAN 経由の音声入力 設計書

作成日: 2026-08-04

## 1. 背景と問題

受信機を繋いだ PC と、GPU を積んだ PC が別である。現在の cw-decorder は
`sounddevice` のローカル入力デバイスからしか音を取れないため、この構成では使えない。

リモートデスクトップのマイク転送は使えない。同じ community-tools 内の
`voice_to_string` が 2026-08-01 に実測しており、RDP は音声入力を狭帯域コーデックで
圧縮してエネルギーの 99% が 1000 Hz 以下になる。CW のトーンは 486〜496 Hz なので
一見通りそうだが、**キーイングの立ち上がり・立ち下がりが鈍る**ため符号長の判別が
壊れる懸念がある。いずれにせよ非圧縮で流せるなら迂回する理由はない。

`voice_to_string` は既にこの問題を解いており、`scripts/audio_send.py`（送信側）と
`src/audio/net_source.py`（受信側）で 16 kHz モノラルの非圧縮 PCM を TCP で素通しする。
本設計はこの方式を cw-decorder に持ち込む。

## 2. ゴールと非ゴール

**ゴール**: 無線機を繋いだ PC で `scripts/audio_send.py` を動かし、GPU 側で
`python scripts/run_app.py --ckpt ... --net-source 192.168.1.20` と指定すると、
ローカルデバイスと同じようにライブデコードできる。

**非ゴール**:

- GUI からの接続先設定（入力デバイスコンボへの追加・設定の永続化）。CLI フラグのみ。
- 暗号化・認証。家庭内 LAN 前提で、`voice_to_string` と同じ扱いとする。
- 複数クライアントの同時接続。送信側は 1 接続のみ受ける。
- 送信側での BPF・AGC。生の音を流し、整形は cw-decorder 側で行う（§6）。

## 3. voice_to_string との関係

**プロトコルは互換に保つ**。マジックは `V2SA` のまま変更しない。名前は
voice_to_string 由来だが、揃えておけば**1 つの送信プロセスをどちらのアプリからも
掴める**利点がある。無線機 PC で送信を 1 つ動かしておけば、CW を聞くときは
cw-decorder、音声を聞くときは voice_to_string、と繋ぎ替えられる。

```
接続時に送信側が 12 バイトのヘッダを 1 回だけ送る (リトルエンディアン)
    magic        4 bytes  b"V2SA"
    sample_rate  uint32   16000
    channels     uint16   1
    bits         uint16   16
以降は int16 のサンプル列が途切れなく続く
```

帯域は 16000 × 2 = 32 kB/s (256 kbps)。

**コードは共有せず、cw-decorder 側に持つ**。2 プロジェクトは別の `.venv` を持ち
現在クロス参照が無い。共通パッケージ化は依存関係と仮想環境の問題を招くため採らない。
重複を許す代わりに、無線機 PC に cw-decorder だけ入れれば完結する。

**受信側の実装は移植であって逐語コピーではない**。理由は 2 つ。

| | voice_to_string | cw-decorder |
|---|---|---|
| 内部サンプルレート | 16 kHz（Whisper がそのまま食う） | **8 kHz**（16→8 のリサンプルが要る） |
| `AudioCapture.drain()` | 単一の `np.ndarray` | `list[np.ndarray]` |
| レベル取得 | `level()` → `InputLevel` | `level_db_rms` プロパティ |

## 4. アーキテクチャ

```
[無線機PC]  受信機 → サウンドカード → scripts/audio_send.py     16 kHz mono int16
                                          ↓ TCP :45678 非圧縮 32 kB/s
[GPU PC]    NetworkAudioCapture → soxr 16k→8k → BPF → AGC → CNN+BiLSTM+CTC
```

`AudioInferenceWorker` が `AudioCapture` に対して使っているのは次の 5 つだけで
（`src/app/workers.py:281,282,352,367,370`）、ここが差し替えの境界になる。

| メンバ | 用途 |
|---|---|
| `start()` | 開始 |
| `stop()` | 停止 |
| `source_sample_rate` | ステータス表示 |
| `drain()` → `list[np.ndarray]` | 8 kHz ブロックの取り出し |
| `level_db_rms` | UI レベルメータ |

`NetworkAudioCapture` はこの 5 つを同じ形で提供する。ワーカーは実デバイスか
LAN 経由かを意識しない。

## 5. コンポーネント

### 5.1 `src/infer/net_audio.py`（新規）

**`parse_endpoint(text, *, default_port=45678) -> tuple[str, int]`**

`"host:port"` / `"host"` / `"[::1]:45678"` を `(host, port)` へ。ポートが数値でなければ
`ValueError`。

**`encode_header(sample_rate=16000) -> bytes` / `encode_samples(block) -> bytes`**

送信側が使う。`encode_samples` は float32 (-1..1) を int16 LE へ。

**`NetworkCaptureError(RuntimeError)`**

接続不能・ヘッダ不正。

**`NetworkAudioCapture`**

- `__init__(host, port=45678, *, target_sample_rate=8000, expected_source_rate=16000, connect_timeout_s=5.0)`
  — `expected_source_rate` はヘッダ検証に使う期待値であって、実際の値はヘッダから受け取る。
- `start()` — 同期的に 1 回目の接続とヘッダ検証を行い、失敗なら `NetworkCaptureError`。
  「繋がっていないのにデコード中と表示する」状態を作らない。成功後は受信スレッドを起こす。
- `stop()` — 二重呼び出しは無害。
- `drain()` — 溜まった受信ぶんを `soxr.ResampleStream` で 8 kHz に変換し
  `list[np.ndarray]` で返す。無ければ空リスト。
- `source_sample_rate` — ヘッダで受け取った実際の値。`start()` 前は `None`
  （既存 `AudioCapture` と同じ規約）。
- `level_db_rms` — 直近ブロックの RMS dBFS。無音時 -120.0。既存 `AudioCapture` と同じ規約。
- `is_connected` / `reconnects` / `last_error` — 状態表示用（ワーカーは今は使わないが
  診断のために持つ）。

リサンプラは `AudioCapture` と同じく **start 時に 1 つ作って使い回す**。
`soxr.ResampleStream` はブロック境界の状態を保持するため、ブロックごとに作り直すと
歪む（Phase 5 で `scipy` の per-block リサンプルが 37 トークン → 1 トークンに破壊した
実績がある。`docs/design.md` §3 参照）。**再接続時はリサンプラを作り直す**。
ストリームが途切れるため状態を引き継ぐと不正になる。

バッファ上限は 30 秒。超えたら古い方から捨て、`dropped_blocks` を増やす。

### 5.2 `scripts/audio_send.py`（新規・移植）

無線機を繋いだ PC で動かす。`src/infer/audio.py` の `AudioCapture` を
`target_sample_rate=16000` で使い、`drain()` が返すリストを連結して送る。

```
python scripts/audio_send.py --list              # 入力デバイス一覧
python scripts/audio_send.py --device 13         # 送信開始 (Ctrl+C で終了)
python scripts/audio_send.py --device 13 --port 45678 --bind 192.168.1.20
```

起動時に自 PC の IP を列挙し、GPU 側で打つコマンドをそのまま表示する。

`client_is_gone()` による切断検出を移植する。TCP は相手が消えても `sendall` が
すぐには失敗せず、放っておくと死んだ接続へ送り続けて `accept` に戻らないため
**次の接続が繋がらなくなる**（voice_to_string で実際に踏んだ問題）。受け側は
データを送ってこないので、読める状態になっていること自体が EOF の合図になる。

### 5.3 `scripts/run_app.py`（変更）

`--net-source HOST[:PORT]` を追加。指定時はメインウィンドウ経由でワーカーへ渡す。

### 5.4 `src/app/workers.py`（変更）

`AudioInferenceWorker` に `set_net_source(endpoint: str | None)` を追加。`start()` で
`endpoint` が設定されていれば `NetworkAudioCapture`、無ければ従来の `AudioCapture` を作る。
ステータス文言は `capture started @ 16000 Hz → 8000 Hz (LAN 192.168.1.20:45678)` の形にし、
どちらの経路で動いているか一目で分かるようにする。

`--net-source` 指定時は入力デバイスコンボを無効化する（選んでも効かないため）。

## 6. BPF は cw-decorder 側で掛ける

送信側では一切フィルタしない。理由は 2 つ。

1. アプリの BPF 設定（中心・帯域）をそのまま効かせられる。無線機 PC 側に同じ設定を
   二重に持たせると食い違いの元になる
2. 生の音を送っておけば、同じ送信プロセスを voice_to_string からも掴める（§3）

**BPF は必ず ON で運用すること。** 2026-08-03 の実測で、BPF 未通過の音を
モデルに入れると打鍵録音 90 件の TER が **97.03%** まで崩壊した（BPF を掛けると
14.44%、無音パディングも加えて 10.60%）。詳細は `docs/phase4_data_collection.md`。
ライブ経路では `_StreamingBPF`（`src/app/workers.py:15`）が担当するため、
LAN 経路でも UI の BPF を ON にしていれば同じ整形が掛かる。

## 7. エラー処理

| 事象 | 挙動 |
|---|---|
| 初回接続失敗 | `NetworkCaptureError` を `error` シグナルへ。「送信側が動いているか」「ファイアウォールがポートを通しているか」を本文に含める |
| magic 不一致 | 「ポート番号を間違えていませんか」と案内して停止 |
| 1ch/16bit 以外 | 未対応として明示エラー |
| サンプルレートが 16000 でない | 明示エラー。8000 が来た場合も弾く（無言で通すとリサンプル比が狂う） |
| 接続断 | 1 秒間隔で自動再接続。`last_error` に理由を保持。送信側を後から起動しても繋がる |
| 消費側の停止 | 30 秒を超えた分を古い方から破棄し `dropped_blocks` を増やす |

## 8. テスト

`tests/test_net_audio.py`（新規）。実ネットワークは使わず、`localhost` にダミーの
送信サーバを立てて検証する。

| # | 内容 |
|---|---|
| 1 | `parse_endpoint` の正常系（`host` / `host:port` / `[::1]:45678`）と異常系（空文字・非数値ポート） |
| 2 | `encode_header` → `struct.unpack` の往復が仕様どおり |
| 3 | `encode_samples` が float32 (-1..1) を int16 LE にし、範囲外をクリップする |
| 4 | ダミーサーバから 16 kHz の既知波形を流し、`drain()` が **8 kHz の長さ**（サンプル数が約半分）で返すこと |
| 5 | magic 不一致・チャンネル数不正・サンプルレート不一致でそれぞれ `NetworkCaptureError` |
| 6 | 接続断後に送信側を上げ直すと `reconnects` が増えて受信が再開すること |
| 7 | `NetworkAudioCapture` が `AudioCapture` と同じ 5 メンバを持つこと（インターフェース適合。片方だけ変更されると壊れるのを防ぐ） |
| 8 | バッファ上限を超えたら古いブロックが捨てられ `dropped_blocks` が増えること |

`soxr` が無い環境ではリサンプル部分をスキップする（既存 `AudioCapture` と同じ扱い）。

## 9. 受け入れ条件

1. 無線機 PC で `audio_send.py` を起動し、GPU PC で `run_app.py --net-source <IP>` を
   実行するとライブデコードが動く
2. 送信側を止めて再起動すると、GPU 側は操作なしで復帰する
3. 接続先を間違えたとき、原因が分かるメッセージが出る
4. `tests/test_net_audio.py` が全パス、既存テストが退行しない
5. ローカルデバイス経路が従来どおり動く（`--net-source` 未指定時）

## 10. 関連

- `voice_to_string/scripts/audio_send.py`、`voice_to_string/src/audio/net_source.py` — 移植元
- `src/infer/audio.py` — `AudioCapture`（差し替え対象のインターフェース）
- `src/app/workers.py` — `_StreamingBPF` / `_AGC` / `AudioInferenceWorker`
- `docs/phase4_data_collection.md` — BPF 必須の根拠
- `docs/design.md` §3 — soxr ステートフルリサンプルの経緯
