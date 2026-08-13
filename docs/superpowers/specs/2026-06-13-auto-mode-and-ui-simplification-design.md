# 自動モード切替 + UI 簡素化 設計書

- 日付: 2026-06-13
- ブランチ: `feature/auto-mode`
- 対象: cw-decorder ライブデコードアプリ

## 1. 背景と目的

Phase 6（スライディングウィンドウ再デコード）の実用化により、ライブ連続モードが
欧文・和文とも実用精度に達した。これを踏まえ次の 2 つを実装する。

1. **自動モード切替**: モデルが出力する `[ホレ]`（和文開始）/`[ラタ]`（和文終了）
   プロサインを検出し、欧文⇄和文の変換表を自動で切替える。
2. **UI 簡素化**: 実用に達したライブ連続モードに一本化し、レガシーな手動デコード経路
   （手動バッファ蓄積・オフライン一括 decode・無音検出自動チャンク）を UI・コード
   ともに撤去する。

この 2 つは同一の `set_mode`/worker 周りを触るため、ひとつの変更として実装する。

## 2. アーキテクチャ原則の確認

- 「音→符号」（NN）と「符号→文字」（変換表）の分離を維持する。
- 自動モードのモード切替判定は **変換表層（converter）** に置く。worker や UI に
  「符号→文字」ロジックを漏らさない。
- NN・SlidingWindowDecoder はモード非依存のまま（符号トークンを出力するだけ）。

## 3. 自動モード変換器

### 3.1 トリガと初期モード

- トリガ: **プロサイン検出**（統計推定は行わない）。
- 初期モード: **欧文**。`[ホレ]` 検出まで欧文。

### 3.2 変換アルゴリズム

トークン列を左から 1 回走査し、ローカルな「現在サブモード」を保持する。
擬似コード（符号は U+30FB 中黒 `・` と U+002D `-`）:

```
current = "european"            # 既定の開始モード
for (code, conf) in tokens:
    if code == "-・・---":        # ホレ (和文開始)
        emit "[ホレ]"
        current = "japanese"
    elif code == "・・・-・":       # ラタ / SN (同符号)
        if current == "japanese":
            emit "[ラタ]"
            current = "european"
        else:
            emit "[SN]"          # 欧文中はプロサイン SN、モード切替なし
    else:
        emit convert(code, table_of(current))
```

### 3.3 同符号 `・・・-・` の扱い（ラタ = SN）

`・・・-・` は和文 `[ラタ]`（和文終了）と欧文プロサイン `[SN]` で完全に同符号。
**現在サブモードが japanese のときだけ** 和文終了として欧文に戻す。european の
ときは `[SN]` として表示し、モードは変えない。これにより文脈依存の曖昧性を解決する。

### 3.4 濁点・半濁点合成

`TokenConverter` の濁点合成（`converter.py` の `self.mode == "japanese"` 判定）は
「**現在サブモード == japanese**」へ置換する。和文セグメント内でのみ前カナと合成され、
欧文セグメントでは合成しない（正しい挙動）。

### 3.5 確定/暫定の開始モード引き継ぎ（重要）

`workers.py:_emit_live_view` は確定トークン列と暫定トークン列を **別々に** `convert()`
する。自動モードでは暫定の開始サブモードが確定末尾のサブモードを継ぐ必要がある。

- `TokenConverter.convert(ids, confs, initial_mode="european")` に開始モード引数を追加。
- `ConvertResult` に `final_mode: Mode` を追加（走査後の末尾サブモード）。
- worker:
  ```
  res_c = converter.convert(committed_ids, committed_confs, initial_mode="european")
  res_p = converter.convert(prov_ids,      prov_confs,      initial_mode=res_c.final_mode)
  ```
- 手動モード（欧文/和文固定）では `initial_mode` は固定値、`final_mode` は無視で従来挙動。
- `convert_timed`（自動モードでは利用しないが）も同様に `initial_mode`/`final_mode` を
  受け渡せるよう揃える（語間ギャップ推定は列全体を見るため、1 回走査方式を維持）。

### 3.6 決定性

worker は毎 redecode でトークン列**全体**を渡して再変換する（`_emit_live_view`）。
自動変換は毎回 european スタートで全列を歩くため、呼び出し間の状態保持は不要で
決定論的。途中の `[ホレ]` が列を確定的に分割する。

### 3.7 ステータス表示

自動モード時、worker は `res_c.final_mode`（= 現在の推定モード）をステータスバーへ
「自動 (現在: 和文)」のように表示する。

## 4. UI 変更

### 4.1 モードコンボ

`[欧文 (european) / 和文 (japanese)]` → `[欧文 / 和文 / 自動]` の 3 択に拡張。

### 4.2 ボタン（上段）

| 変更前 | 変更後 |
|---|---|
| `[開始]` `[停止]` | 据え置き（音声取込みの制御） |
| `[● デコード録音開始]`（チェック式） | `[● デコード中 / デコード停止]` トグルに置換 |
| `[デコード]` | 削除 |

トグル ON = デコード中（スライディング窓へ給餌）、OFF = 停止。`開始` していなければ
押下不可（従来どおり）。「開始 → レベル調整 → デコード中」の段階フローを維持する。

### 4.3 下段

`自動チャンク` チェックボックスを削除。`AGC`/`BPF`/`中心`/`帯域`/スペクトログラム/
`読込…`/`録音開始` は据え置き。

## 5. worker 簡素化（レガシー経路のコード削除）

### 5.1 削除する要素

- メソッド: `decode_and_reset`, `_trigger_auto_decode`, `_process_recording_block`
  のレガシー分岐, `set_auto_chunk_enabled`, `_emit_tokens`。
- 状態: `_accumulated_audio`, `_silence_run_samples`, `auto_chunk_*` 一式。
- シグナル: `full_decode_completed`。
- 引数: `live_continuous`（常時ライブに固定し撤去）, `auto_chunk_*`。
- 旧 `_decoder`（StreamingDecoder）: ライブ `_tick` で未使用のため削除。
  併せて `stop()` の flush 分岐、`workers.py` の StreamingDecoder 構築も除去。
- `chunk_duration_s`/`chunk_overlap_s`: 旧 `StreamingDecoder` 専用引数のため撤去
  （worker 引数・`src/infer/stream.py` 参照とも `_decoder` 削除に伴い不要）。

### 5.1.1 run_app.py / 起動経路への波及

- `run_app.py` の `--chunk`/`--overlap` 引数を削除（旧 stream 用のため）。
- `main_window.main()` の `chunk_duration_s`/`chunk_overlap_s` 引数と、
  これらを使う `[init]` ログ行（`main_window.py` 524-526 付近）を撤去・更新する。
- `src/infer/stream.py`（StreamingDecoder 本体）は他に参照が無ければ削除候補。
  参照状況を実装時に確認し、残テスト（`test_streaming_*` 等）と合わせて整理する。

### 5.2 改名・整理

- `_buffer_recording` → `_decoding`、`set_buffer_recording` → `set_decoding`
  （意味が「デコード ON/OFF」になるため）。
- `_process_recording_block` は `_feed_live_block` の直呼びに簡約。
- `_tick` の給餌ゲートは `if self._decoding: self._feed_live_block(proc_block)`。

### 5.3 main_window 側

削除した signal/slot 接続（`full_decode_completed`, `request_full_decode`,
`request_set_auto_chunk`, `set_buffer_recording` 関連等）を除去。トグルは
`request_set_decoding` 経由で worker の `set_decoding` を呼ぶ。

## 6. 設定移行（settings_version 2 → 3）

### 6.1 変更

- 削除キー: `auto_chunk_enabled`, `auto_chunk_silence_sec`,
  `auto_chunk_min_buffer_sec`, `auto_chunk_silence_amplitude`, `live_continuous`,
  および旧 stream 用 `chunk_duration_s`/`chunk_overlap_s`。
- `mode` の許容値に `"auto"` を追加。
- `settings_version` を 3 に更新。
- 既存の `MIGRATIONS`（`settings.py` の `chunk_duration_s` 1.5→5.0 等、削除キーに
  対する旧マッピング）は対象キー撤去に合わせて整理する。

### 6.2 マイグレーション

既存 `settings.json`（version 2）読込時、削除対象キーを除去し version 3 へ更新。
未知キーは無視して読み込めること。`mode` が `auto` の場合も正しく復元すること。

## 7. テスト方針（TDD）

実装前にテストを書く。

### 7.1 自動モード変換器

- 欧文 → `[ホレ]` → 和文 → `[ラタ]` → 欧文 の往復が正しく切替わる。
- 欧文中の `・・・-・` は `[SN]`（切替なし）、和文中は `[ラタ]`（切替あり）。
- `[ホレ]` の連続・`[ラタ]` 単独（欧文中）の冪等性。
- `final_mode` が走査末尾のサブモードを返す。
- `initial_mode` を与えると途中モードから開始できる（暫定引き継ぎ）。
- 濁点合成が和文セグメント内のみで起き、欧文セグメントでは起きない。

### 7.2 worker

- 自動モードで確定→暫定のモード引き継ぎ（和文の途中で暫定に切れても和文で変換）。
- `set_decoding(True/False)` で給餌が ON/OFF される。

### 7.3 回帰・撤去

- 削除したレガシー（`decode_and_reset`/auto_chunk 系）のテストを除去・置換。
- ライブ連続が唯一経路になったことの確認（B1 競合が構造的に消滅）。

### 7.4 マイグレーション

- version 2 settings.json → version 3 への移行単体テスト。

## 8. 非対象（YAGNI）

- 統計推定によるモード判定（プロサインが無い相手への対応）は本変更では行わない。
- `[ホレ]`/`[ラタ]` を非表示にする/境界マーカー化するオプションは作らない
  （そのまま表示する仕様で確定）。
- PyInstaller 配布・ホレ/ラタ以外のプロサイン自動処理は対象外。

## 9. 受け入れ基準

- 既存テスト + 新規テストが全緑。
- アプリ起動 → モード「自動」→ 和文 QSO（`[ホレ]`…`[ラタ]`）を流すと、ホレ以降が
  和文表、ラタ以降が欧文表で表示される。
- UI 上に手動デコード/自動チャンク関連の操作が残っていない。
- 旧 settings.json から起動してもエラーなく動作し、version 3 に更新される。
