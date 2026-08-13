# LLM テキスト清書機能 設計書

**作成日:** 2026-06-19
**対象:** cw-decorder
**ステータス:** 設計確定 (実装計画作成前)

## 1. 目的

CW デコード結果 (生テキスト) を LLM (Claude / OpenAI / Ollama) に渡し、
次の 3 つを 1 回の呼び出しで実行して「読みやすい日本語」に清書する機能を追加する。

1. **デコード誤りの訂正** (全モード — 欧文・和文とも)
2. **読みやすい日本語への清書** (略語展開・文体整形・カタカナ→漢字かな交じり)
3. **和文モード時の欧文化** — 欧文として意味が通るスパンをカナ⟷欧文対応表で欧文へ

加えて、**LLM が推測・補正した箇所は赤文字で表示**し、確実な情報と区別する。

## 2. アーキテクチャ原則との整合

本機能は「音→符号→文字」三層分離を侵さない **第 4 の独立層 (表示の後段の意味処理)** とする。

- NN・推論エンジン・`TokenConverter`・変換表には一切触れない (原則 1 を堅持)
- カナ⟷欧文対応表は `morse_tokens.py` から**実行時生成**し、二重定義しない (原則 2)
- 生デコード (原文) は清書結果と別に保持し、上書きしない (LLM 誤変換時も原文を失わない)
- 推測箇所の赤表示は、既存の「?の 2 分類 (TABLE_MISS / LOW_CONFIDENCE)」と同じ
  「確実な情報と推測を視覚的に区別する」思想に沿う

## 3. モジュール構成

```
src/llm/
├── __init__.py
├── base.py          # 抽象 LLMProvider, LLMResult, LLMError
├── providers/
│   ├── __init__.py
│   ├── claude.py    # Anthropic Messages API (httpx)
│   ├── openai.py    # OpenAI Chat Completions API (httpx)
│   └── ollama.py    # ローカル /api/chat (httpx)
├── prompt.py        # 清書+誤り訂正+欧文化プロンプト構築 / カナ⟷欧文表生成
├── config.py        # .env 読込 + プロバイダ生成ファクトリ
└── client.py        # 共通 httpx 呼び出し・タイムアウト・例外正規化

src/app/
└── llm_worker.py    # Qt ワーカー (QObject) — UI をブロックしない API 呼び出し
```

## 4. データフロー

```
text_view の確定テキスト (self._committed_text)
   │  「清書」ボタン or 自動清書トグル
   ▼
CWDecoderWindow ──request_llm_transform(text, mode)──▶ LLMWorker (別スレッド)
   │                                                      │ provider.transform()
   ▼  llm_result_ready(str) / llm_error(str)  ◀───────────┘
清書パネル (下部) に表示 (確定=黒 / 推測=赤)
```

- LLM へ送るのは**確定テキストのみ** (暫定グレー文字は不安定なため送らない)
- `mode` (european / japanese / auto) を一緒に渡す。和文・auto のときのみ欧文化指示＋対応表を付与

## 5. プロバイダ層

### 5.1 抽象インターフェース (`base.py`)

```python
@dataclass(frozen=True)
class LLMResult:
    text: str           # 清書後テキスト (推測箇所は ⟦…⟧ マーカー入り)
    provider: str
    model: str

class LLMProvider(Protocol):
    def transform(self, raw_text: str, mode: Mode, *, timeout: float) -> LLMResult: ...

class LLMError(Exception):
    """全プロバイダ共通のエラー型 (キー未設定/オフライン/HTTP/タイムアウト等を正規化)."""
```

### 5.2 3 実装

3 プロバイダはリクエスト/レスポンス形式のみ異なり、`client.py` の共通 httpx 呼び出し
＋例外正規化 (`LLMError`) を共有する。

| プロバイダ | エンドポイント | 認証 |
|---|---|---|
| claude | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| openai | OpenAI Chat Completions API | `OPENAI_API_KEY` |
| ollama | `{ollama_endpoint}/api/chat` (既定 `http://localhost:11434`) | 不要 |

### 5.3 依存

- `httpx>=0.27` — 3 プロバイダ共通の HTTP クライアント (公式 SDK は使わない)
- `python-dotenv>=1.0` — `.env` 読込

## 6. プロンプト (`prompt.py`)

1 回の呼び出しで 3 つの仕事を実行する。

### 6.1 デコード誤りの訂正 (全モード)

- 欧文の既知系統誤り `D↔B, K↔T, Y↔A, 9→O, I→E` 等、`?` (脱落) を文脈から補正
- 和文も同様に文脈補正
- 既知系統誤りをプロンプトのヒントに含める

### 6.2 読みやすい日本語への清書

- 略語展開 (CQ / RST / QTH / OM / TNX / 73 等)、カタカナ→漢字かな交じり、文体整形

### 6.3 和文モード時の欧文化

- `prompt.py` が `morse_tokens.py` の `EUROPEAN_TABLE` / `JAPANESE_TABLE` から
  **カナ→欧文対応表を実行時生成** (同一符号 `code` を共有するエントリのみ)。
  システムプロンプトに正データとして埋め込む
- LLM は「**意味が通る場合**」(コールサイン・RST・数字・Q コード等) を意味的に判断し、
  対応表に従ってそのスパンを欧文へ変換する
- 判断は LLM の強み、変換は対応表で接地 → ハルシネーション抑制
- 欧文・auto モードでは欧文化指示は付けない (混乱防止)

### 6.4 推測箇所のマーキング

- LLM に「直接読めた箇所」と「推測・補正・欧文化した箇所」を区別させ、
  **推測箇所だけ `⟦…⟧` (U+27E6/27E7) で囲って**返させる
- マーカーは CW テキストやプロサイン `<KN>` と衝突しない希少記号を選定
- 対象は 3 種すべて: ①誤り訂正した箇所 ②略語展開で補った箇所 ③欧文化した箇所

## 7. UI

### 7.1 画面構成 (下部に清書パネル追加)

```
┌──────────────────────────────┬──────────┐
│ 生デコード (text_view)         │ レベル    │
│ 確定=黒 / 暫定=グレー          │ メータ    │
├──────────────────────────────┴──────────┤
│ LLM清書パネル (新規 QTextEdit, 読取専用)   │
│  確定箇所=黒 / 推測箇所=赤                  │
├──────────────────────────────────────────┤
│ [プロバイダ▼] [モデル▼] [清書] [☑自動清書]  │
└──────────────────────────────────────────┘
```

### 7.2 推測箇所の赤表示

- ワーカーが受信テキストを `html.escape` した後、`⟦…⟧` を
  `<span style="color:#cc0000;">…</span>` に変換して `setHtml`
- 確定箇所は黒。既存 `text_view` の確定/暫定色分け (`_current_display_html`) と同方式

## 8. 設定 (`AppSettings`)

API キーは含めない (環境変数のみ)。

```python
llm_enabled: bool = False
llm_provider: str = "ollama"          # "claude" | "openai" | "ollama"
llm_model: str = "llama3.1"           # プロバイダ既定モデル
ollama_endpoint: str = "http://localhost:11434"
llm_auto: bool = False                # 自動清書トグル
llm_auto_interval_s: float = 20.0     # 自動清書の最短間隔
llm_timeout_s: float = 30.0
```

- `settings_version` を 3→4 にする。欠損フィールドは既存マイグレーション機構で自動補完
- API キー (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) は `config.py` が
  `python-dotenv` でプロジェクトルートの `.env` から読込
- `.env` を `.gitignore` に追加。`.env.example` (キー名のみ・値空) を同梱
- Ollama はキー不要

## 9. スレッド (既存の鉄則順守)

- `LLMWorker(QObject)` を別 `QThread` に載せる
- UI→ワーカー: Signal (`request_llm_transform`)
- ワーカー→UI: Signal (`llm_result_ready` / `llm_error` / `llm_busy_changed`)
- API 待ち中は「清書」ボタンを無効化＋ステータス表示。多重発火を防止

## 10. 自動清書

- デコード中、確定テキストが前回送信時より伸びていれば `llm_auto_interval_s` ごとに
  1 回だけ送信 (デバウンス)。変化なしならスキップ (コスト抑制)
- 送信中は次の自動清書をスキップ

## 11. エラー処理 (サイレント失敗を作らない)

- キー未設定 / オフライン / タイムアウト / HTTP エラー / Ollama 未起動を `LLMError` に
  正規化し、**ステータスバーに具体的な日本語メッセージ**で表示
- 生デコード・原文は保持し消さない。失敗してもアプリは継続。デコード本体は無影響

## 12. テスト方針 (TDD)

- `prompt.py`: カナ⟷欧文対応表が `morse_tokens.py` から正しく生成されるか (単一ソース照合)
- 各プロバイダ: httpx をモックし、リクエスト整形・レスポンス抽出・エラー正規化を検証
- マーカー→赤 HTML 変換: `⟦…⟧` のエスケープ＋span 化、衝突文字 (`<KN>` 等) の安全性
- 自動清書のデバウンス (変化なしで送らない) ロジック
- 既存テスト (410 passed) は無改修で緑のまま (LLM 層は独立)

## 13. 依存追加

- `httpx>=0.27`
- `python-dotenv>=1.0`

## 14. スコープ外 (YAGNI)

- ストリーミング応答 (まずは一括応答)
- 清書結果のファイル保存・エクスポート (必要なら後続で)
- プロバイダごとの高度なパラメータ (temperature 等) の UI 露出 (既定値固定で開始)
