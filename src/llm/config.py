"""`.env` 読込と AppSettings → LLMProvider 生成ファクトリ."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from src.infer.settings import AppSettings
from src.llm.base import LLMError, LLMProvider
from src.llm.providers.claude import ClaudeProvider
from src.llm.providers.ollama import OllamaProvider
from src.llm.providers.openai import OpenAIProvider

_loaded = False


def _ensure_env_loaded() -> None:
    global _loaded
    if not _loaded:
        load_dotenv()   # プロジェクトルートの .env を読む (無ければ無視)
        _loaded = True


def _require_key(name: str) -> str:
    _ensure_env_loaded()
    key = os.environ.get(name, "").strip()
    if not key:
        raise LLMError(
            f"{name} が設定されていません。.env または環境変数に設定してください。"
        )
    return key


# 一覧が取れなかったときに出す候補 (小さい順)。
# **清書は文字の変換なので小さいモデルで足りる** (運用者の判断)。
FALLBACK_OLLAMA_MODELS: tuple[str, ...] = (
    "qwen3.5:4b", "gemma4:e4b", "gemma4:12b",
)

# クラウドプロバイダのモデル候補 (先頭が既定)。主画面と設定画面の両方が
# ここを参照する (書き写すと必ず食い違う)。
# **ollama はここに書かない** — 候補は実機から取る (``list_ollama_models``)。
# 書き固めると、入っていないモデル名が既定になって「押しても動かない」状態に
# なる (実際に llama3.1 でそうなっていた)。
PROVIDER_MODELS: dict[str, tuple[str, ...]] = {
    # 推論モデルは思考が既定でオンで遅いので、軽い選択肢も出す
    "openai": ("gpt-5.6-luna", "gpt-5", "gpt-5-mini", "gpt-4.1"),
    # Haiku を先頭に (運用者の判断)。清書は文字の変換なので軽い方で足りる
    "claude": ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"),
}


def static_models_for(provider: str, endpoint: str = "") -> list[str]:
    """**通信せずに**出せるモデル候補。

    設定画面の既定はこちら (開くだけで最大 3 秒固まらないため)。
    Ollama の実機照会 (``list_ollama_models``) は主画面が明示的に注入する。
    """
    if provider == "ollama":
        return list(FALLBACK_OLLAMA_MODELS)
    return list(PROVIDER_MODELS.get(provider, ()))


def list_ollama_models(endpoint: str, timeout: float = 3.0) -> list[str]:
    """Ollama にインストール済みのモデル名を返す.

    候補をコードに書き固めると、実際には入っていないモデル名が既定になって
    「押しても動かない」状態が起きる (実際に llama3.1 でそうなっていた)。
    取れなければ ``FALLBACK_OLLAMA_MODELS`` を返す。**例外は投げない**
    (起動時に呼ぶので、Ollama が動いていなくてもアプリは起動できる必要がある)。
    """
    import httpx

    try:
        with httpx.Client(timeout=timeout) as client:
            body = client.get(f"{endpoint.rstrip('/')}/api/tags").json()
        names = [m["name"] for m in body.get("models", []) if m.get("name")]
    except Exception:                                      # noqa: BLE001
        return list(FALLBACK_OLLAMA_MODELS)
    sized = {m["name"]: m.get("size", 0) for m in body.get("models", [])}
    # 埋め込み専用モデルは会話に使えないので候補から外す。
    # クラウドモデルはローカルの実体を持たず size がごく小さく返るので、
    # そのまま並べると「小さい順」の先頭に来てしまう。名前で判定して後ろへ回す
    # (**GPU をローカル LLM に使いたい**のが目的なので、既定はローカルであるべき)。
    def is_cloud(name: str) -> bool:
        return "cloud" in name.lower()

    def sort_key(name: str) -> tuple[int, int]:
        return (1 if is_cloud(name) else 0, sized.get(name, 0))

    names = [n for n in names if "embed" not in n.lower()]
    if not names:
        return list(FALLBACK_OLLAMA_MODELS)
    return sorted(names, key=sort_key)


def create_provider(settings: AppSettings) -> LLMProvider:
    """設定からプロバイダを生成する. キー未設定や未知プロバイダは LLMError."""
    provider = settings.llm_provider
    model = settings.llm_model
    if provider == "ollama":
        return OllamaProvider(model=model, endpoint=settings.ollama_endpoint)
    if provider == "openai":
        return OpenAIProvider(model=model, api_key=_require_key("OPENAI_API_KEY"))
    if provider == "claude":
        return ClaudeProvider(model=model, api_key=_require_key("ANTHROPIC_API_KEY"))
    raise LLMError(f"未知の LLM プロバイダ: {provider!r}")


__all__ = [
    "FALLBACK_OLLAMA_MODELS",
    "PROVIDER_MODELS",
    "create_provider",
    "list_ollama_models",
    "static_models_for",
]
