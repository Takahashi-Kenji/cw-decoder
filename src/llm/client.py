"""全プロバイダ共通の httpx POST と例外正規化."""
from __future__ import annotations

from typing import Any

import httpx

from src.llm.base import LLMError


def _build_client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout)


def post_json(
    url: str,
    json: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
) -> dict[str, Any]:
    """JSON を POST し、レスポンス body (dict) を返す.

    すべての失敗を LLMError に正規化する.
    """
    try:
        with _build_client(timeout) as client:
            resp = client.post(url, json=json, headers=headers)
    except httpx.TimeoutException as exc:
        raise LLMError(f"タイムアウトしました ({timeout:.0f}秒)") from exc
    except httpx.ConnectError as exc:
        raise LLMError(f"接続に失敗しました: {url}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"通信エラー: {exc!r}") from exc

    if resp.status_code >= 400:
        raise LLMError(f"API エラー (HTTP {resp.status_code}): {resp.text[:200]}")

    try:
        return resp.json()
    except ValueError as exc:
        raise LLMError("応答の JSON 解析に失敗しました") from exc


__all__ = ["post_json"]
