"""テスト全体の安全網.

**テストが利用者の実設定ファイルを書き換えないようにする。**
``CWDecoderWindow.closeEvent`` は ``_save_settings()`` を呼ぶので、
ウィンドウを close() するテストは既定の保存先 (``~/.cw-decorder/settings.json``)
を上書きしてしまう。実際に ``llm_auto_interval_s`` を 0.0 に、``llm_model`` を
既定値に書き戻していた (2026-08-08 に発覚)。

本来の対策はウィンドウに ``config_path`` を渡すことだが、渡し忘れても壊れないよう
モジュール定数そのものを一時パスへ向けておく。
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_settings_file(tmp_path_factory) -> None:
    """既定の設定保存先をセッション用の一時パスへ差し替える."""
    import src.infer.settings as settings_module

    original = settings_module.DEFAULT_CONFIG_PATH
    settings_module.DEFAULT_CONFIG_PATH = (
        tmp_path_factory.mktemp("cw-settings") / "settings.json"
    )
    try:
        yield
    finally:
        settings_module.DEFAULT_CONFIG_PATH = original
