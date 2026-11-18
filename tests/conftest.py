"""Общие фикстуры: тесты никогда не трогают настоящие конфиги и историю."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_ryoku_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    monkeypatch.setenv("RYOKU_AI_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("RYOKU_AI_CONFIG_FILE", str(config_dir / "config.toml"))
    monkeypatch.setenv("RYOKU_AI_CREDENTIALS", str(config_dir / "credentials.env"))
    monkeypatch.setenv("RYOKU_AI_STATE_DIR", str(state_dir))
    monkeypatch.setenv("RYOKU_AI_HISTORY_DB", str(state_dir / "history.db"))
    for name in (
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
