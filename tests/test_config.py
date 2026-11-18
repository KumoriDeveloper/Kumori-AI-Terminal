"""Конфигурация, пути и локальное хранение ключей."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ryoku_ai import APP_NAME
from ryoku_ai.config import (
    Config,
    ConfigError,
    Credentials,
    atomic_write_text,
    config_dir,
    config_path,
    credentials_path,
    credentials_permission_issue,
    dump_config,
    history_db_path,
    load_config,
    load_credentials,
    parse_credentials,
    save_config,
    save_credentials,
    state_dir,
    validate_secret_format,
)


def test_default_config_is_safe() -> None:
    config = Config()
    assert config.provider == "auto"
    assert config.provider_order == ["groq", "openrouter"]
    assert config.confirm_commands is True
    assert config.allow_sudo is True
    assert config.max_commands == 3


def test_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "config.toml"
    original = Config(
        provider="openrouter",
        provider_order=["openrouter", "groq"],
        provider_models={"openrouter": "actual-model:free", "unknown": "ignored"},
        max_commands=2,
    )
    saved = save_config(original, path)
    assert saved == path
    loaded = load_config(path)
    assert loaded.provider == "openrouter"
    assert loaded.provider_order == ["openrouter", "groq"]
    assert loaded.provider_models == {"openrouter": "actual-model:free"}
    assert loaded.max_commands == 2
    assert (path.stat().st_mode & 0o777) == 0o600


def test_legacy_gemini_config_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'provider = "gemini"\n'
        'provider_order = ["groq", "gemini", "openrouter"]\n'
        '[provider_models]\n'
        'gemini = "legacy-model"\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.provider == "auto"
    assert config.provider_order == ["groq", "openrouter"]
    assert config.provider_models == {}


def test_dump_config_has_no_secrets() -> None:
    text = dump_config(Config(provider="groq", provider_models={"groq": "m"}))
    assert "API_KEY" not in text
    assert "secret" not in text.lower()
    assert "[provider_models]" in text
    assert 'groq = "m"' in text


def test_invalid_toml_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("provider = [broken", encoding="utf-8")
    with pytest.raises(ConfigError, match="Некорректный TOML"):
        load_config(path)


def test_invalid_config_values_raise(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("max_commands = 99\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Некорректная конфигурация"):
        load_config(path)


def test_empty_provider_order_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("provider_order = []\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_config_keys_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('provider = "auto"\nunknown_key = "value"\n', encoding="utf-8")
    assert load_config(path).provider == "auto"


def test_confirm_commands_cannot_be_disabled(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("confirm_commands = false\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_default_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RYOKU_AI_CONFIG_DIR", raising=False)
    monkeypatch.delenv("RYOKU_AI_CONFIG_FILE", raising=False)
    monkeypatch.delenv("RYOKU_AI_CREDENTIALS", raising=False)
    monkeypatch.delenv("RYOKU_AI_STATE_DIR", raising=False)
    monkeypatch.delenv("RYOKU_AI_HISTORY_DB", raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    assert config_dir() == home / ".config" / APP_NAME
    assert config_path() == home / ".config" / APP_NAME / "config.toml"
    assert credentials_path() == home / ".config" / APP_NAME / "credentials.env"
    assert state_dir() == home / ".local" / "state" / APP_NAME
    assert history_db_path() == home / ".local" / "state" / APP_NAME / "history.db"


def test_parse_credentials_variants() -> None:
    text = """
    # comment
    export GROQ_API_KEY="gsk_abcdefghijklmnopqrstuvwxyz"
    GEMINI_API_KEY = 'AIzaabcdefghijklmnopqrstuvwxyz'
    OPENROUTER_API_KEY=sk-or-v1-abcdefghijklmnop
    UNRELATED=value
    BROKEN LINE
    EMPTY_API_KEY=
    """
    parsed = parse_credentials(text)
    assert parsed["groq"].startswith("gsk_")
    assert parsed["openrouter"].startswith("sk-or-v1-")
    assert set(parsed) == {"groq", "openrouter"}


def test_parse_credentials_ignores_invalid_values() -> None:
    parsed = parse_credentials("GROQ_API_KEY=bad value with spaces\n")
    assert parsed == {}


def test_credentials_saved_with_mode_600(tmp_path: Path) -> None:
    path = tmp_path / "secrets" / "credentials.env"
    saved = save_credentials(
        Credentials(
            groq="gsk_abcdefghijklmnopqrstuvwxyz",
            openrouter="sk-or-v1-abcdefghijklmnop",
        ),
        path,
    )
    assert saved == path
    assert (path.stat().st_mode & 0o777) == 0o600
    text = path.read_text(encoding="utf-8")
    assert "gsk_" in text
    assert "OPENROUTER_API_KEY" in text
    assert "GEMINI_API_KEY" not in text


def test_load_credentials_reads_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "credentials.env"
    path.write_text("GROQ_API_KEY=from_file_value\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "from_env_value")
    credentials = load_credentials(path)
    assert credentials.groq == "from_file_value"
    assert credentials.openrouter is None


def test_credentials_permission_warning(tmp_path: Path) -> None:
    path = tmp_path / "credentials.env"
    path.write_text("GROQ_API_KEY=value\n", encoding="utf-8")
    os.chmod(path, 0o644)
    issue = credentials_permission_issue(path)
    assert issue is not None
    assert "chmod 600" in issue
    os.chmod(path, 0o600)
    assert credentials_permission_issue(path) is None
    assert credentials_permission_issue(tmp_path / "missing") is None


def test_atomic_write_creates_private_parent(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "dir" / "file.txt"
    atomic_write_text(path, "hello", mode=0o600)
    assert path.read_text(encoding="utf-8") == "hello"
    assert (path.stat().st_mode & 0o777) == 0o600
    assert (path.parent.stat().st_mode & 0o777) == 0o700


def test_validate_secret_format() -> None:
    assert validate_secret_format("gsk_abcdefghijklmnopqrstuvwxyz")
    assert not validate_secret_format("short")
    assert not validate_secret_format("with space in value")
    assert not validate_secret_format("")
