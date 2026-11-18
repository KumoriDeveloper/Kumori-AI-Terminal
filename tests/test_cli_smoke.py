"""Дымовые тесты CLI: команды без API и без реальных ключей."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ryoku_ai.cli import app, command_app
from ryoku_ai.config import config_path

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


def invoke_command(*args: str):
    return runner.invoke(command_app, list(args))


def test_version() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert "Ryoku AI Terminal 0.1.0" in result.output


def test_help_lists_diagnostics() -> None:
    result = invoke("--help")
    assert result.exit_code == 0
    for flag in ("--chat", "--network", "--system", "--processes", "--docker", "--models"):
        assert flag in result.output


def test_dashboard_works_without_keys() -> None:
    result = invoke()
    assert result.exit_code == 0
    assert "RYOKU AI TERMINAL" in result.output
    assert "Нужен API-ключ" in result.output


def test_config_init_and_show(tmp_path: Path) -> None:
    init = invoke_command("config", "--init")
    assert init.exit_code == 0, init.output
    assert config_path().exists()
    assert "Создана конфигурация" in init.output

    shown = invoke("--config")
    assert shown.exit_code == 0, shown.output
    assert "Конфигурация" in shown.output
    assert "Подтверждение команд" in shown.output


def test_history_empty() -> None:
    result = invoke("--history")
    assert result.exit_code == 0
    assert "История пуста" in result.output


def test_history_show_missing() -> None:
    result = invoke("--history-show", "999")
    assert result.exit_code == 0
    assert "не найден" in result.output


def test_history_clear_with_yes() -> None:
    result = invoke_command("history", "--clear", "--yes")
    assert result.exit_code == 0
    assert "История очищена" in result.output


def test_models_without_providers_shows_setup_hint() -> None:
    result = invoke("--models")
    assert result.exit_code == 0
    assert "Доступные модели" in result.output
    assert "groq" in result.output
    assert "openrouter" in result.output
    assert result.output.count("нет API-ключа") == 2


def test_doctor_runs() -> None:
    result = invoke_command("doctor")
    assert result.exit_code == 0, result.output
    assert "Проверка окружения" in result.output
    assert "Python" in result.output


def test_query_without_providers_does_not_crash() -> None:
    result = invoke("почему", "медленный", "интернет")
    assert result.exit_code == 0, result.output
    assert "AI не настроен" in result.output


def test_explain_without_providers_does_not_crash() -> None:
    result = invoke("--explain", "ps aux")
    assert result.exit_code == 0, result.output
    assert "AI не настроен" in result.output


def test_temp_directory_query_uses_builtin_inspector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ryoku_ai import cli

    test_directory = tmp_path / "temporary"
    test_directory.mkdir()
    (test_directory / "readable.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(cli, "_inspection_query_paths", lambda _query: [test_directory])

    result = invoke("проанализируй", "размер", "/tmp")

    assert result.exit_code == 0, result.output
    assert "RYOKU AI — DIRECTORY RESULT" in result.output
    assert str(test_directory) in result.output
    assert "Readable size:" in result.output
    assert "scandir" not in result.output.lower()
