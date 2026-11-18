"""SQLite-история: диалоги, команды, диагностика и маскирование."""

from __future__ import annotations

from pathlib import Path

import pytest

from ryoku_ai.storage.database import Database, HistoryError, ensure_private_file
from ryoku_ai.storage.history import HistoryService


def test_history_roundtrip_and_redaction(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "history.db"
    secret = "ultra-secret-value-123"
    history = HistoryService(db_path, secrets=[secret])

    conversation_id = history.start_conversation(
        title=f"Тема с {secret}",
        cwd="/tmp/project",
    )
    assert conversation_id is not None

    history.add_message(conversation_id, "user", f"Мой ключ {secret}")
    history.add_message(
        conversation_id,
        "assistant",
        "Ответ без секрета",
        provider="groq",
        model="llama",
    )
    history.add_command(
        conversation_id,
        argv=["cat", f"/tmp/{secret}"],
        description=f"Проверка {secret}",
        risk="low",
        requires_sudo=False,
        approved=True,
        executed=True,
        exit_code=0,
        duration_ms=4,
        output_preview=f"output {secret}",
        note="ok",
    )
    history.add_diagnostic(
        "network",
        f"Сводка {secret}",
        {"nested": [f"value {secret}", {"token": secret}], "ok": True},
    )

    conversation = history.get_conversation(conversation_id)
    assert conversation is not None
    assert secret not in (conversation.title or "")
    assert conversation.message_count == 2

    messages = history.list_messages(conversation_id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert secret not in messages[0].content
    assert messages[1].provider == "groq"

    commands = history.list_commands()
    assert len(commands) == 1
    assert secret not in " ".join(commands[0]["argv"])
    assert secret not in (commands[0]["output_preview"] or "")
    assert commands[0]["exit_code"] == 0

    stats = history.stats()
    assert stats == {
        "conversations": 1,
        "messages": 2,
        "commands": 1,
        "diagnostics": 1,
    }
    assert (db_path.stat().st_mode & 0o777) == 0o600


def test_history_preserves_large_limited_diagnostic_output(tmp_path: Path) -> None:
    history = HistoryService(tmp_path / "history.db")
    conversation_id = history.start_conversation(title="long output")
    assert conversation_id is not None
    output = "x" * 250000

    history.add_command(
        conversation_id,
        argv=["du", "-sh", "/tmp"],
        description="diagnostic",
        risk="low",
        requires_sudo=False,
        approved=True,
        executed=True,
        exit_code=1,
        output_preview=output,
    )

    assert history.list_commands(1)[0]["output_preview"] == output


def test_recent_messages_order_and_limit(tmp_path: Path) -> None:
    history = HistoryService(tmp_path / "history.db")
    conversation_id = history.start_conversation(title="t")
    assert conversation_id is not None
    for index in range(5):
        history.add_message(conversation_id, "user", f"message {index}")
    recent = history.recent_messages(conversation_id, limit=3)
    assert [message.content for message in recent] == [
        "message 2",
        "message 3",
        "message 4",
    ]
    assert history.recent_messages(conversation_id, limit=0) == []


def test_history_prune_and_clear(tmp_path: Path) -> None:
    history = HistoryService(tmp_path / "history.db")
    for index in range(12):
        conversation_id = history.start_conversation(title=f"dialog {index}")
        assert conversation_id is not None
        history.add_message(conversation_id, "user", "hello")
    assert history.stats()["conversations"] == 12

    removed = history.prune(10)
    assert removed == 2
    assert history.stats()["conversations"] == 10

    history.clear()
    assert history.stats() == {
        "conversations": 0,
        "messages": 0,
        "commands": 0,
        "diagnostics": 0,
    }


def test_disabled_history_does_not_create_files(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "history.db"
    history = HistoryService(db_path, enabled=False)
    assert history.start_conversation(title="x") is None
    history.add_message(1, "user", "x")
    history.add_command(
        1,
        argv=["ps"],
        description=None,
        risk=None,
        requires_sudo=False,
        approved=False,
        executed=False,
    )
    history.add_diagnostic("x", "y", {"a": 1})
    assert history.stats()["conversations"] == 0
    assert history.list_conversations() == []
    assert history.list_commands() == []
    assert history.prune(10) == 0
    history.clear()
    assert not db_path.exists()


def test_database_rejects_invalid_role(tmp_path: Path) -> None:
    database = Database(tmp_path / "history.db")
    database.initialize()
    conversation_id = database.create_conversation(title="t")
    with pytest.raises(ValueError):
        database.add_message(conversation_id, "hacker", "text")


def test_database_missing_path_error(tmp_path: Path) -> None:
    database = Database(tmp_path / "file" / "history.db")
    (tmp_path / "file").write_text("not a directory", encoding="utf-8")
    with pytest.raises(HistoryError):
        database.initialize()


def test_ensure_private_file(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_text("x", encoding="utf-8")
    path.chmod(0o644)
    ensure_private_file(path)
    assert (path.stat().st_mode & 0o777) == 0o600
    ensure_private_file(tmp_path / "missing")
