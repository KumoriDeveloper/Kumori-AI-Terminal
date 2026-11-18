"""SQLite-хранилище истории разговоров и выполненных команд."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT,
    provider TEXT,
    model TEXT,
    cwd TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    provider TEXT,
    model TEXT
);

CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    argv_json TEXT NOT NULL,
    description TEXT,
    risk TEXT,
    requires_sudo INTEGER NOT NULL DEFAULT 0,
    approved INTEGER NOT NULL DEFAULT 0,
    executed INTEGER NOT NULL DEFAULT 0,
    exit_code INTEGER,
    duration_ms INTEGER,
    output_preview TEXT,
    note TEXT
);

CREATE TABLE IF NOT EXISTS diagnostics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    summary TEXT,
    payload TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations(updated_at);
CREATE INDEX IF NOT EXISTS idx_commands_created
    ON commands(created_at);
"""


class HistoryError(Exception):
    """Ошибка работы с историей."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class ConversationRow:
    id: int
    started_at: str
    updated_at: str
    title: str | None
    provider: str | None
    model: str | None
    cwd: str | None
    message_count: int = 0


@dataclass(slots=True)
class MessageRow:
    id: int
    role: str
    content: str
    created_at: str
    provider: str | None = None
    model: str | None = None


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser()

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with suppress(OSError):
                self.path.parent.chmod(0o700)
            conn = sqlite3.connect(self.path, timeout=10)
        except OSError as exc:
            raise HistoryError(f"Не удалось открыть базу истории: {exc}") from exc

        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise HistoryError(f"Ошибка базы истории: {exc}") from exc
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._session() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        with suppress(OSError):
            self.path.chmod(0o600)

    def create_conversation(
        self,
        *,
        title: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        cwd: str | None = None,
    ) -> int:
        now = utc_now()
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversations(started_at, updated_at, title, provider, model, cwd)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (now, now, _clip(title, 200), provider, model, cwd),
            )
            return int(cursor.lastrowid or 0)

    def update_conversation(
        self,
        conversation_id: int,
        *,
        provider: str | None = None,
        model: str | None = None,
        title: str | None = None,
    ) -> None:
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if provider is not None:
            fields.append("provider = ?")
            values.append(provider)
        if model is not None:
            fields.append("model = ?")
            values.append(model)
        if title is not None:
            fields.append("title = ?")
            values.append(_clip(title, 200))
        values.append(conversation_id)
        with self._session() as conn:
            conn.execute(
                f"UPDATE conversations SET {', '.join(fields)} WHERE id = ?",  # noqa: S608
                values,
            )

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Недопустимая роль сообщения: {role}")
        now = utc_now()
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages(conversation_id, role, content, created_at, provider, model)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, now, provider, model),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            return int(cursor.lastrowid or 0)

    def recent_messages(self, conversation_id: int, limit: int = 6) -> list[MessageRow]:
        limit = max(0, limit)
        if limit == 0:
            return []
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, created_at, provider, model
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [
            MessageRow(
                id=int(row["id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
                provider=row["provider"],
                model=row["model"],
            )
            for row in reversed(rows)
        ]

    def add_command(
        self,
        *,
        conversation_id: int | None,
        argv: list[str],
        description: str | None,
        risk: str | None,
        requires_sudo: bool,
        approved: bool,
        executed: bool,
        exit_code: int | None = None,
        duration_ms: int | None = None,
        output_preview: str | None = None,
        note: str | None = None,
    ) -> int:
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO commands(
                    conversation_id, created_at, argv_json, description, risk,
                    requires_sudo, approved, executed, exit_code, duration_ms,
                    output_preview, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    utc_now(),
                    json.dumps(argv, ensure_ascii=False),
                    _clip(description, 1000),
                    risk,
                    int(requires_sudo),
                    int(approved),
                    int(executed),
                    exit_code,
                    duration_ms,
                    _clip(output_preview, 410000),
                    _clip(note, 1000),
                ),
            )
            return int(cursor.lastrowid or 0)

    def add_diagnostic(self, kind: str, summary: str, payload: Any | None = None) -> int:
        payload_text: str | None = None
        if payload is not None:
            try:
                payload_text = json.dumps(payload, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                payload_text = str(payload)
            payload_text = _clip(payload_text, 100000)
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO diagnostics(created_at, kind, summary, payload)
                VALUES (?, ?, ?, ?)
                """,
                (utc_now(), kind, _clip(summary, 2000), payload_text),
            )
            return int(cursor.lastrowid or 0)

    def list_conversations(self, limit: int = 20) -> list[ConversationRow]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.started_at, c.updated_at, c.title, c.provider, c.model, c.cwd,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.id DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [
            ConversationRow(
                id=int(row["id"]),
                started_at=str(row["started_at"]),
                updated_at=str(row["updated_at"]),
                title=row["title"],
                provider=row["provider"],
                model=row["model"],
                cwd=row["cwd"],
                message_count=int(row["message_count"]),
            )
            for row in rows
        ]

    def get_conversation(self, conversation_id: int) -> ConversationRow | None:
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT c.id, c.started_at, c.updated_at, c.title, c.provider, c.model, c.cwd,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.id = ?
                GROUP BY c.id
                """,
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return ConversationRow(
            id=int(row["id"]),
            started_at=str(row["started_at"]),
            updated_at=str(row["updated_at"]),
            title=row["title"],
            provider=row["provider"],
            model=row["model"],
            cwd=row["cwd"],
            message_count=int(row["message_count"]),
        )

    def list_messages(self, conversation_id: int, limit: int = 200) -> list[MessageRow]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT id, role, content, created_at, provider, model
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id
                LIMIT ?
                """,
                (conversation_id, max(1, limit)),
            ).fetchall()
        return [
            MessageRow(
                id=int(row["id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
                provider=row["provider"],
                model=row["model"],
            )
            for row in rows
        ]

    def list_commands(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, created_at, argv_json, description, risk, requires_sudo,
                       approved, executed, exit_code, duration_ms, output_preview, note
                FROM commands
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["argv"] = json.loads(item.pop("argv_json"))
            except (TypeError, ValueError):
                item["argv"] = []
                item.pop("argv_json", None)
            result.append(item)
        return result

    def stats(self) -> dict[str, int]:
        with self._session() as conn:
            conversations = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            commands = conn.execute("SELECT COUNT(*) FROM commands").fetchone()[0]
            diagnostics = conn.execute("SELECT COUNT(*) FROM diagnostics").fetchone()[0]
        return {
            "conversations": int(conversations),
            "messages": int(messages),
            "commands": int(commands),
            "diagnostics": int(diagnostics),
        }

    def prune(self, max_conversations: int = 1000) -> int:
        with self._session() as conn:
            cursor = conn.execute(
                """
                DELETE FROM conversations
                WHERE id NOT IN (
                    SELECT id FROM conversations ORDER BY id DESC LIMIT ?
                )
                """,
                (max(10, max_conversations),),
            )
            return int(cursor.rowcount or 0)

    def clear_all(self) -> None:
        with self._session() as conn:
            for table in ("commands", "diagnostics", "messages", "conversations"):
                conn.execute(f"DELETE FROM {table}")  # noqa: S608
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN "
                         "('commands','diagnostics','messages','conversations')")

    def vacuum(self) -> None:
        with self._session() as conn:
            conn.execute("VACUUM")


def _clip(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def ensure_private_file(path: Path) -> None:
    with suppress(OSError):
        os.chmod(path, 0o600)
