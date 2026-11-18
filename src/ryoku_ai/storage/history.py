"""Высокоуровневый сервис истории с обязательным маскированием секретов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..redaction import redact_text
from .database import ConversationRow, Database, MessageRow


class HistoryService:
    def __init__(
        self,
        path: Path | str,
        *,
        enabled: bool = True,
        secrets: list[str] | None = None,
    ):
        self.db = Database(path)
        self.enabled = enabled
        self._secrets = [item for item in (secrets or []) if item]
        self._initialized = False

    def _db(self) -> Database:
        if not self._initialized:
            self.db.initialize()
            self._initialized = True
        return self.db

    def _clean(self, text: str, limit: int | None = None) -> str:
        cleaned = redact_text(text, extra_secrets=self._secrets)
        if limit is not None and len(cleaned) > limit:
            cleaned = cleaned[: limit - 1] + "…"
        return cleaned

    def start_conversation(
        self,
        *,
        title: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        cwd: str | None = None,
    ) -> int | None:
        if not self.enabled:
            return None
        return self._db().create_conversation(
            title=self._clean(title, 200) if title else None,
            provider=provider,
            model=model,
            cwd=cwd,
        )

    def update_conversation(
        self,
        conversation_id: int | None,
        *,
        provider: str | None = None,
        model: str | None = None,
        title: str | None = None,
    ) -> None:
        if not self.enabled or conversation_id is None:
            return
        self._db().update_conversation(
            conversation_id,
            provider=provider,
            model=model,
            title=self._clean(title, 200) if title else None,
        )

    def add_message(
        self,
        conversation_id: int | None,
        role: str,
        content: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        if not self.enabled or conversation_id is None:
            return
        self._db().add_message(
            conversation_id,
            role,
            self._clean(content, 50000),
            provider=provider,
            model=model,
        )

    def recent_messages(self, conversation_id: int | None, limit: int = 6) -> list[MessageRow]:
        if not self.enabled or conversation_id is None:
            return []
        return self._db().recent_messages(conversation_id, limit)

    def add_command(
        self,
        conversation_id: int | None,
        *,
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
    ) -> None:
        if not self.enabled:
            return
        clean_argv = [self._clean(part, 500) for part in argv]
        self._db().add_command(
            conversation_id=conversation_id,
            argv=clean_argv,
            description=self._clean(description, 1000) if description else None,
            risk=risk,
            requires_sudo=requires_sudo,
            approved=approved,
            executed=executed,
            exit_code=exit_code,
            duration_ms=duration_ms,
            output_preview=self._clean(output_preview, 410000) if output_preview else None,
            note=self._clean(note, 1000) if note else None,
        )

    def add_diagnostic(self, kind: str, summary: str, payload: Any | None = None) -> None:
        if not self.enabled:
            return
        clean_summary = self._clean(summary, 2000)
        if payload is not None and not isinstance(payload, (str, int, float, bool, dict, list)):
            payload = str(payload)
        self._db().add_diagnostic(kind, clean_summary, self._redact_payload(payload))

    def _redact_payload(self, payload: Any) -> Any:
        if isinstance(payload, str):
            return self._clean(payload, 100000)
        if isinstance(payload, dict):
            return {
                self._clean(str(key), 200): self._redact_payload(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [self._redact_payload(item) for item in payload]
        return payload

    def list_conversations(self, limit: int = 20) -> list[ConversationRow]:
        if not self.enabled:
            return []
        return self._db().list_conversations(limit)

    def get_conversation(self, conversation_id: int) -> ConversationRow | None:
        if not self.enabled:
            return None
        return self._db().get_conversation(conversation_id)

    def list_messages(self, conversation_id: int, limit: int = 200) -> list[MessageRow]:
        if not self.enabled:
            return []
        return self._db().list_messages(conversation_id, limit)

    def list_commands(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        return self._db().list_commands(limit)

    def latest_executed_command(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        commands = self._db().list_commands(100)
        return next((item for item in commands if item.get("executed")), None)

    def stats(self) -> dict[str, int]:
        if not self.enabled:
            return {"conversations": 0, "messages": 0, "commands": 0, "diagnostics": 0}
        return self._db().stats()

    def prune(self, max_conversations: int) -> int:
        if not self.enabled:
            return 0
        return self._db().prune(max_conversations)

    def clear(self) -> None:
        if not self.enabled:
            return
        self._db().clear_all()
        self._db().vacuum()

    def close(self) -> None:
        # Соединения открываются на операцию и закрываются автоматически.
        return None
