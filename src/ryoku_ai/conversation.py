"""Управление диалогом и его сохранение."""

from __future__ import annotations

from pathlib import Path

from .ai.base import AIMessage, ChatRequest
from .config import Config
from .storage.history import HistoryService


class ConversationSession:
    def __init__(
        self,
        config: Config,
        history: HistoryService,
        *,
        title: str | None = None,
        cwd: Path | None = None,
    ):
        self.config = config
        self.history = history
        self.cwd = (cwd or Path.cwd()).resolve()
        self.messages: list[AIMessage] = []
        self.provider: str | None = None
        self.model: str | None = None
        self.conversation_id = history.start_conversation(
            title=title,
            cwd=str(self.cwd),
        )

    def add_user(self, content: str) -> None:
        message = AIMessage(role="user", content=content)
        self.messages.append(message)
        self.history.add_message(self.conversation_id, "user", content)
        if not self.messages[:-1]:
            self.history.update_conversation(
                self.conversation_id,
                title=content.strip().splitlines()[0][:120] if content.strip() else "Запрос",
            )

    def add_assistant(self, content: str, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.messages.append(AIMessage(role="assistant", content=content))
        self.history.add_message(
            self.conversation_id,
            "assistant",
            content,
            provider=provider,
            model=model,
        )
        self.history.update_conversation(
            self.conversation_id,
            provider=provider,
            model=model,
        )

    def add_system(self, content: str) -> None:
        self.messages.append(AIMessage(role="system", content=content))

    def history_window(self) -> list[AIMessage]:
        limit = self.config.history_context_messages
        if limit <= 0:
            return []
        return self.messages[-limit:]

    def build_request(
        self,
        system_prompt: str,
        *,
        model: str | None = None,
        json_mode: bool = True,
    ) -> ChatRequest:
        messages = [AIMessage(role="system", content=system_prompt)]
        messages.extend(self.history_window())
        return ChatRequest(
            messages=messages,
            model=model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            json_mode=json_mode,
        )
