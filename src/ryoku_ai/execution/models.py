"""Структурированные модели команд. Никакого исполнения текста модели."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RiskLevel = Literal["low", "medium", "high"]


class ProposedCommand(BaseModel):
    argv: list[str] = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=1000)
    risk: RiskLevel = "low"
    requires_sudo: bool = False


class ProposedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=1000)
    executable: str = Field(min_length=1, max_length=100)
    arguments: list[str] = Field(default_factory=list, max_length=16)
    required_privileges: list[Literal["root"]] = Field(default_factory=list, max_length=1)
    risk_level: RiskLevel
    expected_effect: str = Field(min_length=1, max_length=500)
    rollback_available: bool

    @field_validator("arguments")
    @classmethod
    def _valid_arguments(cls, value: list[str]) -> list[str]:
        if any(not arg or len(arg) > 256 or "\x00" in arg for arg in value):
            raise ValueError("arguments содержит пустое или недопустимое значение")
        return value


class AIAction(BaseModel):
    message: str = ""
    commands: list[ProposedCommand] = Field(default_factory=list)
    actions: list[ProposedAction] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)

    def limited(self, maximum: int, maximum_actions: int = 5) -> AIAction:
        command_limit = max(0, maximum)
        action_limit = max(0, maximum_actions)
        rejected = list(self.rejected)
        if len(self.commands) > command_limit:
            rejected.append(f"Отброшено команд сверх лимита: {len(self.commands) - command_limit}")
        if len(self.actions) > action_limit:
            rejected.append(f"Отброшено действий сверх лимита: {len(self.actions) - action_limit}")
        if len(rejected) == len(self.rejected):
            return self
        return self.model_copy(
            update={
                "commands": self.commands[:command_limit],
                "actions": self.actions[:action_limit],
                "rejected": rejected,
            }
        )


@dataclass(frozen=True, slots=True)
class ValidatedCommand:
    argv: tuple[str, ...]
    executable: str
    description: str
    risk: RiskLevel
    requires_sudo: bool
    policy_note: str
    digest: str

    def display(self, *, with_sudo: bool = True) -> str:
        parts = list(self.argv)
        if self.requires_sudo and with_sudo:
            parts = ["sudo", "-n", "--", *parts]
        return shlex.join(parts)

    def command_hash(self) -> str:
        return self.digest


def compute_digest(
    *,
    argv: tuple[str, ...],
    executable: str,
    requires_sudo: bool,
) -> str:
    payload = json.dumps(
        {
            "argv": list(argv),
            "executable": executable,
            "requires_sudo": requires_sudo,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ValidationError(Exception):
    def __init__(self, reason: str, *, category: str = "policy"):
        super().__init__(reason)
        self.reason = reason
        self.category = category
