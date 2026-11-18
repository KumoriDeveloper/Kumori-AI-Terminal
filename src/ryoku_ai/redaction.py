"""Маскирование секретов перед записью в историю и отправкой модели.

Модуль не пытается угадать все секреты. Он закрывает типовые форматы ключей,
Authorization-заголовки, присваивания паролей и приватные SSH-ключи. Любой
вывод команд всё равно следует считать потенциально чувствительным.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

# Токены конкретных провайдеров + распространённые форматы.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bgsk_[0-9A-Za-z]{20,}\b"),
    re.compile(r"\bsk-or-v1-[0-9A-Za-z]{16,}\b"),
    re.compile(r"\bsk-[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[0-9A-Za-z]{20,}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bBearer\s+[0-9A-Za-z._~+/=-]{8,}"),
)

_ASSIGNMENT = re.compile(
    r"(?i)\b("
    r"api[_-]?key|secret(?:[_-]?key)?|token|access[_-]?token|refresh[_-]?token|"
    r"password|passwd|pwd|authorization|client[_-]?secret"
    r")\b(\s*[:=]\s*)(['\"]?)([^\s'\";,]{3,})\3"
)

_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

# Ключи в формате credentials.env (GROQ_API_KEY=...).
_ENV_KEY = re.compile(
    r"(?m)^(?P<name>[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)"
    r"(?P<sep>\s*=\s*)(?P<value>\S+)"
)


def redact_text(text: str, *, extra_secrets: list[str] | None = None) -> str:
    """Вернуть текст с замаскированными секретами.

    ``extra_secrets`` позволяет скрыть конкретные значения (например, ключи,
    загруженные из окружения), даже если их формат не совпадает с шаблонами.
    """

    if not text:
        return text

    result = text
    for pattern in _TOKEN_PATTERNS:
        result = pattern.sub(REDACTED, result)

    result = _PRIVATE_KEY.sub(REDACTED, result)
    result = _ENV_KEY.sub(lambda m: f"{m.group('name')}{m.group('sep')}{REDACTED}", result)
    result = _ASSIGNMENT.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}{m.group(3)}",
        result,
    )

    for secret in extra_secrets or []:
        secret = secret.strip()
        if len(secret) >= 8:
            result = result.replace(secret, REDACTED)

    return result


def contains_redaction(text: str) -> bool:
    """True, если в тексте уже есть маркер маскирования."""

    return REDACTED in text
