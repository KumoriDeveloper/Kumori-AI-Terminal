"""Маскирование ключей и секретов."""

from __future__ import annotations

from ryoku_ai.redaction import REDACTED, contains_redaction, redact_text


def test_provider_tokens_are_redacted() -> None:
    text = (
        "gemini AIzaabcdefghijklmnopqrstuvwxyz\n"
        "groq gsk_abcdefghijklmnopqrstuvwxyz\n"
        "openrouter sk-or-v1-abcdefghijklmnopqrstuv\n"
        "generic sk-abcdefghijklmnopqrstuvwxyz\n"
    )
    cleaned = redact_text(text)
    assert "AIza" not in cleaned
    assert "gsk_" not in cleaned
    assert "sk-or-v1-" not in cleaned
    assert "sk-abc" not in cleaned
    assert cleaned.count(REDACTED) == 4


def test_bearer_and_assignments_are_redacted() -> None:
    text = 'Authorization: Bearer abcdefghijklmnop\npassword = "hunter2"\napi_key: "value123"'
    cleaned = redact_text(text)
    assert "abcdefghijklmnop" not in cleaned
    assert "hunter2" not in cleaned
    assert "value123" not in cleaned


def test_credentials_env_format_is_redacted() -> None:
    text = "GROQ_API_KEY=gsk_abcdefghijklmnopqrstuvwxyz\nNORMAL_VALUE=ok"
    cleaned = redact_text(text)
    assert "GROQ_API_KEY=[REDACTED]" in cleaned
    assert "NORMAL_VALUE=ok" in cleaned


def test_private_key_is_redacted() -> None:
    text = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "very-secret-body\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    cleaned = redact_text(text)
    assert "very-secret-body" not in cleaned
    assert REDACTED in cleaned


def test_extra_secrets_are_redacted_even_without_known_format() -> None:
    secret = "custom-secret-12345"
    cleaned = redact_text(
        f"value={secret} short",
        extra_secrets=[secret, "", "short"],
    )
    assert secret not in cleaned
    assert "short" in cleaned


def test_contains_redaction() -> None:
    assert contains_redaction(f"x {REDACTED} y")
    assert not contains_redaction("clean text")


def test_empty_text_is_safe() -> None:
    assert redact_text("") == ""
