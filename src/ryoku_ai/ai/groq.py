"""Провайдер Groq (OpenAI-совместимый API)."""

from __future__ import annotations

from .openai_compat import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    title = "Groq"
    base_url = "https://api.groq.com/openai/v1"
    preferred_models = (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    )
    def _is_chat_model(self, model_id: str) -> bool:
        if not super()._is_chat_model(model_id):
            return False
        lowered = model_id.lower()
        return not any(
            token in lowered for token in ("compound", "playai", "distil-whisper")
        )
