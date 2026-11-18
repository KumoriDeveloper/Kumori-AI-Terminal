"""Общая реализация OpenAI-совместимых Chat Completions API."""

from __future__ import annotations

from typing import Any

from .base import (
    AIProvider,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    ProviderAPIError,
    ProviderUnavailable,
)


class OpenAICompatibleProvider(AIProvider):
    base_url: str = ""

    def _model_free(self, item: dict[str, Any]) -> bool | None:
        return None

    def _context_length(self, item: dict[str, Any]) -> int | None:
        candidates = (
            item.get("context_length"),
            item.get("context_window"),
            (item.get("top_provider") or {}).get("context_length"),
        )
        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value
        return None

    async def list_models(self) -> list[ModelInfo]:
        cached = self._cached_models()
        if cached is not None:
            return cached
        async with self._client() as client:
            payload = await self._get_json(client, self._url(self.models_path))
        data = payload.get("data")
        if not isinstance(data, list):
            raise ProviderAPIError(
                f"{self.title}: неожиданный формат списка моделей",
                provider=self.name,
            )
        models: list[ModelInfo] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            models.append(
                ModelInfo(
                    id=model_id,
                    provider=self.name,
                    label=item.get("name") if isinstance(item.get("name"), str) else None,
                    context_length=self._context_length(item),
                    free=self._model_free(item),
                )
            )
        return self._cache_models(models)

    def _payload(self, request: ChatRequest, model: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message.as_dict() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _parse_response(self, payload: dict[str, Any], model: str) -> ChatResponse:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderAPIError(
                f"{self.title}: пустой ответ модели",
                provider=self.name,
            )
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
            ]
            content = "".join(parts)
        if not isinstance(content, str) or not content.strip():
            raise ProviderAPIError(
                f"{self.title}: модель вернула пустой текст",
                provider=self.name,
            )
        usage_raw = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        usage = {
            key: int(value)
            for key, value in usage_raw.items()
            if isinstance(value, int) and value >= 0
        }
        return ChatResponse(
            text=content.strip(),
            provider=self.name,
            model=str(payload.get("model") or model),
            usage=usage,
        )

    async def complete(self, request: ChatRequest) -> ChatResponse:
        model = request.model
        if not model:
            models = await self.choose_models(limit=1)
            if not models:
                raise ProviderUnavailable(
                    f"{self.title}: нет доступной модели",
                    provider=self.name,
                )
            model = models[0]

        payload = self._payload(request, model)
        async with self._client() as client:
            try:
                response = await self._post_json(
                    client, self._url("/chat/completions"), payload
                )
            except ProviderAPIError as exc:
                # Некоторые бесплатные модели не поддерживают response_format.
                if request.json_mode and exc.status == 400 and "json" in str(exc).lower():
                    payload.pop("response_format", None)
                    response = await self._post_json(
                        client, self._url("/chat/completions"), payload
                    )
                else:
                    raise
        return self._parse_response(response, model)
