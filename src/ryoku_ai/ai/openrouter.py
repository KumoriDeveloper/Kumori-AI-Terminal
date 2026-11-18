"""Провайдер OpenRouter (OpenAI-совместимый API, бесплатные модели :free)."""

from __future__ import annotations

from typing import Any

from .base import ModelInfo, ProviderUnavailable
from .openai_compat import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"
    title = "OpenRouter"
    base_url = "https://openrouter.ai/api/v1"
    preferred_models = ()

    def headers(self) -> dict[str, str]:
        headers = super().headers()
        headers["X-Title"] = "Ryoku AI Terminal"
        return headers

    def _model_free(self, item: dict[str, Any]) -> bool | None:
        model_id = item.get("id")
        return isinstance(model_id, str) and model_id.endswith(":free")

    def _is_chat_model(self, model_id: str) -> bool:
        if not super()._is_chat_model(model_id):
            return False
        lowered = model_id.lower()
        excluded = ("roleplay", "uncensored", "nsfw", "erotic", "embed", "rerank")
        return not any(token in lowered for token in excluded)

    async def list_models(self) -> list[ModelInfo]:
        cached = self._cached_models()
        if cached is not None:
            return cached
        async with self._client() as client:
            payload = await self._get_json(client, self._url(self.models_path))
        data = payload.get("data")
        if not isinstance(data, list):
            raise ProviderUnavailable(
                "OpenRouter: неожиданный формат списка моделей",
                provider=self.name,
            )
        models: list[ModelInfo] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            architecture = item.get("architecture")
            if isinstance(architecture, dict):
                outputs = architecture.get("output_modalities")
                if isinstance(outputs, list) and outputs and "text" not in outputs:
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

    async def choose_models(self, *, limit: int = 2, free_only: bool = True) -> list[str]:
        models = await self.list_models()
        usable = [model for model in models if self._is_chat_model(model.id)]
        usable = [model for model in usable if model.id.endswith(":free")]
        ordered = self._sort_models(usable)
        result = [model.id for model in ordered[:limit]]
        if not result:
            raise ProviderUnavailable(
                "OpenRouter: не найдено доступных бесплатных текстовых моделей",
                provider=self.name,
            )
        return result
