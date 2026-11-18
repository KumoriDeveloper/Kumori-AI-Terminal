"""Провайдер Google Gemini (native generateContent API)."""

from __future__ import annotations

import re
from typing import Any

import httpx

from .base import (
    AIProvider,
    AuthError,
    ChatRequest,
    ChatResponse,
    ContentBlockedError,
    ModelInfo,
    ProviderAPIError,
    ProviderUnavailable,
    RateLimitError,
)

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


class GeminiProvider(AIProvider):
    name = "gemini"
    title = "Google Gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"
    models_path = "/models"
    preferred_models = (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
        "gemini-flash-latest",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    )
    # Официальные модели из документации Gemini на момент разработки.
    # Используются только если список моделей временно недоступен.
    fallback_models = (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "gemini-4-hydrogen",
    )

    def headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ryoku-ai-terminal/0.1",
        }

    def _is_chat_model(self, model_id: str) -> bool:
        if not super()._is_chat_model(model_id):
            return False
        lowered = model_id.lower()
        excluded = (
            "imagen",
            "veo",
            "lyria",
            "live",
            "native-audio",
            "robotics",
            "transcribe",
            "computer-use",
            "deep-research",
            "antigravity",
            "omni",
            "decision",
        )
        return not any(token in lowered for token in excluded)

    async def list_models(self) -> list[ModelInfo]:
        cached = self._cached_models()
        if cached is not None:
            return cached

        models: list[ModelInfo] = []
        params: dict[str, Any] = {"pageSize": 1000}
        async with self._client() as client:
            for _ in range(3):
                payload = await self._get_json(client, self._url(self.models_path), params=params)
                data = payload.get("models")
                if not isinstance(data, list):
                    raise ProviderAPIError(
                        "Google Gemini: неожиданный формат списка моделей",
                        provider=self.name,
                    )
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if not isinstance(name, str):
                        continue
                    methods = item.get("supportedGenerationMethods") or []
                    if "generateContent" not in methods:
                        continue
                    model_id = name.removeprefix("models/")
                    models.append(
                        ModelInfo(
                            id=model_id,
                            provider=self.name,
                            label=item.get("displayName")
                            if isinstance(item.get("displayName"), str)
                            else None,
                            context_length=item.get("inputTokenLimit")
                            if isinstance(item.get("inputTokenLimit"), int)
                            else None,
                            free=None,
                        )
                    )
                token = payload.get("nextPageToken")
                if not isinstance(token, str) or not token:
                    break
                params["pageToken"] = token

        return self._cache_models(models)

    def _raise_http_error(self, response: httpx.Response) -> None:
        body = response.text
        if response.status_code == 429:
            match = _RETRY_DELAY_RE.search(body)
            if match:
                raise RateLimitError(
                    "Google Gemini: лимит квоты исчерпан",
                    provider=self.name,
                    status=429,
                    retry_after=float(match.group(1)),
                )
        if response.status_code == 400 and (
            "API key not valid" in body or "API_KEY_INVALID" in body
        ):
            raise AuthError(
                "Google Gemini: API-ключ недействителен",
                provider=self.name,
                status=400,
            )
        super()._raise_http_error(response)

    def _normalize_model(self, model: str) -> str:
        cleaned = model.strip().removeprefix("models/")
        if not _MODEL_RE.match(cleaned):
            raise ProviderAPIError(
                "Google Gemini: недопустимое имя модели",
                provider=self.name,
            )
        return cleaned

    def _build_payload(self, request: ChatRequest) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for message in request.messages:
            if message.role == "system":
                if message.content.strip():
                    system_parts.append(message.content.strip())
                continue
            role = "model" if message.role == "assistant" else "user"
            text = message.content.strip()
            if not text:
                continue
            if contents and contents[-1]["role"] == role:
                parts = contents[-1]["parts"]
                parts.append({"text": text})
            else:
                contents.append({"role": role, "parts": [{"text": text}]})

        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Ответь кратко."}]}]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        if request.json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        return payload

    def _parse_response(self, payload: dict[str, Any], model: str) -> ChatResponse:
        feedback = payload.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise ContentBlockedError(
                "Google Gemini: запрос заблокирован фильтрами безопасности",
                provider=self.name,
            )

        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderAPIError(
                "Google Gemini: пустой ответ модели",
                provider=self.name,
            )
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        finish_reason = str(candidate.get("finishReason") or "")
        if finish_reason in {
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
        }:
            raise ContentBlockedError(
                f"Google Gemini: ответ заблокирован ({finish_reason})",
                provider=self.name,
            )

        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        texts: list[str] = []
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("thought"):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        text = "\n".join(texts).strip()
        if not text:
            raise ProviderAPIError(
                "Google Gemini: модель вернула пустой текст",
                provider=self.name,
            )

        usage_raw = payload.get("usageMetadata")
        usage: dict[str, int] = {}
        if isinstance(usage_raw, dict):
            mapping = {
                "promptTokenCount": "prompt_tokens",
                "candidatesTokenCount": "completion_tokens",
                "totalTokenCount": "total_tokens",
            }
            for source, target in mapping.items():
                value = usage_raw.get(source)
                if isinstance(value, int) and value >= 0:
                    usage[target] = value

        return ChatResponse(
            text=text,
            provider=self.name,
            model=model,
            usage=usage,
        )

    async def complete(self, request: ChatRequest) -> ChatResponse:
        model = request.model
        if not model:
            models = await self.choose_models(limit=1)
            if not models:
                raise ProviderUnavailable(
                    "Google Gemini: нет доступной модели",
                    provider=self.name,
                )
            model = models[0]
        model = self._normalize_model(model)

        payload = self._build_payload(request)
        url = self._url(f"/models/{model}:generateContent")
        async with self._client() as client:
            try:
                response = await self._post_json(client, url, payload)
            except ProviderAPIError as exc:
                # Часть моделей не поддерживает responseMimeType.
                if request.json_mode and exc.status == 400 and "response" in str(exc).lower():
                    payload["generationConfig"].pop("responseMimeType", None)
                    response = await self._post_json(client, url, payload)
                else:
                    raise
        return self._parse_response(response, model)
