"""Базовые типы и безопасная обработка ошибок AI-провайдеров."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from ..redaction import redact_text

MessageRole = Literal["system", "user", "assistant"]


class AIMessage(BaseModel):
    role: MessageRole
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ChatRequest(BaseModel):
    messages: list[AIMessage]
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 2048
    json_mode: bool = True


class ChatResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    route_notes: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    id: str
    provider: str
    label: str | None = None
    context_length: int | None = None
    free: bool | None = None


class AIError(Exception):
    """Базовая ошибка AI-слоя."""

    def __init__(self, message: str, *, provider: str | None = None):
        super().__init__(message)
        self.message = message
        self.provider = provider

    def __str__(self) -> str:
        return self.message


class ProviderError(AIError):
    """Ошибка конкретного провайдера."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status: int | None = None,
        retry_after: float | None = None,
        transient: bool = False,
    ):
        super().__init__(message, provider=provider)
        self.status = status
        self.retry_after = retry_after
        self.transient = transient


class AuthError(ProviderError):
    """Неверный или отозванный API-ключ."""


class RateLimitError(ProviderError):
    """Исчерпана квота или сработал rate limit."""


class ProviderTimeout(ProviderError):
    """Провайдер не ответил за отведённое время."""


class ModelNotFoundError(ProviderError):
    """Модель недоступна или устарела."""


class ContentBlockedError(ProviderError):
    """Провайдер отклонил содержимое запроса."""


class ProviderUnavailable(ProviderError):
    """Сетевая ошибка или недоступность сервиса."""


class ProviderAPIError(ProviderError):
    """Иная ошибка HTTP/API."""


class NoProvidersConfigured(AIError):
    """Не настроен ни один API-ключ."""


class AllProvidersFailed(AIError):
    """Ни один провайдер не смог выполнить запрос."""

    def __init__(self, attempts: list[tuple[str, str]]):
        self.attempts = attempts
        details = "; ".join(f"{name}: {reason}" for name, reason in attempts)
        message = "Ни один AI-провайдер не доступен."
        if details:
            message += f" Попытки: {details}"
        super().__init__(message)

    def summary_lines(self) -> list[str]:
        return [f"{name}: {reason}" for name, reason in self.attempts]


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        delta = (when - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    return max(0.0, seconds)


def _excerpt(
    text: str,
    limit: int = 300,
    *,
    extra_secrets: list[str] | None = None,
) -> str:
    cleaned = redact_text(text, extra_secrets=extra_secrets).replace("\n", " ").strip()
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1] + "…"
    return cleaned


class AIProvider(ABC):
    """Общий интерфейс провайдера."""

    name: str = "base"
    title: str = "AI"
    models_path: str = "/models"
    preferred_models: tuple[str, ...] = ()
    fallback_models: tuple[str, ...] = ()
    model_cache_ttl = 300.0

    def __init__(
        self,
        api_key: str,
        *,
        timeout: httpx.Timeout | float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str | None = None,
    ):
        self._api_key = api_key.strip()
        if isinstance(timeout, (int, float)):
            timeout = httpx.Timeout(float(timeout))
        self.timeout = timeout
        self.transport = transport
        if base_url is not None:
            self.base_url = base_url.rstrip("/")
        self._models_cache: list[ModelInfo] | None = None
        self._models_cached_at = 0.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(configured={bool(self._api_key)})"

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
            "User-Agent": "ryoku-ai-terminal/0.1",
        }

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.timeout,
            transport=self.transport,
            headers=self.headers(),
            follow_redirects=False,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _cached_models(self) -> list[ModelInfo] | None:
        if self._models_cache is None:
            return None
        if time.monotonic() - self._models_cached_at >= self.model_cache_ttl:
            self.invalidate_model_cache()
            return None
        return self._models_cache

    def _cache_models(self, models: list[ModelInfo]) -> list[ModelInfo]:
        self._models_cache = models
        self._models_cached_at = time.monotonic()
        return models

    def invalidate_model_cache(self) -> None:
        self._models_cache = None
        self._models_cached_at = 0.0

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    def _is_chat_model(self, model_id: str) -> bool:
        lowered = model_id.lower()
        excluded = (
            "whisper",
            "embedding",
            "tts",
            "audio",
            "image",
            "imagen",
            "veo",
            "lyria",
            "moderation",
            "guard",
            "rerank",
        )
        return not any(token in lowered for token in excluded)

    def _sort_models(self, models: list[ModelInfo]) -> list[ModelInfo]:
        preference = {name: index for index, name in enumerate(self.preferred_models)}

        def key(model: ModelInfo) -> tuple[int, int, str]:
            if model.id in preference:
                return (0, preference[model.id], model.id)
            lowered = model.id.lower()
            for index, preferred in enumerate(self.preferred_models):
                if lowered.startswith(preferred.lower()):
                    return (1, index, model.id)
            return (2, 0, model.id)

        return sorted(models, key=key)

    async def choose_models(self, *, limit: int = 2, free_only: bool = True) -> list[str]:
        models = await self.list_models()

        usable = [model for model in models if self._is_chat_model(model.id)]
        if free_only:
            free = [model for model in usable if model.free is not False]
            usable = free or usable
        ordered = self._sort_models(usable)
        result = [model.id for model in ordered[:limit]]
        if not result:
            raise ProviderUnavailable(
                f"{self.title}: не найдено доступных текстовых моделей",
                provider=self.name,
            )
        return result

    async def _get_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"{self.title}: превышено время ожидания",
                provider=self.name,
                transient=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"{self.title}: сетевая ошибка ({type(exc).__name__})",
                provider=self.name,
                transient=True,
            ) from exc

        if response.status_code >= 400:
            self._raise_http_error(response)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderAPIError(
                f"{self.title}: некорректный JSON в ответе",
                provider=self.name,
                status=response.status_code,
            ) from exc

    async def _post_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"{self.title}: превышено время ожидания",
                provider=self.name,
                transient=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"{self.title}: сетевая ошибка ({type(exc).__name__})",
                provider=self.name,
                transient=True,
            ) from exc

        if response.status_code >= 400:
            self._raise_http_error(response)
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderAPIError(
                f"{self.title}: некорректный JSON в ответе",
                provider=self.name,
                status=response.status_code,
            ) from exc

    def _raise_http_error(self, response: httpx.Response) -> None:
        status = response.status_code
        retry_after = parse_retry_after(response.headers.get("retry-after"))
        try:
            body = response.text
        except Exception:
            body = ""
        excerpt = _excerpt(body, extra_secrets=[self._api_key])
        suffix = f" ({excerpt})" if excerpt else ""

        if status == 429:
            raise RateLimitError(
                f"{self.title}: лимит запросов исчерпан{suffix}",
                provider=self.name,
                status=status,
                retry_after=retry_after,
            )
        if status in {401, 403}:
            raise AuthError(
                f"{self.title}: ключ отклонён или нет доступа{suffix}",
                provider=self.name,
                status=status,
            )
        if status == 404:
            raise ModelNotFoundError(
                f"{self.title}: модель или ресурс не найдены{suffix}",
                provider=self.name,
                status=status,
            )
        if status == 402:
            raise ProviderAPIError(
                f"{self.title}: требуется оплата или закончились кредиты{suffix}",
                provider=self.name,
                status=status,
            )
        if status == 400 and "model" in body.lower():
            raise ModelNotFoundError(
                f"{self.title}: модель недоступна{suffix}",
                provider=self.name,
                status=status,
            )
        if status >= 500:
            raise ProviderAPIError(
                f"{self.title}: ошибка сервера {status}{suffix}",
                provider=self.name,
                status=status,
                transient=True,
                retry_after=retry_after,
            )
        raise ProviderAPIError(
            f"{self.title}: ошибка API {status}{suffix}",
            provider=self.name,
            status=status,
            retry_after=retry_after,
        )
