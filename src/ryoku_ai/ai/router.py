"""Маршрутизатор AI-запросов с резервными провайдерами и cooldown."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from ..config import Config, Credentials
from .base import (
    AIError,
    AIProvider,
    AllProvidersFailed,
    AuthError,
    ChatRequest,
    ChatResponse,
    ContentBlockedError,
    ModelInfo,
    ModelNotFoundError,
    NoProvidersConfigured,
    ProviderAPIError,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitError,
)
from .groq import GroqProvider
from .openrouter import OpenRouterProvider

PROVIDER_CLASSES: dict[str, type[AIProvider]] = {
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
}

PROVIDER_LABELS = {
    "groq": "Groq",
    "openrouter": "OpenRouter",
}


@dataclass(slots=True)
class _Cooldown:
    until: float
    reason: str


def build_providers(
    config: Config,
    credentials: Credentials,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, AIProvider]:
    timeout = httpx.Timeout(config.request_timeout, connect=config.connect_timeout)
    providers: dict[str, AIProvider] = {}
    for name in ("groq", "openrouter"):
        key = getattr(credentials, name, None)
        if not key:
            continue
        cls = PROVIDER_CLASSES[name]
        providers[name] = cls(key, timeout=timeout, transport=transport)
    return providers


class Router:
    """Попытки идут по порядку; ошибки не раскрывают ключи."""

    def __init__(
        self,
        config: Config,
        credentials: Credentials | None = None,
        *,
        providers: dict[str, AIProvider] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        clock=time.monotonic,
    ):
        self.config = config
        self.credentials = credentials or Credentials()
        self._providers = (
            providers
            if providers is not None
            else build_providers(config, self.credentials, transport=transport)
        )
        self._cooldowns: dict[str, _Cooldown] = {}
        self._clock = clock
        self.last_route_notes: list[str] = []

    @property
    def providers(self) -> dict[str, AIProvider]:
        return self._providers

    def has_any_provider(self) -> bool:
        return bool(self._providers)

    def configured_providers(self) -> list[str]:
        ordered = [name for name in self.config.provider_order if name in self._providers]
        extras = [name for name in self._providers if name not in ordered]
        return ordered + extras

    def provider_label(self, name: str) -> str:
        return PROVIDER_LABELS.get(name, name)

    def cooldown_remaining(self, key: str) -> float:
        item = self._cooldowns.get(key)
        if item is None:
            return 0.0
        remaining = item.until - self._clock()
        if remaining <= 0:
            self._cooldowns.pop(key, None)
            return 0.0
        return remaining

    def _is_cooling(self, key: str) -> bool:
        return self.cooldown_remaining(key) > 0

    def _cool(self, key: str, seconds: float, reason: str) -> None:
        self._cooldowns[key] = _Cooldown(self._clock() + max(1.0, seconds), reason)

    def _provider_order(self, preferred: str | None) -> list[str]:
        if preferred and preferred != "auto":
            order = [preferred]
            if self.config.fallback_enabled:
                order.extend(
                    name
                    for name in self.config.provider_order
                    if name not in order
                )
        elif self.config.provider != "auto":
            order = [self.config.provider]
            if self.config.fallback_enabled:
                order.extend(
                    name
                    for name in self.config.provider_order
                    if name not in order
                )
        else:
            order = list(self.config.provider_order)
        return [name for name in dict.fromkeys(order) if name in self._providers]

    async def _models_for(
        self, provider: AIProvider, explicit_model: str | None
    ) -> tuple[list[str], str | None]:
        try:
            available = await provider.list_models()
            selected = explicit_model or self.config.model_for(provider.name)
            if selected:
                if not any(
                    item.id == selected
                    and provider._is_chat_model(item.id)
                    and (provider.name != "openrouter" or item.id.endswith(":free"))
                    for item in available
                ):
                    return [], f"модель {selected} отсутствует в списке доступных моделей"
                return [selected], None
            return await provider.choose_models(
                limit=2, free_only=True
            ), None
        except ProviderError as exc:
            return [], str(exc)

    async def complete(
        self,
        request: ChatRequest,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        preferred = provider or self.config.provider
        if not self._providers:
            raise NoProvidersConfigured(
                "Не настроен ни один AI-провайдер. Запустите: ryoku-ai setup"
            )

        order = self._provider_order(preferred)
        if not order:
            raise AllProvidersFailed(
                [(preferred or "config", "провайдер не настроен или недоступен")]
            )

        explicit_model = model or request.model or self.config.model
        attempts: list[tuple[str, str]] = []
        notes: list[str] = []

        for pname in order:
            provider_obj = self._providers[pname]
            provider_cool = self.cooldown_remaining(pname)
            if provider_cool > 0:
                attempts.append(
                    (self.provider_label(pname), f"пауза после ошибки, {int(provider_cool)} с")
                )
                continue

            provider_model = explicit_model if pname == order[0] else None
            models, list_error = await self._models_for(provider_obj, provider_model)
            if list_error:
                attempts.append((self.provider_label(pname), list_error))
                continue
            if not models:
                attempts.append((self.provider_label(pname), "нет доступных моделей"))
                continue

            provider_failed = False
            for index, model_name in enumerate(models[:2]):
                cooldown_key = f"{pname}:{model_name}"
                if self._is_cooling(cooldown_key):
                    remaining = int(self.cooldown_remaining(cooldown_key))
                    attempts.append(
                        (self.provider_label(pname), f"{model_name}: пауза {remaining} с")
                    )
                    continue
                sub_request = request.model_copy(update={"model": model_name})
                try:
                    response = await provider_obj.complete(sub_request)
                except ContentBlockedError:
                    raise
                except AuthError as exc:
                    self._cool(pname, 300, "auth")
                    attempts.append((self.provider_label(pname), exc.message))
                    provider_failed = True
                    break
                except RateLimitError as exc:
                    seconds = exc.retry_after if exc.retry_after else 60.0
                    self._cool(cooldown_key, seconds, "rate-limit")
                    attempts.append(
                        (
                            self.provider_label(pname),
                            f"{model_name}: лимит квоты, пауза {int(seconds)} с",
                        )
                    )
                    continue
                except ModelNotFoundError as exc:
                    provider_obj.invalidate_model_cache()
                    attempts.append(
                        (self.provider_label(pname), f"{model_name}: {exc.message}")
                    )
                    continue
                except (ProviderTimeout, ProviderUnavailable) as exc:
                    self._cool(pname, 20, "network")
                    attempts.append((self.provider_label(pname), exc.message))
                    provider_failed = True
                    break
                except ProviderAPIError as exc:
                    attempts.append((self.provider_label(pname), exc.message))
                    if exc.transient:
                        self._cool(pname, 20, "server")
                        provider_failed = True
                        break
                    continue
                except AIError as exc:
                    attempts.append((self.provider_label(pname), str(exc)))
                    provider_failed = True
                    break

                if attempts:
                    notes.append(
                        f"Использован резервный вариант: {self.provider_label(pname)}."
                    )
                if index > 0:
                    notes.append(f"Выбрана альтернативная модель: {response.model}.")
                response.route_notes = notes
                self.last_route_notes = notes
                return response

            if provider_failed:
                continue

        raise AllProvidersFailed(attempts)

    async def list_models(self, provider: str) -> list[ModelInfo]:
        provider_obj = self._providers.get(provider)
        if provider_obj is None:
            raise NoProvidersConfigured(f"{self.provider_label(provider)} не настроен")
        return await provider_obj.list_models()

    def selected_model_hint(self, provider: str) -> str:
        return self.config.model_for(provider) or self.config.model or "авто"

    async def close(self) -> None:
        # Клиенты httpx создаются на один запрос и закрываются автоматически.
        return None
