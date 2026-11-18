"""Маршрутизация, резервные провайдеры и cooldown."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ryoku_ai.ai.base import (
    AIMessage,
    AIProvider,
    AllProvidersFailed,
    AuthError,
    ChatRequest,
    ChatResponse,
    ContentBlockedError,
    ModelInfo,
    NoProvidersConfigured,
    ProviderUnavailable,
    RateLimitError,
)
from ryoku_ai.ai.router import Router
from ryoku_ai.config import Config, Credentials


class StubProvider(AIProvider):
    """Провайдер-заглушка с заранее заданным сценарием ответов."""

    def __init__(
        self,
        name: str,
        scenario: list[Any] | None = None,
        models: list[ModelInfo] | None = None,
    ):
        super().__init__("test-key")
        self.name = name
        self.title = name.title()
        self.scenario = list(scenario or [])
        self.models = list(models) if models is not None else [
            ModelInfo(id="g1", provider=name, free=True),
            ModelInfo(id="g2", provider=name, free=True),
            ModelInfo(id="or1:free", provider=name, free=True),
            ModelInfo(id="m1:free", provider=name, free=True),
        ]
        self.calls: list[str] = []
        self.list_calls = 0

    async def list_models(self) -> list[ModelInfo]:
        self.list_calls += 1
        return self.models

    async def complete(self, request: ChatRequest) -> ChatResponse:
        assert request.model is not None
        self.calls.append(request.model)
        action = self.scenario.pop(0) if self.scenario else None
        if isinstance(action, Exception):
            raise action
        return ChatResponse(text="ok", provider=self.name, model=request.model)


def make_request() -> ChatRequest:
    return ChatRequest(messages=[AIMessage(role="user", content="тест")])


def make_config(**kwargs: Any) -> Config:
    defaults = {
        "provider": "auto",
        "provider_order": ["groq", "openrouter"],
        "fallback_enabled": True,
    }
    defaults.update(kwargs)
    return Config(**defaults)


def test_no_providers_configured() -> None:
    router = Router(make_config(), Credentials(), providers={})
    with pytest.raises(NoProvidersConfigured, match="setup"):
        asyncio.run(router.complete(make_request()))


def test_preferred_provider_goes_first() -> None:
    groq = StubProvider("groq")
    openrouter = StubProvider("openrouter")
    config = make_config(
        provider="groq",
        provider_models={"groq": "g1", "openrouter": "m1:free"},
    )
    router = Router(config, providers={"groq": groq, "openrouter": openrouter})
    response = asyncio.run(router.complete(make_request(), provider="openrouter"))
    assert response.provider == "openrouter"
    assert groq.calls == []
    assert openrouter.calls == ["m1:free"]


def test_global_model_is_used_for_primary_provider() -> None:
    groq = StubProvider("groq")
    openrouter = StubProvider("openrouter")
    router = Router(
        make_config(model="g2"),
        providers={"groq": groq, "openrouter": openrouter},
    )

    response = asyncio.run(router.complete(make_request()))

    assert response.provider == "groq"
    assert response.model == "g2"
    assert groq.calls == ["g2"]
    assert openrouter.calls == []


def test_fallback_after_auth_error_and_cooldown() -> None:
    clock = [100.0]
    groq = StubProvider(
        "groq",
        [AuthError("неверный ключ", provider="groq")],
    )
    openrouter = StubProvider("openrouter")
    config = make_config(provider_models={"groq": "g1", "openrouter": "m1:free"})
    router = Router(
        config,
        providers={"groq": groq, "openrouter": openrouter},
        clock=lambda: clock[0],
    )

    response = asyncio.run(router.complete(make_request()))
    assert response.provider == "openrouter"
    assert groq.calls == ["g1"]
    assert openrouter.calls == ["m1:free"]
    assert any("резерв" in note.lower() for note in response.route_notes)
    assert router.cooldown_remaining("groq") > 0

    response2 = asyncio.run(router.complete(make_request()))
    assert response2.provider == "openrouter"
    assert groq.calls == ["g1"], "провайдер на cooldown не должен вызываться"
    assert openrouter.calls == ["m1:free", "m1:free"]


def test_rate_limit_switches_to_second_model() -> None:
    models = [
        ModelInfo(id="m1", provider="groq", free=True),
        ModelInfo(id="m2", provider="groq", free=True),
    ]
    groq = StubProvider(
        "groq",
        [RateLimitError("quota", provider="groq", retry_after=30), None],
        models=models,
    )
    config = make_config(provider="groq", fallback_enabled=False)
    router = Router(config, providers={"groq": groq})

    response = asyncio.run(router.complete(make_request()))
    assert response.provider == "groq"
    assert response.model == "m2"
    assert groq.calls == ["m1", "m2"]
    assert router.cooldown_remaining("groq:m1") > 0

    response2 = asyncio.run(router.complete(make_request()))
    assert response2.model == "m2"
    assert groq.calls == ["m1", "m2", "m2"]


def test_all_providers_failed_summary() -> None:
    groq = StubProvider("groq", [AuthError("bad key", provider="groq")])
    openrouter = StubProvider(
        "openrouter",
        [ProviderUnavailable("network down", provider="openrouter", transient=True)],
    )
    config = make_config(
        provider="auto",
        provider_models={"groq": "g1", "openrouter": "m1:free"},
    )
    router = Router(config, providers={"groq": groq, "openrouter": openrouter})
    with pytest.raises(AllProvidersFailed) as exc_info:
        asyncio.run(router.complete(make_request()))
    lines = exc_info.value.summary_lines()
    assert any("Groq" in line for line in lines)
    assert any("OpenRouter" in line for line in lines)


@pytest.mark.parametrize(
    "failure",
    [
        RateLimitError("quota", provider="groq", retry_after=30),
        ProviderUnavailable("network down", provider="groq", transient=True),
    ],
)
def test_groq_rate_limit_or_unavailability_uses_openrouter(failure: Exception) -> None:
    groq = StubProvider("groq", [failure], models=[ModelInfo(id="g1", provider="groq")])
    openrouter = StubProvider("openrouter")
    config = make_config(
        provider_models={"groq": "g1", "openrouter": "m1:free"},
    )
    router = Router(config, providers={"groq": groq, "openrouter": openrouter})

    response = asyncio.run(router.complete(make_request()))

    assert response.provider == "openrouter"
    assert groq.calls == ["g1"]
    assert openrouter.calls == ["m1:free"]


def test_unavailable_explicit_model_is_not_called_and_falls_back() -> None:
    groq = StubProvider("groq", models=[ModelInfo(id="g1", provider="groq")])
    openrouter = StubProvider("openrouter")
    router = Router(
        make_config(provider_models={"openrouter": "m1:free"}),
        providers={"groq": groq, "openrouter": openrouter},
    )

    response = asyncio.run(
        router.complete(make_request(), provider="groq", model="not-in-live-list")
    )

    assert response.provider == "openrouter"
    assert groq.calls == []
    assert openrouter.calls == ["m1:free"]


def test_fallback_disabled_stops_after_first_provider() -> None:
    groq = StubProvider("groq", [AuthError("bad key", provider="groq")])
    openrouter = StubProvider("openrouter")
    config = make_config(
        provider="groq",
        fallback_enabled=False,
        provider_models={"groq": "g1", "openrouter": "m1:free"},
    )
    router = Router(config, providers={"groq": groq, "openrouter": openrouter})
    with pytest.raises(AllProvidersFailed):
        asyncio.run(router.complete(make_request()))
    assert openrouter.calls == []


def test_content_blocked_is_not_retried_on_other_provider() -> None:
    groq = StubProvider(
        "groq",
        [ContentBlockedError("blocked", provider="groq")],
    )
    openrouter = StubProvider("openrouter")
    config = make_config(provider_models={"groq": "g1", "openrouter": "m1:free"})
    router = Router(config, providers={"groq": groq, "openrouter": openrouter})
    with pytest.raises(ContentBlockedError):
        asyncio.run(router.complete(make_request()))
    assert openrouter.calls == []


def test_model_list_failure_falls_back() -> None:
    class BrokenListProvider(StubProvider):
        async def list_models(self) -> list[ModelInfo]:
            self.list_calls += 1
            raise ProviderUnavailable("models unavailable", provider=self.name)

    groq = BrokenListProvider("groq")
    openrouter = StubProvider("openrouter")
    config = make_config(provider_models={"openrouter": "m1:free"})
    router = Router(config, providers={"groq": groq, "openrouter": openrouter})
    response = asyncio.run(router.complete(make_request()))
    assert response.provider == "openrouter"
    assert groq.list_calls == 1


def test_configured_providers_follow_config_order() -> None:
    config = make_config(provider_order=["openrouter", "groq"])
    router = Router(
        config,
        providers={"groq": StubProvider("groq"), "openrouter": StubProvider("openrouter")},
    )
    assert router.configured_providers() == ["openrouter", "groq"]
    assert router.has_any_provider()
    assert router.selected_model_hint("groq") == "авто"
