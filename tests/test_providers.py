"""Провайдеры на httpx.MockTransport: URL, заголовки, коды и разбор ответов."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from ryoku_ai.ai.base import (
    AIMessage,
    AuthError,
    ChatRequest,
    ContentBlockedError,
    ModelNotFoundError,
    ProviderAPIError,
    ProviderUnavailable,
    RateLimitError,
    parse_retry_after,
)
from ryoku_ai.ai.gemini import GeminiProvider
from ryoku_ai.ai.groq import GroqProvider
from ryoku_ai.ai.openrouter import OpenRouterProvider


def chat_request(model: str | None = None, *, json_mode: bool = True) -> ChatRequest:
    return ChatRequest(
        messages=[
            AIMessage(role="system", content="system text"),
            AIMessage(role="user", content="привет"),
        ],
        model=model,
        temperature=0.1,
        max_tokens=256,
        json_mode=json_mode,
    )


def test_parse_retry_after_variants() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after("invalid") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is not None


# --- Groq -----------------------------------------------------------------


def test_groq_list_models_and_choose() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "llama-3.1-8b-instant"},
                    {"id": "llama-3.3-70b-versatile"},
                    {"id": "whisper-large-v3"},
                    {"id": "text-embedding-3-small"},
                ]
            },
        )

    provider = GroqProvider("gsk_test", transport=httpx.MockTransport(handler))
    models = asyncio.run(provider.list_models())
    assert [model.id for model in models][0] == "llama-3.1-8b-instant"
    assert seen["path"] == "/openai/v1/models"
    assert seen["authorization"] == "Bearer gsk_test"

    chosen = asyncio.run(provider.choose_models(limit=2, free_only=True))
    assert chosen[0] == "llama-3.3-70b-versatile"
    assert "whisper-large-v3" not in chosen


def test_groq_model_cache_expires() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"data": [{"id": "chat-model"}]})

    provider = GroqProvider("gsk_test", transport=httpx.MockTransport(handler))
    provider.model_cache_ttl = 0

    asyncio.run(provider.list_models())
    asyncio.run(provider.list_models())

    assert requests == 2


def test_groq_complete_payload_and_usage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"content": '{"message":"ok"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    provider = GroqProvider("gsk_test", transport=httpx.MockTransport(handler))
    response = asyncio.run(provider.complete(chat_request("llama-3.3-70b-versatile")))
    assert response.text == '{"message":"ok"}'
    assert response.provider == "groq"
    assert response.model == "llama-3.3-70b-versatile"
    assert response.usage["total_tokens"] == 7
    assert seen["path"] == "/openai/v1/chat/completions"
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["messages"][0]["role"] == "system"


@pytest.mark.parametrize(
    ("status", "body", "headers", "error_type"),
    [
        (401, {"error": "bad key"}, {}, AuthError),
        (429, {"error": "quota"}, {"retry-after": "7"}, RateLimitError),
        (500, {"error": "server"}, {}, ProviderAPIError),
        (400, {"error": "model not found"}, {}, ModelNotFoundError),
    ],
)
def test_groq_http_errors(
    status: int,
    body: dict[str, Any],
    headers: dict[str, str],
    error_type: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, headers=headers)

    provider = GroqProvider("gsk_test", transport=httpx.MockTransport(handler))
    with pytest.raises(error_type) as exc_info:
        asyncio.run(provider.complete(chat_request("model")))
    if status == 429:
        assert exc_info.value.retry_after == 7.0  # type: ignore[attr-defined]
    if status == 500:
        assert exc_info.value.transient is True  # type: ignore[attr-defined]


def test_groq_network_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    provider = GroqProvider("gsk_test", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderUnavailable):
        asyncio.run(provider.list_models())


def test_provider_error_body_redacts_exact_custom_key() -> None:
    secret = "custom-provider-secret-12345"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": f"rejected {secret}"})

    provider = GroqProvider(secret, transport=httpx.MockTransport(handler))
    with pytest.raises(AuthError) as exc_info:
        asyncio.run(provider.list_models())

    assert secret not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_openai_compatible_retries_without_json_mode() -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": "json mode unsupported"})
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "plain text"}}],
            },
        )

    provider = GroqProvider("gsk_test", transport=httpx.MockTransport(handler))
    response = asyncio.run(provider.complete(chat_request("m")))
    assert response.text == "plain text"
    assert len(calls) == 2
    assert "response_format" not in calls[1]


# --- OpenRouter -----------------------------------------------------------


def test_openrouter_models_free_filter_and_headers() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["title"] = request.headers.get("x-title")
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "z-model:free",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "context_length": 4096,
                    },
                    {
                        "id": "a-model:free",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "top_provider": {"context_length": 8192},
                    },
                    {
                        "id": "paid-model",
                        "pricing": {"prompt": "0.1", "completion": "0.2"},
                    },
                    {
                        "id": "zero-priced-without-tag",
                        "pricing": {"prompt": "0", "completion": "0"},
                    },
                    {
                        "id": "image-model",
                        "architecture": {"output_modalities": ["image"]},
                    },
                ]
            },
        )

    provider = OpenRouterProvider("sk-or-test", transport=httpx.MockTransport(handler))
    models = {model.id: model for model in asyncio.run(provider.list_models())}
    assert models["z-model:free"].free is True
    assert models["a-model:free"].context_length == 8192
    assert models["paid-model"].free is False
    assert models["zero-priced-without-tag"].free is False
    assert "image-model" not in models
    assert seen["title"] == "Ryoku AI Terminal"

    chosen = asyncio.run(provider.choose_models(limit=2, free_only=True))
    assert chosen == ["a-model:free", "z-model:free"]


def test_openrouter_requires_free_models_when_free_only() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"id": "paid", "pricing": {"prompt": "1", "completion": "1"}}]},
        )

    provider = OpenRouterProvider("sk-or-test", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderUnavailable, match="бесплатных"):
        asyncio.run(provider.choose_models())


# --- Gemini ---------------------------------------------------------------


def test_gemini_list_models_filters_and_paginates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-goog-api-key") == "AIza_test"
        assert "key=" not in str(request.url)
        page = request.url.params.get("pageToken")
        if page is None:
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-3.8-flash",
                            "displayName": "Gemini 3.8 Flash",
                            "supportedGenerationMethods": ["generateContent"],
                            "inputTokenLimit": 1000000,
                        },
                        {
                            "name": "models/text-embedding-004",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                    ],
                    "nextPageToken": "page2",
                },
            )
        assert page == "page2"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-3.5-flash-lite",
                        "supportedGenerationMethods": ["generateContent"],
                    }
                ]
            },
        )

    provider = GeminiProvider("AIza_test", transport=httpx.MockTransport(handler))
    models = asyncio.run(provider.list_models())
    assert [model.id for model in models] == [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
    ]
    assert models[0].context_length == 1000000


def test_gemini_complete_native_payload() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = dict(request.url.params)
        seen["header_key"] = request.headers.get("x-goog-api-key")
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": '{"message":"ok"}'},
                                {"thought": True, "text": "internal"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 4,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 7,
                },
            },
        )

    provider = GeminiProvider("AIza_test", transport=httpx.MockTransport(handler))
    response = asyncio.run(provider.complete(chat_request("gemini-3.8-flash")))
    assert response.text == '{"message":"ok"}'
    assert response.model == "gemini-3.8-flash"
    assert response.usage == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }
    assert seen["path"] == "/v1beta/models/gemini-3.8-flash:generateContent"
    assert seen["query"] == {}
    assert seen["header_key"] == "AIza_test"
    assert "systemInstruction" in seen["payload"]
    assert seen["payload"]["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["payload"]["contents"] == [{"role": "user", "parts": [{"text": "привет"}]}]


def test_gemini_rate_limit_retry_delay() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"details": [{"retryDelay": "12s"}]}},
        )

    provider = GeminiProvider("AIza_test", transport=httpx.MockTransport(handler))
    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(provider.complete(chat_request("gemini-3.8-flash")))
    assert exc_info.value.retry_after == 12.0


def test_gemini_invalid_key_is_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "API key not valid"}})

    provider = GeminiProvider("bad", transport=httpx.MockTransport(handler))
    with pytest.raises(AuthError):
        asyncio.run(provider.complete(chat_request("gemini-3.8-flash")))


def test_gemini_prompt_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"promptFeedback": {"blockReason": "SAFETY"}},
        )

    provider = GeminiProvider("AIza_test", transport=httpx.MockTransport(handler))
    with pytest.raises(ContentBlockedError):
        asyncio.run(provider.complete(chat_request("gemini-3.8-flash")))


def test_gemini_finish_reason_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]},
        )

    provider = GeminiProvider("AIza_test", transport=httpx.MockTransport(handler))
    with pytest.raises(ContentBlockedError):
        asyncio.run(provider.complete(chat_request("gemini-3.8-flash")))


def test_gemini_invalid_model_name_is_rejected_before_request() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    provider = GeminiProvider("AIza_test", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderAPIError, match="недопустимое имя модели"):
        asyncio.run(provider.complete(chat_request("../../etc/passwd")))
    assert calls == 0
