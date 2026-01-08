from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import openai
from openai import AsyncOpenAI, OpenAI

from src.models.retry import with_retries, with_retries_async
from src.types import Message


class ToolCallError(Exception):
    pass


@dataclass
class GenerateRequest:
    messages: list[Message]
    temperature: float = 1.0
    tools: list[dict[str, Any]] | None = None


@dataclass
class BatchResult:
    response: str | None
    error: Exception | None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> str:
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    def error_details(self) -> str | None:
        if self.error is None:
            return None

        details = [f"{type(self.error).__name__}: {self.error}"]

        if hasattr(self.error, "status_code"):
            details.append(f"Status code: {self.error.status_code}")
        if hasattr(self.error, "body"):
            details.append(f"Body: {self.error.body}")
        if hasattr(self.error, "response"):
            resp = self.error.response
            if hasattr(resp, "status_code"):
                details.append(f"Response status: {resp.status_code}")
            if hasattr(resp, "text"):
                details.append(f"Response text: {resp.text}")
        if hasattr(self.error, "__cause__") and self.error.__cause__:
            details.append(f"Caused by: {self.error.__cause__}")

        return "\n  ".join(details)


class OpenAICompatibleClient(ABC):
    """Base class for OpenAI-compatible API providers."""

    @property
    @abstractmethod
    def _api_key_env_var(self) -> str: ...

    @property
    @abstractmethod
    def _base_url(self) -> str: ...

    @property
    @abstractmethod
    def _default_model(self) -> str: ...

    _model_aliases: dict[str, str] = {}

    def _resolve_model_name(self, model_name: str | None) -> tuple[str, str]:
        """Returns (canonical_name, provider_model_name)."""
        if model_name is None:
            return (self._default_model, self._default_model)
        if model_name in self._model_aliases:
            return (model_name, self._model_aliases[model_name])
        return (model_name, model_name)

    def __init__(
        self,
        model_name: str | None = None,
        max_new_tokens: int = 256,
    ):
        self.canonical_model_name, self.model_name = self._resolve_model_name(model_name)
        self.max_new_tokens = max_new_tokens
        self._api_key = os.environ[self._api_key_env_var]
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _create_async_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _parse_response(
        self,
        message: Any,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        if tools is not None:
            if not message.tool_calls:
                raise ToolCallError(
                    f"Expected tool call but got text: {message.content}"
                )
            tool_call = message.tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
                return json.dumps(args)
            except json.JSONDecodeError as e:
                raise ToolCallError(
                    f"Invalid JSON in tool arguments: {tool_call.function.arguments}"
                ) from e
        return (message.content or "").strip()

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_new_tokens,
        }

        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = with_retries(lambda: self.client.chat.completions.create(**kwargs))
        except Exception as e:
            raise ToolCallError(f"API call failed: {e}") from e

        return self._parse_response(response.choices[0].message, tools)

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]:
        response = with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=max_tokens,
            )
        )
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        return {lp.token: lp.logprob for lp in top_logprobs}

    async def _generate_batch_async(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int,
        on_complete: Callable[[], None] | None = None,
    ) -> list[BatchResult]:
        semaphore = asyncio.Semaphore(max_concurrent)
        async_client = self._create_async_client()

        async def process_one(request: GenerateRequest) -> BatchResult:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": self.max_new_tokens,
            }
            if request.tools is not None:
                kwargs["tools"] = request.tools
                kwargs["tool_choice"] = "auto"

            async with semaphore:
                try:
                    response = await with_retries_async(
                        lambda: async_client.chat.completions.create(**kwargs)
                    )
                    text = self._parse_response(
                        response.choices[0].message, request.tools
                    )
                    if on_complete:
                        on_complete()
                    return BatchResult(response=text, error=None)
                except Exception as e:
                    if on_complete:
                        on_complete()
                    return BatchResult(response=None, error=e)

        async def process_with_index(i: int, request: GenerateRequest) -> tuple[int, BatchResult]:
            result = await process_one(request)
            return (i, result)

        tasks = [process_with_index(i, r) for i, r in enumerate(requests)]
        results: list[BatchResult | None] = [None] * len(requests)

        try:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results[idx] = result
            return results  # type: ignore[return-value]
        finally:
            await async_client.close()

    def generate_batch(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int = 10,
        on_complete: Callable[[], None] | None = None,
    ) -> list[BatchResult]:
        return asyncio.run(self._generate_batch_async(requests, max_concurrent, on_complete))


class HyperbolicClient(OpenAICompatibleClient):
    _api_key_env_var = "HYPERBOLIC_API_KEY"
    _base_url = "https://api.hyperbolic.xyz/v1"
    _default_model = "llama-3.1-8b"
    default_max_concurrent = 50
    _model_aliases = {
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    }


class CerebrasClient(OpenAICompatibleClient):
    _api_key_env_var = "CEREBRAS_API_KEY"
    _base_url = "https://api.cerebras.ai/v1"
    _default_model = "llama-3.1-8b"
    default_max_concurrent = 150
    _model_aliases = {
        "llama-3.1-8b": "llama3.1-8b",
        "llama-3.1-70b": "llama3.1-70b",
    }
