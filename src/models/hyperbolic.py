"""Hyperbolic API model implementation."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import openai
from openai import AsyncOpenAI, OpenAI

from src.types import Message


class ToolCallError(Exception):
    """Raised when tool call parsing fails."""

    pass


@dataclass
class GenerateRequest:
    """A single generation request for batch processing."""

    messages: list[Message]
    temperature: float = 1.0
    tools: list[dict[str, Any]] | None = None


@dataclass
class BatchResult:
    """Result of a single request in a batch operation."""

    response: str | None
    error: Exception | None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> str:
        """Return response or raise the error."""
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    def error_details(self) -> str | None:
        """Return detailed error information for debugging API issues."""
        if self.error is None:
            return None

        details = [f"{type(self.error).__name__}: {self.error}"]

        # Extract OpenAI/API-specific error attributes
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


class HyperbolicModel:
    """Model via Hyperbolic API (OpenAI-compatible)."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._api_key = os.environ.get("HYPERBOLIC_API_KEY")
        self._base_url = "https://api.hyperbolic.xyz/v1"
        self.client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _create_async_client(self) -> AsyncOpenAI:
        """Create a fresh async client for batch operations."""
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _parse_response(
        self,
        message: Any,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        """Parse API response message into string result."""
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
        """Generate a response for the given messages.

        Args:
            messages: The conversation messages.
            temperature: Sampling temperature.
            tools: Optional list of tool definitions for function calling.
                When provided, tool_choice is set to "required".

        Returns:
            If tools are provided: JSON string of the tool call arguments.
            Otherwise: The model's text response.

        Raises:
            ToolCallError: If tool use was requested but failed.
        """
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
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ToolCallError(f"API call failed: {e}") from e

        return self._parse_response(response.choices[0].message, tools)

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]:
        """Get log probabilities for next tokens."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=max_tokens,
        )
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        return {lp.token: lp.logprob for lp in top_logprobs}

    async def _generate_batch_async(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int,
        on_complete: Callable[[], None] | None = None,
        timeout: float = 10.0,
    ) -> list[BatchResult]:
        """Run all requests concurrently with limited parallelism."""
        semaphore = asyncio.Semaphore(max_concurrent)
        # Create a fresh async client for this event loop
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

            max_retries = 3
            for attempt in range(max_retries):
                async with semaphore:
                    try:
                        response = await asyncio.wait_for(
                            async_client.chat.completions.create(**kwargs),
                            timeout=timeout,
                        )
                        text = self._parse_response(
                            response.choices[0].message, request.tools
                        )
                        if on_complete:
                            on_complete()
                        return BatchResult(response=text, error=None)
                    except asyncio.TimeoutError as e:
                        if on_complete:
                            on_complete()
                        return BatchResult(response=None, error=TimeoutError(f"Request timed out after {timeout}s"))
                    except openai.RateLimitError as e:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        if on_complete:
                            on_complete()
                        return BatchResult(response=None, error=e)
                    except Exception as e:
                        if on_complete:
                            on_complete()
                        return BatchResult(response=None, error=e)
            # Should never reach here, but satisfy type checker
            return BatchResult(response=None, error=RuntimeError("Retry loop exited unexpectedly"))

        try:
            return await asyncio.gather(*[process_one(r) for r in requests])
        finally:
            await async_client.close()

    def generate_batch(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int = 10,
        on_complete: Callable[[], None] | None = None,
        timeout: float = 60.0,
    ) -> list[BatchResult]:
        """Generate responses for multiple requests in parallel.

        Args:
            requests: List of GenerateRequest objects.
            max_concurrent: Maximum number of concurrent API calls.
            on_complete: Optional callback invoked after each request completes.
            timeout: Per-request timeout in seconds (default 60s).

        Returns:
            List of BatchResult objects. Use .ok to check success,
            .unwrap() to get response or raise the error.
            Order matches input requests.
        """
        return asyncio.run(self._generate_batch_async(requests, max_concurrent, on_complete, timeout))
