from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable

import openai


class EmptyResponseError(Exception):
    """Raised when the API returns an empty/null response content."""


RETRYABLE_ERRORS = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError, asyncio.TimeoutError, EmptyResponseError)
MAX_RETRIES = 3
VERBOSE = os.getenv("VERBOSE", "0") == "1"


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 1s, 2s, 4s, clamped to 8s."""
    return min(2 ** attempt, 8)


def with_retries[T](fn: Callable[[], T]) -> T:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except RETRYABLE_ERRORS as e:
            last_error = e
            if VERBOSE:
                print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff_seconds(attempt))
    raise last_error  # type: ignore[misc]
