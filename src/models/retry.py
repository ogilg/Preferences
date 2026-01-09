from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable

import openai

RETRYABLE_ERRORS = (openai.RateLimitError, openai.APIConnectionError)
MAX_RETRIES = 2
VERBOSE = os.getenv("VERBOSE", "0") == "1"


def with_retries[T](fn: Callable[[], T]) -> T:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            raise
    raise last_error  # type: ignore[misc]


async def with_retries_async[T](fn: Callable[[], Awaitable[T]]) -> T:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return await fn()
        except RETRYABLE_ERRORS as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                if VERBOSE:
                    error_type = "rate-limit" if isinstance(e, openai.RateLimitError) else "connection-error"
                    sleep_time = 2**attempt
                    print(f"  [{error_type}] retry in {sleep_time}s (attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(2**attempt)
                continue
            raise
    raise last_error  # type: ignore[misc]
