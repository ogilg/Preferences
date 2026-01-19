from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

RETRYABLE_ERRORS = (openai.RateLimitError, openai.APIConnectionError, asyncio.TimeoutError)
MAX_RETRIES = 2
VERBOSE = os.getenv("VERBOSE", "0") == "1"

logger = logging.getLogger(__name__)
if VERBOSE:
    logging.basicConfig(level=logging.INFO)

# wait_exponential: wait = multiplier * 2^attempt, clamped to [min, max]
# attempt 1: 1s, attempt 2: 2s (matches original 2^attempt behavior)
_retry_policy = retry(
    retry=retry_if_exception_type(RETRYABLE_ERRORS),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=0.5, min=1, max=2),
    before_sleep=before_sleep_log(logger, logging.INFO) if VERBOSE else None,
    reraise=True,
)


def with_retries[T](fn: Callable[[], T]) -> T:
    return _retry_policy(fn)()


async def with_retries_async[T](fn: Callable[[], Awaitable[T]]) -> T:
    return await _retry_policy(fn)()
