from __future__ import annotations

from typing import Protocol, Any

import numpy as np

from src.types import Message
from .openai_compatible import GenerateRequest, BatchResult


class Model(Protocol):
    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> str: ...

    def generate_batch(
        self,
        requests: list["GenerateRequest"],
        max_concurrent: int = 10,
    ) -> list["BatchResult"]: ...

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]: ...

    def get_activations(
        self,
        messages: list[Message],
        layers: list[int | float],
    ) -> dict[int, np.ndarray]: ...


class ConfigurableMockModel:
    """Mock model that returns configurable responses for testing."""

    def __init__(self, response: str = "a"):
        self.response = response
        self.last_messages: list[Message] | None = None
        self.last_temperature: float | None = None
        self.last_tools: list[dict[str, Any]] | None = None

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        self.last_messages = messages
        self.last_temperature = temperature
        self.last_tools = tools
        return self.response

    def generate_batch(
        self,
        requests: list["GenerateRequest"],
        max_concurrent: int = 10,
    ) -> list["BatchResult"]:
        from .openai_compatible import BatchResult

        return [BatchResult(response=self.response, error=None) for _ in requests]

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]:
        return {}