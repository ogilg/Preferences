from __future__ import annotations

from enum import Enum
from typing import Protocol, Any

import numpy as np

from src.types import Message
from .openai_compatible import GenerateRequest, BatchResult


class TokenPosition(Enum):
    LAST = "last"
    FIRST = "first"  # First token of assistant response


class ActivationReduction(Enum):
    LAST = "last"
    MEAN = "mean"
    CHUNKED_MEAN = "chunked_mean"


class ActivationDtype(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_numpy_dtype(self) -> np.dtype:
        """Return numpy dtype for storage (bfloat16 stored as uint16)."""
        if self == ActivationDtype.FLOAT32:
            return np.dtype("float32")
        elif self == ActivationDtype.FLOAT16:
            return np.dtype("float16")
        elif self == ActivationDtype.BFLOAT16:
            return np.dtype("uint16")
        raise ValueError(f"Unknown dtype: {self}")

    @staticmethod
    def from_numpy_dtype(dtype: np.dtype) -> "ActivationDtype":
        """Infer ActivationDtype from numpy dtype."""
        if dtype == np.float32:
            return ActivationDtype.FLOAT32
        elif dtype == np.float16:
            return ActivationDtype.FLOAT16
        elif dtype == np.uint16:
            return ActivationDtype.BFLOAT16
        raise ValueError(f"Cannot infer ActivationDtype from numpy dtype: {dtype}")

    def validate_array(self, arr: np.ndarray) -> None:
        """Raise if array dtype doesn't match expected storage dtype."""
        expected = self.to_numpy_dtype()
        if arr.dtype != expected:
            raise TypeError(
                f"Array dtype {arr.dtype} doesn't match expected {expected} for {self.value}"
            )

    def to_float32(self, arr: np.ndarray) -> np.ndarray:
        """Convert array stored in this dtype back to float32."""
        self.validate_array(arr)
        if self == ActivationDtype.FLOAT32:
            return arr
        elif self == ActivationDtype.FLOAT16:
            return arr.astype(np.float32)
        elif self == ActivationDtype.BFLOAT16:
            import torch
            return torch.from_numpy(arr).view(torch.bfloat16).float().numpy()
        raise ValueError(f"Unknown dtype: {self}")


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