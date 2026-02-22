from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Any, Callable

import numpy as np
import torch

from src.types import Message
from .openai_compatible import GenerateRequest, BatchResult


@dataclass
class GenerationResult:
    completion: str
    activations: dict[int, np.ndarray] | dict[str, dict[int, np.ndarray]]
    prompt_tokens: int
    completion_tokens: int


# SteeringHook takes (resid, prompt_len) and returns modified resid
SteeringHook = Callable[[torch.Tensor, int], torch.Tensor]

TokenSelectorFn = Callable[[torch.Tensor, int], torch.Tensor]


def autoregressive_steering(steering_tensor: torch.Tensor) -> SteeringHook:
    """Steer only the last token position. Works with KV caching during generation."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resid[:, -1, :] += steering_tensor
        return resid
    return hook


def all_tokens_steering(steering_tensor: torch.Tensor) -> SteeringHook:
    """Steer all token positions."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resid += steering_tensor
        return resid
    return hook


STEERING_MODES = {
    "autoregressive": autoregressive_steering,
    "all_tokens": all_tokens_steering,
}


def position_selective_steering(
    steering_tensor: torch.Tensor, start: int, end: int
) -> SteeringHook:
    """Steer only tokens in [start, end) during prompt processing."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prompt processing, not autoregressive
            resid[:, start:end, :] += steering_tensor
        return resid
    return hook


def differential_steering(
    steering_tensor: torch.Tensor,
    pos_start: int,
    pos_end: int,
    neg_start: int,
    neg_end: int,
) -> SteeringHook:
    """Apply +direction on [pos_start, pos_end) and -direction on [neg_start, neg_end)."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prompt processing only
            resid[:, pos_start:pos_end, :] += steering_tensor
            resid[:, neg_start:neg_end, :] -= steering_tensor
        return resid
    return hook


def noop_steering() -> SteeringHook:
    """No-op hook for control conditions."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        return resid
    return hook


def select_last(activations: torch.Tensor, first_completion_idx: int) -> torch.Tensor:
    return activations[-1, :]


def select_first(activations: torch.Tensor, first_completion_idx: int) -> torch.Tensor:
    return activations[first_completion_idx, :]


def select_mean(activations: torch.Tensor, first_completion_idx: int) -> torch.Tensor:
    return activations[first_completion_idx:, :].mean(dim=0)


def select_prompt_last(activations: torch.Tensor, first_completion_idx: int) -> torch.Tensor:
    return activations[first_completion_idx - 1, :]


SELECTOR_REGISTRY: dict[str, TokenSelectorFn] = {
    "last": select_last,
    "first": select_first,
    "mean": select_mean,
    "prompt_last": select_prompt_last,
}

# Selectors that require a completion (assistant message)
COMPLETION_SELECTORS = {"first", "last", "mean"}

# Batched selectors: operate on (batch, seq_len, d_model) tensors
BatchedTokenSelectorFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def select_last_batched(
    activations: torch.Tensor,
    first_completion_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Select last non-padding token for each sample. Returns (batch, d_model)."""
    batch_size = activations.shape[0]
    last_indices = seq_lengths - 1
    return activations[torch.arange(batch_size, device=activations.device), last_indices, :]


def select_first_batched(
    activations: torch.Tensor,
    first_completion_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Select first completion token for each sample. Returns (batch, d_model)."""
    batch_size = activations.shape[0]
    return activations[torch.arange(batch_size, device=activations.device), first_completion_indices, :]


def select_mean_batched(
    activations: torch.Tensor,
    first_completion_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean over completion tokens for each sample. Returns (batch, d_model)."""
    batch_size, max_seq_len, d_model = activations.shape
    device = activations.device

    # Create mask: True for completion tokens (from first_completion_idx to seq_length)
    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)  # (1, max_seq_len)
    mask = (positions >= first_completion_indices.unsqueeze(1)) & (positions < seq_lengths.unsqueeze(1))
    mask = mask.unsqueeze(-1)  # (batch, max_seq_len, 1)

    # Masked mean
    masked_acts = activations * mask
    completion_lengths = (seq_lengths - first_completion_indices).unsqueeze(-1).float()  # (batch, 1)
    return masked_acts.sum(dim=1) / completion_lengths


def select_prompt_mean_batched(
    activations: torch.Tensor,
    first_completion_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean over prompt tokens for each sample. Returns (batch, d_model)."""
    batch_size, max_seq_len, d_model = activations.shape
    device = activations.device

    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)  # (1, max_seq_len)
    mask = positions < first_completion_indices.unsqueeze(1)
    mask = mask.unsqueeze(-1)  # (batch, max_seq_len, 1)

    masked_acts = activations * mask
    prompt_lengths = first_completion_indices.unsqueeze(-1).float()  # (batch, 1)
    return masked_acts.sum(dim=1) / prompt_lengths


def select_prompt_last_batched(
    activations: torch.Tensor,
    first_completion_indices: torch.Tensor,
    seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Last token before completion (final assistant tag token). Returns (batch, d_model)."""
    batch_size = activations.shape[0]
    indices = first_completion_indices - 1
    return activations[torch.arange(batch_size, device=activations.device), indices, :]


BATCHED_SELECTOR_REGISTRY: dict[str, BatchedTokenSelectorFn] = {
    "last": select_last_batched,
    "first": select_first_batched,
    "mean": select_mean_batched,
    "prompt_last": select_prompt_last_batched,
    "prompt_mean": select_prompt_mean_batched,
}


class TokenPosition(Enum):
    """Legacy enum for backwards compatibility."""
    LAST = "last"
    FIRST = "first"
    MEAN = "mean"  # Mean over completion tokens


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