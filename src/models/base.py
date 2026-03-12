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


# LayerHook takes (resid, prompt_len) and returns modified resid
LayerHook = Callable[[torch.Tensor, int], torch.Tensor]

def autoregressive_steering(steering_tensor: torch.Tensor) -> LayerHook:
    """Steer only the last token position. Works with KV caching during generation."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        resid[:, -1, :] += steering_tensor
        return resid
    return hook


def all_tokens_steering(steering_tensor: torch.Tensor) -> LayerHook:
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
) -> LayerHook:
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
) -> LayerHook:
    """Apply +direction on [pos_start, pos_end) and -direction on [neg_start, neg_end)."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prompt processing only
            resid[:, pos_start:pos_end, :] += steering_tensor
            resid[:, neg_start:neg_end, :] -= steering_tensor
        return resid
    return hook


def last_token_steering(steering_tensor: torch.Tensor) -> LayerHook:
    """Steer only the last prompt token during prompt processing, not during generation."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:  # prompt processing, not autoregressive
            resid[:, -1, :] += steering_tensor
        return resid
    return hook


def noop_steering() -> LayerHook:
    """No-op hook for control conditions."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        return resid
    return hook


def swap_positions(pos_a: int, pos_b: int) -> LayerHook:
    """Swap activations at two token positions during prefill."""
    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            a_act = resid[:, pos_a, :].clone()
            resid[:, pos_a, :] = resid[:, pos_b, :]
            resid[:, pos_b, :] = a_act
        return resid
    return hook


def swap_spans(a_start: int, a_end: int, b_start: int, b_end: int) -> LayerHook:
    """Swap activations across two token spans during prefill.

    Right-aligns when spans differ in length: swaps the last min(len_a, len_b)
    tokens of each span.
    """
    swap_len = min(a_end - a_start, b_end - b_start)
    a_swap_start = a_end - swap_len
    b_swap_start = b_end - swap_len

    def hook(resid: torch.Tensor, prompt_len: int) -> torch.Tensor:
        if resid.shape[1] > 1:
            a_act = resid[:, a_swap_start:a_end, :].clone()
            resid[:, a_swap_start:a_end, :] = resid[:, b_swap_start:b_end, :]
            resid[:, b_swap_start:b_end, :] = a_act
        return resid
    return hook


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

    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
    mask = (positions >= first_completion_indices.unsqueeze(1)) & (positions < seq_lengths.unsqueeze(1))
    mask = mask.unsqueeze(-1)

    masked_acts = activations * mask
    completion_lengths = (seq_lengths - first_completion_indices).unsqueeze(-1).float()
    return masked_acts.sum(dim=1) / completion_lengths


BATCHED_SELECTOR_REGISTRY: dict[str, BatchedTokenSelectorFn] = {
    "last": select_last_batched,
    "first": select_first_batched,
    "mean": select_mean_batched,
}

# Selectors that need input_ids (handled as special cases in _apply_selectors)
TOKEN_ID_SELECTORS = {"eot"}

# --- Anchored offset selectors: "prefix:N" where N is an integer offset from an anchor ---
# turn_boundary:N  → anchor = first_completion_index (start of generated content)
#   e.g. turn_boundary:-2 = model token, -5 = <end_of_turn> (for Gemma-3 IT)
# assistant_tb:N   → anchor = start of follow-up user content (after first assistant turn)
#   Same offsets select the same structural tokens as turn_boundary:N
ANCHORED_OFFSET_PREFIXES = {
    "turn_boundary:": "first_completion",
    "assistant_tb:": "assistant_to_user",
}


def parse_anchored_offset(selector_name: str) -> tuple[str, int] | None:
    """Parse 'prefix:N' -> (anchor_name, offset), or None if not an anchored offset selector."""
    for prefix, anchor_name in ANCHORED_OFFSET_PREFIXES.items():
        if selector_name.startswith(prefix):
            offset_str = selector_name[len(prefix):]
            try:
                offset = int(offset_str)
            except ValueError:
                raise ValueError(f"Invalid offset in {selector_name!r}: {offset_str!r} (must be an integer)")
            if prefix == ASSISTANT_TB_PREFIX and offset >= 0:
                raise ValueError(
                    f"Invalid offset in {selector_name!r}: assistant_tb offsets must be negative "
                    f"(offset 0 is already past the assistant turn)"
                )
            return anchor_name, offset
    return None


def is_anchored_offset_selector(name: str) -> bool:
    return any(name.startswith(p) for p in ANCHORED_OFFSET_PREFIXES)


def requires_chat_template(selector_name: str) -> bool:
    return (selector_name in TOKEN_ID_SELECTORS
            or selector_name in ASSISTANT_SELECTORS
            or is_anchored_offset_selector(selector_name))


# --- Task selectors: operate on user task prompt token spans [start, end) ---
TaskSelectorFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def select_task_last_batched(
    activations: torch.Tensor,
    task_starts: torch.Tensor,
    task_ends: torch.Tensor,
) -> torch.Tensor:
    """Last token of the task prompt for each sample. Returns (batch, d_model)."""
    batch_size = activations.shape[0]
    return activations[torch.arange(batch_size, device=activations.device), task_ends - 1, :]


def select_task_mean_batched(
    activations: torch.Tensor,
    task_starts: torch.Tensor,
    task_ends: torch.Tensor,
) -> torch.Tensor:
    """Mean over task prompt tokens for each sample. Returns (batch, d_model)."""
    batch_size, max_seq_len, d_model = activations.shape
    device = activations.device

    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
    mask = (positions >= task_starts.unsqueeze(1).to(device)) & (
        positions < task_ends.unsqueeze(1).to(device)
    )
    mask = mask.unsqueeze(-1)

    masked_acts = activations * mask
    task_lengths = (task_ends - task_starts).unsqueeze(-1).float().to(device)
    return masked_acts.sum(dim=1) / task_lengths


TASK_SELECTOR_REGISTRY: dict[str, TaskSelectorFn] = {
    "task_last": select_task_last_batched,
    "task_mean": select_task_mean_batched,
}
TASK_SELECTORS = set(TASK_SELECTOR_REGISTRY)


# --- Assistant selectors: operate on first assistant message token spans ---
# Reuse TaskSelectorFn signature: (activations, starts, ends) -> (batch, d_model)
# assistant_mean uses the span registry; assistant_tb:N uses the anchored offset system

ASSISTANT_SELECTOR_REGISTRY: dict[str, TaskSelectorFn] = {
    "assistant_mean": select_task_mean_batched,  # same logic, different span
}
ASSISTANT_SELECTORS = set(ASSISTANT_SELECTOR_REGISTRY)
ASSISTANT_TB_PREFIX = "assistant_tb:"


def needs_assistant_content_span(selector_names: list[str]) -> bool:
    """Check if any selector requires (assistant_starts, assistant_ends) content span."""
    return bool(set(selector_names) & ASSISTANT_SELECTORS)


def needs_assistant_tb_anchor(selector_names: list[str]) -> bool:
    """Check if any selector requires the assistant→user turn boundary anchor."""
    return any(name.startswith(ASSISTANT_TB_PREFIX) for name in selector_names)


# --- Selector validation ---
FIXED_SELECTOR_NAMES = set(BATCHED_SELECTOR_REGISTRY) | TOKEN_ID_SELECTORS | TASK_SELECTORS | ASSISTANT_SELECTORS


def is_valid_selector(name: str) -> bool:
    return name in FIXED_SELECTOR_NAMES or is_anchored_offset_selector(name)


def validate_selectors(names: list[str]) -> None:
    for name in names:
        if not is_valid_selector(name):
            raise ValueError(
                f"Unknown selector: {name}. Valid fixed: {sorted(FIXED_SELECTOR_NAMES)}, "
                f"or anchored offsets: turn_boundary:N, assistant_tb:N"
            )
        if is_anchored_offset_selector(name):
            parse_anchored_offset(name)  # validates the offset


def find_eot_indices(
    input_ids: torch.Tensor,
    eot_token_id: int,
    first_completion_indices: torch.Tensor,
) -> torch.Tensor:
    """Find the last end-of-turn token before first_completion_idx per sample.

    Fully vectorized. Works with any model's end-of-turn token ID.
    """
    input_ids_cpu = input_ids.cpu()
    first_comp_cpu = first_completion_indices.cpu()
    seq_len = input_ids_cpu.shape[1]
    positions = torch.arange(seq_len)

    match = input_ids_cpu == eot_token_id
    before_completion = positions.unsqueeze(0) < first_comp_cpu.unsqueeze(1)
    valid = match & before_completion

    if not valid.any(dim=1).all():
        missing = (~valid.any(dim=1)).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"No end-of-turn token (id={eot_token_id}) found in samples {missing}"
        )

    # Last valid position per row: mask invalid positions to -1, take argmax
    scored = torch.where(valid, positions.unsqueeze(0), -1)
    return scored.max(dim=1).values


def find_first_eot_after(
    input_ids: torch.Tensor,
    eot_token_id: int,
    start_indices: torch.Tensor,
) -> torch.Tensor:
    """Find the first end-of-turn token at or after start_indices per sample."""
    input_ids_cpu = input_ids.cpu()
    start_cpu = start_indices.cpu()
    seq_len = input_ids_cpu.shape[1]
    positions = torch.arange(seq_len)

    match = input_ids_cpu == eot_token_id
    after_start = positions.unsqueeze(0) >= start_cpu.unsqueeze(1)
    valid = match & after_start

    if not valid.any(dim=1).all():
        missing = (~valid.any(dim=1)).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"No end-of-turn token (id={eot_token_id}) found after assistant start in samples {missing}"
        )

    # First valid position per row: mask invalid positions to seq_len, take argmin
    scored = torch.where(valid, positions.unsqueeze(0), seq_len)
    return scored.min(dim=1).values


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