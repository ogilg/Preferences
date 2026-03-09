"""Steered HuggingFace model that duck-types as OpenAICompatibleClient."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.models.base import (
    STEERING_MODES,
    LayerHook,
    differential_steering,
    noop_steering,
    position_selective_steering,
)
from src.models.huggingface_model import HuggingFaceModel
from src.models.openai_compatible import BatchResult, GenerateRequest
from src.steering.tokenization import find_pairwise_task_spans


class SteeredHFClient:
    """HuggingFaceModel with steering, duck-typed as OpenAICompatibleClient."""

    def __init__(
        self,
        hf_model: HuggingFaceModel,
        layer: int,
        steering_direction: np.ndarray,
        coefficient: float,
        steering_mode: str = "all_tokens",
        a_marker: str = "Task A:",
        b_marker: str = "Task B:",
    ):
        self.hf_model = hf_model
        self.layer = layer
        self._direction = steering_direction
        self.coefficient = coefficient
        self.steering_mode = steering_mode
        self.a_marker = a_marker
        self.b_marker = b_marker

        self.canonical_model_name = hf_model.canonical_model_name
        self.model_name = hf_model.model_name
        self.max_new_tokens = hf_model.max_new_tokens

        # Pre-compute scaled steering tensor on GPU
        scaled = steering_direction * coefficient
        self._steering_tensor = torch.tensor(
            scaled, dtype=torch.bfloat16, device=hf_model.device
        )

    @property
    def direction(self) -> np.ndarray:
        return self._direction

    def with_coefficient(self, coefficient: float) -> SteeredHFClient:
        """Return a new client with a different coefficient, sharing the same model."""
        return SteeredHFClient(
            self.hf_model,
            self.layer,
            self._direction,
            coefficient,
            self.steering_mode,
            self.a_marker,
            self.b_marker,
        )

    def _make_layer_hook(self) -> LayerHook:
        return STEERING_MODES[self.steering_mode](self._steering_tensor)

    def _resolve_hook(self, messages: list, task_prompts: list[str] | None = None) -> LayerHook:
        if self.coefficient == 0:
            return noop_steering()
        if task_prompts is not None and len(task_prompts) == 2 and self.steering_mode == "differential":
            return self._make_pairwise_hook(messages, task_prompts[0], task_prompts[1])
        return self._make_layer_hook()

    def _make_pairwise_hook(
        self, messages: list, task_a_text: str, task_b_text: str
    ) -> LayerHook:
        """Create a per-prompt hook using token spans of each task in the prompt."""
        formatted = self.hf_model.format_messages(messages)
        tokenizer = self.hf_model.tokenizer
        a_span, b_span = find_pairwise_task_spans(
            tokenizer, formatted, task_a_text, task_b_text,
            self.a_marker, self.b_marker,
        )
        if self.steering_mode == "differential":
            return differential_steering(
                self._steering_tensor, a_span[0], a_span[1], b_span[0], b_span[1]
            )
        # steer_task_a: steer only task A's tokens
        return position_selective_steering(
            self._steering_tensor, a_span[0], a_span[1]
        )

    def generate(self, messages, temperature=1.0, task_prompts: list[str] | None = None) -> str:
        hook = self._resolve_hook(messages, task_prompts)
        return self.hf_model.generate_with_hook(
            messages=messages, layer=self.layer, hook=hook, temperature=temperature,
        )

    def generate_n(self, messages, n: int, temperature: float = 1.0, task_prompts: list[str] | None = None) -> list[str]:
        hook = self._resolve_hook(messages, task_prompts)
        return self.hf_model.generate_with_hook_n(
            messages=messages, layer=self.layer, hook=hook, n=n, temperature=temperature,
        )

    def _run_batch(
        self,
        requests: list[GenerateRequest],
        on_complete: Callable[[], None] | None = None,
    ) -> list[BatchResult]:
        results: list[BatchResult] = []
        for request in requests:
            if request.tools is not None:
                raise ValueError("SteeredHFClient does not support tool use")
            try:
                response = self.generate(request.messages, temperature=request.temperature, task_prompts=request.task_prompts)
                results.append(BatchResult(response=response, error=None))
            except Exception as e:
                results.append(BatchResult(response=None, error=e))
            if on_complete:
                on_complete()
        return results

    async def generate_batch_async(
        self,
        requests: list[GenerateRequest],
        semaphore: asyncio.Semaphore,
        on_complete: Callable[[], None] | None = None,
        enable_reasoning: bool = False,
    ) -> list[BatchResult]:
        # Async signature required because callers await this, but HF inference
        # is synchronous — semaphore is accepted for interface compat only.
        return self._run_batch(requests, on_complete)

    def generate_batch(
        self,
        requests: list[GenerateRequest],
        max_concurrent: int = 10,
        on_complete: Callable[[], None] | None = None,
        enable_reasoning: bool = False,
    ) -> list[BatchResult]:
        return self._run_batch(requests, on_complete)


def create_steered_client(
    model_name: str,
    layer: int,
    direction: np.ndarray,
    coefficient: float,
    steering_mode: str = "all_tokens",
    max_new_tokens: int = 256,
    a_marker: str = "Task A:",
    b_marker: str = "Task B:",
) -> SteeredHFClient:
    """Load model, return a steered client ready for measurement."""
    hf_model = HuggingFaceModel(model_name, max_new_tokens=max_new_tokens)
    return SteeredHFClient(
        hf_model, layer, direction, coefficient, steering_mode, a_marker, b_marker
    )
