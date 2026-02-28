"""Steered HuggingFace model that duck-types as OpenAICompatibleClient."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.models.base import (
    STEERING_MODES,
    SteeringHook,
    differential_steering,
    position_selective_steering,
)
from src.models.huggingface_model import HuggingFaceModel
from src.models.openai_compatible import BatchResult, GenerateRequest
from src.probes.core.storage import load_probe_direction
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

    def _make_hook(self) -> SteeringHook:
        return STEERING_MODES[self.steering_mode](self._steering_tensor)

    def _resolve_hook(self, messages: list, task_prompts: list[str] | None = None) -> SteeringHook:
        if task_prompts is not None and len(task_prompts) == 2 and self.steering_mode == "differential":
            return self._make_pairwise_hook(messages, task_prompts[0], task_prompts[1])
        return self._make_hook()

    def _make_pairwise_hook(
        self, messages: list, task_a_text: str, task_b_text: str
    ) -> SteeringHook:
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

    def generate_pairwise(
        self,
        messages: list,
        task_a_text: str,
        task_b_text: str,
        temperature: float = 1.0,
    ) -> str:
        """Generate with per-prompt differential/position-selective steering.

        Use for pairwise preference prompts where steering depends on task token spans.
        For static modes (all_tokens, autoregressive), use generate() instead.
        """
        if self.coefficient == 0:
            return self.hf_model.generate(messages, temperature=temperature)
        hook = self._make_pairwise_hook(messages, task_a_text, task_b_text)
        return self.hf_model.generate_with_steering(
            messages=messages,
            layer=self.layer,
            steering_hook=hook,
            temperature=temperature,
        )

    def generate_with_hook(
        self, messages: list, hook: SteeringHook, temperature: float = 1.0
    ) -> str:
        """Generate with a caller-supplied steering hook, bypassing mode/coefficient."""
        return self.hf_model.generate_with_steering(
            messages=messages,
            layer=self.layer,
            steering_hook=hook,
            temperature=temperature,
        )

    def generate_with_hook_n(
        self, messages: list, hook: SteeringHook, n: int, temperature: float = 1.0
    ) -> list[str]:
        """Generate n completions with a caller-supplied hook (shared prefill)."""
        return self.hf_model.generate_with_steering_n(
            messages=messages,
            layer=self.layer,
            steering_hook=hook,
            n=n,
            temperature=temperature,
        )

    def generate(self, messages, temperature=1.0, task_prompts: list[str] | None = None) -> str:
        if self.coefficient == 0:
            return self.hf_model.generate(messages, temperature=temperature)
        hook = self._resolve_hook(messages, task_prompts)
        return self.hf_model.generate_with_steering(
            messages=messages,
            layer=self.layer,
            steering_hook=hook,
            temperature=temperature,
        )

    def generate_n(self, messages, n: int, temperature: float = 1.0) -> list[str]:
        """Generate n completions in a single forward pass (shared prefill)."""
        if self.coefficient == 0:
            return self.hf_model.generate_n(messages, n=n, temperature=temperature)
        return self.hf_model.generate_with_steering_n(
            messages=messages,
            layer=self.layer,
            steering_hook=self._make_hook(),
            n=n,
            temperature=temperature,
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
                response = self._generate_one(request)
                results.append(BatchResult(response=response, error=None))
            except Exception as e:
                results.append(BatchResult(response=None, error=e))
            if on_complete:
                on_complete()
        return results

    def _generate_one(self, request: GenerateRequest) -> str:
        return self.generate(request.messages, temperature=request.temperature, task_prompts=request.task_prompts)

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
    probe_manifest_dir: Path,
    probe_id: str,
    coefficient: float,
    steering_mode: str = "all_tokens",
    max_new_tokens: int = 256,
    a_marker: str = "Task A:",
    b_marker: str = "Task B:",
) -> SteeredHFClient:
    """Load model + probe, return a steered client ready for measurement."""
    hf_model = HuggingFaceModel(model_name, max_new_tokens=max_new_tokens)
    layer, direction = load_probe_direction(probe_manifest_dir, probe_id)
    return SteeredHFClient(
        hf_model, layer, direction, coefficient, steering_mode, a_marker, b_marker
    )
