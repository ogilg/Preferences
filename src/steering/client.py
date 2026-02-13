"""Steered HuggingFace model that duck-types as OpenAICompatibleClient."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.models.base import STEERING_MODES
from src.models.huggingface_model import HuggingFaceModel
from src.models.openai_compatible import BatchResult, GenerateRequest
from src.probes.core.storage import load_probe_direction


class SteeredHFClient:
    """HuggingFaceModel with steering, duck-typed as OpenAICompatibleClient."""

    def __init__(
        self,
        hf_model: HuggingFaceModel,
        layer: int,
        steering_direction: np.ndarray,
        coefficient: float,
        steering_mode: str = "all_tokens",
    ):
        self.hf_model = hf_model
        self.layer = layer
        self.coefficient = coefficient
        self.steering_mode = steering_mode

        self.canonical_model_name = hf_model.canonical_model_name
        self.model_name = hf_model.model_name
        self.max_new_tokens = hf_model.max_new_tokens

        # Pre-compute scaled steering tensor on GPU
        scaled = steering_direction * coefficient
        self._steering_tensor = torch.tensor(
            scaled, dtype=torch.bfloat16, device=hf_model.device
        )

    def _make_hook(self):
        return STEERING_MODES[self.steering_mode](self._steering_tensor)

    def generate(self, messages, temperature=1.0) -> str:
        if self.coefficient == 0:
            return self.hf_model.generate(messages, temperature=temperature)
        return self.hf_model.generate_with_steering(
            messages=messages,
            layer=self.layer,
            steering_hook=self._make_hook(),
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
                response = self.generate(
                    request.messages, temperature=request.temperature
                )
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
    probe_manifest_dir: Path,
    probe_id: str,
    coefficient: float,
    steering_mode: str = "all_tokens",
    max_new_tokens: int = 256,
) -> SteeredHFClient:
    """Load model + probe, return a steered client ready for measurement."""
    hf_model = HuggingFaceModel(model_name, max_new_tokens=max_new_tokens)
    layer, direction = load_probe_direction(probe_manifest_dir, probe_id)
    return SteeredHFClient(hf_model, layer, direction, coefficient, steering_mode)
