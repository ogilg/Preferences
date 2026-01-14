"""Tests for activation extraction with nnsight.

Run with:
    pytest tests/test_activation_extraction.py -v

Skip with:
    pytest -m "not gpu"
"""

import logging
from dataclasses import dataclass

import numpy as np
import pytest

try:
    import torch
    import nnsight  # noqa: F401
except ImportError as e:
    raise RuntimeError(
        "nnsight and torch are required for activation tests. "
        "Install with `pip install -e .` or `uv sync`."
    ) from e

from src.models.nnsight_model import NnsightModel

if not torch.cuda.is_available():
    pytestmark = pytest.mark.skip(reason="CUDA not available")
else:
    pytestmark = pytest.mark.gpu

logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass
class GenerationResult:
    model: NnsightModel
    messages: list[dict]
    completion: str
    activations: dict[int, np.ndarray]
    layers: list[int]


@pytest.fixture(scope="module")
def generation() -> GenerationResult:
    logger.info("Loading model %s", MODEL_NAME)
    model = NnsightModel(model_name=MODEL_NAME, max_new_tokens=16)
    messages = [{"role": "user", "content": "Hi"}]
    layers = [0, model.n_layers // 2, model.n_layers - 1]

    completion, activations = model.generate_with_activations(messages, layers=layers)

    return GenerationResult(
        model=model,
        messages=messages,
        completion=completion,
        activations=activations,
        layers=layers,
    )


class TestGenerateWithActivations:

    def test_returns_completion(self, generation):
        assert isinstance(generation.completion, str)
        assert len(generation.completion) > 0

    def test_returns_activations_for_all_layers(self, generation):
        assert set(generation.activations.keys()) == set(generation.layers)

    def test_activations_are_1d(self, generation):
        for layer in generation.layers:
            assert generation.activations[layer].ndim == 1

    def test_activations_differ_between_layers(self, generation):
        first, last = generation.layers[0], generation.layers[-1]
        assert not np.allclose(generation.activations[first], generation.activations[last])

    def test_get_activations_matches_generate_with_activations(self, generation):
        full_messages = generation.messages + [{"role": "assistant", "content": generation.completion}]
        standalone_acts = generation.model.get_activations(full_messages, layers=generation.layers)

        for layer in generation.layers:
            assert np.allclose(generation.activations[layer], standalone_acts[layer], rtol=1e-5)

    def test_resolve_layer_with_float(self, generation):
        model = generation.model
        assert model.resolve_layer(0.5) == model.n_layers // 2
        assert model.resolve_layer(0.0) == 0
        assert model.resolve_layer(1) == 1
