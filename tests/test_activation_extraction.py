"""Tests for activation extraction with nnsight and transformer_lens.

Run with:
    pytest tests/test_activation_extraction.py -v

Skip with:
    pytest -m "not gpu"
"""

import pytest

pytestmark = pytest.mark.probes

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

try:
    import torch
except ImportError as e:
    raise RuntimeError("torch is required for activation tests.") from e

from src.models.nnsight_model import NnsightModel
from src.models.transformer_lens import TransformerLensModel

if not torch.cuda.is_available():
    pytestmark = pytest.mark.skip(reason="CUDA not available")
else:
    pytestmark = pytest.mark.gpu

logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.1-8b"
TEST_MESSAGES = [{"role": "user", "content": "Hi"}]


@dataclass
class GenerationResult:
    model: Any
    model_name: str
    messages: list[dict]
    completion: str
    activations: dict[int, np.ndarray]
    layers: list[int]


@pytest.fixture(scope="module")
def nnsight_model():
    logger.info("Loading NnsightModel %s", MODEL_NAME)
    return NnsightModel(model_name=MODEL_NAME, max_new_tokens=16)


@pytest.fixture(scope="module")
def transformer_lens_model():
    logger.info("Loading TransformerLensModel %s", MODEL_NAME)
    return TransformerLensModel(model_name=MODEL_NAME, max_new_tokens=16)


@pytest.fixture(scope="module")
def nnsight_generation(nnsight_model) -> GenerationResult:
    layers = [0, nnsight_model.n_layers // 2, nnsight_model.n_layers - 1]
    result = nnsight_model.generate_with_activations(TEST_MESSAGES, layers=layers)
    return GenerationResult(
        model=nnsight_model,
        model_name="nnsight",
        messages=TEST_MESSAGES,
        completion=result.completion,
        activations=result.activations,
        layers=layers,
    )


@pytest.fixture(scope="module")
def transformer_lens_generation(transformer_lens_model) -> GenerationResult:
    layers = [0, transformer_lens_model.n_layers // 2, transformer_lens_model.n_layers - 1]
    result = transformer_lens_model.generate_with_activations(TEST_MESSAGES, layers=layers)
    return GenerationResult(
        model=transformer_lens_model,
        model_name="transformer_lens",
        messages=TEST_MESSAGES,
        completion=result.completion,
        activations=result.activations,
        layers=layers,
    )


@pytest.fixture(params=["nnsight", "transformer_lens"])
def generation(request, nnsight_generation, transformer_lens_generation) -> GenerationResult:
    if request.param == "nnsight":
        return nnsight_generation
    return transformer_lens_generation


class TestGenerateWithActivations:

    def test_returns_completion_and_activations(self, generation):
        assert isinstance(generation.completion, str)
        assert len(generation.completion) > 0
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


class TestCrossModelComparison:

    def test_activations_match_between_models(self, nnsight_model, transformer_lens_model):
        """Verify NnsightModel and TransformerLensModel extract similar activations."""
        messages = [{"role": "user", "content": "The capital of France is"}]
        layers = [0, nnsight_model.n_layers // 2, nnsight_model.n_layers - 1]

        nnsight_acts = nnsight_model.get_activations(messages, layers=layers)
        tl_acts = transformer_lens_model.get_activations(messages, layers=layers)

        for layer in layers:
            correlation = np.corrcoef(nnsight_acts[layer], tl_acts[layer])[0, 1]
            assert correlation > 0.99, f"Layer {layer}: correlation {correlation:.4f} < 0.99"


class TestFirstTokenPosition:
    """Tests for extracting activations at the first assistant token position."""

    def test_first_and_last_differ_for_multitoken_response(self, transformer_lens_model):
        """FIRST and LAST should give different activations for multi-token responses."""
        from src.models.base import TokenPosition

        messages = [
            {"role": "user", "content": "Count to five"},
            {"role": "assistant", "content": "One two three four five"},
        ]
        layers = [transformer_lens_model.n_layers // 2]

        first_acts = transformer_lens_model.get_activations(
            messages, layers=layers, token_position=TokenPosition.FIRST
        )
        last_acts = transformer_lens_model.get_activations(
            messages, layers=layers, token_position=TokenPosition.LAST
        )

        layer = layers[0]
        assert not np.allclose(first_acts[layer], last_acts[layer])

    def test_first_position_tokenization_correctness(self, transformer_lens_model):
        """Verify the position calculation matches manual tokenization."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Get prompt without assistant content
        prompt_only = [{"role": "user", "content": "Hello"}]
        prompt_with_header = transformer_lens_model._format_messages(
            prompt_only, add_generation_prompt=True
        )
        prompt_tokens = transformer_lens_model.model.to_tokens(prompt_with_header)
        expected_first_pos = prompt_tokens.shape[1]

        # Get full sequence
        full_text = transformer_lens_model._format_messages(messages, add_generation_prompt=False)
        full_tokens = transformer_lens_model.model.to_tokens(full_text)

        # Decode tokens around the expected position to verify
        first_assistant_token = full_tokens[0, expected_first_pos].item()
        decoded = transformer_lens_model.tokenizer.decode([first_assistant_token])

        # The first token of "Hi there!" should be "Hi" or start with "H"
        assert "H" in decoded or "Hi" in decoded, f"Expected 'Hi' but got '{decoded}'"

    def test_first_position_error_without_assistant_message(self, transformer_lens_model):
        """Should raise error when using FIRST without an assistant message."""
        from src.models.base import TokenPosition

        messages = [{"role": "user", "content": "Hello"}]
        layers = [0]

        with pytest.raises(ValueError, match="must end with an assistant message"):
            transformer_lens_model.get_activations(
                messages, layers=layers, token_position=TokenPosition.FIRST
            )

    def test_first_position_single_token_equals_last(self, transformer_lens_model):
        """For single-token response, FIRST and LAST should be identical."""
        from src.models.base import TokenPosition

        messages = [
            {"role": "user", "content": "Reply with just the letter A"},
            {"role": "assistant", "content": "A"},
        ]
        layers = [transformer_lens_model.n_layers // 2]

        first_acts = transformer_lens_model.get_activations(
            messages, layers=layers, token_position=TokenPosition.FIRST
        )
        last_acts = transformer_lens_model.get_activations(
            messages, layers=layers, token_position=TokenPosition.LAST
        )

        layer = layers[0]
        assert np.allclose(first_acts[layer], last_acts[layer])
