"""Tests for activation extraction with TransformerLens.

Run with:
    pytest tests/test_activation_extraction.py -v

Skip with:
    pytest -m "not gpu"
"""

import gc
import logging

import numpy as np
import pytest

try:
    import torch
except ImportError as e:
    raise RuntimeError("torch is required for activation tests.") from e

from src.models.transformer_lens import TransformerLensModel

if not torch.cuda.is_available():
    pytestmark = pytest.mark.skip(reason="CUDA not available")
else:
    pytestmark = pytest.mark.gpu

logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.2-1b"
TEST_MESSAGES = [{"role": "user", "content": "Hi"}]


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache after each test."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def transformer_lens_model():
    logger.info("Loading TransformerLensModel %s", MODEL_NAME)
    return TransformerLensModel(model_name=MODEL_NAME, max_new_tokens=16)


class TestGenerateWithActivations:

    def test_returns_completion_and_activations(self, transformer_lens_model):
        layers = [0, transformer_lens_model.n_layers // 2, transformer_lens_model.n_layers - 1]
        result = transformer_lens_model.generate_with_activations(
            TEST_MESSAGES, layers=layers, selector_names=["last"]
        )

        assert isinstance(result.completion, str)
        assert len(result.completion) > 0
        for layer in layers:
            assert result.activations["last"][layer].ndim == 1

    def test_activations_differ_between_layers(self, transformer_lens_model):
        layers = [0, transformer_lens_model.n_layers // 2, transformer_lens_model.n_layers - 1]
        result = transformer_lens_model.generate_with_activations(
            TEST_MESSAGES, layers=layers, selector_names=["last"]
        )

        first, last = layers[0], layers[-1]
        assert not np.allclose(result.activations["last"][first], result.activations["last"][last])

    def test_get_activations_matches_generate_with_activations(self, transformer_lens_model):
        layers = [0, transformer_lens_model.n_layers // 2, transformer_lens_model.n_layers - 1]
        result = transformer_lens_model.generate_with_activations(
            TEST_MESSAGES, layers=layers, selector_names=["last"]
        )

        full_messages = TEST_MESSAGES + [{"role": "assistant", "content": result.completion}]
        standalone_acts = transformer_lens_model.get_activations(
            full_messages, layers=layers, selector_names=["last"]
        )

        for layer in layers:
            assert np.allclose(
                result.activations["last"][layer],
                standalone_acts["last"][layer],
                rtol=1e-5,
            )


class TestTokenSelectors:
    """Tests for different token position selectors (first, last, mean)."""

    def test_first_and_last_differ_for_multitoken_response(self, transformer_lens_model):
        """first and last selectors should give different activations for multi-token responses."""
        messages = [
            {"role": "user", "content": "Count to five"},
            {"role": "assistant", "content": "One two three four five"},
        ]
        layers = [transformer_lens_model.n_layers // 2]

        acts = transformer_lens_model.get_activations(
            messages, layers=layers, selector_names=["first", "last"]
        )

        layer = layers[0]
        assert not np.allclose(acts["first"][layer], acts["last"][layer])

    def test_mean_differs_from_endpoints(self, transformer_lens_model):
        """mean selector should differ from first and last."""
        messages = [
            {"role": "user", "content": "Count to five"},
            {"role": "assistant", "content": "One two three four five"},
        ]
        layers = [transformer_lens_model.n_layers // 2]

        acts = transformer_lens_model.get_activations(
            messages, layers=layers, selector_names=["first", "last", "mean"]
        )

        layer = layers[0]
        assert not np.allclose(acts["mean"][layer], acts["first"][layer])
        assert not np.allclose(acts["mean"][layer], acts["last"][layer])

    def test_multiple_selectors_in_single_call(self, transformer_lens_model):
        """All selectors can be extracted in a single forward pass."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        layers = [0, transformer_lens_model.n_layers - 1]

        acts = transformer_lens_model.get_activations(
            messages, layers=layers, selector_names=["first", "last", "mean"]
        )

        assert set(acts.keys()) == {"first", "last", "mean"}
        for selector in ["first", "last", "mean"]:
            for layer in layers:
                assert acts[selector][layer].ndim == 1

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

    def test_error_without_assistant_message(self, transformer_lens_model):
        """Should raise error when messages don't end with assistant message."""
        messages = [{"role": "user", "content": "Hello"}]
        layers = [0]

        with pytest.raises(ValueError, match="must end with an assistant message"):
            transformer_lens_model.get_activations(
                messages, layers=layers, selector_names=["first"]
            )

