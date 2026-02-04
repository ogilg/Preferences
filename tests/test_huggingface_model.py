"""Tests for HuggingFaceModel against TransformerLens baseline.

These tests verify that HuggingFaceModel produces identical results to TransformerLensModel.
"""

import pytest
import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.transformer_lens import TransformerLensModel, autoregressive_steering


# Use a small model for fast testing
TEST_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.fixture(scope="module")
def hf_model():
    """Load HuggingFace model once per test module."""
    return HuggingFaceModel(TEST_MODEL, device="cuda", dtype="bfloat16")


@pytest.fixture(scope="module")
def tl_model():
    """Load TransformerLens model once per test module."""
    return TransformerLensModel(TEST_MODEL, device="cuda", dtype="bfloat16")


@pytest.mark.api
class TestActivationsMatchTransformerLens:
    """Verify HF activations match TL activations."""

    def test_tokenization_matches(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        """Both models should produce the same tokens."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        hf_prompt = hf_model._format_messages(messages, add_generation_prompt=False)
        tl_prompt = tl_model._format_messages(messages, add_generation_prompt=False)
        assert hf_prompt == tl_prompt

        hf_tokens = hf_model._tokenize(hf_prompt)[0].tolist()
        tl_tokens = tl_model.model.to_tokens(tl_prompt)[0].tolist()
        assert hf_tokens == tl_tokens

    def test_activations_match_last(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        """Last token activations should match within tolerance."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        layers = [0, hf_model.n_layers // 2, hf_model.n_layers - 1]

        hf_acts = hf_model.get_activations(messages, layers, ["last"])
        tl_acts = tl_model.get_activations(messages, layers, ["last"])

        for layer in layers:
            hf_act = hf_acts["last"][layer]
            tl_act = tl_acts["last"][layer]
            assert hf_act.shape == tl_act.shape, f"Shape mismatch at layer {layer}"
            assert np.allclose(hf_act, tl_act, rtol=1e-4, atol=1e-5), (
                f"Activation mismatch at layer {layer}: max diff = {np.abs(hf_act - tl_act).max()}"
            )

    def test_activations_match_first(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        """First completion token activations should match."""
        messages = [
            {"role": "user", "content": "Count to three"},
            {"role": "assistant", "content": "One, two, three."},
        ]
        layers = [0, hf_model.n_layers - 1]

        hf_acts = hf_model.get_activations(messages, layers, ["first"])
        tl_acts = tl_model.get_activations(messages, layers, ["first"])

        for layer in layers:
            hf_act = hf_acts["first"][layer]
            tl_act = tl_acts["first"][layer]
            assert np.allclose(hf_act, tl_act, rtol=1e-4, atol=1e-5)

    def test_activations_match_mean(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        """Mean completion activations should match."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        layers = [hf_model.n_layers // 2]

        hf_acts = hf_model.get_activations(messages, layers, ["mean"])
        tl_acts = tl_model.get_activations(messages, layers, ["mean"])

        for layer in layers:
            hf_act = hf_acts["mean"][layer]
            tl_act = tl_acts["mean"][layer]
            assert np.allclose(hf_act, tl_act, rtol=1e-4, atol=1e-5)


@pytest.mark.api
class TestGenerationMatches:
    """Verify generation produces same results."""

    def test_generation_matches_temperature_zero(
        self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel
    ):
        """At temperature=0, both models should produce identical output."""
        messages = [{"role": "user", "content": "What is 1+1? Answer with just the number."}]

        hf_output = hf_model.generate(messages, temperature=0, max_new_tokens=5)
        tl_output = tl_model.generate(messages, temperature=0, max_new_tokens=5)

        assert hf_output == tl_output


@pytest.mark.api
class TestSteeringMatches:
    """Verify steering produces same results."""

    def test_steering_matches(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        """Steering should produce same results."""
        messages = [{"role": "user", "content": "Hi"}]
        layer = hf_model.n_layers // 2

        steering_vec = torch.randn(hf_model.hidden_dim, device="cuda", dtype=torch.bfloat16)
        hook = autoregressive_steering(steering_vec)

        hf_output = hf_model.generate_with_steering(messages, layer, hook, temperature=0, max_new_tokens=10)
        tl_output = tl_model.generate_with_steering(messages, layer, hook, temperature=0, max_new_tokens=10)

        assert hf_output == tl_output


@pytest.mark.api
class TestModelProperties:
    """Test model property accessors."""

    def test_n_layers_matches(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        assert hf_model.n_layers == tl_model.n_layers

    def test_hidden_dim_matches(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        assert hf_model.hidden_dim == tl_model.hidden_dim

    def test_resolve_layer(self, hf_model: HuggingFaceModel):
        assert hf_model.resolve_layer(0) == 0
        assert hf_model.resolve_layer(0.5) == hf_model.n_layers // 2
        assert hf_model.resolve_layer(1.0) == hf_model.n_layers


@pytest.mark.api
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_assistant_message_raises(self, hf_model: HuggingFaceModel):
        """Empty assistant message should raise ValueError."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
        ]
        with pytest.raises(ValueError, match="beyond sequence length"):
            hf_model.get_activations(messages, [0], ["last"])

    def test_no_assistant_message_raises(self, hf_model: HuggingFaceModel):
        """Missing assistant message should raise ValueError."""
        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(ValueError, match="must end with an assistant message"):
            hf_model.get_activations(messages, [0], ["first"])

    def test_multiple_selectors(self, hf_model: HuggingFaceModel):
        """Multiple selectors should work in single call."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        layers = [0]
        selectors = ["last", "first", "mean"]

        acts = hf_model.get_activations(messages, layers, selectors)

        assert set(acts.keys()) == set(selectors)
        for selector in selectors:
            assert 0 in acts[selector]
            assert acts[selector][0].shape == (hf_model.hidden_dim,)


@pytest.mark.api
class TestBatchedActivations:
    """Test batched activation extraction."""

    def test_batch_matches_individual(self, hf_model: HuggingFaceModel):
        """Batched extraction should match individual extraction."""
        messages_batch = [
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}],
            [{"role": "user", "content": "Say something long"}, {"role": "assistant", "content": "Here is a longer response with more tokens."}],
        ]
        layers = [0, hf_model.n_layers - 1]
        selectors = ["last", "first", "mean"]

        # Get batched results
        batch_acts = hf_model.get_activations_batch(messages_batch, layers, selectors)

        # Get individual results
        for i, messages in enumerate(messages_batch):
            individual_acts = hf_model.get_activations(messages, layers, selectors)
            for selector in selectors:
                for layer in layers:
                    batch_vec = batch_acts[selector][layer][i]
                    individual_vec = individual_acts[selector][layer]
                    assert np.allclose(batch_vec, individual_vec, rtol=1e-4, atol=1e-5), (
                        f"Mismatch at sample {i}, selector {selector}, layer {layer}: "
                        f"max diff = {np.abs(batch_vec - individual_vec).max()}"
                    )

    def test_batch_output_shape(self, hf_model: HuggingFaceModel):
        """Batched output should have correct shape."""
        messages_batch = [
            [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}],
            [{"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}],
        ]
        layers = [0]
        selectors = ["last"]

        acts = hf_model.get_activations_batch(messages_batch, layers, selectors)

        assert acts["last"][0].shape == (2, hf_model.hidden_dim)

    def test_batch_variable_lengths(self, hf_model: HuggingFaceModel):
        """Batching should handle variable-length sequences."""
        messages_batch = [
            [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
            [{"role": "user", "content": "A much longer prompt here"}, {"role": "assistant", "content": "And a much longer response too!"}],
        ]
        layers = [0]
        selectors = ["last", "first", "mean"]

        # Should not raise
        acts = hf_model.get_activations_batch(messages_batch, layers, selectors)

        for selector in selectors:
            assert acts[selector][0].shape == (2, hf_model.hidden_dim)
