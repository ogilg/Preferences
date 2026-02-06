"""Tests for HuggingFaceModel against TransformerLens baseline.

These tests verify that HuggingFaceModel produces identical results to TransformerLensModel.
Run with: pytest -m gpu tests/test_huggingface_model.py -v
"""

import gc

import pytest
import numpy as np
import torch

from src.models.huggingface_model import HuggingFaceModel
from src.models.transformer_lens import TransformerLensModel, autoregressive_steering


# Use a small model for fast testing
TEST_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    yield
    gc.collect()
    torch.cuda.empty_cache()


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def hf_model():
    attn_impl = "flash_attention_2" if _flash_attn_available() else "eager"
    return HuggingFaceModel(TEST_MODEL, device="cuda", dtype="bfloat16", attn_implementation=attn_impl)


@pytest.fixture(scope="module")
def tl_model():
    return TransformerLensModel(TEST_MODEL, device="cuda", dtype="bfloat16")


# Tolerance for same-backend comparisons (batch vs individual, etc.)
SAME_BACKEND_ATOL = 1e-5


def cross_backend_atol(layer: int, n_layers: int) -> float:
    """Tolerance for HF-vs-TL comparison, scaling with layer depth.

    Different attention implementations (HF eager vs TransformerLens) produce
    different bfloat16 rounding. Errors compound through layers:
    ~0.002 at layer 0, ~0.008 at mid layers, ~0.04 at final layer.
    """
    return 0.005 + 0.05 * (layer / n_layers)


class TestActivationsMatchTransformerLens:
    """Verify HF activations match TL activations on pre-existing completions."""

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
        """Last token activations should match across all layers."""
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
            np.testing.assert_allclose(
                hf_act, tl_act, rtol=0, atol=cross_backend_atol(layer, hf_model.n_layers),
                err_msg=f"Activation mismatch at layer {layer}",
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
            np.testing.assert_allclose(
                hf_acts["first"][layer], tl_acts["first"][layer],
                rtol=0, atol=cross_backend_atol(layer, hf_model.n_layers),
                err_msg=f"First-token mismatch at layer {layer}",
            )

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
            np.testing.assert_allclose(
                hf_acts["mean"][layer], tl_acts["mean"][layer],
                rtol=0, atol=cross_backend_atol(layer, hf_model.n_layers),
                err_msg=f"Mean-token mismatch at layer {layer}",
            )

    def test_first_and_last_differ_for_multitoken_completion(
        self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel,
    ):
        """Sanity check: first and last selectors should give different vectors
        for a multi-token completion (catches off-by-one making them identical)."""
        messages = [
            {"role": "user", "content": "Count to five"},
            {"role": "assistant", "content": "One, two, three, four, five."},
        ]
        layer = hf_model.n_layers // 2

        for model, name in [(hf_model, "HF"), (tl_model, "TL")]:
            acts = model.get_activations(messages, [layer], ["first", "last"])
            first = acts["first"][layer]
            last = acts["last"][layer]
            assert not np.allclose(first, last, rtol=1e-2), (
                f"{name}: first and last activations are identical — likely an off-by-one bug"
            )


class TestGenerateWithActivationsMatch:
    """Verify generate_with_activations produces matching results.

    This is the codepath used by the extraction pipeline: generate a completion,
    then extract activations from the full (prompt + completion) sequence.
    """

    def test_same_completion(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        """At temperature=0 both backends should produce the same completion text."""
        messages = [{"role": "user", "content": "What is 1+1? Answer with just the number."}]
        layers = [hf_model.n_layers // 2]
        selectors = ["last", "first", "mean"]

        hf_result = hf_model.generate_with_activations(
            messages, layers, selectors, temperature=0, max_new_tokens=5,
        )
        tl_result = tl_model.generate_with_activations(
            messages, layers, selectors, temperature=0, max_new_tokens=5,
        )

        assert hf_result.completion == tl_result.completion, (
            f"Completions differ: HF={hf_result.completion!r} vs TL={tl_result.completion!r}"
        )

    def test_activations_match_after_generation(
        self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel,
    ):
        """Activations extracted after generation should match between backends."""
        messages = [{"role": "user", "content": "Say hello in French."}]
        layers = [0, hf_model.n_layers // 2, hf_model.n_layers - 1]
        selectors = ["last", "first", "mean"]

        hf_result = hf_model.generate_with_activations(
            messages, layers, selectors, temperature=0, max_new_tokens=16,
        )
        tl_result = tl_model.generate_with_activations(
            messages, layers, selectors, temperature=0, max_new_tokens=16,
        )

        # Completions must match first — otherwise activations are for different sequences
        assert hf_result.completion == tl_result.completion

        for selector in selectors:
            for layer in layers:
                np.testing.assert_allclose(
                    hf_result.activations[selector][layer],
                    tl_result.activations[selector][layer],
                    rtol=0, atol=cross_backend_atol(layer, hf_model.n_layers),
                    err_msg=f"generate_with_activations mismatch: selector={selector}, layer={layer}",
                )

    def test_activations_are_from_completion_not_prompt(
        self, hf_model: HuggingFaceModel,
    ):
        """Verify that extracted activations change when completion changes.

        If the extractor accidentally used prompt-only activations (ignoring
        the completion), changing the completion would not change the result.
        """
        messages = [{"role": "user", "content": "Hi"}]
        layer = hf_model.n_layers // 2

        acts_a = hf_model.get_activations(
            messages + [{"role": "assistant", "content": "Hello! How are you doing today?"}],
            [layer], ["last"],
        )
        acts_b = hf_model.get_activations(
            messages + [{"role": "assistant", "content": "Goodbye forever."}],
            [layer], ["last"],
        )

        assert not np.allclose(acts_a["last"][layer], acts_b["last"][layer], rtol=1e-2), (
            "Last-token activations are identical for different completions — "
            "extractor may be ignoring the completion"
        )


class TestSteeringMatches:
    """Verify steering produces same results."""

    def test_steering_matches(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        messages = [{"role": "user", "content": "Hi"}]
        layer = hf_model.n_layers // 2

        steering_vec = torch.randn(hf_model.hidden_dim, device="cuda", dtype=torch.bfloat16)
        hook = autoregressive_steering(steering_vec)

        hf_output = hf_model.generate_with_steering(messages, layer, hook, temperature=0, max_new_tokens=10)
        tl_output = tl_model.generate_with_steering(messages, layer, hook, temperature=0, max_new_tokens=10)

        assert hf_output == tl_output


class TestModelProperties:
    """Test model property accessors."""

    def test_n_layers_matches(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        assert hf_model.n_layers == tl_model.n_layers

    def test_hidden_dim_matches(self, hf_model: HuggingFaceModel, tl_model: TransformerLensModel):
        assert hf_model.hidden_dim == tl_model.hidden_dim


class TestEdgeCases:

    def test_empty_assistant_message_raises(self, hf_model: HuggingFaceModel):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
        ]
        with pytest.raises(ValueError, match="no content tokens"):
            hf_model.get_activations(messages, [0], ["last"])

    def test_no_assistant_message_raises(self, hf_model: HuggingFaceModel):
        messages = [{"role": "user", "content": "Hi"}]
        with pytest.raises(ValueError, match="must end with an assistant message"):
            hf_model.get_activations(messages, [0], ["first"])

    def test_multiple_selectors(self, hf_model: HuggingFaceModel):
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


class TestBatchedActivations:

    def test_batch_matches_individual(self, hf_model: HuggingFaceModel):
        """Batched extraction should match individual extraction."""
        messages_batch = [
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}],
            [{"role": "user", "content": "Say something long"}, {"role": "assistant", "content": "Here is a longer response with more tokens."}],
        ]
        layers = [0, hf_model.n_layers - 1]
        selectors = ["last", "first", "mean"]

        batch_acts = hf_model.get_activations_batch(messages_batch, layers, selectors)

        for i, messages in enumerate(messages_batch):
            individual_acts = hf_model.get_activations(messages, layers, selectors)
            for selector in selectors:
                for layer in layers:
                    np.testing.assert_allclose(
                        batch_acts[selector][layer][i],
                        individual_acts[selector][layer],
                        rtol=0, atol=SAME_BACKEND_ATOL,
                        err_msg=f"Batch vs individual mismatch: sample {i}, {selector}, layer {layer}",
                    )

    def test_batch_output_shape(self, hf_model: HuggingFaceModel):
        messages_batch = [
            [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}],
            [{"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}],
        ]

        acts = hf_model.get_activations_batch(messages_batch, [0], ["last"])
        assert acts["last"][0].shape == (2, hf_model.hidden_dim)

    def test_batch_variable_lengths(self, hf_model: HuggingFaceModel):
        messages_batch = [
            [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
            [{"role": "user", "content": "A much longer prompt here"}, {"role": "assistant", "content": "And a much longer response too!"}],
        ]
        selectors = ["last", "first", "mean"]

        acts = hf_model.get_activations_batch(messages_batch, [0], selectors)

        for selector in selectors:
            assert acts[selector][0].shape == (2, hf_model.hidden_dim)
