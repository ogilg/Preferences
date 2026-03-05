"""End-to-end GPU tests for steering primitives.

Run with: pytest tests/test_steering_e2e.py -v -s
Requires GPU and ~3GB VRAM (llama-3.2-1b).
"""

import gc
import time

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu

from src.models.base import (
    all_tokens_steering,
    position_selective_steering,
    differential_steering,
    noop_steering,
)
from src.models.huggingface_model import HuggingFaceModel
from src.steering.client import SteeredHFClient
from src.steering.tokenization import find_text_span, find_pairwise_task_spans


MODEL_NAME = "llama-3.2-1b"
# Llama-3.2-1B has 16 layers, 2048 hidden dim
STEER_LAYER = 8
HIDDEN_DIM = 2048

# Use a fixed seed for the random direction so results are deterministic
RNG = np.random.default_rng(42)
DIRECTION = RNG.standard_normal(HIDDEN_DIM).astype(np.float32)
DIRECTION = DIRECTION / np.linalg.norm(DIRECTION)


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests in this module."""
    m = HuggingFaceModel(MODEL_NAME, max_new_tokens=32)
    yield m
    del m
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def steered_client(model):
    """Create a SteeredHFClient wrapping the shared model."""
    return SteeredHFClient(
        model, layer=STEER_LAYER, steering_direction=DIRECTION,
        coefficient=0.0, steering_mode="all_tokens",
    )


SIMPLE_PROMPT = [{"role": "user", "content": "What is 2 + 2?"}]


class TestSteeringChangesOutput:
    """Verify that steering actually modifies the model's generation."""

    def test_large_coefficient_changes_output(self, model):
        """With a large enough coefficient, output must differ from unsteered."""
        unsteered = model.generate(SIMPLE_PROMPT, temperature=0)

        tensor = torch.tensor(DIRECTION * 5000, dtype=torch.bfloat16, device="cuda")
        hook = all_tokens_steering(tensor)
        steered = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER, hook=hook, temperature=0,
        )

        assert steered != unsteered, (
            f"Steering with coef=5000 should change output.\n"
            f"Unsteered: {unsteered!r}\nSteered: {steered!r}"
        )

    def test_noop_hook_matches_unsteered(self, model):
        """Noop hook must produce identical output to no hook at temperature=0."""
        unsteered = model.generate(SIMPLE_PROMPT, temperature=0)

        hook = noop_steering()
        with_noop = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER, hook=hook, temperature=0,
        )

        assert with_noop == unsteered, (
            f"Noop hook should match unsteered.\n"
            f"Unsteered: {unsteered!r}\nNoop: {with_noop!r}"
        )


class TestPositionSelectiveHook:
    """Verify position-selective steering only affects specified token range."""

    def test_selective_differs_from_all_tokens(self, model):
        """Position-selective should produce different output than all-tokens steering."""
        tensor = torch.tensor(DIRECTION * 5000, dtype=torch.bfloat16, device="cuda")

        all_hook = all_tokens_steering(tensor)
        all_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER, hook=all_hook, temperature=0,
        )

        # Steer only the first 3 tokens
        sel_hook = position_selective_steering(tensor, start=0, end=3)
        sel_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER, hook=sel_hook, temperature=0,
        )

        assert sel_out != all_out, (
            "Position-selective (0:3) should differ from all-tokens steering"
        )


class TestDifferentialHook:
    def test_differential_differs_from_noop(self, model):
        """Differential hook should change output vs noop."""
        noop_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER, hook=noop_steering(), temperature=0,
        )

        tensor = torch.tensor(DIRECTION * 3000, dtype=torch.bfloat16, device="cuda")
        diff_hook = differential_steering(tensor, pos_start=0, pos_end=5, neg_start=5, neg_end=10)
        diff_out = model.generate_with_hook(
            SIMPLE_PROMPT, layer=STEER_LAYER, hook=diff_hook, temperature=0,
        )

        assert diff_out != noop_out, "Differential hook should change output vs noop"


class TestSteeredHFClientE2E:
    def test_with_coefficient_changes_output(self, steered_client):
        """with_coefficient(large) should change output vs coefficient=0."""
        baseline = steered_client.generate(SIMPLE_PROMPT, temperature=0)

        strong = steered_client.with_coefficient(5000.0)
        steered = strong.generate(SIMPLE_PROMPT, temperature=0)

        assert steered != baseline

    def test_direction_property_matches(self, steered_client):
        assert np.array_equal(steered_client.direction, DIRECTION)


class TestFindTextSpanRealTokenizer:
    """Test find_text_span with the actual Llama tokenizer."""

    def test_finds_correct_span(self, model):
        text = "The quick brown fox jumps over the lazy dog"
        start, end = find_text_span(model.tokenizer, text, "brown fox")

        # Decode the span tokens and verify they cover "brown fox"
        encoding = model.tokenizer(text, add_special_tokens=False)
        span_ids = encoding["input_ids"][start:end]
        decoded = model.tokenizer.decode(span_ids)
        assert "brown fox" in decoded, f"Decoded span: {decoded!r}"

    def test_pairwise_spans_real_tokenizer(self, model):
        prompt = "Task A: Write a poem about the ocean. Task B: Solve a calculus problem."
        a_span, b_span = find_pairwise_task_spans(
            model.tokenizer, prompt,
            "Write a poem about the ocean",
            "Solve a calculus problem",
            a_marker="Task A",
            b_marker="Task B",
        )

        encoding = model.tokenizer(prompt, add_special_tokens=False)
        a_decoded = model.tokenizer.decode(encoding["input_ids"][a_span[0]:a_span[1]])
        b_decoded = model.tokenizer.decode(encoding["input_ids"][b_span[0]:b_span[1]])

        assert "poem" in a_decoded and "ocean" in a_decoded, f"A span: {a_decoded!r}"
        assert "calculus" in b_decoded, f"B span: {b_decoded!r}"
        # Spans should not overlap
        assert a_span[1] <= b_span[0], f"Spans overlap: A={a_span}, B={b_span}"


class TestContrastiveMiniExperiment:
    """Mini steering experiment: opposite all-tokens steering produces different outputs.

    Uses +coef vs -coef with all_tokens_steering on the same random direction.
    This tests the full SteeredHFClient pipeline end-to-end: a coefficient sweep
    where opposite signs should produce distinguishable generations.
    """

    def test_opposite_coefficients_produce_different_outputs(self, steered_client, model):
        messages = [{"role": "user", "content": "Tell me a short story."}]

        # Baseline (coef=0)
        baseline = steered_client.generate(messages, temperature=0)

        # Positive direction
        pos_client = steered_client.with_coefficient(5000.0)
        out_pos = pos_client.generate(messages, temperature=0)

        # Negative direction
        neg_client = steered_client.with_coefficient(-5000.0)
        out_neg = neg_client.generate(messages, temperature=0)

        print(f"\n--- Contrastive coefficient sweep ---")
        print(f"Baseline (0):    {baseline[:80]!r}")
        print(f"Positive (5000): {out_pos[:80]!r}")
        print(f"Negative (-5000):{out_neg[:80]!r}")

        # Both steered conditions should differ from baseline
        assert out_pos != baseline, "Positive steering should change output"
        assert out_neg != baseline, "Negative steering should change output"
        # And differ from each other
        assert out_pos != out_neg, (
            f"+coef and -coef should produce different outputs.\n"
            f"Positive: {out_pos!r}\nNegative: {out_neg!r}"
        )

    def test_differential_hook_changes_output_from_baseline(self, model):
        """Differential hook with large coefficient should change output vs baseline."""
        messages = [{"role": "user", "content": "What is your favorite color?"}]

        baseline = model.generate(messages, temperature=0)

        formatted = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        n_tokens = len(model.tokenizer(formatted, add_special_tokens=False)["input_ids"])
        mid = n_tokens // 2

        tensor = torch.tensor(DIRECTION * 10000, dtype=torch.bfloat16, device="cuda")
        hook = differential_steering(tensor, pos_start=0, pos_end=mid, neg_start=mid, neg_end=n_tokens)
        steered = model.generate_with_hook(
            messages, layer=STEER_LAYER, hook=hook, temperature=0,
        )

        print(f"\n--- Differential hook test ---")
        print(f"Baseline: {baseline[:80]!r}")
        print(f"Steered:  {steered[:80]!r}")

        assert steered != baseline, "Differential hook with large coef should change output"


class TestGenerateN:
    """Test batched generation via generate_n / generate_with_hook_n."""

    def test_generate_n_returns_list(self, model):
        results = model.generate_n(SIMPLE_PROMPT, n=5, temperature=1.0)
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, str) for r in results)

    def test_generate_still_returns_str(self, model):
        result = model.generate(SIMPLE_PROMPT, temperature=0)
        assert isinstance(result, str)

    def test_generate_with_hook_n_returns_list(self, model):
        tensor = torch.tensor(DIRECTION * 3000, dtype=torch.bfloat16, device="cuda")
        hook = all_tokens_steering(tensor)
        results = model.generate_with_hook_n(
            SIMPLE_PROMPT, layer=STEER_LAYER, hook=hook,
            n=5, temperature=1.0,
        )
        assert isinstance(results, list)
        assert len(results) == 5

    def test_client_generate_n(self, steered_client):
        client = steered_client.with_coefficient(3000.0)
        results = client.generate_n(SIMPLE_PROMPT, n=5, temperature=1.0)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_batched_has_variation(self, model):
        results = model.generate_n(SIMPLE_PROMPT, n=10, temperature=1.0)
        unique = set(results)
        assert len(unique) > 1, (
            f"10 samples at temperature=1.0 should show variation, got all identical: {results[0]!r}"
        )

    def test_steered_batched_has_variation(self, model):
        tensor = torch.tensor(DIRECTION * 1000, dtype=torch.bfloat16, device="cuda")
        hook = all_tokens_steering(tensor)
        results = model.generate_with_hook_n(
            SIMPLE_PROMPT, layer=STEER_LAYER, hook=hook,
            n=10, temperature=1.0,
        )
        unique = set(results)
        assert len(unique) > 1, (
            f"10 steered samples at temperature=1.0 should show variation, got all identical: {results[0]!r}"
        )

    def test_batched_faster_than_serial(self, model):
        n = 10
        messages = [{"role": "user", "content": "Tell me a short story about a cat."}]

        t0 = time.perf_counter()
        for _ in range(n):
            model.generate(messages, temperature=1.0)
        serial_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.generate_n(messages, n=n, temperature=1.0)
        batched_time = time.perf_counter() - t0

        speedup = serial_time / batched_time
        print(f"\n--- Batch speedup: {speedup:.1f}x (serial={serial_time:.2f}s, batched={batched_time:.2f}s)")
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.1f}x"

    def test_steered_batched_faster_than_serial(self, model):
        n = 10
        messages = [{"role": "user", "content": "Tell me a short story about a cat."}]
        tensor = torch.tensor(DIRECTION * 1000, dtype=torch.bfloat16, device="cuda")
        hook = all_tokens_steering(tensor)

        t0 = time.perf_counter()
        for _ in range(n):
            model.generate_with_hook(
                messages, layer=STEER_LAYER, hook=hook, temperature=1.0,
            )
        serial_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        model.generate_with_hook_n(
            messages, layer=STEER_LAYER, hook=hook,
            n=n, temperature=1.0,
        )
        batched_time = time.perf_counter() - t0

        speedup = serial_time / batched_time
        print(f"\n--- Steered batch speedup: {speedup:.1f}x (serial={serial_time:.2f}s, batched={batched_time:.2f}s)")
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.1f}x"
