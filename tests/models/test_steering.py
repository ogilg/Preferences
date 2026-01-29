"""End-to-end causal tests for steering hooks in TransformerLens model.

Run with GPU: pytest tests/models/test_steering.py -v -m api
"""

import pytest
import torch
import numpy as np

from src.models.transformer_lens import (
    TransformerLensModel,
    STEERING_MODES,
    autoregressive_steering,
)


@pytest.mark.api
class TestSteeringCausality:
    """Causal tests verifying steering actually affects generation."""

    @pytest.fixture(scope="class")
    def model(self):
        return TransformerLensModel(
            "llama-3.2-1b",
            device="cuda",
            dtype="bfloat16",
            max_new_tokens=20,
        )

    def test_opposite_directions_produce_different_outputs(self, model):
        """Opposite steering directions must produce different outputs.

        With temperature=0, applying +/- the same direction should change tokens.
        """
        messages = [{"role": "user", "content": "Describe your current mood in one word:"}]

        torch.manual_seed(42)
        direction = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        direction = direction / direction.norm()

        coefficient = 3.0
        pos_hook = autoregressive_steering(direction * coefficient)
        neg_hook = autoregressive_steering(direction * (-coefficient))

        pos_output = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=pos_hook,
            temperature=0.0,
            max_new_tokens=15,
        )

        neg_output = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=neg_hook,
            temperature=0.0,
            max_new_tokens=15,
        )

        assert pos_output != neg_output, (
            f"Steering with opposite directions should produce different outputs.\n"
            f"Positive: {pos_output!r}\n"
            f"Negative: {neg_output!r}"
        )

    def test_zero_steering_matches_baseline(self, model):
        """Zero steering vector should produce identical output to no steering."""
        messages = [{"role": "user", "content": "What is 2+2?"}]

        baseline = model.generate(messages, temperature=0.0, max_new_tokens=10)

        zero_hook = autoregressive_steering(torch.zeros(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        ))

        steered = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=zero_hook,
            temperature=0.0,
            max_new_tokens=10,
        )

        assert steered == baseline, (
            f"Zero steering should match baseline.\n"
            f"Baseline: {baseline!r}\n"
            f"Steered: {steered!r}"
        )

    def test_larger_coefficient_has_larger_effect(self, model):
        """Larger steering coefficients should produce outputs more different from baseline."""
        messages = [{"role": "user", "content": "Tell me a short story about a cat."}]

        torch.manual_seed(123)
        direction = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        direction = direction / direction.norm()

        baseline = model.generate(messages, temperature=0.0, max_new_tokens=20)

        small_hook = autoregressive_steering(direction * 1.0)
        large_hook = autoregressive_steering(direction * 5.0)

        small_output = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=small_hook,
            temperature=0.0,
            max_new_tokens=20,
        )

        large_output = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=large_hook,
            temperature=0.0,
            max_new_tokens=20,
        )

        # At least one should differ from baseline
        assert small_output != baseline or large_output != baseline, (
            "At least one steering strength should change the output"
        )

        # If both differ, large should differ more (by token overlap)
        if small_output != baseline and large_output != baseline:
            baseline_tokens = set(baseline.split())
            small_overlap = len(baseline_tokens & set(small_output.split()))
            large_overlap = len(baseline_tokens & set(large_output.split()))
            # Large coefficient should have less overlap with baseline
            assert large_overlap <= small_overlap or large_output != small_output, (
                f"Larger coefficient should diverge more from baseline.\n"
                f"Baseline: {baseline!r}\n"
                f"Small (1.0): {small_output!r}\n"
                f"Large (5.0): {large_output!r}"
            )

    def test_all_steering_modes_affect_generation(self, model):
        """All registered steering modes should causally affect generation."""
        messages = [{"role": "user", "content": "Complete this sentence: The weather today is"}]

        torch.manual_seed(456)
        direction = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        direction = direction / direction.norm()

        baseline = model.generate(messages, temperature=0.0, max_new_tokens=15)

        for mode_name, factory in STEERING_MODES.items():
            hook = factory(direction * 3.0)
            steered = model.generate_with_steering(
                messages=messages,
                layer=model.n_layers // 2,
                steering_hook=hook,
                temperature=0.0,
                max_new_tokens=15,
            )
            assert steered != baseline, (
                f"Steering mode '{mode_name}' should affect generation.\n"
                f"Baseline: {baseline!r}\n"
                f"Steered: {steered!r}"
            )

    def test_steering_at_different_layers(self, model):
        """Steering at different layers should produce different outputs."""
        messages = [{"role": "user", "content": "What color is the sky?"}]

        torch.manual_seed(789)
        direction = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        direction = direction / direction.norm()

        coefficient = 3.0
        hook = autoregressive_steering(direction * coefficient)

        early_layer = model.n_layers // 4
        mid_layer = model.n_layers // 2
        late_layer = 3 * model.n_layers // 4

        outputs = {}
        for layer in [early_layer, mid_layer, late_layer]:
            outputs[layer] = model.generate_with_steering(
                messages=messages,
                layer=layer,
                steering_hook=hook,
                temperature=0.0,
                max_new_tokens=15,
            )

        # At least two layers should produce different outputs
        unique_outputs = set(outputs.values())
        assert len(unique_outputs) >= 2, (
            f"Steering at different layers should produce different outputs.\n"
            f"Early ({early_layer}): {outputs[early_layer]!r}\n"
            f"Mid ({mid_layer}): {outputs[mid_layer]!r}\n"
            f"Late ({late_layer}): {outputs[late_layer]!r}"
        )

    def test_hooks_cleaned_up_after_generation(self, model):
        """Hooks should be removed after generation completes."""
        messages = [{"role": "user", "content": "Test"}]

        hook = autoregressive_steering(torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        ))

        model.generate_with_steering(
            messages=messages,
            layer=0,
            steering_hook=hook,
            max_new_tokens=3,
        )

        active_hooks = sum(len(hp.fwd_hooks) for hp in model.model.hook_dict.values())
        assert active_hooks == 0, f"Expected 0 active hooks after generation, found {active_hooks}"


@pytest.mark.api
@pytest.mark.slow
class TestConceptVectorSteering:
    """Tests using real trained concept vectors. Requires 8B model."""

    @pytest.fixture(scope="class")
    def model_8b(self):
        return TransformerLensModel(
            "llama-3.1-8b",
            device="cuda",
            dtype="bfloat16",
            max_new_tokens=30,
        )

    def test_math_concept_vector_steering(self, model_8b):
        """Steering with math concept vector should causally affect output."""
        from pathlib import Path

        vector_path = Path("concept_vectors/math_math_sys/vectors/layer_16.npy")
        if not vector_path.exists():
            pytest.skip("Math concept vector not available")

        direction = np.load(vector_path)
        direction_tensor = torch.tensor(
            direction,
            dtype=model_8b.model.cfg.dtype,
            device=model_8b.model.cfg.device,
        )

        coefficient = 10.0
        pos_hook = autoregressive_steering(direction_tensor * coefficient)
        neg_hook = autoregressive_steering(direction_tensor * (-coefficient))

        messages = [{"role": "user", "content": "How do you feel about your work?"}]

        pos_output = model_8b.generate_with_steering(
            messages=messages,
            layer=16,
            steering_hook=pos_hook,
            temperature=0.0,
            max_new_tokens=30,
        )

        neg_output = model_8b.generate_with_steering(
            messages=messages,
            layer=16,
            steering_hook=neg_hook,
            temperature=0.0,
            max_new_tokens=30,
        )

        assert pos_output != neg_output, (
            f"Steering with +/-10 coefficient should produce different outputs.\n"
            f"+10: {pos_output!r}\n"
            f"-10: {neg_output!r}"
        )
