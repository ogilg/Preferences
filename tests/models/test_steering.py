"""End-to-end tests for steering hooks in TransformerLens model."""

import pytest
import torch
import numpy as np

from src.models.transformer_lens import (
    TransformerLensModel,
    SteeringHook,
    STEERING_MODES,
    last_token_steering,
    all_tokens_steering,
    generation_only_steering,
)


class TestSteeringHookShapes:
    """Test that steering hooks handle tensor shapes correctly."""

    def test_last_token_steering_shape(self):
        """last_token_steering should modify only the last position."""
        d_model = 768
        steering_tensor = torch.randn(d_model)
        hook = last_token_steering(steering_tensor)

        # Simulate residual stream: (batch=1, seq_len=5, d_model)
        resid = torch.zeros(1, 5, d_model)
        prompt_len = 3

        result = hook(resid, prompt_len)

        # Only last position should be modified
        assert torch.allclose(result[:, -1, :], steering_tensor.unsqueeze(0))
        assert torch.allclose(result[:, :-1, :], torch.zeros(1, 4, d_model))

    def test_all_tokens_steering_shape(self):
        """all_tokens_steering should modify all positions."""
        d_model = 768
        steering_tensor = torch.randn(d_model)
        hook = all_tokens_steering(steering_tensor)

        resid = torch.zeros(1, 5, d_model)
        prompt_len = 3

        result = hook(resid, prompt_len)

        # All positions should be modified equally
        for i in range(5):
            assert torch.allclose(result[:, i, :], steering_tensor.unsqueeze(0))

    def test_generation_only_steering_shape(self):
        """generation_only_steering should modify only positions >= prompt_len."""
        d_model = 768
        steering_tensor = torch.randn(d_model)
        hook = generation_only_steering(steering_tensor)

        resid = torch.zeros(1, 5, d_model)
        prompt_len = 3

        result = hook(resid, prompt_len)

        # Prompt positions should be unchanged
        assert torch.allclose(result[:, :prompt_len, :], torch.zeros(1, prompt_len, d_model))
        # Generated positions should be steered
        for i in range(prompt_len, 5):
            assert torch.allclose(result[:, i, :], steering_tensor.unsqueeze(0))

    def test_generation_only_with_no_generation(self):
        """generation_only_steering with prompt_len == seq_len should do nothing."""
        d_model = 768
        steering_tensor = torch.randn(d_model)
        hook = generation_only_steering(steering_tensor)

        resid = torch.zeros(1, 5, d_model)
        prompt_len = 5  # No new tokens yet

        result = hook(resid, prompt_len)

        # Nothing should be modified
        assert torch.allclose(result, torch.zeros(1, 5, d_model))


class TestSteeringModes:
    """Test that STEERING_MODES registry is correct."""

    def test_all_modes_registered(self):
        assert "last_token" in STEERING_MODES
        assert "all_tokens" in STEERING_MODES
        assert "generation_only" in STEERING_MODES

    def test_modes_return_callables(self):
        d_model = 768
        steering_tensor = torch.randn(d_model)

        for name, factory in STEERING_MODES.items():
            hook = factory(steering_tensor)
            assert callable(hook), f"{name} factory should return callable"


@pytest.mark.api
class TestSteeringEndToEnd:
    """End-to-end tests with actual model generation. Run with GPU."""

    @pytest.fixture(scope="class")
    def model(self):
        return TransformerLensModel(
            "llama-3.2-1b",
            device="cuda",
            dtype="bfloat16",
            max_new_tokens=10,
        )

    def test_generate_with_steering_runs(self, model):
        """Basic smoke test that steering doesn't crash."""
        messages = [{"role": "user", "content": "Say hello"}]

        steering_tensor = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        hook = last_token_steering(steering_tensor)

        # Should not raise
        result = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=hook,
            temperature=1.0,
            max_new_tokens=5,
        )
        assert isinstance(result, str)

    def test_all_steering_modes_run(self, model):
        """All steering modes should run without error."""
        messages = [{"role": "user", "content": "Count to 3"}]

        steering_tensor = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )

        for mode_name, factory in STEERING_MODES.items():
            hook = factory(steering_tensor)
            result = model.generate_with_steering(
                messages=messages,
                layer=model.n_layers // 2,
                steering_hook=hook,
                temperature=1.0,
                max_new_tokens=5,
            )
            assert isinstance(result, str), f"{mode_name} should return string"

    def test_steering_causally_affects_generation(self, model):
        """Opposite steering directions must produce different outputs.

        This tests that steering has a causal effect on generation by applying
        the same random direction with positive and negative coefficients.
        With temperature=0 (greedy decoding), different activations should
        produce different token sequences.
        """
        messages = [{"role": "user", "content": "Describe your current mood in one word:"}]

        # Use a fixed random direction
        torch.manual_seed(42)
        direction = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        direction = direction / direction.norm()  # Unit normalize

        coefficient = 3.0  # Strong enough to change behavior
        pos_steering = direction * coefficient
        neg_steering = direction * (-coefficient)

        pos_hook = generation_only_steering(pos_steering)
        neg_hook = generation_only_steering(neg_steering)

        pos_output = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=pos_hook,
            temperature=0.0,
            max_new_tokens=20,
        )

        neg_output = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=neg_hook,
            temperature=0.0,
            max_new_tokens=20,
        )

        assert pos_output != neg_output, (
            f"Steering with opposite directions should produce different outputs.\n"
            f"Positive: {pos_output!r}\n"
            f"Negative: {neg_output!r}"
        )

    def test_zero_steering_matches_baseline(self, model):
        """Zero steering vector should produce same output as no steering."""
        messages = [{"role": "user", "content": "Hello"}]

        # Baseline
        torch.manual_seed(123)
        baseline = model.generate(messages, temperature=0.0, max_new_tokens=5)

        # Zero steering
        steering_tensor = torch.zeros(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        hook = generation_only_steering(steering_tensor)

        torch.manual_seed(123)
        steered = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=hook,
            temperature=0.0,
            max_new_tokens=5,
        )

        assert steered == baseline

    def test_hooks_are_cleaned_up(self, model):
        """Hooks should be removed after generation."""
        messages = [{"role": "user", "content": "Test"}]

        steering_tensor = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        hook = last_token_steering(steering_tensor)

        # Run steering
        model.generate_with_steering(
            messages=messages,
            layer=0,
            steering_hook=hook,
            max_new_tokens=3,
        )

        # Check hooks are cleaned up
        assert len(model.model.hook_dict) == 0 or all(
            len(hooks) == 0 for hooks in model.model.hook_dict.values()
        )

    def test_multi_turn_conversation(self, model):
        """Steering should work with multi-turn conversations."""
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Give an example."},
        ]

        steering_tensor = torch.randn(
            model.hidden_dim,
            dtype=model.model.cfg.dtype,
            device=model.model.cfg.device,
        )
        hook = generation_only_steering(steering_tensor)

        # Should not raise
        result = model.generate_with_steering(
            messages=messages,
            layer=model.n_layers // 2,
            steering_hook=hook,
            temperature=1.0,
            max_new_tokens=5,
        )
        assert isinstance(result, str)


@pytest.mark.api
class TestConceptVectorSteering:
    """Tests using real trained concept vectors."""

    @pytest.fixture(scope="class")
    def model_8b(self):
        return TransformerLensModel(
            "llama-3.1-8b",
            device="cuda",
            dtype="bfloat16",
            max_new_tokens=50,
        )

    def test_math_concept_vector_steering(self, model_8b):
        """Steering with math concept vector should causally affect output.

        Uses the math_math_sys concept vector at layer 16 with +/- 10 coefficient.
        """
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
        pos_steering = direction_tensor * coefficient
        neg_steering = direction_tensor * (-coefficient)

        messages = [{"role": "user", "content": "How do you feel about your work?"}]

        pos_hook = generation_only_steering(pos_steering)
        neg_hook = generation_only_steering(neg_steering)

        pos_output = model_8b.generate_with_steering(
            messages=messages,
            layer=16,
            steering_hook=pos_hook,
            temperature=0.0,
            max_new_tokens=50,
        )

        neg_output = model_8b.generate_with_steering(
            messages=messages,
            layer=16,
            steering_hook=neg_hook,
            temperature=0.0,
            max_new_tokens=50,
        )

        assert pos_output != neg_output, (
            f"Steering with +/-10 coefficient should produce different outputs.\n"
            f"+10: {pos_output!r}\n"
            f"-10: {neg_output!r}"
        )
