"""Integration tests for activation extraction pipeline.

Run with:
    pytest tests/test_activation_extraction.py -v

Skip with:
    pytest -m "not gpu"

Requires:
    pip install -e ".[probe]"
    GPU with sufficient VRAM for the model
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformer_lens")

from src.models.transformer_lens import TransformerLensModel
from src.probes.data import ProbeDataPoint, save_probe_dataset, load_probe_dataset


pytestmark = pytest.mark.gpu

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"


@pytest.fixture(scope="module")
def model():
    return TransformerLensModel(model_name=MODEL_NAME, max_new_tokens=64)


class TestActivationExtractionIntegration:
    """End-to-end integration tests for activation extraction."""

    def test_extract_activations_and_save_load(self, model):
        """Full pipeline: extract activations -> save -> load -> verify."""
        prompts = [
            "Write a haiku about mountains.",
            "Solve: 2 + 2 = ?",
            "Explain gravity in one sentence.",
        ]

        layers = [0.25, 0.5, 0.75]
        resolved_layers = [model.resolve_layer(l) for l in layers]

        data_points = []
        for i, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            completion = model.generate(messages, temperature=0.0)
            messages.append({"role": "assistant", "content": completion})

            activations = model.get_activations(messages, layers=layers)

            data_points.append(ProbeDataPoint(
                task_id=f"task_{i}",
                activations=activations,
                score=float(i),
                completion=completion,
                raw_rating_response=str(i),
            ))

        # Verify extraction worked
        d_model = model.model.cfg.d_model
        for dp in data_points:
            for layer in resolved_layers:
                assert layer in dp.activations
                assert dp.activations[layer].shape == (d_model,)

        # Activations should differ between tasks
        assert not np.allclose(
            data_points[0].activations[resolved_layers[1]],
            data_points[1].activations[resolved_layers[1]],
        )

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_probe_dataset(data_points, output_dir, metadata={"model": MODEL_NAME})
            loaded = load_probe_dataset(output_dir)

        # Verify loaded data matches
        assert len(loaded) == len(data_points)
        for original, reloaded in zip(data_points, loaded):
            assert original.task_id == reloaded.task_id
            assert original.score == reloaded.score
            assert original.completion == reloaded.completion
            for layer in resolved_layers:
                assert np.allclose(original.activations[layer], reloaded.activations[layer])

    def test_activations_stackable_for_probe_training(self, model):
        """Activations can be stacked into a matrix for training probes."""
        prompts = ["Hello", "Goodbye", "Thanks", "Sorry"]
        layer = 0.5
        resolved = model.resolve_layer(layer)

        activations_list = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            acts = model.get_activations(messages, layers=[layer])
            activations_list.append(acts[resolved])

        X = np.stack(activations_list)
        assert X.shape == (len(prompts), model.model.cfg.d_model)

    def test_activation_patching_changes_output(self, model):
        """Verify we extract correct activations by patching them into another forward pass.

        If patching A's activations into B's forward pass changes B's output toward A's,
        we know we're extracting from the right place.
        """
        from functools import partial

        prompt_a = "The capital of France is"
        prompt_b = "The largest ocean is the"

        # Get clean outputs
        tokens_a = model.model.to_tokens(prompt_a)
        tokens_b = model.model.to_tokens(prompt_b)

        logits_a_clean = model.model(tokens_a)
        logits_b_clean = model.model(tokens_b)

        # Get next token predictions
        pred_a_clean = logits_a_clean[0, -1].argmax().item()
        pred_b_clean = logits_b_clean[0, -1].argmax().item()

        # They should predict different things
        assert pred_a_clean != pred_b_clean, "Test inputs should have different predictions"

        # Extract activations from A at a mid-layer
        layer = model.n_layers // 2
        _, cache_a = model.model.run_with_cache(tokens_a)
        activation_a = cache_a["resid_post", layer]

        # Patch A's activation into B's forward pass
        def patch_hook(activation, hook, patched_activation):
            # Only patch the last token position
            activation[0, -1, :] = patched_activation[0, -1, :]
            return activation

        hook_fn = partial(patch_hook, patched_activation=activation_a)
        logits_b_patched = model.model.run_with_hooks(
            tokens_b,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
        )

        pred_b_patched = logits_b_patched[0, -1].argmax().item()

        # The patched prediction should differ from clean B
        # (it may or may not equal A, but it should be influenced)
        logits_diff = (logits_b_patched[0, -1] - logits_b_clean[0, -1]).abs().mean().item()
        assert logits_diff > 0.1, "Patching should meaningfully change the output distribution"

    def test_get_activations_matches_direct_cache(self, model):
        """Verify get_activations returns the same values as direct cache access."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        layers = [0, model.n_layers // 2, model.n_layers - 1]

        # Get activations via our method
        activations = model.get_activations(messages, layers=layers)

        # Get activations directly from cache
        prompt = model._format_messages(messages, add_generation_prompt=False)
        tokens = model.model.to_tokens(prompt)
        _, cache = model.model.run_with_cache(tokens)

        for layer in layers:
            direct = cache["resid_post", layer][0, -1, :].cpu().numpy()
            assert np.allclose(activations[layer], direct, rtol=1e-5), \
                f"Layer {layer}: get_activations doesn't match direct cache access"
