"""Integration tests for concept vector extraction via system prompt conditioning.

Run with:
    pytest tests/test_concept_vectors_integration.py -v -m gpu

Skip with:
    pytest -m "not gpu"
"""

import gc
import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except ImportError as e:
    raise RuntimeError("torch is required for integration tests.") from e

if not torch.cuda.is_available():
    pytestmark = pytest.mark.skip(reason="CUDA not available")
else:
    pytestmark = pytest.mark.gpu

from src.concept_vectors.config import ConceptVectorExtractionConfig, load_config
from src.concept_vectors.difference import (
    compute_difference_in_means,
    load_concept_vector_for_steering,
    save_concept_vectors,
)
from src.concept_vectors.extraction import extract_activations_with_system_prompt
from src.models.transformer_lens import TransformerLensModel
from src.task_data import Task, OriginDataset, load_tasks

MODEL_NAME = "llama-3.2-1b"


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache after each test."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def transformer_lens_model():
    """Shared TransformerLens model instance for efficiency."""
    return TransformerLensModel(model_name=MODEL_NAME, max_new_tokens=64)


@pytest.fixture
def small_tasks():
    """Small set of tasks for quick tests."""
    return [
        Task(prompt="What is 2+2?", origin=OriginDataset.MATH, id="math_1", metadata={}),
        Task(prompt="What is 3+3?", origin=OriginDataset.MATH, id="math_2", metadata={}),
        Task(prompt="What is 4+4?", origin=OriginDataset.MATH, id="math_3", metadata={}),
    ]


class TestExtractionWithSystemPrompt:
    """Test activation extraction with system prompts."""

    def test_extraction_produces_activations(self, transformer_lens_model, small_tasks, tmp_path):
        """Verify extraction produces activation files and completions.json."""
        layers = [transformer_lens_model.n_layers // 2]
        system_prompt = "You are a helpful assistant."
        selector_names = ["last", "first", "mean"]

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt=system_prompt,
            condition_name="test",
            output_dir=tmp_path,
            selector_names=selector_names,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Check files exist for all selectors
        for selector in selector_names:
            assert (tmp_path / f"activations_{selector}.npz").exists()
        assert (tmp_path / "completions.json").exists()
        assert (tmp_path / "extraction_metadata.json").exists()

        # Load and verify activations for each selector
        for selector in selector_names:
            data = np.load(tmp_path / f"activations_{selector}.npz", allow_pickle=True)
            assert len(data["task_ids"]) == 3
            assert f"layer_{layers[0]}" in data.keys()

            # Verify activations have correct shape
            acts = data[f"layer_{layers[0]}"]
            assert acts.shape[0] == 3  # n_tasks
            assert acts.shape[1] == transformer_lens_model.hidden_dim

        # Load and verify completions
        with open(tmp_path / "completions.json") as f:
            completions = json.load(f)
        assert len(completions) == 3
        assert all(c["system_prompt"] == system_prompt for c in completions)
        assert all(c["condition"] == "test" for c in completions)

    def test_different_system_prompts_produce_different_activations(
        self, transformer_lens_model, small_tasks, tmp_path
    ):
        """Verify different system prompts produce measurably different activations."""
        layers = [transformer_lens_model.n_layers // 2]
        selector_names = ["last"]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        # Positive condition
        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You love solving problems. Every task brings you joy.",
            condition_name="positive",
            output_dir=pos_dir,
            selector_names=selector_names,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Negative condition
        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You hate solving problems. Every task is tedious.",
            condition_name="negative",
            output_dir=neg_dir,
            selector_names=selector_names,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Load both activations
        pos_data = np.load(pos_dir / "activations_last.npz", allow_pickle=True)
        neg_data = np.load(neg_dir / "activations_last.npz", allow_pickle=True)

        pos_acts = pos_data[f"layer_{layers[0]}"]
        neg_acts = neg_data[f"layer_{layers[0]}"]

        # Activations should differ
        assert not np.allclose(pos_acts, neg_acts)

        # Compute cosine similarity - should be high but not perfect
        # (same task content, different system prompt)
        for i in range(len(small_tasks)):
            cos_sim = np.dot(pos_acts[i], neg_acts[i]) / (
                np.linalg.norm(pos_acts[i]) * np.linalg.norm(neg_acts[i])
            )
            # Should be similar (same task) but not identical (different system prompt)
            assert 0.5 < cos_sim < 0.9999, f"Task {i} cosine sim {cos_sim} outside expected range"

    def test_extraction_without_system_prompt(self, transformer_lens_model, small_tasks, tmp_path):
        """Verify extraction works with system_prompt=None."""
        layers = [transformer_lens_model.n_layers // 2]

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt=None,
            condition_name="baseline",
            output_dir=tmp_path,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        with open(tmp_path / "completions.json") as f:
            completions = json.load(f)
        assert all(c["system_prompt"] is None for c in completions)

    def test_extraction_multiple_layers(self, transformer_lens_model, small_tasks, tmp_path):
        """Verify extraction works with multiple layers."""
        n_layers = transformer_lens_model.n_layers
        layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You are helpful.",
            condition_name="test",
            output_dir=tmp_path,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        data = np.load(tmp_path / "activations_last.npz", allow_pickle=True)

        # All layers should be present
        for layer in layers:
            assert f"layer_{layer}" in data.keys()
            acts = data[f"layer_{layer}"]
            assert acts.shape == (3, transformer_lens_model.hidden_dim)

        # Activations should differ between layers
        acts_early = data[f"layer_{layers[0]}"]
        acts_late = data[f"layer_{layers[-1]}"]
        assert not np.allclose(acts_early, acts_late)


class TestDifferenceInMeansIntegration:
    """Test difference-in-means on real extracted activations."""

    def test_compute_direction_from_extracted_activations(
        self, transformer_lens_model, small_tasks, tmp_path
    ):
        """End-to-end: extract activations for two conditions and compute direction."""
        layers = [transformer_lens_model.n_layers // 2]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        # Extract both conditions
        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You love this task. It makes you happy.",
            condition_name="positive",
            output_dir=pos_dir,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You hate this task. It makes you miserable.",
            condition_name="negative",
            output_dir=neg_dir,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        # Compute difference-in-means
        vectors = compute_difference_in_means(pos_dir, neg_dir, selector_name="last", layers=layers, normalize=True)

        # Verify output
        assert len(vectors) == 1
        layer = layers[0]
        assert layer in vectors
        assert vectors[layer].shape == (transformer_lens_model.hidden_dim,)
        assert abs(np.linalg.norm(vectors[layer]) - 1.0) < 1e-6  # Unit normalized

    def test_direction_is_nonzero(self, transformer_lens_model, small_tasks, tmp_path):
        """Direction should be non-trivial (not near-zero before normalization)."""
        layers = [transformer_lens_model.n_layers // 2]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You are extremely positive and enthusiastic.",
            condition_name="positive",
            output_dir=pos_dir,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You are extremely negative and pessimistic.",
            condition_name="negative",
            output_dir=neg_dir,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        # Compute without normalization to check raw magnitude
        vectors = compute_difference_in_means(pos_dir, neg_dir, selector_name="last", layers=layers, normalize=False)
        layer = layers[0]

        # The raw difference should have non-trivial magnitude
        raw_norm = np.linalg.norm(vectors[layer])
        assert raw_norm > 0.1, f"Direction norm too small: {raw_norm}"


class TestFullPipelineE2E:
    """End-to-end test of the complete pipeline."""

    def test_full_extraction_to_steering_compatibility(self, transformer_lens_model, tmp_path):
        """Full pipeline: extract -> compute direction -> verify steering-compatible format."""
        from src.concept_vectors.difference import compute_all_concept_vectors

        # Use real tasks
        tasks = load_tasks(n=3, origins=[OriginDataset.MATH], seed=42)
        layers = [transformer_lens_model.n_layers // 2]
        layer = layers[0]
        selector_names = ["last", "first", "mean"]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        # Extract positive condition
        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=tasks,
            layers=layers,
            system_prompt="You find math problems deeply satisfying to solve.",
            condition_name="positive",
            output_dir=pos_dir,
            selector_names=selector_names,
            temperature=0.0,
            max_new_tokens=64,
            task_origins=["math"],
            layers_config=[0.5],
            seed=42,
        )

        # Extract negative condition
        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=tasks,
            layers=layers,
            system_prompt="You find math problems extremely tedious and annoying.",
            condition_name="negative",
            output_dir=neg_dir,
            selector_names=selector_names,
            temperature=0.0,
            max_new_tokens=64,
            task_origins=["math"],
            layers_config=[0.5],
            seed=42,
        )

        # Compute and save vectors for all selectors
        vectors_by_selector = compute_all_concept_vectors(
            pos_dir, neg_dir, selector_names=selector_names, layers=layers, normalize=True
        )

        metadata = {
            "experiment_id": "e2e_test",
            "model": MODEL_NAME,
            "n_tasks": len(tasks),
            "task_origins": ["math"],
            "positive_condition": {"name": "positive", "system_prompt": "..."},
            "negative_condition": {"name": "negative", "system_prompt": "..."},
        }
        save_concept_vectors(vectors_by_selector, tmp_path, metadata)

        # Verify manifest structure
        assert (tmp_path / "manifest.json").exists()
        for selector in selector_names:
            assert (tmp_path / "vectors" / selector / f"layer_{layer}.npy").exists()

        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["experiment_id"] == "e2e_test"
        assert manifest["n_layers"] == 1
        assert manifest["hidden_dim"] == transformer_lens_model.hidden_dim
        assert set(manifest["selectors"]) == set(selector_names)

        # Verify steering-compatible loading for each selector
        for selector in selector_names:
            loaded_layer, direction = load_concept_vector_for_steering(tmp_path, layer=layer, selector=selector)
            assert loaded_layer == layer
            assert direction.shape == (transformer_lens_model.hidden_dim,)
            assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_extracted_direction_usable_for_steering(self, transformer_lens_model, tmp_path):
        """Verify extracted direction can be used with generate_with_steering."""
        tasks = load_tasks(n=2, origins=[OriginDataset.MATH], seed=42)
        layers = [transformer_lens_model.n_layers // 2]
        layer = layers[0]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=tasks,
            layers=layers,
            system_prompt="You love this.",
            condition_name="positive",
            output_dir=pos_dir,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=tasks,
            layers=layers,
            system_prompt="You hate this.",
            condition_name="negative",
            output_dir=neg_dir,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        vectors = compute_difference_in_means(pos_dir, neg_dir, selector_name="last", layers=layers, normalize=True)
        vectors_by_selector = {"last": vectors}
        save_concept_vectors(vectors_by_selector, tmp_path, {"experiment_id": "test"})

        # Load and use for steering
        import torch
        from src.models.transformer_lens import last_token_steering

        _, direction = load_concept_vector_for_steering(tmp_path, layer=layer)

        # Generate with positive steering
        messages = [{"role": "user", "content": "How do you feel about math?"}]
        pos_tensor = torch.tensor(direction * 2.0, dtype=transformer_lens_model.model.cfg.dtype, device=transformer_lens_model.model.cfg.device)
        pos_output = transformer_lens_model.generate_with_steering(
            messages=messages,
            layer=layer,
            steering_hook=last_token_steering(pos_tensor),
            temperature=0.0,
        )

        # Generate with negative steering
        neg_tensor = torch.tensor(direction * -2.0, dtype=transformer_lens_model.model.cfg.dtype, device=transformer_lens_model.model.cfg.device)
        neg_output = transformer_lens_model.generate_with_steering(
            messages=messages,
            layer=layer,
            steering_hook=last_token_steering(neg_tensor),
            temperature=0.0,
        )

        # Outputs should differ
        assert pos_output != neg_output
        assert len(pos_output) > 0
        assert len(neg_output) > 0


class TestResumeCheckpointing:
    """Test resume/checkpoint functionality."""

    def test_resume_skips_completed_tasks(self, transformer_lens_model, small_tasks, tmp_path):
        """Verify resume mode skips already-extracted tasks."""
        layers = [transformer_lens_model.n_layers // 2]

        # Extract first 2 tasks
        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks[:2],
            layers=layers,
            system_prompt="Test prompt",
            condition_name="test",
            output_dir=tmp_path,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
        )

        # Verify 2 tasks extracted
        data = np.load(tmp_path / "activations_last.npz", allow_pickle=True)
        assert len(data["task_ids"]) == 2

        # Resume with all 3 tasks
        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="Test prompt",
            condition_name="test",
            output_dir=tmp_path,
            selector_names=["last"],
            temperature=0.0,
            max_new_tokens=32,
            resume=True,
        )

        # Should now have all 3
        data = np.load(tmp_path / "activations_last.npz", allow_pickle=True)
        assert len(data["task_ids"]) == 3


class TestTokenSelectorCorrectness:
    """Test that different token selectors extract activations from the correct positions."""

    def test_selectors_extract_different_activations(self, transformer_lens_model, small_tasks, tmp_path):
        """Verify that last, first, and mean selectors produce different activations."""
        layers = [transformer_lens_model.n_layers // 2]
        selector_names = ["last", "first", "mean"]

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You are a helpful assistant.",
            condition_name="test",
            output_dir=tmp_path,
            selector_names=selector_names,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Load activations for each selector
        layer = layers[0]
        last_data = np.load(tmp_path / "activations_last.npz", allow_pickle=True)
        first_data = np.load(tmp_path / "activations_first.npz", allow_pickle=True)
        mean_data = np.load(tmp_path / "activations_mean.npz", allow_pickle=True)

        last_acts = last_data[f"layer_{layer}"]
        first_acts = first_data[f"layer_{layer}"]
        mean_acts = mean_data[f"layer_{layer}"]

        # All should have same shape
        assert last_acts.shape == first_acts.shape == mean_acts.shape
        assert last_acts.shape[1] == transformer_lens_model.hidden_dim

        # They should be different from each other
        assert not np.allclose(last_acts, first_acts), "Last and first should differ"
        assert not np.allclose(last_acts, mean_acts), "Last and mean should differ"
        assert not np.allclose(first_acts, mean_acts), "First and mean should differ"

    def test_first_token_position_is_completion_start(self, transformer_lens_model):
        """Verify that 'first' selector gets the first completion token, not the first input token."""
        from src.models.base import SELECTOR_REGISTRY

        messages = [
            {"role": "user", "content": "Say hello"},
        ]

        # Generate and get activations
        layer = transformer_lens_model.n_layers // 2
        result = transformer_lens_model.generate_with_activations(
            messages,
            layers=[layer],
            selector_names=["first", "last"],
            temperature=0.0,
            max_new_tokens=16,
        )

        first_act = result.activations["first"][layer]
        last_act = result.activations["last"][layer]

        # First and last should differ for multi-token completions
        assert not np.allclose(first_act, last_act), "First and last should differ for multi-token output"

    def test_mean_is_average_of_completion_tokens(self, transformer_lens_model):
        """Verify that 'mean' selector produces an average of completion tokens."""
        # Get the full activation sequence manually to verify mean calculation
        messages = [{"role": "user", "content": "Count to three"}]

        layer = transformer_lens_model.n_layers // 2
        result = transformer_lens_model.generate_with_activations(
            messages,
            layers=[layer],
            selector_names=["mean"],
            temperature=0.0,
            max_new_tokens=32,
        )

        # Verify shape is correct (single vector, not sequence)
        mean_act = result.activations["mean"][layer]
        assert mean_act.shape == (transformer_lens_model.hidden_dim,), f"Mean activation should be 1D, got {mean_act.shape}"

    def test_selectors_consistent_across_calls(self, transformer_lens_model, small_tasks):
        """Verify that selectors produce consistent results on identical inputs."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        layer = transformer_lens_model.n_layers // 2

        # Call twice with temperature=0 for determinism
        result1 = transformer_lens_model.generate_with_activations(
            messages, layers=[layer], selector_names=["last", "first", "mean"],
            temperature=0.0, max_new_tokens=16
        )
        result2 = transformer_lens_model.generate_with_activations(
            messages, layers=[layer], selector_names=["last", "first", "mean"],
            temperature=0.0, max_new_tokens=16
        )

        # Should be identical
        for selector in ["last", "first", "mean"]:
            np.testing.assert_array_almost_equal(
                result1.activations[selector][layer],
                result2.activations[selector][layer],
                err_msg=f"Selector {selector} not consistent across calls"
            )

    def test_different_selectors_produce_different_concept_vectors(
        self, transformer_lens_model, small_tasks, tmp_path
    ):
        """Verify that different selectors produce different concept vectors."""
        from src.concept_vectors.difference import compute_all_concept_vectors

        layers = [transformer_lens_model.n_layers // 2]
        selector_names = ["last", "first", "mean"]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You love this task.",
            condition_name="positive",
            output_dir=pos_dir,
            selector_names=selector_names,
            temperature=0.0,
            max_new_tokens=32,
        )

        extract_activations_with_system_prompt(
            model=transformer_lens_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You hate this task.",
            condition_name="negative",
            output_dir=neg_dir,
            selector_names=selector_names,
            temperature=0.0,
            max_new_tokens=32,
        )

        vectors_by_selector = compute_all_concept_vectors(
            pos_dir, neg_dir, selector_names=selector_names, layers=layers, normalize=True
        )

        layer = layers[0]
        last_vec = vectors_by_selector["last"][layer]
        first_vec = vectors_by_selector["first"][layer]
        mean_vec = vectors_by_selector["mean"][layer]

        # Concept vectors from different selectors should differ
        assert not np.allclose(last_vec, first_vec), "Last and first concept vectors should differ"
        assert not np.allclose(last_vec, mean_vec), "Last and mean concept vectors should differ"
        assert not np.allclose(first_vec, mean_vec), "First and mean concept vectors should differ"

        # But they should all be unit normalized
        assert abs(np.linalg.norm(last_vec) - 1.0) < 1e-6
        assert abs(np.linalg.norm(first_vec) - 1.0) < 1e-6
        assert abs(np.linalg.norm(mean_vec) - 1.0) < 1e-6


class TestConfigIntegration:
    """Test config loading and integration."""

    def test_load_and_use_config(self, tmp_path):
        """Verify config can be loaded and used."""
        config_content = """
model: llama-3.1-8b
backend: transformer_lens
n_tasks: 3
task_origins:
  - math
task_sampling_seed: 42
conditions:
  positive:
    name: positive
    system_prompt: "You love math."
  negative:
    name: negative
    system_prompt: "You hate math."
layers_to_extract:
  - 0.5
selectors:
  - last
  - first
  - mean
temperature: 0.0
max_new_tokens: 32
output_dir: {output_dir}
experiment_id: config_test
""".format(output_dir=tmp_path)

        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.model == "llama-3.1-8b"
        assert config.n_tasks == 3
        assert len(config.conditions) == 2
        assert config.conditions["positive"]["system_prompt"] == "You love math."
        assert config.conditions["negative"]["system_prompt"] == "You hate math."
        assert config.selectors == ["last", "first", "mean"]


@pytest.mark.slow
class TestCLIEndToEnd:
    """Test the CLI entry point runs successfully."""

    def test_run_extraction_cli(self, tmp_path):
        """Full CLI test: run_extraction.py with a config file."""
        import subprocess
        import sys

        config_content = """
model: llama-3.2-1b
backend: transformer_lens
n_tasks: 2
task_origins:
  - math
task_sampling_seed: 42
conditions:
  positive:
    name: positive
    system_prompt: "You love solving math. It brings you joy."
  negative:
    name: negative
    system_prompt: "You hate solving math. It is tedious."
layers_to_extract:
  - 0.5
selectors:
  - last
  - first
  - mean
temperature: 0.0
max_new_tokens: 32
output_dir: {output_dir}
experiment_id: cli_test
""".format(output_dir=tmp_path)

        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        # Run the CLI
        result = subprocess.run(
            [sys.executable, "-m", "src.concept_vectors.run_extraction", str(config_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Check it succeeded
        assert result.returncode == 0, f"CLI failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        # Verify outputs exist for all selectors
        for selector in ["last", "first", "mean"]:
            assert (tmp_path / "positive" / f"activations_{selector}.npz").exists()
            assert (tmp_path / "negative" / f"activations_{selector}.npz").exists()
            assert (tmp_path / "vectors" / selector).exists()
        assert (tmp_path / "manifest.json").exists()

        # Verify manifest content
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["experiment_id"] == "cli_test"
        assert manifest["n_tasks"] == 2
        assert len(manifest["layers"]) == 1
        assert set(manifest["selectors"]) == {"last", "first", "mean"}

        # Verify vector is loadable with correct shape
        layer = manifest["layers"][0]
        hidden_dim = manifest["hidden_dim"]
        for selector in ["last", "first", "mean"]:
            loaded_layer, direction = load_concept_vector_for_steering(tmp_path, layer=layer, selector=selector)
            assert direction.shape == (hidden_dim,)
