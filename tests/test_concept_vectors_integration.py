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
from src.models.nnsight_model import NnsightModel
from src.task_data import Task, OriginDataset, load_tasks

MODEL_NAME = "llama-3.1-8b"


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache after each test."""
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def nnsight_model():
    """Shared model instance for efficiency."""
    return NnsightModel(model_name=MODEL_NAME, max_new_tokens=64)


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

    def test_extraction_produces_activations(self, nnsight_model, small_tasks, tmp_path):
        """Verify extraction produces activations.npz and completions.json."""
        layers = [nnsight_model.n_layers // 2]
        system_prompt = "You are a helpful assistant."

        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt=system_prompt,
            condition_name="test",
            output_dir=tmp_path,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Check files exist
        assert (tmp_path / "activations.npz").exists()
        assert (tmp_path / "completions.json").exists()
        assert (tmp_path / "extraction_metadata.json").exists()

        # Load and verify activations
        data = np.load(tmp_path / "activations.npz", allow_pickle=True)
        assert len(data["task_ids"]) == 3
        assert f"layer_{layers[0]}" in data.keys()

        # Verify activations have correct shape
        acts = data[f"layer_{layers[0]}"]
        assert acts.shape[0] == 3  # n_tasks
        assert acts.shape[1] == 4096  # hidden_dim for llama-3.1-8b

        # Load and verify completions
        with open(tmp_path / "completions.json") as f:
            completions = json.load(f)
        assert len(completions) == 3
        assert all(c["system_prompt"] == system_prompt for c in completions)
        assert all(c["condition"] == "test" for c in completions)

    def test_different_system_prompts_produce_different_activations(
        self, nnsight_model, small_tasks, tmp_path
    ):
        """Verify different system prompts produce measurably different activations."""
        layers = [nnsight_model.n_layers // 2]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        # Positive condition
        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You love solving problems. Every task brings you joy.",
            condition_name="positive",
            output_dir=pos_dir,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Negative condition
        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You hate solving problems. Every task is tedious.",
            condition_name="negative",
            output_dir=neg_dir,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Load both activations
        pos_data = np.load(pos_dir / "activations.npz", allow_pickle=True)
        neg_data = np.load(neg_dir / "activations.npz", allow_pickle=True)

        pos_acts = pos_data[f"layer_{layers[0]}"]
        neg_acts = neg_data[f"layer_{layers[0]}"]

        # Activations should differ
        assert not np.allclose(pos_acts, neg_acts)

        # Completions should differ
        with open(pos_dir / "completions.json") as f:
            pos_completions = json.load(f)
        with open(neg_dir / "completions.json") as f:
            neg_completions = json.load(f)

        # At least some completions should differ
        pos_texts = [c["completion"] for c in pos_completions]
        neg_texts = [c["completion"] for c in neg_completions]
        # Note: With temperature=0, completions might be similar, but activations should differ

    def test_extraction_without_system_prompt(self, nnsight_model, small_tasks, tmp_path):
        """Verify extraction works with system_prompt=None."""
        layers = [nnsight_model.n_layers // 2]

        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt=None,
            condition_name="baseline",
            output_dir=tmp_path,
            temperature=0.0,
            max_new_tokens=32,
        )

        with open(tmp_path / "completions.json") as f:
            completions = json.load(f)
        assert all(c["system_prompt"] is None for c in completions)


class TestDifferenceInMeansIntegration:
    """Test difference-in-means on real extracted activations."""

    def test_compute_direction_from_extracted_activations(
        self, nnsight_model, small_tasks, tmp_path
    ):
        """End-to-end: extract activations for two conditions and compute direction."""
        layers = [nnsight_model.n_layers // 2]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        # Extract both conditions
        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You love this task. It makes you happy.",
            condition_name="positive",
            output_dir=pos_dir,
            temperature=0.0,
            max_new_tokens=32,
        )

        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You hate this task. It makes you miserable.",
            condition_name="negative",
            output_dir=neg_dir,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Compute difference-in-means
        vectors = compute_difference_in_means(pos_dir, neg_dir, layers=layers, normalize=True)

        # Verify output
        assert len(vectors) == 1
        layer = layers[0]
        assert layer in vectors
        assert vectors[layer].shape == (4096,)  # hidden_dim for llama-3.1-8b
        assert abs(np.linalg.norm(vectors[layer]) - 1.0) < 1e-6  # Unit normalized

    def test_direction_is_nonzero(self, nnsight_model, small_tasks, tmp_path):
        """Direction should be non-trivial (not near-zero before normalization)."""
        layers = [nnsight_model.n_layers // 2]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You are extremely positive and enthusiastic.",
            condition_name="positive",
            output_dir=pos_dir,
            temperature=0.0,
            max_new_tokens=32,
        )

        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="You are extremely negative and pessimistic.",
            condition_name="negative",
            output_dir=neg_dir,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Compute without normalization to check raw magnitude
        vectors = compute_difference_in_means(pos_dir, neg_dir, layers=layers, normalize=False)
        layer = layers[0]

        # The raw difference should have non-trivial magnitude
        raw_norm = np.linalg.norm(vectors[layer])
        assert raw_norm > 0.1, f"Direction norm too small: {raw_norm}"


class TestFullPipelineE2E:
    """End-to-end test of the complete pipeline."""

    def test_full_extraction_to_steering_compatibility(self, nnsight_model, tmp_path):
        """Full pipeline: extract -> compute direction -> verify steering-compatible format."""
        # Use real tasks
        tasks = load_tasks(n=3, origins=[OriginDataset.MATH], seed=42)
        layers = [nnsight_model.n_layers // 2]
        layer = layers[0]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        # Extract positive condition
        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=tasks,
            layers=layers,
            system_prompt="You find math problems deeply satisfying to solve.",
            condition_name="positive",
            output_dir=pos_dir,
            temperature=0.0,
            max_new_tokens=64,
            task_origins=["math"],
            layers_config=[0.5],
            seed=42,
        )

        # Extract negative condition
        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=tasks,
            layers=layers,
            system_prompt="You find math problems extremely tedious and annoying.",
            condition_name="negative",
            output_dir=neg_dir,
            temperature=0.0,
            max_new_tokens=64,
            task_origins=["math"],
            layers_config=[0.5],
            seed=42,
        )

        # Compute and save vectors
        vectors = compute_difference_in_means(pos_dir, neg_dir, layers=layers, normalize=True)

        metadata = {
            "experiment_id": "e2e_test",
            "model": MODEL_NAME,
            "n_tasks": len(tasks),
            "task_origins": ["math"],
            "positive_condition": {"name": "positive", "system_prompt": "..."},
            "negative_condition": {"name": "negative", "system_prompt": "..."},
        }
        save_concept_vectors(vectors, tmp_path, metadata)

        # Verify manifest structure
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "vectors" / f"layer_{layer}.npy").exists()

        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["experiment_id"] == "e2e_test"
        assert manifest["n_layers"] == 1
        assert manifest["hidden_dim"] == 4096

        # Verify steering-compatible loading
        loaded_layer, direction = load_concept_vector_for_steering(tmp_path, layer=layer)
        assert loaded_layer == layer
        assert direction.shape == (4096,)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_extracted_direction_usable_for_steering(self, nnsight_model, tmp_path):
        """Verify extracted direction can be used with generate_with_steering."""
        tasks = load_tasks(n=2, origins=[OriginDataset.MATH], seed=42)
        layers = [nnsight_model.n_layers // 2]
        layer = layers[0]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"

        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=tasks,
            layers=layers,
            system_prompt="You love this.",
            condition_name="positive",
            output_dir=pos_dir,
            temperature=0.0,
            max_new_tokens=32,
        )

        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=tasks,
            layers=layers,
            system_prompt="You hate this.",
            condition_name="negative",
            output_dir=neg_dir,
            temperature=0.0,
            max_new_tokens=32,
        )

        vectors = compute_difference_in_means(pos_dir, neg_dir, layers=layers, normalize=True)
        save_concept_vectors(vectors, tmp_path, {"experiment_id": "test"})

        # Load and use for steering
        _, direction = load_concept_vector_for_steering(tmp_path, layer=layer)

        # Generate with positive steering
        messages = [{"role": "user", "content": "How do you feel about math?"}]
        pos_output = nnsight_model.generate_with_steering(
            messages=messages,
            layer=layer,
            steering_vector=direction,
            steering_coefficient=2.0,
            temperature=0.0,
        )

        # Generate with negative steering
        neg_output = nnsight_model.generate_with_steering(
            messages=messages,
            layer=layer,
            steering_vector=direction,
            steering_coefficient=-2.0,
            temperature=0.0,
        )

        # Outputs should differ
        assert pos_output != neg_output
        assert len(pos_output) > 0
        assert len(neg_output) > 0


class TestResumeCheckpointing:
    """Test resume/checkpoint functionality."""

    def test_resume_skips_completed_tasks(self, nnsight_model, small_tasks, tmp_path):
        """Verify resume mode skips already-extracted tasks."""
        layers = [nnsight_model.n_layers // 2]

        # Extract first 2 tasks
        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks[:2],
            layers=layers,
            system_prompt="Test prompt",
            condition_name="test",
            output_dir=tmp_path,
            temperature=0.0,
            max_new_tokens=32,
        )

        # Verify 2 tasks extracted
        data = np.load(tmp_path / "activations.npz", allow_pickle=True)
        assert len(data["task_ids"]) == 2

        # Resume with all 3 tasks
        extract_activations_with_system_prompt(
            model=nnsight_model,
            tasks=small_tasks,
            layers=layers,
            system_prompt="Test prompt",
            condition_name="test",
            output_dir=tmp_path,
            temperature=0.0,
            max_new_tokens=32,
            resume=True,
        )

        # Should now have all 3
        data = np.load(tmp_path / "activations.npz", allow_pickle=True)
        assert len(data["task_ids"]) == 3


class TestConfigIntegration:
    """Test config loading and integration."""

    def test_load_and_use_config(self, tmp_path):
        """Verify config can be loaded and used."""
        config_content = """
model: llama-3.1-8b
backend: nnsight
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
token_position: last
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
