"""Unit tests for concept vector extraction and difference-in-means computation."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.concept_vectors.config import (
    ConceptVectorExtractionConfig,
    ConditionDict,
    load_config,
)
from src.concept_vectors.difference import (
    compute_difference_in_means,
    load_concept_vector,
    load_concept_vector_for_steering,
    save_concept_vectors,
)




class TestDifferenceInMeans:
    def test_compute_difference_basic(self, tmp_path):
        """Test basic difference-in-means computation."""
        n_tasks = 5
        hidden_dim = 32
        layer = 16

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"
        pos_dir.mkdir()
        neg_dir.mkdir()

        task_ids = np.array([f"task_{i}" for i in range(n_tasks)])

        # Create activations where positive has known offset
        pos_acts = np.ones((n_tasks, hidden_dim), dtype=np.float32)
        neg_acts = np.zeros((n_tasks, hidden_dim), dtype=np.float32)

        np.savez(pos_dir / "activations_last.npz", task_ids=task_ids, layer_16=pos_acts)
        np.savez(neg_dir / "activations_last.npz", task_ids=task_ids, layer_16=neg_acts)

        # Without normalization: should be all 1s
        vectors = compute_difference_in_means(pos_dir, neg_dir, selector_name="last", normalize=False)
        expected = np.ones(hidden_dim, dtype=np.float32)
        np.testing.assert_array_almost_equal(vectors[layer], expected)

        # With normalization: should be unit vector
        vectors_normalized = compute_difference_in_means(pos_dir, neg_dir, selector_name="last", normalize=True)
        norm = np.linalg.norm(vectors_normalized[layer])
        assert abs(norm - 1.0) < 1e-6

    def test_partial_task_overlap_uses_intersection(self, tmp_path):
        """Test that partial task overlap uses intersection."""
        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"
        pos_dir.mkdir()
        neg_dir.mkdir()

        hidden_dim = 64
        rng = np.random.default_rng(42)

        # Different task IDs with some overlap
        pos_task_ids = ["task_0", "task_1", "task_2"]
        neg_task_ids = ["task_0", "task_1", "task_3"]

        pos_acts = rng.standard_normal((3, hidden_dim)).astype(np.float32)
        neg_acts = rng.standard_normal((3, hidden_dim)).astype(np.float32)

        np.savez(pos_dir / "activations_last.npz", task_ids=np.array(pos_task_ids), layer_8=pos_acts)
        np.savez(neg_dir / "activations_last.npz", task_ids=np.array(neg_task_ids), layer_8=neg_acts)

        # Should use intersection (task_0, task_1)
        vectors = compute_difference_in_means(pos_dir, neg_dir, selector_name="last")
        assert 8 in vectors

    def test_multiple_layers(self, tmp_path):
        """Test computation across multiple layers."""
        n_tasks = 3
        hidden_dim = 16
        layers = [4, 8, 12]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"
        pos_dir.mkdir()
        neg_dir.mkdir()

        task_ids = np.array([f"task_{i}" for i in range(n_tasks)])
        rng = np.random.default_rng(42)

        pos_data = {"task_ids": task_ids}
        neg_data = {"task_ids": task_ids}

        for layer in layers:
            pos_data[f"layer_{layer}"] = rng.standard_normal((n_tasks, hidden_dim)).astype(
                np.float32
            )
            neg_data[f"layer_{layer}"] = rng.standard_normal((n_tasks, hidden_dim)).astype(
                np.float32
            )

        np.savez(pos_dir / "activations_last.npz", **pos_data)
        np.savez(neg_dir / "activations_last.npz", **neg_data)

        vectors = compute_difference_in_means(pos_dir, neg_dir, selector_name="last", layers=layers)

        assert set(vectors.keys()) == set(layers)
        for layer, vec in vectors.items():
            assert vec.shape == (hidden_dim,)
            assert vec.dtype == np.float32

    def test_layer_subset_selection(self, tmp_path):
        """Test selecting subset of available layers."""
        n_tasks = 3
        hidden_dim = 16
        all_layers = [4, 8, 12, 16]

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"
        pos_dir.mkdir()
        neg_dir.mkdir()

        task_ids = np.array([f"task_{i}" for i in range(n_tasks)])
        rng = np.random.default_rng(42)

        pos_data = {"task_ids": task_ids}
        neg_data = {"task_ids": task_ids}

        for layer in all_layers:
            pos_data[f"layer_{layer}"] = rng.standard_normal((n_tasks, hidden_dim)).astype(
                np.float32
            )
            neg_data[f"layer_{layer}"] = rng.standard_normal((n_tasks, hidden_dim)).astype(
                np.float32
            )

        np.savez(pos_dir / "activations_last.npz", **pos_data)
        np.savez(neg_dir / "activations_last.npz", **neg_data)

        # Only request subset of layers
        selected_layers = [8, 16]
        vectors = compute_difference_in_means(pos_dir, neg_dir, selector_name="last", layers=selected_layers)

        assert set(vectors.keys()) == set(selected_layers)

    def test_task_ordering_alignment(self, tmp_path):
        """Test that tasks are correctly aligned even with different orderings."""
        n_tasks = 4
        hidden_dim = 8
        layer = 8

        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"
        pos_dir.mkdir()
        neg_dir.mkdir()

        # Same task IDs but different order
        pos_task_ids = np.array(["task_a", "task_b", "task_c", "task_d"])
        neg_task_ids = np.array(["task_c", "task_a", "task_d", "task_b"])

        # Create distinct activations for each task
        pos_acts = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        # Reorder to match neg_task_ids order (c, a, d, b)
        neg_acts = np.array(
            [[3, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )

        np.savez(pos_dir / "activations_last.npz", task_ids=pos_task_ids, layer_8=pos_acts)
        np.savez(neg_dir / "activations_last.npz", task_ids=neg_task_ids, layer_8=neg_acts)

        # When aligned correctly, pos - neg should be 0 for all tasks
        vectors = compute_difference_in_means(pos_dir, neg_dir, selector_name="last", normalize=False)

        # pos_acts mean = [2.5, 0, ...], neg_acts (aligned) mean = [2.5, 0, ...]
        # Difference should be near zero
        np.testing.assert_array_almost_equal(vectors[layer], np.zeros(hidden_dim))


class TestSaveLoadConceptVectors:
    def test_save_and_load_vectors(self, tmp_path):
        """Test saving and loading concept vectors."""
        layers = [8, 16, 24]
        hidden_dim = 64

        vectors_by_selector = {}
        for selector in ["last", "first", "mean"]:
            vectors_by_selector[selector] = {
                layer: np.random.randn(hidden_dim).astype(np.float32) for layer in layers
            }
            for layer in layers:
                vectors_by_selector[selector][layer] /= np.linalg.norm(vectors_by_selector[selector][layer])

        metadata = {
            "experiment_id": "test_001",
            "model": "llama-3.1-8b",
            "n_tasks": 100,
        }

        save_concept_vectors(vectors_by_selector, tmp_path, metadata)

        # Check files exist
        assert (tmp_path / "manifest.json").exists()
        for selector in ["last", "first", "mean"]:
            for layer in layers:
                assert (tmp_path / "vectors" / selector / f"layer_{layer}.npy").exists()

        # Load and verify
        for layer in layers:
            loaded = load_concept_vector(tmp_path, layer, selector="last")
            np.testing.assert_array_almost_equal(loaded, vectors_by_selector["last"][layer])

    def test_load_nonexistent_layer_raises(self, tmp_path):
        """Test that loading non-existent layer raises ValueError."""
        vectors_by_selector = {"last": {8: np.random.randn(32).astype(np.float32)}}
        metadata = {"experiment_id": "test"}

        save_concept_vectors(vectors_by_selector, tmp_path, metadata)

        with pytest.raises(ValueError, match="Layer 16 not available"):
            load_concept_vector(tmp_path, 16)

    def test_load_for_steering_default_layer(self, tmp_path):
        """Test load_concept_vector_for_steering with default middle layer."""
        layers = [8, 16, 24]
        hidden_dim = 32

        vectors = {layer: np.random.randn(hidden_dim).astype(np.float32) for layer in layers}
        for layer in vectors:
            vectors[layer] /= np.linalg.norm(vectors[layer])

        vectors_by_selector = {"last": vectors}
        metadata = {"experiment_id": "test"}
        save_concept_vectors(vectors_by_selector, tmp_path, metadata)

        # Default should pick middle layer (16)
        layer, direction = load_concept_vector_for_steering(tmp_path)
        assert layer == 16
        np.testing.assert_array_almost_equal(direction, vectors[16])
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_load_for_steering_specific_layer(self, tmp_path):
        """Test load_concept_vector_for_steering with specific layer."""
        layers = [8, 16, 24]
        hidden_dim = 32

        vectors = {layer: np.random.randn(hidden_dim).astype(np.float32) for layer in layers}
        for layer in vectors:
            vectors[layer] /= np.linalg.norm(vectors[layer])

        vectors_by_selector = {"last": vectors}
        metadata = {"experiment_id": "test"}
        save_concept_vectors(vectors_by_selector, tmp_path, metadata)

        layer, direction = load_concept_vector_for_steering(tmp_path, layer=24)
        assert layer == 24
        np.testing.assert_array_almost_equal(direction, vectors[24])

    def test_manifest_metadata(self, tmp_path):
        """Test that manifest contains expected metadata."""
        vectors_by_selector = {"last": {8: np.random.randn(64).astype(np.float32)}}
        metadata = {
            "experiment_id": "test_exp",
            "model": "llama-3.1-8b",
            "n_tasks": 500,
            "positive_condition": {"name": "positive", "system_prompt": "You love this."},
            "negative_condition": {"name": "negative", "system_prompt": "You hate this."},
        }

        save_concept_vectors(vectors_by_selector, tmp_path, metadata)

        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["experiment_id"] == "test_exp"
        assert manifest["model"] == "llama-3.1-8b"
        assert manifest["n_tasks"] == 500
        assert manifest["n_layers"] == 1
        assert manifest["layers"] == [8]
        assert manifest["hidden_dim"] == 64
        assert manifest["selectors"] == ["last"]
        assert "created_at" in manifest
        assert "vector_files" in manifest


class TestConfig:
    def test_load_config(self, tmp_path):
        """Test loading config from YAML."""
        config_content = """
model: llama-3.1-8b
backend: nnsight
n_tasks: 100
task_origins:
  - math
task_sampling_seed: 42
conditions:
  positive:
    name: positive
    system_prompt: "You love this."
  negative:
    name: negative
    system_prompt: "You hate this."
layers_to_extract:
  - 0.25
  - 0.5
  - 0.75
token_position: last
temperature: 1.0
max_new_tokens: 512
output_dir: results/test
experiment_id: test_001
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        config = load_config(config_path)

        assert config.model == "llama-3.1-8b"
        assert config.n_tasks == 100
        assert config.task_origins == ["math"]
        assert config.task_sampling_seed == 42
        assert "positive" in config.conditions
        assert "negative" in config.conditions
        assert config.conditions["positive"]["system_prompt"] == "You love this."
        assert config.layers_to_extract == [0.25, 0.5, 0.75]
        assert config.output_dir == Path("results/test")
        assert config.experiment_id == "test_001"

    def test_config_validation(self, tmp_path):
        """Test that config validation catches missing fields."""
        config_content = """
model: llama-3.1-8b
n_tasks: 100
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        with pytest.raises(Exception):  # Pydantic ValidationError
            load_config(config_path)
