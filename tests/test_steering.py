"""Tests for steering validation experiment infrastructure."""

import gc
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

from src.steering.config import SteeringExperimentConfig, load_steering_config
from src.probes.storage import load_probe_direction


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache before and after each test."""
    yield
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class TestSteeringConfig:
    """Test SteeringExperimentConfig validation and loading."""

    def test_config_default_values(self):
        """Verify default config values are sensible."""
        config = SteeringExperimentConfig(
            model="llama-3.1-8b",
            probe_manifest_dir=Path("probe_data/manifests/test"),
            experiment_id="test_001",
        )
        assert config.probe_id == "0004"
        assert config.steering_coefficients == [-2.0, -1.0, 0.0, 1.0, 2.0]
        assert config.n_tasks == 25
        assert 0.0 in config.steering_coefficients  # Baseline must be included

    def test_config_custom_values(self):
        """Verify custom values are correctly set."""
        config = SteeringExperimentConfig(
            model="llama-3.1-8b",
            probe_manifest_dir=Path("custom/path"),
            probe_id="0001",
            steering_coefficients=[-1.0, 0.0, 1.0],
            n_tasks=10,
            task_origins=["math"],
            rating_seeds=[0, 1],
            experiment_id="custom_exp",
        )
        assert config.probe_manifest_dir == Path("custom/path")
        assert config.probe_id == "0001"
        assert config.steering_coefficients == [-1.0, 0.0, 1.0]
        assert config.n_tasks == 10
        assert config.task_origins == ["math"]
        assert config.rating_seeds == [0, 1]

    def test_config_loads_from_yaml(self):
        """Verify config loads correctly from YAML file."""
        yaml_content = """
model: llama-3.1-8b
probe_manifest_dir: probe_data/manifests/probe_4_all_datasets
probe_id: "0004"
steering_coefficients: [-2.0, 0.0, 2.0]
n_tasks: 5
task_origins:
  - wildchat
experiment_id: yaml_test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_steering_config(Path(f.name))

        assert config.probe_manifest_dir == Path("probe_data/manifests/probe_4_all_datasets")
        assert config.probe_id == "0004"
        assert config.steering_coefficients == [-2.0, 0.0, 2.0]
        assert config.n_tasks == 5
        assert config.task_origins == ["wildchat"]
        assert config.experiment_id == "yaml_test"

    def test_config_serializes_to_json(self):
        """Verify config can be serialized for storage in results."""
        config = SteeringExperimentConfig(
            model="llama-3.1-8b",
            probe_manifest_dir=Path("test/path"),
            experiment_id="test",
        )
        serialized = config.model_dump(mode="json")

        assert isinstance(serialized, dict)
        assert serialized["probe_manifest_dir"] == "test/path"
        assert serialized["experiment_id"] == "test"
        assert isinstance(serialized["steering_coefficients"], list)


class TestLoadProbeDirection:
    """Test probe direction loading for steering."""

    def test_load_probe_direction_real_data(self):
        """Verify probe direction loads correctly from actual probe data."""
        manifest_dir = Path("probe_data/manifests/probe_4_all_datasets")
        if not manifest_dir.exists():
            pytest.skip("Probe data not available")

        layer, direction = load_probe_direction(manifest_dir, "0004")

        assert layer == 16
        assert direction.shape == (4096,)
        # Should be unit normalized
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6
        assert direction.dtype == np.float32 or direction.dtype == np.float64

    def test_load_probe_direction_different_probes(self):
        """Verify different probes load with correct metadata."""
        manifest_dir = Path("probe_data/manifests/probe_4_all_datasets")
        if not manifest_dir.exists():
            pytest.skip("Probe data not available")

        # Load probe 0001 (wildchat only)
        layer1, dir1 = load_probe_direction(manifest_dir, "0001")
        # Load probe 0004 (all datasets, English template)
        layer2, dir2 = load_probe_direction(manifest_dir, "0004")

        # Both should be at same layer
        assert layer1 == layer2 == 16
        # But directions should differ
        assert not np.allclose(dir1, dir2)

    def test_load_probe_direction_invalid_id_raises(self):
        """Verify invalid probe ID raises ValueError."""
        manifest_dir = Path("probe_data/manifests/probe_4_all_datasets")
        if not manifest_dir.exists():
            pytest.skip("Probe data not available")

        with pytest.raises(ValueError, match="not found in manifest"):
            load_probe_direction(manifest_dir, "9999")

    def test_direction_is_unit_normalized(self):
        """Verify returned direction is always unit normalized."""
        manifest_dir = Path("probe_data/manifests/probe_4_all_datasets")
        if not manifest_dir.exists():
            pytest.skip("Probe data not available")

        for probe_id in ["0001", "0002", "0004"]:
            _, direction = load_probe_direction(manifest_dir, probe_id)
            norm = np.linalg.norm(direction)
            assert abs(norm - 1.0) < 1e-6, f"Probe {probe_id} direction not unit normalized"


class TestAnalysisFunctions:
    """Test steering analysis functions."""

    def test_aggregate_by_coefficient(self):
        """Verify aggregation groups scores correctly."""
        from src.analysis.steering.analyze_steering_experiment import aggregate_by_coefficient

        mock_results = {
            "results": [
                {
                    "task_id": "task_1",
                    "conditions": [
                        {"steering_coefficient": -1.0, "parsed_value": -1.0},
                        {"steering_coefficient": 0.0, "parsed_value": 1.0},
                        {"steering_coefficient": 1.0, "parsed_value": 1.0},
                    ],
                },
                {
                    "task_id": "task_2",
                    "conditions": [
                        {"steering_coefficient": -1.0, "parsed_value": -1.0},
                        {"steering_coefficient": 0.0, "parsed_value": -1.0},
                        {"steering_coefficient": 1.0, "parsed_value": 1.0},
                    ],
                },
            ],
        }

        by_coef = aggregate_by_coefficient(mock_results)

        assert set(by_coef.keys()) == {-1.0, 0.0, 1.0}
        assert by_coef[-1.0] == [-1.0, -1.0]
        assert by_coef[0.0] == [1.0, -1.0]
        assert by_coef[1.0] == [1.0, 1.0]

    def test_compute_statistics(self):
        """Verify statistics are computed correctly."""
        from src.analysis.steering.analyze_steering_experiment import compute_statistics

        by_coef = {
            -1.0: [-0.5, -0.3, -0.4],
            0.0: [0.0, 0.1, 0.05],
            1.0: [0.5, 0.4, 0.45],
        }

        stats = compute_statistics(by_coef)

        assert stats["coefficients"] == [-1.0, 0.0, 1.0]
        assert len(stats["means"]) == 3
        assert len(stats["stds"]) == 3
        assert len(stats["sems"]) == 3
        assert stats["n_per_condition"] == [3, 3, 3]

        # Verify monotonic increase in means
        assert stats["means"][0] < stats["means"][1] < stats["means"][2]

        # Effect size should be positive (positive > negative)
        assert stats["cohens_d"] > 0

        # Regression slope should be positive
        assert stats["regression_slope"] > 0

    def test_compute_statistics_with_varied_n(self):
        """Verify statistics handle different n per condition."""
        from src.analysis.steering.analyze_steering_experiment import compute_statistics

        by_coef = {
            -1.0: [-0.5, -0.3],
            0.0: [0.0, 0.1, 0.05, 0.02],
            1.0: [0.5],
        }

        stats = compute_statistics(by_coef)

        assert stats["n_per_condition"] == [2, 4, 1]


class TestAnalysisPlotting:
    """Test plotting functions (without actually displaying plots)."""

    def test_plot_dose_response_creates_file(self):
        """Verify plot function creates output file."""
        from src.analysis.steering.analyze_steering_experiment import plot_dose_response

        by_coef = {
            -1.0: [-0.5, -0.3],
            0.0: [0.0, 0.1],
            1.0: [0.5, 0.4],
        }
        stats = {
            "coefficients": [-1.0, 0.0, 1.0],
            "means": [-0.4, 0.05, 0.45],
            "stds": [0.1, 0.05, 0.05],
            "sems": [0.07, 0.035, 0.035],
            "cohens_d": 1.5,
            "regression_slope": 0.425,
            "regression_intercept": 0.05,
            "regression_r2": 0.95,
            "regression_p_value": 0.001,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_plot.png"
            plot_dose_response(by_coef, stats, output_path)
            assert output_path.exists()


class TestResultsSerialization:
    """Test results can be saved and loaded correctly."""

    def test_results_json_structure(self):
        """Verify results JSON has correct structure."""
        from src.steering.runner import TaskSteeringResults, SteeringConditionResult

        task_result = TaskSteeringResults(
            task_id="test_task",
            task_origin="WILDCHAT",
            completion="Test completion",
            conditions=[
                SteeringConditionResult(
                    steering_coefficient=-1.0,
                    rating_seed=0,
                    measurement_prompt="Was this good or bad?",
                    preference_expression="bad",
                    parsed_value=-1.0,
                ),
                SteeringConditionResult(
                    steering_coefficient=1.0,
                    rating_seed=0,
                    measurement_prompt="Was this good or bad?",
                    preference_expression="good",
                    parsed_value=1.0,
                ),
            ],
        )

        # Verify serialization matches expected format
        serialized = {
            "task_id": task_result.task_id,
            "task_origin": task_result.task_origin,
            "completion": task_result.completion,
            "conditions": [
                {
                    "steering_coefficient": c.steering_coefficient,
                    "rating_seed": c.rating_seed,
                    "measurement_prompt": c.measurement_prompt,
                    "preference_expression": c.preference_expression,
                    "parsed_value": c.parsed_value,
                }
                for c in task_result.conditions
            ],
        }

        # Should be JSON serializable
        json_str = json.dumps(serialized)
        loaded = json.loads(json_str)

        assert loaded["task_id"] == "test_task"
        assert len(loaded["conditions"]) == 2
        assert loaded["conditions"][0]["steering_coefficient"] == -1.0
        assert loaded["conditions"][0]["parsed_value"] == -1.0


class TestBuildRatingPrompt:
    """Test rating prompt construction."""

    def test_build_rating_prompt_structure(self):
        """Verify rating prompt has correct three-turn structure."""
        from src.steering.runner import _build_rating_prompt
        from src.measurement.elicitation.response_format import RegexQualitativeFormat, BINARY_QUALITATIVE_VALUES, BINARY_QUALITATIVE_TO_NUMERIC
        from src.task_data import Task, OriginDataset

        task = Task(
            prompt="Write a haiku",
            origin=OriginDataset.WILDCHAT,
            id="test_task",
            metadata={},
        )
        completion = "Cherry blossoms fall\nSoft petals on still water\nSpring whispers goodbye"
        response_format = RegexQualitativeFormat(
            values=BINARY_QUALITATIVE_VALUES,
            value_to_score=BINARY_QUALITATIVE_TO_NUMERIC,
        )

        messages, measurement_prompt = _build_rating_prompt(task, completion, "001", response_format)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Write a haiku"
        assert messages[1]["role"] == "assistant"
        assert "Cherry blossoms" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        # Should contain template text about good/bad experience
        content_lower = messages[2]["content"].lower()
        assert "good" in content_lower and "bad" in content_lower
        assert measurement_prompt == messages[2]["content"]

    def test_build_rating_prompt_invalid_id_raises(self):
        """Verify invalid template id raises ValueError."""
        from src.steering.runner import _build_rating_prompt
        from src.measurement.elicitation.response_format import RegexQualitativeFormat, BINARY_QUALITATIVE_VALUES, BINARY_QUALITATIVE_TO_NUMERIC
        from src.task_data import Task, OriginDataset

        task = Task(prompt="test", origin=OriginDataset.WILDCHAT, id="t", metadata={})
        response_format = RegexQualitativeFormat(
            values=BINARY_QUALITATIVE_VALUES,
            value_to_score=BINARY_QUALITATIVE_TO_NUMERIC,
        )

        with pytest.raises(ValueError, match="No template found"):
            _build_rating_prompt(task, "completion", "999", response_format)


@pytest.mark.gpu
@pytest.mark.slow
class TestSteeringExperimentE2E:
    """End-to-end test for steering experiment.

    Runs full experiment with small config. Run with: pytest -m "gpu and slow"
    """

    def test_full_experiment_pipeline(self):
        """Run full steering experiment with minimal config."""
        from src.steering.runner import run_steering_experiment
        from src.steering.config import SteeringExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = SteeringExperimentConfig(
                model="llama-3.1-8b",
                probe_manifest_dir=Path("probe_data/manifests/probe_4_all_datasets"),
                probe_id="0004",
                steering_coefficients=[-1.0, 0.0, 1.0],
                n_tasks=2,
                task_origins=["wildchat"],
                rating_seeds=[0],
                experiment_id="e2e_test",
                output_dir=Path(tmpdir),
                completions_path=Path("results/completions/llama-3.1-8b_seed0/completions.json"),
            )

            results = run_steering_experiment(config)

            # Verify results structure
            assert "config" in results
            assert "results" in results
            assert len(results["results"]) == 2  # n_tasks
            for task_result in results["results"]:
                assert len(task_result["conditions"]) == 3  # 3 coefficients * 1 seed

    def test_analysis_on_experiment_results(self):
        """Run analysis on experiment results."""
        from src.analysis.steering.analyze_steering_experiment import analyze_steering_experiment

        rng = np.random.default_rng(42)
        # Create mock results file
        mock_results = {
            "config": {
                "experiment_id": "test",
                "model": "llama-3.1-8b",
                "probe_id": "0004",
            },
            "results": [
                {
                    "task_id": f"task_{i}",
                    "task_origin": "WILDCHAT",
                    "completion": "test",
                    "conditions": [
                        {"steering_coefficient": -1.0, "rating_seed": 0, "measurement_prompt": "...", "preference_expression": "bad", "parsed_value": -1.0 if rng.random() > 0.3 else 1.0},
                        {"steering_coefficient": 0.0, "rating_seed": 0, "measurement_prompt": "...", "preference_expression": "good", "parsed_value": 1.0 if rng.random() > 0.5 else -1.0},
                        {"steering_coefficient": 1.0, "rating_seed": 0, "measurement_prompt": "...", "preference_expression": "good", "parsed_value": 1.0 if rng.random() > 0.3 else -1.0},
                    ],
                }
                for i in range(10)
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            results_path = tmpdir / "steering_results.json"
            with open(results_path, "w") as f:
                json.dump(mock_results, f)

            stats = analyze_steering_experiment(tmpdir)

            # Verify analysis outputs
            assert stats["regression_slope"] > 0  # Positive slope expected
            assert (tmpdir / "steering_statistics.json").exists()
            # Plot should be created with today's date
            plot_files = list(tmpdir.glob("plot_*.png"))
            assert len(plot_files) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
