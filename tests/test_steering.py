"""Tests for steering library primitives."""

import gc
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.models.base import (
    position_selective_steering,
    differential_steering,
    noop_steering,
)
from src.steering.tokenization import find_text_span, find_pairwise_task_spans
from src.steering.calibration import suggest_coefficient_range
from src.probes.core.storage import load_probe_direction


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    yield
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer where each whitespace-separated word is one token."""
    tokenizer = MagicMock()

    def tokenize_with_offsets(text, return_offsets_mapping=False, add_special_tokens=True):
        words = text.split(" ")
        offsets = []
        pos = 0
        for w in words:
            start = text.find(w, pos)
            offsets.append((start, start + len(w)))
            pos = start + len(w)
        result = {"input_ids": list(range(len(words)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    tokenizer.side_effect = tokenize_with_offsets
    tokenizer.__call__ = tokenize_with_offsets
    return tokenizer


class TestPositionSelectiveSteering:
    def test_steers_correct_positions(self):
        steering_tensor = torch.ones(8)
        hook = position_selective_steering(steering_tensor, start=2, end=5)

        resid = torch.zeros(1, 10, 8)
        result = hook(resid, prompt_len=10)

        assert torch.all(result[0, 2:5, :] == 1.0)
        assert torch.all(result[0, :2, :] == 0.0)
        assert torch.all(result[0, 5:, :] == 0.0)

    def test_no_op_during_autoregressive(self):
        steering_tensor = torch.ones(8)
        hook = position_selective_steering(steering_tensor, start=0, end=1)

        resid = torch.zeros(1, 1, 8)
        result = hook(resid, prompt_len=10)
        assert torch.all(result == 0.0)


class TestDifferentialSteering:
    def test_positive_and_negative_spans(self):
        steering_tensor = torch.ones(8) * 2.0
        hook = differential_steering(steering_tensor, pos_start=0, pos_end=3, neg_start=5, neg_end=8)

        resid = torch.zeros(1, 10, 8)
        result = hook(resid, prompt_len=10)

        assert torch.all(result[0, 0:3, :] == 2.0)
        assert torch.all(result[0, 5:8, :] == -2.0)
        assert torch.all(result[0, 3:5, :] == 0.0)
        assert torch.all(result[0, 8:, :] == 0.0)

    def test_no_op_during_autoregressive(self):
        steering_tensor = torch.ones(8)
        hook = differential_steering(steering_tensor, 0, 1, 2, 3)

        resid = torch.zeros(1, 1, 8)
        result = hook(resid, prompt_len=10)
        assert torch.all(result == 0.0)


class TestNoopSteering:
    def test_returns_unchanged(self):
        hook = noop_steering()
        resid = torch.randn(1, 10, 8)
        original = resid.clone()
        result = hook(resid, prompt_len=10)
        assert torch.all(result == original)


class TestFindTextSpan:
    def test_finds_span(self, mock_tokenizer):
        text = "Hello world foo bar baz"
        start, end = find_text_span(mock_tokenizer, text, "foo bar")
        assert start == 2
        assert end == 4

    def test_search_after(self, mock_tokenizer):
        text = "foo bar foo bar baz"
        start, end = find_text_span(mock_tokenizer, text, "foo bar", search_after=4)
        assert start == 2
        assert end == 4

    def test_not_found_raises(self, mock_tokenizer):
        with pytest.raises(ValueError, match="not found"):
            find_text_span(mock_tokenizer, "Hello world", "missing")


class TestFindPairwiseTaskSpans:
    def test_finds_both_spans(self, mock_tokenizer):
        prompt = "Task A Write code Task B Fix bugs"
        a_span, b_span = find_pairwise_task_spans(
            mock_tokenizer, prompt, "Write code", "Fix bugs"
        )
        assert a_span == (2, 4)
        assert b_span == (6, 8)

    def test_missing_marker_raises(self, mock_tokenizer):
        with pytest.raises(ValueError, match="Marker"):
            find_pairwise_task_spans(
                mock_tokenizer, "no markers here", "text_a", "text_b"
            )


class TestSuggestCoefficientRange:
    def test_returns_scaled_coefficients(self):
        with (
            patch("src.steering.calibration.load_probe_direction") as mock_load,
            patch("src.steering.calibration.compute_activation_norms") as mock_norms,
        ):
            mock_load.return_value = (31, np.ones(3072))
            mock_norms.return_value = {31: 1000.0}

            coeffs = suggest_coefficient_range(
                Path("fake/activations.npz"),
                Path("fake/manifest"),
                "ridge_L31",
                multipliers=[-0.1, 0.0, 0.1],
            )

            assert coeffs == [-100.0, 0.0, 100.0]
            mock_norms.assert_called_once_with(Path("fake/activations.npz"), layers=[31])

    def test_default_multipliers(self):
        with (
            patch("src.steering.calibration.load_probe_direction") as mock_load,
            patch("src.steering.calibration.compute_activation_norms") as mock_norms,
        ):
            mock_load.return_value = (16, np.ones(4096))
            mock_norms.return_value = {16: 500.0}

            coeffs = suggest_coefficient_range(
                Path("fake.npz"), Path("fake"), "probe_0004"
            )

            assert len(coeffs) == 5
            assert coeffs[2] == 0.0


class TestSteeredHFClientDirection:
    def test_direction_property(self):
        from src.steering.client import SteeredHFClient

        mock_model = MagicMock()
        mock_model.device = "cpu"
        direction = np.random.randn(128).astype(np.float32)

        client = SteeredHFClient(mock_model, layer=16, steering_direction=direction, coefficient=1.0)
        assert np.array_equal(client.direction, direction)

    def test_generate_with_hook(self):
        from src.steering.client import SteeredHFClient

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate_with_steering.return_value = "steered output"

        direction = np.random.randn(128).astype(np.float32)
        client = SteeredHFClient(mock_model, layer=16, steering_direction=direction, coefficient=0.0)

        custom_hook = noop_steering()
        result = client.generate_with_hook([{"role": "user", "content": "hi"}], custom_hook)

        assert result == "steered output"
        mock_model.generate_with_steering.assert_called_once()
        call_kwargs = mock_model.generate_with_steering.call_args
        assert call_kwargs.kwargs["layer"] == 16


class TestLoadProbeDirection:
    def test_load_probe_direction_real_data(self):
        manifest_dir = Path("probe_data/manifests/probe_4_all_datasets")
        if not manifest_dir.exists():
            pytest.skip("Probe data not available")

        layer, direction = load_probe_direction(manifest_dir, "0004")

        assert layer == 16
        assert direction.shape == (4096,)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-6

    def test_load_probe_direction_invalid_id_raises(self):
        manifest_dir = Path("probe_data/manifests/probe_4_all_datasets")
        if not manifest_dir.exists():
            pytest.skip("Probe data not available")

        with pytest.raises(ValueError, match="not found in manifest"):
            load_probe_direction(manifest_dir, "9999")


class TestAnalysisFunctions:
    def test_aggregate_by_coefficient(self):
        from src.steering.analysis import aggregate_by_coefficient

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
        from src.steering.analysis import compute_statistics

        by_coef = {
            -1.0: [-0.5, -0.3, -0.4],
            0.0: [0.0, 0.1, 0.05],
            1.0: [0.5, 0.4, 0.45],
        }

        stats = compute_statistics(by_coef)

        assert stats["coefficients"] == [-1.0, 0.0, 1.0]
        assert len(stats["means"]) == 3
        assert stats["means"][0] < stats["means"][1] < stats["means"][2]
        assert stats["cohens_d"] > 0
        assert stats["regression_slope"] > 0

    def test_compute_statistics_with_varied_n(self):
        from src.steering.analysis import compute_statistics

        by_coef = {
            -1.0: [-0.5, -0.3],
            0.0: [0.0, 0.1, 0.05, 0.02],
            1.0: [0.5],
        }

        stats = compute_statistics(by_coef)
        assert stats["n_per_condition"] == [2, 4, 1]


class TestAnalysisPlotting:
    def test_plot_dose_response_creates_file(self):
        from src.steering.analysis import plot_dose_response

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
