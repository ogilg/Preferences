"""Tests for lightweight activation extraction."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.probes.extraction.simple import extract_activations
from src.task_data import Task, OriginDataset


def _make_tasks(n: int) -> list[Task]:
    return [
        Task(prompt=f"Do task {i}", origin=OriginDataset.WILDCHAT, id=f"t{i}", metadata={})
        for i in range(n)
    ]


def _mock_model(layers: list[int], selectors: list[str], d_model: int = 16):
    model = MagicMock()

    def fake_get_activations_batch(messages_batch, layers_, selectors_):
        batch_size = len(messages_batch)
        return {
            s: {layer: np.random.randn(batch_size, d_model).astype(np.float32) for layer in layers_}
            for s in selectors_
        }

    model.get_activations_batch = MagicMock(side_effect=fake_get_activations_batch)
    return model


class TestExtractActivations:
    def test_returns_correct_shape(self):
        tasks = _make_tasks(5)
        layers = [10, 20]
        selectors = ["prompt_last"]
        model = _mock_model(layers, selectors)

        result = extract_activations(model, tasks, layers, selectors, batch_size=3)

        assert set(result.keys()) == {"prompt_last"}
        assert set(result["prompt_last"].keys()) == {10, 20}
        for layer in layers:
            assert result["prompt_last"][layer].shape == (5, 16)

    def test_messages_have_user_role(self):
        tasks = _make_tasks(2)
        model = _mock_model([10], ["prompt_last"])

        extract_activations(model, tasks, [10], ["prompt_last"], batch_size=10)

        call_args = model.get_activations_batch.call_args_list[0]
        messages_batch = call_args[0][0]
        assert len(messages_batch) == 2
        assert messages_batch[0] == [{"role": "user", "content": "Do task 0"}]
        assert messages_batch[1] == [{"role": "user", "content": "Do task 1"}]

    def test_system_prompt_prepended(self):
        tasks = _make_tasks(1)
        model = _mock_model([10], ["prompt_last"])

        extract_activations(
            model, tasks, [10], ["prompt_last"], batch_size=10, system_prompt="Be helpful."
        )

        messages_batch = model.get_activations_batch.call_args_list[0][0][0]
        assert messages_batch[0] == [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Do task 0"},
        ]

    def test_multiple_selectors(self):
        tasks = _make_tasks(3)
        layers = [5]
        selectors = ["prompt_last", "mean"]
        model = _mock_model(layers, selectors)

        result = extract_activations(model, tasks, layers, selectors, batch_size=10)

        assert set(result.keys()) == {"prompt_last", "mean"}
        assert result["prompt_last"][5].shape == (3, 16)
        assert result["mean"][5].shape == (3, 16)

    def test_save_path_creates_npz(self, tmp_path):
        tasks = _make_tasks(4)
        layers = [10, 20]
        selectors = ["prompt_last"]
        model = _mock_model(layers, selectors)

        result = extract_activations(
            model, tasks, layers, selectors, batch_size=10, save_path=tmp_path
        )

        npz_file = tmp_path / "activations_prompt_last.npz"
        assert npz_file.exists()
        data = np.load(npz_file, allow_pickle=True)
        assert list(data["task_ids"]) == ["t0", "t1", "t2", "t3"]
        np.testing.assert_array_equal(data["layer_10"], result["prompt_last"][10])
        np.testing.assert_array_equal(data["layer_20"], result["prompt_last"][20])

    def test_batching_calls_model_multiple_times(self):
        tasks = _make_tasks(5)
        model = _mock_model([10], ["prompt_last"])

        extract_activations(model, tasks, [10], ["prompt_last"], batch_size=2)

        # 5 tasks, batch_size=2 → 3 calls (2+2+1)
        assert model.get_activations_batch.call_count == 3
