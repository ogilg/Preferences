"""Tests for measure_dataset_preferences.py - dataset-wide preference measurement."""

import pytest
from unittest.mock import Mock

from src.task_data import Task, OriginDataset
from src.types import PreferencePrompt, MeasurementResponse
from src.preferences.config import DatasetMeasurementConfig, PairingStrategy
from src.preferences.measure_dataset_preferences import (
    measure_dataset_preferences,
    _sample_measurement,
    _generate_pairs,
)
from src.models.base import ConfigurableMockModel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def task_a():
    return Task(
        prompt="Write a haiku about spring.",
        origin=OriginDataset.WILDCHAT,
        id="task_a",
        metadata={"topic": "creative"},
    )


@pytest.fixture
def task_b():
    return Task(
        prompt="Solve x^2 = 4",
        origin=OriginDataset.MATH,
        id="task_b",
        metadata={"topic": "math"},
    )


@pytest.fixture
def task_c():
    return Task(
        prompt="Explain photosynthesis.",
        origin=OriginDataset.ALPACA,
        id="task_c",
        metadata={"topic": "science"},
    )


@pytest.fixture
def tasks(task_a, task_b, task_c):
    return [task_a, task_b, task_c]


@pytest.fixture
def mock_model():
    return ConfigurableMockModel(response="a")


@pytest.fixture
def mock_measurer():
    """Create a mock measurer that returns MeasurementResponse objects."""
    measurer = Mock()
    measurer.parse.side_effect = lambda text, prompt: MeasurementResponse(
        text=text,
        source_prompt=prompt,
        result=Mock(),
    )
    return measurer


@pytest.fixture
def mock_prompt():
    """Create a mock PreferencePrompt."""
    prompt = Mock(spec=PreferencePrompt)
    prompt.messages = [{"role": "user", "content": "test"}]
    return prompt


@pytest.fixture
def mock_rating_builder(mock_measurer, mock_prompt):
    """Create a mock rating builder."""
    builder = Mock()
    mock_prompt.measurer = mock_measurer
    builder.build.return_value = mock_prompt
    return builder


@pytest.fixture
def mock_binary_builder(mock_measurer, mock_prompt):
    """Create a mock binary builder."""
    builder = Mock()
    mock_prompt.measurer = mock_measurer
    builder.build.return_value = mock_prompt
    return builder


# =============================================================================
# Tests for _generate_pairs
# =============================================================================


class TestGeneratePairs:
    """Tests for task pairing strategies."""

    def test_all_pairs_strategy(self, tasks):
        """ALL_PAIRS should generate n(n-1)/2 unique pairs."""
        config = DatasetMeasurementConfig(pairing_strategy=PairingStrategy.ALL_PAIRS)

        pairs = _generate_pairs(tasks, config)

        # 3 tasks -> 3 pairs: (a,b), (a,c), (b,c)
        assert len(pairs) == 3
        # Check all tasks appear in at least one pair
        all_tasks_in_pairs = [t for pair in pairs for t in pair]
        for task in tasks:
            assert task in all_tasks_in_pairs

    def test_adjacent_pairs_strategy(self, tasks):
        """ADJACENT_PAIRS should generate n-1 pairs of consecutive tasks."""
        config = DatasetMeasurementConfig(pairing_strategy=PairingStrategy.ADJACENT_PAIRS)

        pairs = _generate_pairs(tasks, config)

        # 3 tasks -> 2 pairs: (a,b), (b,c)
        assert len(pairs) == 2
        assert pairs[0] == (tasks[0], tasks[1])
        assert pairs[1] == (tasks[1], tasks[2])

    def test_random_pairs_strategy_samples_subset(self, tasks):
        """RANDOM_PAIRS with max_pairs should sample at most max_pairs pairs."""
        config = DatasetMeasurementConfig(
            pairing_strategy=PairingStrategy.RANDOM_PAIRS,
            max_pairs=2,
            seed=42,
        )

        pairs = _generate_pairs(tasks, config)

        assert len(pairs) == 2

    def test_random_pairs_without_max_returns_all(self, tasks):
        """RANDOM_PAIRS without max_pairs should return all pairs."""
        config = DatasetMeasurementConfig(
            pairing_strategy=PairingStrategy.RANDOM_PAIRS,
            max_pairs=None,
            seed=42,
        )

        pairs = _generate_pairs(tasks, config)

        # Should return all 3 pairs since max_pairs is None
        assert len(pairs) == 3

    def test_random_pairs_with_seed_is_reproducible(self, tasks):
        """Same seed should produce same pairs."""
        config = DatasetMeasurementConfig(
            pairing_strategy=PairingStrategy.RANDOM_PAIRS,
            max_pairs=2,
            seed=123,
        )

        pairs1 = _generate_pairs(tasks, config)
        pairs2 = _generate_pairs(tasks, config)

        assert pairs1 == pairs2

    def test_max_pairs_exceeds_available_pairs(self, task_a, task_b):
        """max_pairs larger than available pairs should return all pairs."""
        tasks = [task_a, task_b]  # Only 1 possible pair
        config = DatasetMeasurementConfig(
            pairing_strategy=PairingStrategy.RANDOM_PAIRS,
            max_pairs=100,
            seed=42,
        )

        pairs = _generate_pairs(tasks, config)

        assert len(pairs) == 1  # Only (a,b) is possible

    def test_empty_task_list(self):
        """Empty task list should return empty pairs."""
        config = DatasetMeasurementConfig(pairing_strategy=PairingStrategy.ALL_PAIRS)

        pairs = _generate_pairs([], config)

        assert pairs == []

    def test_single_task_list(self, task_a):
        """Single task should return empty pairs."""
        config = DatasetMeasurementConfig(pairing_strategy=PairingStrategy.ALL_PAIRS)

        pairs = _generate_pairs([task_a], config)

        assert pairs == []


# =============================================================================
# Tests for _sample_measurement
# =============================================================================


class TestSampleMeasurement:
    """Tests for individual measurement sampling."""

    def test_runs_correct_number_of_samples(self, mock_model, mock_rating_builder):
        """Should run num_samples iterations."""
        config = DatasetMeasurementConfig(num_samples=3)

        samples = _sample_measurement(
            model=mock_model,
            builder=mock_rating_builder,
            config=config,
            args=(Mock(),),
        )

        assert len(samples) == 3
        assert mock_rating_builder.build.call_count == 3

    def test_sample_index_increments(self, mock_model, mock_rating_builder):
        """Each sample should have incrementing sample_index."""
        config = DatasetMeasurementConfig(num_samples=4)

        samples = _sample_measurement(
            model=mock_model,
            builder=mock_rating_builder,
            config=config,
            args=(Mock(),),
        )

        indices = [s["sample_index"] for s in samples]
        assert indices == [0, 1, 2, 3]

    def test_uses_configured_temperature(self, mock_rating_builder):
        """Should pass configured temperature to model.generate."""
        model = ConfigurableMockModel()
        config = DatasetMeasurementConfig(num_samples=1, temperature=0.7)

        _sample_measurement(
            model=model,
            builder=mock_rating_builder,
            config=config,
            args=(Mock(),),
        )

        assert model.last_temperature == 0.7

    def test_passes_args_to_builder(self, mock_model, mock_rating_builder, task_a, task_b):
        """Args should be passed to builder.build()."""
        config = DatasetMeasurementConfig(num_samples=1)

        _sample_measurement(
            model=mock_model,
            builder=mock_rating_builder,
            config=config,
            args=(task_a, task_b),
        )

        mock_rating_builder.build.assert_called_once_with(task_a, task_b)


# =============================================================================
# Tests for measure_dataset_preferences
# =============================================================================


class TestMeasureDatasetPreferences:
    """Integration tests for measure_dataset_preferences."""

    def test_returns_config_in_result(self, mock_model, mock_rating_builder, tasks):
        """Result should include the config used."""
        config = DatasetMeasurementConfig(
            measurement_types=frozenset({"rating"}),
            temperature=0.8,
        )

        result = measure_dataset_preferences(
            model=mock_model,
            tasks=tasks,
            rating_builder=mock_rating_builder,
            config=config,
        )

        assert result["config"] is config

    def test_uses_default_config_when_none_provided(
        self, mock_model, mock_rating_builder, mock_binary_builder, tasks
    ):
        """Should use default DatasetMeasurementConfig when config is None."""
        result = measure_dataset_preferences(
            model=mock_model,
            tasks=tasks,
            rating_builder=mock_rating_builder,
            binary_builder=mock_binary_builder,
            config=None,
        )

        assert isinstance(result["config"], DatasetMeasurementConfig)

    def test_runs_only_rating_when_configured(
        self, mock_model, mock_rating_builder, mock_binary_builder, tasks
    ):
        """Should only run rating measurements when only 'rating' in measurement_types."""
        config = DatasetMeasurementConfig(measurement_types=frozenset({"rating"}))

        result = measure_dataset_preferences(
            model=mock_model,
            tasks=tasks,
            rating_builder=mock_rating_builder,
            binary_builder=mock_binary_builder,
            config=config,
        )

        assert len(result["task_ratings"]) == 3
        assert result["binary_comparisons"] == []

    def test_runs_only_binary_when_configured(
        self, mock_model, mock_rating_builder, mock_binary_builder, tasks
    ):
        """Should only run binary measurements when only 'binary' in measurement_types."""
        config = DatasetMeasurementConfig(
            measurement_types=frozenset({"binary"}),
            pairing_strategy=PairingStrategy.ALL_PAIRS,
        )

        result = measure_dataset_preferences(
            model=mock_model,
            tasks=tasks,
            rating_builder=mock_rating_builder,
            binary_builder=mock_binary_builder,
            config=config,
        )

        assert result["task_ratings"] == []
        assert len(result["binary_comparisons"]) == 3  # 3 tasks -> 3 pairs

    def test_runs_both_when_configured(
        self, mock_model, mock_rating_builder, mock_binary_builder, tasks
    ):
        """Should run both measurement types when both in measurement_types."""
        config = DatasetMeasurementConfig(
            measurement_types=frozenset({"rating", "binary"}),
        )

        result = measure_dataset_preferences(
            model=mock_model,
            tasks=tasks,
            rating_builder=mock_rating_builder,
            binary_builder=mock_binary_builder,
            config=config,
        )

        assert len(result["task_ratings"]) == 3
        assert len(result["binary_comparisons"]) == 3

    def test_raises_when_rating_builder_missing(self, mock_model, mock_binary_builder, tasks):
        """Should raise ValueError when rating requested but builder missing."""
        config = DatasetMeasurementConfig(measurement_types=frozenset({"rating"}))

        with pytest.raises(ValueError, match="rating_builder required"):
            measure_dataset_preferences(
                model=mock_model,
                tasks=tasks,
                rating_builder=None,
                binary_builder=mock_binary_builder,
                config=config,
            )

    def test_raises_when_binary_builder_missing(self, mock_model, mock_rating_builder, tasks):
        """Should raise ValueError when binary requested but builder missing."""
        config = DatasetMeasurementConfig(measurement_types=frozenset({"binary"}))

        with pytest.raises(ValueError, match="binary_builder required"):
            measure_dataset_preferences(
                model=mock_model,
                tasks=tasks,
                rating_builder=mock_rating_builder,
                binary_builder=None,
                config=config,
            )

    def test_respects_pairing_strategy(
        self, mock_model, mock_binary_builder, tasks
    ):
        """Pairing strategy should affect number of binary comparisons."""
        config_all = DatasetMeasurementConfig(
            measurement_types=frozenset({"binary"}),
            pairing_strategy=PairingStrategy.ALL_PAIRS,
        )
        config_adjacent = DatasetMeasurementConfig(
            measurement_types=frozenset({"binary"}),
            pairing_strategy=PairingStrategy.ADJACENT_PAIRS,
        )

        result_all = measure_dataset_preferences(
            model=mock_model,
            tasks=tasks,
            binary_builder=mock_binary_builder,
            config=config_all,
        )
        result_adjacent = measure_dataset_preferences(
            model=mock_model,
            tasks=tasks,
            binary_builder=mock_binary_builder,
            config=config_adjacent,
        )

        assert len(result_all["binary_comparisons"]) == 3  # n(n-1)/2
        assert len(result_adjacent["binary_comparisons"]) == 2  # n-1

    def test_empty_measurement_types_returns_empty_results(self, mock_model, tasks):
        """Empty measurement_types should return empty result lists."""
        config = DatasetMeasurementConfig(measurement_types=frozenset())

        result = measure_dataset_preferences(
            model=mock_model,
            tasks=tasks,
            config=config,
        )

        assert result["task_ratings"] == []
        assert result["binary_comparisons"] == []

    def test_num_samples_affects_all_measurements(
        self, mock_model, mock_rating_builder, mock_binary_builder, task_a, task_b
    ):
        """num_samples should control samples per task and per pair."""
        config = DatasetMeasurementConfig(
            measurement_types=frozenset({"rating", "binary"}),
            num_samples=7,
        )

        result = measure_dataset_preferences(
            model=mock_model,
            tasks=[task_a, task_b],
            rating_builder=mock_rating_builder,
            binary_builder=mock_binary_builder,
            config=config,
        )

        # Check rating samples
        for task_result in result["task_ratings"]:
            assert len(task_result["samples"]) == 7

        # Check binary samples
        for comparison in result["binary_comparisons"]:
            assert len(comparison["samples"]) == 7
