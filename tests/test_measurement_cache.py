from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.preferences.storage.cache import MeasurementCache, reconstruct_measurements
from src.preferences.templates import PromptTemplate, BINARY_PLACEHOLDERS
from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType


class MockClient:
    """Mock that matches inference client interface for testing."""

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.canonical_model_name = model_name


@pytest.fixture
def sample_template() -> PromptTemplate:
    return PromptTemplate(
        template="Choose: {task_a} or {task_b}? {format_instruction}",
        name="binary_choice_v1",
        required_placeholders=BINARY_PLACEHOLDERS,
        tags=frozenset(["phrasing:1", "task_a_label:A", "task_b_label:B"]),
    )


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        Task(prompt="Write a poem", origin=OriginDataset.WILDCHAT, id="task_1", metadata={}),
        Task(prompt="Solve an equation", origin=OriginDataset.WILDCHAT, id="task_2", metadata={}),
        Task(prompt="Explain gravity", origin=OriginDataset.WILDCHAT, id="task_3", metadata={}),
    ]


@pytest.fixture
def sample_measurements(sample_tasks: list[Task]) -> list[BinaryPreferenceMeasurement]:
    return [
        BinaryPreferenceMeasurement(
            task_a=sample_tasks[0],
            task_b=sample_tasks[1],
            choice="a",
            preference_type=PreferenceType.PRE_TASK_STATED,
        ),
        BinaryPreferenceMeasurement(
            task_a=sample_tasks[1],
            task_b=sample_tasks[2],
            choice="b",
            preference_type=PreferenceType.PRE_TASK_STATED,
        ),
    ]


class TestMeasurementCacheUnit:
    """Unit tests for MeasurementCache - tests individual methods."""

    def test_cache_dir_naming_and_get_existing_pairs_on_empty(
        self, tmp_path: Path, sample_template
    ):
        """Test cache directory is named correctly and empty cache returns empty set."""
        model = MockClient()
        cache = MeasurementCache(sample_template, model, results_dir=tmp_path)

        # Directory naming uses template name, shortened model name, response format, and order
        assert cache.cache_dir == tmp_path / "binary_choice_v1_llama-3.1-8b_regex_canonical"

        # Empty cache returns empty set
        existing = cache.get_existing_pairs()
        assert existing == set()

    def test_append_creates_config_and_measurements_files(
        self, tmp_path: Path, sample_template, sample_measurements
    ):
        """Test append creates both config and measurements files on first write."""
        model = MockClient()
        cache = MeasurementCache(sample_template, model, results_dir=tmp_path)

        cache.append(sample_measurements)

        # Verify both files exist
        assert cache._config_path.exists()
        assert cache._measurements_path.exists()

        # Verify config content
        with open(cache._config_path) as f:
            config = yaml.safe_load(f)
        assert config["template_name"] == "binary_choice_v1"
        assert config["model"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert config["model_short"] == "llama-3.1-8b"

        # Verify measurements content
        with open(cache._measurements_path) as f:
            data = yaml.safe_load(f)
        assert len(data) == 2
        assert data[0] == {"task_a": "task_1", "task_b": "task_2", "choice": "a"}
        assert data[1] == {"task_a": "task_2", "task_b": "task_3", "choice": "b"}

    def test_append_empty_list_does_nothing(self, tmp_path: Path, sample_template):
        """Appending empty list should not create any files."""
        model = MockClient()
        cache = MeasurementCache(sample_template, model, results_dir=tmp_path)

        cache.append([])

        assert not cache._config_path.exists()
        assert not cache._measurements_path.exists()

    def test_get_measurements_filters_by_task_ids(
        self, tmp_path: Path, sample_template, sample_measurements
    ):
        """Test filtering measurements by task IDs."""
        model = MockClient()
        cache = MeasurementCache(sample_template, model, results_dir=tmp_path)
        cache.append(sample_measurements)

        # Get all measurements
        all_data = cache.get_measurements()
        assert len(all_data) == 2

        # Filter to only pairs involving task_1 and task_2
        filtered = cache.get_measurements(task_ids={"task_1", "task_2"})
        assert len(filtered) == 1
        assert filtered[0]["task_a"] == "task_1"
        assert filtered[0]["task_b"] == "task_2"

        # Filter with task set that excludes all pairs
        empty_result = cache.get_measurements(task_ids={"task_1"})
        assert len(empty_result) == 0


class TestReconstructMeasurements:
    """Tests for the reconstruct_measurements helper function."""

    def test_reconstructs_measurements_from_raw_dicts(self, sample_tasks):
        """Test that raw dicts are reconstructed into proper measurement objects."""
        tasks_dict = {t.id: t for t in sample_tasks}
        raw = [
            {"task_a": "task_1", "task_b": "task_2", "choice": "a"},
            {"task_a": "task_2", "task_b": "task_3", "choice": "b"},
        ]

        measurements = reconstruct_measurements(raw, tasks_dict)

        assert len(measurements) == 2
        assert measurements[0].task_a.id == "task_1"
        assert measurements[0].task_b.id == "task_2"
        assert measurements[0].choice == "a"
        assert measurements[0].preference_type == PreferenceType.PRE_TASK_STATED

        assert measurements[1].task_a.id == "task_2"
        assert measurements[1].choice == "b"

    def test_uses_provided_preference_type(self, sample_tasks):
        """Test that the provided preference_type is used."""
        tasks_dict = {t.id: t for t in sample_tasks}
        raw = [{"task_a": "task_1", "task_b": "task_2", "choice": "a"}]

        measurements = reconstruct_measurements(
            raw, tasks_dict, preference_type=PreferenceType.POST_TASK_STATED
        )

        assert measurements[0].preference_type == PreferenceType.POST_TASK_STATED


class TestMeasurementCacheIntegration:
    """Integration tests - test the full workflow as would be used in practice."""

    def test_incremental_append_accumulates_and_tracks_pairs(
        self, tmp_path: Path, sample_template, sample_tasks
    ):
        """Test that multiple appends accumulate measurements and get_existing_pairs tracks them.

        This tests the core use case: collecting measurements in batches over time.
        """
        model = MockClient()
        cache = MeasurementCache(sample_template, model, results_dir=tmp_path)

        # First batch
        batch1 = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache.append(batch1)

        assert cache.get_existing_pairs() == {("task_1", "task_2")}
        assert len(cache.get_measurements()) == 1

        # Second batch - different pair
        batch2 = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[1], task_b=sample_tasks[2],
                choice="b", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache.append(batch2)

        # Both pairs now tracked
        assert cache.get_existing_pairs() == {("task_1", "task_2"), ("task_2", "task_3")}
        assert len(cache.get_measurements()) == 2

        # Third batch - reversed order of first pair (should be distinct)
        batch3 = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[1], task_b=sample_tasks[0],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache.append(batch3)

        # Now have 3 pairs, (1,2) and (2,1) are distinct
        existing = cache.get_existing_pairs()
        assert ("task_1", "task_2") in existing
        assert ("task_2", "task_1") in existing
        assert len(existing) == 3
        assert len(cache.get_measurements()) == 3

    def test_cache_persists_across_instances(
        self, tmp_path: Path, sample_template, sample_tasks
    ):
        """Test that measurements persist when creating a new cache instance.

        Simulates resuming a measurement session.
        """
        model = MockClient()

        # First session - append some measurements
        cache1 = MeasurementCache(sample_template, model, results_dir=tmp_path)
        measurements = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[2],
                choice="b", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache1.append(measurements)

        # Second session - new instance, same parameters
        cache2 = MeasurementCache(sample_template, model, results_dir=tmp_path)

        # Should see the same data
        assert cache2.get_existing_pairs() == {("task_1", "task_2"), ("task_1", "task_3")}
        data = cache2.get_measurements()
        assert len(data) == 2
        assert data[0]["choice"] == "a"
        assert data[1]["choice"] == "b"

    def test_roundtrip_with_reconstruct(
        self, tmp_path: Path, sample_template, sample_tasks
    ):
        """Test full round-trip: append -> persist -> load -> reconstruct.

        Verifies that measurements can be fully reconstructed with Task objects.
        """
        model = MockClient()
        cache = MeasurementCache(sample_template, model, results_dir=tmp_path)

        original = [
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[1], task_b=sample_tasks[2],
                choice="b", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ]
        cache.append(original)

        # Load raw and reconstruct
        raw = cache.get_measurements()
        tasks_dict = {t.id: t for t in sample_tasks}
        reconstructed = reconstruct_measurements(raw, tasks_dict)

        # Verify structural equivalence
        assert len(reconstructed) == 2
        assert reconstructed[0].task_a.prompt == sample_tasks[0].prompt
        assert reconstructed[0].task_b.prompt == sample_tasks[1].prompt
        assert reconstructed[0].choice == "a"
        assert reconstructed[1].task_a.prompt == sample_tasks[1].prompt
        assert reconstructed[1].choice == "b"

    def test_different_template_model_pairs_isolated(
        self, tmp_path: Path, sample_tasks
    ):
        """Test that different template/model combinations use separate storage.

        Each (template, model) pair should have its own cache directory.
        """
        template1 = PromptTemplate(
            template="Choose: {task_a} or {task_b}? {format_instruction}",
            name="template_one",
            required_placeholders=BINARY_PLACEHOLDERS,
        )
        template2 = PromptTemplate(
            template="Pick: {task_a} or {task_b}? {format_instruction}",
            name="template_two",
            required_placeholders=BINARY_PLACEHOLDERS,
        )
        model1 = MockClient("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model2 = MockClient("meta-llama/Meta-Llama-3.1-70B-Instruct")

        cache1 = MeasurementCache(template1, model1, results_dir=tmp_path)
        cache2 = MeasurementCache(template2, model1, results_dir=tmp_path)
        cache3 = MeasurementCache(template1, model2, results_dir=tmp_path)

        # Each has unique directory
        assert cache1.cache_dir != cache2.cache_dir
        assert cache1.cache_dir != cache3.cache_dir
        assert cache2.cache_dir != cache3.cache_dir

        # Write to cache1 only
        cache1.append([
            BinaryPreferenceMeasurement(
                task_a=sample_tasks[0], task_b=sample_tasks[1],
                choice="a", preference_type=PreferenceType.PRE_TASK_STATED,
            ),
        ])

        # Others should be empty
        assert len(cache1.get_measurements()) == 1
        assert len(cache2.get_measurements()) == 0
        assert len(cache3.get_measurements()) == 0
