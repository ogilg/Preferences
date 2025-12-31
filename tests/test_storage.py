from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.task_data import Task, OriginDataset
from src.types import BinaryPreferenceMeasurement, PreferenceType
from src.preferences.templates import PromptTemplate, BINARY_PLACEHOLDERS
from src.preferences.ranking import PairwiseData, fit_thurstonian
from src.preferences.storage import (
    save_measurements,
    save_run,
    load_run,
    list_runs,
    update_index,
)


class MockModel:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name


@pytest.fixture
def sample_tasks() -> list[Task]:
    return [
        Task(prompt="Task 1", origin=OriginDataset.WILDCHAT, id="task_1", metadata={}),
        Task(prompt="Task 2", origin=OriginDataset.WILDCHAT, id="task_2", metadata={}),
        Task(prompt="Task 3", origin=OriginDataset.WILDCHAT, id="task_3", metadata={}),
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
            task_a=sample_tasks[0],
            task_b=sample_tasks[2],
            choice="a",
            preference_type=PreferenceType.PRE_TASK_STATED,
        ),
        BinaryPreferenceMeasurement(
            task_a=sample_tasks[1],
            task_b=sample_tasks[2],
            choice="a",
            preference_type=PreferenceType.PRE_TASK_STATED,
        ),
    ]


@pytest.fixture
def sample_template() -> PromptTemplate:
    return PromptTemplate(
        template="Choose: {task_a} or {task_b}? {format_instruction}",
        name="binary_choice_001",
        required_placeholders=BINARY_PLACEHOLDERS,
        tags=frozenset(["phrasing:1", "language:en"]),
    )


class TestSaveMeasurements:
    def test_serializes_to_expected_format(self, tmp_path: Path, sample_measurements):
        path = tmp_path / "measurements.yaml"
        save_measurements(sample_measurements, path)

        with open(path) as f:
            data = yaml.safe_load(f)

        assert len(data) == 3
        assert data[0] == {"task_a": "task_1", "task_b": "task_2", "choice": "a"}
        assert data[1] == {"task_a": "task_1", "task_b": "task_3", "choice": "a"}


class TestSaveAndLoadRun:
    def test_round_trip_preserves_data(
        self, tmp_path: Path, sample_template, sample_tasks, sample_measurements
    ):
        """Save a run and load it back, verify all data is preserved."""
        model = MockModel()
        thurstonian = fit_thurstonian(
            PairwiseData.from_comparisons(sample_measurements, sample_tasks)
        )

        run_path = save_run(
            template=sample_template,
            template_file="test_templates.yaml",
            model=model,
            temperature=1.0,
            tasks=sample_tasks,
            measurements=sample_measurements,
            thurstonian=thurstonian,
            results_dir=tmp_path,
        )

        # Verify files created
        assert (run_path / "config.yaml").exists()
        assert (run_path / "measurements.yaml").exists()
        assert (run_path / "thurstonian.yaml").exists()

        # Load and verify config
        loaded = load_run(run_path, tasks=sample_tasks)

        assert loaded.config.template_id == "001"
        assert loaded.config.template_name == "binary_choice_001"
        assert loaded.config.model == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert loaded.config.model_short == "llama-3.1-8b"
        assert loaded.config.temperature == 1.0
        assert loaded.config.n_tasks == 3
        assert loaded.config.template_tags["phrasing"] == "1"
        assert loaded.config.template_tags["language"] == "en"

        # Verify measurements
        assert len(loaded.measurements) == 3
        assert loaded.measurements[0]["choice"] == "a"

        # Verify thurstonian
        assert loaded.thurstonian is not None
        assert loaded.thurstonian.converged == thurstonian.converged


class TestListRuns:
    def test_filters_by_template_id(
        self, tmp_path: Path, sample_tasks, sample_measurements
    ):
        model = MockModel()
        thurstonian = fit_thurstonian(
            PairwiseData.from_comparisons(sample_measurements, sample_tasks)
        )

        template1 = PromptTemplate(
            template="Choose: {task_a} or {task_b}? {format_instruction}",
            name="test_001",
            required_placeholders=BINARY_PLACEHOLDERS,
            tags=frozenset(["phrasing:1"]),
        )
        template2 = PromptTemplate(
            template="Pick: {task_a} or {task_b}? {format_instruction}",
            name="test_002",
            required_placeholders=BINARY_PLACEHOLDERS,
            tags=frozenset(["phrasing:2"]),
        )

        save_run(template1, "t.yaml", model, 1.0, sample_tasks, sample_measurements, thurstonian, tmp_path)
        save_run(template2, "t.yaml", model, 1.0, sample_tasks, sample_measurements, thurstonian, tmp_path)

        # Filter by template_id
        runs = list_runs(tmp_path, template_id="001")
        assert len(runs) == 1
        assert runs[0].template_id == "001"

        # Get all
        all_runs = list_runs(tmp_path)
        assert len(all_runs) == 2

    def test_filters_by_template_tags(
        self, tmp_path: Path, sample_tasks, sample_measurements
    ):
        model = MockModel()
        thurstonian = fit_thurstonian(
            PairwiseData.from_comparisons(sample_measurements, sample_tasks)
        )

        template1 = PromptTemplate(
            template="Choose: {task_a} or {task_b}? {format_instruction}",
            name="test_001",
            required_placeholders=BINARY_PLACEHOLDERS,
            tags=frozenset(["phrasing:1", "language:en"]),
        )
        template2 = PromptTemplate(
            template="Pick: {task_a} or {task_b}? {format_instruction}",
            name="test_002",
            required_placeholders=BINARY_PLACEHOLDERS,
            tags=frozenset(["phrasing:2", "language:fr"]),
        )

        save_run(template1, "t.yaml", model, 1.0, sample_tasks, sample_measurements, thurstonian, tmp_path)
        save_run(template2, "t.yaml", model, 1.0, sample_tasks, sample_measurements, thurstonian, tmp_path)

        runs = list_runs(tmp_path, language="en")
        assert len(runs) == 1
        assert runs[0].template_tags["language"] == "en"


class TestUpdateIndex:
    def test_regenerates_index_from_dirs(
        self, tmp_path: Path, sample_template, sample_tasks, sample_measurements
    ):
        model = MockModel()
        thurstonian = fit_thurstonian(
            PairwiseData.from_comparisons(sample_measurements, sample_tasks)
        )

        save_run(
            template=sample_template,
            template_file="test_templates.yaml",
            model=model,
            temperature=1.0,
            tasks=sample_tasks,
            measurements=sample_measurements,
            thurstonian=thurstonian,
            results_dir=tmp_path,
        )

        # Delete index and verify list_runs fails
        (tmp_path / "index.yaml").unlink()
        assert list_runs(tmp_path) == []

        # Regenerate and verify
        update_index(tmp_path)

        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0].template_id == "001"
