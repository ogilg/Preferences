"""End-to-end tests for post-task measurement pipeline.

Tests the full workflow: completion generation -> storage -> measurement.
Uses a mock client to avoid API calls.

Run with:
    pytest tests/test_post_task_pipeline.py -v
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.task_data import Task, OriginDataset
from src.measurement_storage.completions import (
    CompletionStore,
    TaskCompletion,
    generate_completions,
    load_completions,
    completions_exist,
)
from src.measurement_storage import save_yaml, load_yaml
from src.preference_measurement import (
    measure_post_task_stated,
    measure_post_task_revealed,
    StatedScoreMeasurer,
    RevealedPreferenceMeasurer,
    RATING_FORMATS,
    CHOICE_FORMATS,
)
from src.prompt_templates import (
    PostTaskStatedPromptBuilder,
    PostTaskRevealedPromptBuilder,
    POST_TASK_STATED_TEMPLATE,
    POST_TASK_REVEALED_TEMPLATE,
    post_task_stated_template,
)
from src.types import TaskScore, BinaryPreferenceMeasurement, PreferenceType
from src.models.openai_compatible import BatchResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tasks():
    return [
        Task(prompt="What is 2 + 2?", origin=OriginDataset.MATH, id="math_1", metadata={}),
        Task(prompt="Write a haiku about rain.", origin=OriginDataset.WILDCHAT, id="creative_1", metadata={}),
        Task(prompt="Explain gravity.", origin=OriginDataset.WILDCHAT, id="science_1", metadata={}),
    ]


@pytest.fixture
def task_lookup(sample_tasks):
    return {t.id: t for t in sample_tasks}


@pytest.fixture
def temp_results_dir():
    """Temporary directory for test results."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_client():
    """Mock client that returns predictable completions and ratings."""
    client = MagicMock()
    client.canonical_model_name = "test-model"
    client.model_name = "test-model"
    return client


# =============================================================================
# CompletionStore Tests
# =============================================================================


class TestCompletionStore:
    """Test completion storage and retrieval."""

    def test_store_and_load_completions(self, sample_tasks, task_lookup, temp_results_dir, mock_client):
        """Should store completions and load them back."""
        with patch("src.measurement_storage.completions.COMPLETIONS_DIR", temp_results_dir):
            store = CompletionStore(client=mock_client, seed=42)

            completions = [
                TaskCompletion(task=sample_tasks[0], completion="The answer is 4."),
                TaskCompletion(task=sample_tasks[1], completion="Rain falls gently down\nPuddles form on the wet ground\nNature's soft refrain"),
            ]

            store.save(completions, config={"model": "test-model", "seed": 42})

            assert store.exists()
            assert store.get_existing_task_ids() == {"math_1", "creative_1"}

            loaded = store.load(task_lookup)
            assert len(loaded) == 2
            assert loaded[0].task.id == "math_1"
            assert loaded[0].completion == "The answer is 4."

    def test_incremental_save(self, sample_tasks, task_lookup, temp_results_dir, mock_client):
        """Should append new completions without duplicating existing ones."""
        with patch("src.measurement_storage.completions.COMPLETIONS_DIR", temp_results_dir):
            store = CompletionStore(client=mock_client, seed=42)

            # First save
            completions1 = [TaskCompletion(task=sample_tasks[0], completion="Answer 1")]
            store.save(completions1, config={"model": "test-model", "seed": 42})

            # Second save with overlap
            completions2 = [
                TaskCompletion(task=sample_tasks[0], completion="Should be ignored"),
                TaskCompletion(task=sample_tasks[1], completion="Answer 2"),
            ]
            store.save(completions2, config={"model": "test-model", "seed": 42})

            loaded = store.load(task_lookup)
            assert len(loaded) == 2
            # Original should be kept, not replaced
            assert loaded[0].completion == "Answer 1"
            assert loaded[1].completion == "Answer 2"

    def test_load_without_task_lookup(self, sample_tasks, temp_results_dir, mock_client):
        """Should reconstruct Task objects from stored data when no lookup provided."""
        with patch("src.measurement_storage.completions.COMPLETIONS_DIR", temp_results_dir):
            store = CompletionStore(client=mock_client, seed=42)

            completions = [TaskCompletion(task=sample_tasks[0], completion="Test")]
            store.save(completions, config={"model": "test-model", "seed": 42})

            loaded = store.load(task_lookup=None)
            assert len(loaded) == 1
            assert loaded[0].task.id == "math_1"
            assert loaded[0].task.prompt == "What is 2 + 2?"
            assert loaded[0].task.origin == OriginDataset.MATH


class TestGenerateCompletions:
    """Test completion generation with mock client."""

    def test_generates_completions_for_all_tasks(self, sample_tasks, mock_client):
        """Should generate completions for all tasks and return successful ones."""
        mock_client.generate_batch.return_value = [
            BatchResult(response="Answer 1", error=None),
            BatchResult(response="Answer 2", error=None),
            BatchResult(response="Answer 3", error=None),
        ]

        completions = generate_completions(
            client=mock_client,
            tasks=sample_tasks,
            temperature=1.0,
            max_concurrent=10,
            seed=42,
        )

        assert len(completions) == 3
        assert completions[0].task.id == "math_1"
        assert completions[0].completion == "Answer 1"

    def test_handles_failed_requests(self, sample_tasks, mock_client):
        """Should skip failed requests and return only successful completions."""
        mock_client.generate_batch.return_value = [
            BatchResult(response="Answer 1", error=None),
            BatchResult(response=None, error=Exception("API error")),
            BatchResult(response="Answer 3", error=None),
        ]

        completions = generate_completions(
            client=mock_client,
            tasks=sample_tasks,
            temperature=1.0,
            max_concurrent=10,
            seed=42,
        )

        assert len(completions) == 2
        assert completions[0].task.id == "math_1"
        assert completions[1].task.id == "science_1"


# =============================================================================
# Post-Task Stated Measurement Tests
# =============================================================================


class TestPostTaskStatedMeasurement:
    """Test post-task stated preference measurement."""

    def test_measures_post_task_ratings(self, sample_tasks, mock_client):
        """Should measure ratings after task completion."""
        # Mock returns valid rating responses
        mock_client.generate_batch.return_value = [
            BatchResult(response="I rate this task 7 out of 10.", error=None),
            BatchResult(response="Rating: 8", error=None),
        ]

        data = [
            (sample_tasks[0], "The answer is 4."),
            (sample_tasks[1], "Rain haiku here."),
        ]

        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS["regex"](1, 10),
            template=POST_TASK_STATED_TEMPLATE,
        )

        batch = measure_post_task_stated(
            client=mock_client,
            data=data,
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
            seed=42,
        )

        assert len(batch.successes) == 2
        assert isinstance(batch.successes[0], TaskScore)
        assert batch.successes[0].score == 7.0
        assert batch.successes[1].score == 8.0
        assert batch.successes[0].preference_type == PreferenceType.POST_TASK_STATED

    def test_prompt_structure_is_multi_turn(self, sample_tasks):
        """Post-task prompt should have 3 messages: user, assistant, user."""
        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS["regex"](1, 10),
            template=POST_TASK_STATED_TEMPLATE,
        )

        prompt = builder.build(sample_tasks[0], "The answer is 4.")

        assert len(prompt.messages) == 3
        assert prompt.messages[0]["role"] == "user"
        assert prompt.messages[1]["role"] == "assistant"
        assert prompt.messages[2]["role"] == "user"
        assert sample_tasks[0].prompt in prompt.messages[0]["content"]
        assert "The answer is 4." in prompt.messages[1]["content"]


# =============================================================================
# Post-Task Revealed Measurement Tests
# =============================================================================


class TestPostTaskRevealedMeasurement:
    """Test post-task revealed preference measurement."""

    def test_measures_post_task_binary_choice(self, sample_tasks, mock_client):
        """Should measure binary preference after completing both tasks."""
        mock_client.generate_batch.return_value = [
            BatchResult(response="I preferred Task A.", error=None),
        ]

        data = [
            (sample_tasks[0], sample_tasks[1], "Answer to math", "Answer to creative"),
        ]

        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CHOICE_FORMATS["regex"]("Task A", "Task B"),
            template=POST_TASK_REVEALED_TEMPLATE,
        )

        batch = measure_post_task_revealed(
            client=mock_client,
            data=data,
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
            seed=42,
        )

        assert len(batch.successes) == 1
        assert isinstance(batch.successes[0], BinaryPreferenceMeasurement)
        assert batch.successes[0].choice == "a"
        assert batch.successes[0].preference_type == PreferenceType.POST_TASK_REVEALED

    def test_prompt_structure_is_five_turn(self, sample_tasks):
        """Post-task revealed prompt should have 5 messages."""
        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CHOICE_FORMATS["regex"]("Task A", "Task B"),
            template=POST_TASK_REVEALED_TEMPLATE,
        )

        prompt = builder.build(
            sample_tasks[0], sample_tasks[1],
            "Completion A", "Completion B"
        )

        assert len(prompt.messages) == 5
        assert prompt.messages[0]["role"] == "user"
        assert prompt.messages[1]["role"] == "assistant"
        assert prompt.messages[2]["role"] == "user"
        assert prompt.messages[3]["role"] == "assistant"
        assert prompt.messages[4]["role"] == "user"


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================


class TestFullPostTaskPipeline:
    """End-to-end tests for the complete post-task measurement workflow."""

    def test_completion_to_stated_measurement_pipeline(
        self, sample_tasks, task_lookup, temp_results_dir, mock_client
    ):
        """Full pipeline: generate completions -> store -> load -> measure stated."""
        with patch("src.measurement_storage.completions.COMPLETIONS_DIR", temp_results_dir):
            # Step 1: Generate completions
            mock_client.generate_batch.return_value = [
                BatchResult(response="4", error=None),
                BatchResult(response="Haiku here", error=None),
                BatchResult(response="Gravity pulls", error=None),
            ]

            completions = generate_completions(
                client=mock_client, tasks=sample_tasks,
                temperature=1.0, max_concurrent=10, seed=42,
            )

            # Step 2: Store completions
            store = CompletionStore(client=mock_client, seed=42)
            store.save(completions, config={"model": "test-model", "seed": 42})

            # Step 3: Load completions
            loaded = store.load(task_lookup)
            assert len(loaded) == 3

            # Step 4: Measure stated preferences
            mock_client.generate_batch.return_value = [
                BatchResult(response="Rating: 7", error=None),
                BatchResult(response="I give it an 8", error=None),
                BatchResult(response="9 out of 10", error=None),
            ]

            data = [(tc.task, tc.completion) for tc in loaded]
            builder = PostTaskStatedPromptBuilder(
                measurer=StatedScoreMeasurer(),
                response_format=RATING_FORMATS["regex"](1, 10),
                template=POST_TASK_STATED_TEMPLATE,
            )

            batch = measure_post_task_stated(
                client=mock_client, data=data, builder=builder,
                temperature=1.0, max_concurrent=10, seed=42,
            )

            assert len(batch.successes) == 3
            scores = [s.score for s in batch.successes]
            assert scores == [7.0, 8.0, 9.0]

    def test_completion_to_revealed_measurement_pipeline(
        self, sample_tasks, task_lookup, temp_results_dir, mock_client
    ):
        """Full pipeline: generate completions -> store -> load -> measure revealed."""
        with patch("src.measurement_storage.completions.COMPLETIONS_DIR", temp_results_dir):
            # Step 1: Generate completions
            mock_client.generate_batch.return_value = [
                BatchResult(response="Completion A", error=None),
                BatchResult(response="Completion B", error=None),
                BatchResult(response="Completion C", error=None),
            ]

            completions = generate_completions(
                client=mock_client, tasks=sample_tasks,
                temperature=1.0, max_concurrent=10, seed=42,
            )

            # Step 2: Store completions
            store = CompletionStore(client=mock_client, seed=42)
            store.save(completions, config={"model": "test-model", "seed": 42})

            # Step 3: Load and build completion lookup
            loaded = store.load(task_lookup)
            completion_lookup = {tc.task.id: tc.completion for tc in loaded}

            # Step 4: Generate pairs and measure revealed preferences
            pairs = [
                (sample_tasks[0], sample_tasks[1]),
                (sample_tasks[0], sample_tasks[2]),
                (sample_tasks[1], sample_tasks[2]),
            ]

            mock_client.generate_batch.return_value = [
                BatchResult(response="Task A was better", error=None),
                BatchResult(response="I prefer Task B", error=None),
                BatchResult(response="Task A", error=None),
            ]

            data = [
                (a, b, completion_lookup[a.id], completion_lookup[b.id])
                for a, b in pairs
            ]

            builder = PostTaskRevealedPromptBuilder(
                measurer=RevealedPreferenceMeasurer(),
                response_format=CHOICE_FORMATS["regex"]("Task A", "Task B"),
                template=POST_TASK_REVEALED_TEMPLATE,
            )

            batch = measure_post_task_revealed(
                client=mock_client, data=data, builder=builder,
                temperature=1.0, max_concurrent=10, seed=42,
            )

            assert len(batch.successes) == 3
            choices = [m.choice for m in batch.successes]
            assert choices == ["a", "b", "a"]

    def test_multiple_seeds_stored_separately(
        self, sample_tasks, task_lookup, temp_results_dir, mock_client
    ):
        """Different seeds should create separate storage directories."""
        with patch("src.measurement_storage.completions.COMPLETIONS_DIR", temp_results_dir):
            mock_client.generate_batch.return_value = [
                BatchResult(response="Seed 0 completion", error=None),
            ]

            # Store with seed 0
            store0 = CompletionStore(client=mock_client, seed=0)
            store0.save(
                [TaskCompletion(task=sample_tasks[0], completion="Seed 0")],
                config={"seed": 0}
            )

            # Store with seed 1
            store1 = CompletionStore(client=mock_client, seed=1)
            store1.save(
                [TaskCompletion(task=sample_tasks[0], completion="Seed 1")],
                config={"seed": 1}
            )

            # Verify separate storage
            assert store0.store_dir != store1.store_dir
            assert store0.load()[0].completion == "Seed 0"
            assert store1.load()[0].completion == "Seed 1"


# =============================================================================
# Response Format Tests
# =============================================================================


class TestPostTaskResponseFormats:
    """Test different response formats work with post-task measurements."""

    @pytest.mark.parametrize("format_name", ["regex", "xml"])
    def test_stated_rating_formats(self, sample_tasks, mock_client, format_name):
        """Should parse ratings from different formats."""
        responses = {
            "regex": "I rate this 7 out of 10",
            "xml": "<rating>7</rating>",
        }

        mock_client.generate_batch.return_value = [
            BatchResult(response=responses[format_name], error=None),
        ]

        # Use a simple template without scale placeholders
        simple_template = post_task_stated_template(
            name="test_template",
            template="Rate that task.\n{format_instruction}",
        )

        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=RATING_FORMATS[format_name](scale_min=1, scale_max=10),
            template=simple_template,
        )

        batch = measure_post_task_stated(
            client=mock_client,
            data=[(sample_tasks[0], "Test completion")],
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
        )

        assert len(batch.successes) == 1
        assert batch.successes[0].score == 7.0

    @pytest.mark.parametrize("format_name", ["regex", "xml"])
    def test_revealed_choice_formats(self, sample_tasks, mock_client, format_name):
        """Should parse choices from different formats."""
        responses = {
            "regex": "I prefer Task A",
            "xml": "<choice>Task A</choice>",
        }

        mock_client.generate_batch.return_value = [
            BatchResult(response=responses[format_name], error=None),
        ]

        builder = PostTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=CHOICE_FORMATS[format_name]("Task A", "Task B"),
            template=POST_TASK_REVEALED_TEMPLATE,
        )

        batch = measure_post_task_revealed(
            client=mock_client,
            data=[(sample_tasks[0], sample_tasks[1], "Comp A", "Comp B")],
            builder=builder,
            temperature=1.0,
            max_concurrent=10,
        )

        assert len(batch.successes) == 1
        assert batch.successes[0].choice == "a"
