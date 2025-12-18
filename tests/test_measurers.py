"""Tests for measurers - classes that own parsing logic and measurement-specific properties."""

import pytest
from src.task_data import Task, OriginDataset
from src.preferences import (
    BinaryPreferenceMeasurement,
    TaskScore,
    PreferenceType,
    PreferencePrompt,
    RegexChoiceFormat,
    RegexRatingFormat,
)
from src.preferences.measurer import (
    Measurer,
    BinaryPreferenceMeasurer,
    TaskScoreMeasurer,
)


@pytest.fixture
def sample_task_a():
    return Task(
        prompt="Write a haiku about spring.",
        origin=OriginDataset.WILDCHAT,
        id="task_a",
        metadata={},
    )


@pytest.fixture
def sample_task_b():
    return Task(
        prompt="Solve x^2 = 4.",
        origin=OriginDataset.MATH,
        id="task_b",
        metadata={},
    )


class TestBinaryPreferenceMeasurer:
    """Tests for BinaryPreferenceMeasurer."""

    def test_parse_returns_response_with_measurement(self, sample_task_a, sample_task_b):
        """Should parse response and return BinaryPreferenceMeasurement."""
        measurer = BinaryPreferenceMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Choose A or B"}],
            tasks=[sample_task_a, sample_task_b],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexChoiceFormat(),
        )

        response = measurer.parse("A", prompt)

        assert isinstance(response.result, BinaryPreferenceMeasurement)
        assert response.result.choice == "a"
        assert response.result.task_a == sample_task_a
        assert response.result.task_b == sample_task_b
        assert response.result.preference_type == PreferenceType.PRE_TASK_STATED

    def test_parse_choice_b(self, sample_task_a, sample_task_b):
        """Should correctly parse choice B."""
        measurer = BinaryPreferenceMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Choose A or B"}],
            tasks=[sample_task_a, sample_task_b],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexChoiceFormat(),
        )

        response = measurer.parse("B", prompt)

        assert response.result.choice == "b"

    def test_parse_raises_on_ambiguous(self, sample_task_a, sample_task_b):
        """Should raise ValueError on ambiguous response."""
        measurer = BinaryPreferenceMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Choose A or B"}],
            tasks=[sample_task_a, sample_task_b],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexChoiceFormat(),
        )

        with pytest.raises(ValueError):
            measurer.parse("Both are good", prompt)


class TestTaskScoreMeasurer:
    """Tests for TaskScoreMeasurer."""

    def test_default_scale(self):
        """Should have default scale 1-10."""
        measurer = TaskScoreMeasurer()

        assert measurer.scale_min == 1
        assert measurer.scale_max == 10

    def test_custom_scale(self):
        """Should accept custom scale bounds."""
        measurer = TaskScoreMeasurer(scale_min=-5, scale_max=5)

        assert measurer.scale_min == -5
        assert measurer.scale_max == 5

    def test_parse_returns_response_with_score(self, sample_task_a):
        """Should parse response and return TaskScore."""
        measurer = TaskScoreMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Rate this task"}],
            tasks=[sample_task_a],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexRatingFormat(1, 10),
        )

        response = measurer.parse("7", prompt)

        assert isinstance(response.result, TaskScore)
        assert response.result.score == 7.0
        assert response.result.task == sample_task_a
        assert response.result.preference_type == PreferenceType.PRE_TASK_STATED

    def test_parse_extracts_float(self, sample_task_a):
        """Should parse float ratings."""
        measurer = TaskScoreMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Rate this task"}],
            tasks=[sample_task_a],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexRatingFormat(1, 10),
        )

        response = measurer.parse("7.5", prompt)

        assert response.result.score == 7.5

    def test_parse_extracts_from_text(self, sample_task_a):
        """Should extract number from surrounding text."""
        measurer = TaskScoreMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Rate this task"}],
            tasks=[sample_task_a],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexRatingFormat(1, 10),
        )

        response = measurer.parse("I'd rate this a 7", prompt)

        assert response.result.score == 7.0


class TestMeasurerIsABC:
    """Tests that Measurer is properly abstract."""

    def test_cannot_instantiate_measurer(self):
        """Should not be able to instantiate Measurer directly."""
        with pytest.raises(TypeError):
            Measurer()
