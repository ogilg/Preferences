import pytest
from src.task_data import Task, OriginDataset
from src.preference_measurement import (
    BinaryPreferenceMeasurement,
    TaskScore,
    PreferenceType,
    PreferencePrompt,
    RegexChoiceFormat,
    RegexRatingFormat,
)
from src.preference_measurement import (
    Measurer,
    RevealedPreferenceMeasurer,
    StatedScoreMeasurer,
)
from src.prompt_templates import PromptTemplate


@pytest.fixture
def dummy_template():
    return PromptTemplate(
        template="{format_instruction}",
        name="dummy",
        required_placeholders=frozenset({"format_instruction"}),
    )


class TestRevealedPreferenceMeasurer:
    """Tests for RevealedPreferenceMeasurer."""

    def test_parse_returns_response_with_measurement(
        self, sample_task_a, sample_task_b, dummy_template
    ):
        """Should parse response and return BinaryPreferenceMeasurement."""
        measurer = RevealedPreferenceMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Choose A or B"}],
            tasks=[sample_task_a, sample_task_b],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexChoiceFormat(),
            template=dummy_template,
        )

        response = measurer.parse("Task A", prompt)

        assert isinstance(response.result, BinaryPreferenceMeasurement)
        assert response.result.choice == "a"
        assert response.result.task_a == sample_task_a
        assert response.result.task_b == sample_task_b
        assert response.result.preference_type == PreferenceType.PRE_TASK_STATED

    def test_parse_choice_b(self, sample_task_a, sample_task_b, dummy_template):
        """Should correctly parse choice B."""
        measurer = RevealedPreferenceMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Choose A or B"}],
            tasks=[sample_task_a, sample_task_b],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexChoiceFormat(),
            template=dummy_template,
        )

        response = measurer.parse("Task B", prompt)

        assert response.result.choice == "b"

    def test_parse_raises_on_ambiguous(self, sample_task_a, sample_task_b, dummy_template):
        """Should raise ValueError on ambiguous response."""
        measurer = RevealedPreferenceMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Choose A or B"}],
            tasks=[sample_task_a, sample_task_b],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexChoiceFormat(),
            template=dummy_template,
        )

        with pytest.raises(ValueError):
            measurer.parse("Both are good", prompt)


class TestStatedScoreMeasurer:
    """Tests for StatedScoreMeasurer."""

    def test_parse_returns_response_with_score(self, sample_task_a, dummy_template):
        """Should parse response and return TaskScore."""
        measurer = StatedScoreMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Rate this task"}],
            tasks=[sample_task_a],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexRatingFormat(1, 10),
            template=dummy_template,
        )

        response = measurer.parse("7", prompt)

        assert isinstance(response.result, TaskScore)
        assert response.result.score == 7.0
        assert response.result.task == sample_task_a
        assert response.result.preference_type == PreferenceType.PRE_TASK_STATED

    def test_parse_extracts_float(self, sample_task_a, dummy_template):
        """Should parse float ratings."""
        measurer = StatedScoreMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Rate this task"}],
            tasks=[sample_task_a],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexRatingFormat(1, 10),
            template=dummy_template,
        )

        response = measurer.parse("7.5", prompt)

        assert response.result.score == 7.5

    def test_parse_extracts_from_text(self, sample_task_a, dummy_template):
        """Should extract number from surrounding text."""
        measurer = StatedScoreMeasurer()
        prompt = PreferencePrompt(
            messages=[{"role": "user", "content": "Rate this task"}],
            tasks=[sample_task_a],
            kind=PreferenceType.PRE_TASK_STATED,
            measurer=measurer,
            response_format=RegexRatingFormat(1, 10),
            template=dummy_template,
        )

        response = measurer.parse("I'd rate this a 7", prompt)

        assert response.result.score == 7.0


class TestMeasurerIsABC:
    """Tests that Measurer is properly abstract."""

    def test_cannot_instantiate_measurer(self):
        """Should not be able to instantiate Measurer directly."""
        with pytest.raises(TypeError):
            Measurer()
