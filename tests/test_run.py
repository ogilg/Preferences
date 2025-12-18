"""Tests for measurement orchestration."""

import pytest
from src.model import ConfigurableMockModel
from src.task_data import Task, OriginDataset
from src.types import PreferencePrompt
from src.preferences import (
    BinaryPromptBuilder,
    PreTaskRatingPromptBuilder,
    PostTaskRatingPromptBuilder,
    BINARY_CHOICE_TEMPLATE,
    PRE_TASK_RATING_TEMPLATE,
    POST_TASK_RATING_TEMPLATE,
    RegexChoiceFormat,
    RegexRatingFormat,
    BinaryPreferenceMeasurement,
    TaskScore,
    TaskCompletion,
    PreferenceType,
)
from src.preferences.measurer import (
    BinaryPreferenceMeasurer,
    TaskScoreMeasurer,
)
from src.preferences.run import run_measurement, run_with_prompt


def get_all_content(prompt: PreferencePrompt) -> str:
    """Helper to get all message content concatenated for simple assertions."""
    return "\n".join(msg["content"] for msg in prompt.messages)


@pytest.fixture
def task_a():
    return Task(
        prompt="Write a haiku about spring.",
        origin=OriginDataset.WILDCHAT,
        id="task_a",
        metadata={},
    )


@pytest.fixture
def task_b():
    return Task(
        prompt="Solve x^2 = 4.",
        origin=OriginDataset.MATH,
        id="task_b",
        metadata={},
    )


class TestRunMeasurement:
    """Tests for run_measurement function."""

    def test_binary_preference_returns_response(self, task_a, task_b):
        """Should return Response with BinaryPreferenceMeasurement."""
        model = ConfigurableMockModel(response="A")
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        response = run_measurement(model, builder, task_a, task_b)

        assert isinstance(response.result, BinaryPreferenceMeasurement)
        assert response.result.choice == "a"

    def test_binary_preference_choice_b(self, task_a, task_b):
        """Should correctly parse choice B."""
        model = ConfigurableMockModel(response="B")
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        response = run_measurement(model, builder, task_a, task_b)

        assert response.result.choice == "b"

    def test_pre_task_rating_returns_response(self, task_a):
        """Should return Response with TaskScore for pre-task rating."""
        model = ConfigurableMockModel(response="7")
        measurer = TaskScoreMeasurer()
        builder = PreTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
            template=PRE_TASK_RATING_TEMPLATE,
        )

        response = run_measurement(model, builder, task_a)

        assert isinstance(response.result, TaskScore)
        assert response.result.score == 7.0

    def test_post_task_rating_returns_response(self, task_a):
        """Should return Response with TaskScore for post-task rating."""
        model = ConfigurableMockModel(response="8")
        measurer = TaskScoreMeasurer()
        builder = PostTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
            template=POST_TASK_RATING_TEMPLATE,
        )
        completion = TaskCompletion(task=task_a, text="Here is my haiku...")

        response = run_measurement(model, builder, task_a, completion)

        assert isinstance(response.result, TaskScore)
        assert response.result.score == 8.0

    def test_passes_correct_messages_to_model(self, task_a, task_b):
        """Should pass the builder's built messages to the model."""
        model = ConfigurableMockModel(response="A")
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        run_measurement(model, builder, task_a, task_b)

        expected_prompt = builder.build(task_a, task_b)
        assert model.last_messages == expected_prompt.messages

    def test_response_has_source_prompt(self, task_a, task_b):
        """Response should include the source prompt."""
        model = ConfigurableMockModel(response="A")
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        response = run_measurement(model, builder, task_a, task_b)

        assert response.source_prompt is not None
        prompt_content = get_all_content(response.source_prompt)
        assert task_a.prompt in prompt_content
        assert task_b.prompt in prompt_content

    def test_response_has_raw_text(self, task_a, task_b):
        """Response should include the raw model output."""
        model = ConfigurableMockModel(response="A")
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        response = run_measurement(model, builder, task_a, task_b)

        assert response.text == "A"


class TestRunWithPrompt:
    """Tests for run_with_prompt function."""

    def test_runs_with_prebuilt_prompt(self, task_a, task_b):
        """Should work with a pre-built prompt."""
        model = ConfigurableMockModel(response="A")
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )
        prompt = builder.build(task_a, task_b)

        response = run_with_prompt(model, prompt)

        assert isinstance(response.result, BinaryPreferenceMeasurement)
        assert response.result.choice == "a"
