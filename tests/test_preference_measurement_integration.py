"""Integration tests for preference measurement with real API calls.

Tests the full pipeline: prompt building -> model generation -> response parsing.

Run with:
    pytest tests/test_preference_measurement_integration.py -v

Skip with:
    pytest -m "not api"
"""

import pytest
from dotenv import load_dotenv

load_dotenv()

from src.models import HyperbolicModel
from src.task_data import Task, OriginDataset
from src.preferences import (
    BinaryPromptBuilder,
    PreTaskRatingPromptBuilder,
    PostTaskRatingPromptBuilder,
    BinaryPreferenceMeasurer,
    TaskScoreMeasurer,
    RegexChoiceFormat,
    XMLChoiceFormat,
    RegexRatingFormat,
    XMLRatingFormat,
    ToolUseChoiceFormat,
    ToolUseRatingFormat,
    BINARY_CHOICE_TEMPLATE,
    PRE_TASK_RATING_TEMPLATE,
    POST_TASK_RATING_TEMPLATE,
    BinaryPreferenceMeasurement,
    TaskScore,
    TaskCompletion,
    PreferenceType,
    measure_dataset_preferences,
    DatasetMeasurementConfig,
    PairingStrategy,
)


pytestmark = pytest.mark.api


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def model():
    """Shared model instance to minimize setup overhead."""
    return HyperbolicModel(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_new_tokens=32,
    )


@pytest.fixture
def math_task():
    return Task(
        prompt="Solve: 2 + 2 = ?",
        origin=OriginDataset.MATH,
        id="math_simple",
        metadata={},
    )


@pytest.fixture
def creative_task():
    return Task(
        prompt="Write a haiku about the ocean.",
        origin=OriginDataset.WILDCHAT,
        id="creative_haiku",
        metadata={},
    )


# =============================================================================
# Binary Choice Tests
# =============================================================================


class TestBinaryChoiceRegexFormat:
    """Test binary preference measurement with RegexChoiceFormat."""

    def test_parses_choice_successfully(self, model, math_task, creative_task):
        """Should parse A or B from model response."""
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        response_text = model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")
        assert result.text == response_text


class TestBinaryChoiceXMLFormat:
    """Test binary preference measurement with XMLChoiceFormat."""

    def test_parses_choice_from_xml(self, model, math_task, creative_task):
        """Should parse choice from XML tags."""
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=XMLChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        response_text = model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")


class TestBinaryChoicePreTaskStated:
    """Test binary preference measurement with PRE_TASK_STATED preference type."""

    def test_pre_task_stated_regex_format(self, model, math_task, creative_task):
        """Should parse choice with PRE_TASK_STATED preference type using Regex."""
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_STATED,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        assert prompt.kind == PreferenceType.PRE_TASK_STATED

        response_text = model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")

    def test_pre_task_stated_xml_format(self, model, math_task, creative_task):
        """Should parse choice with PRE_TASK_STATED preference type using XML."""
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_STATED,
            response_format=XMLChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        assert prompt.kind == PreferenceType.PRE_TASK_STATED

        response_text = model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")


class TestBinaryChoiceToolUseFormat:
    """Test binary preference measurement with ToolUseChoiceFormat."""

    def test_tool_call_returns_valid_choice(self, model, math_task, creative_task):
        """Tool call should return valid JSON that parses to a choice."""
        response_format = ToolUseChoiceFormat()
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=response_format,
            template=BINARY_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)

        # Call API with tools
        response_text = model.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        # Response should be valid JSON with choice
        result = prompt.measurer.parse(response_text, prompt)
        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")

    def test_tool_call_response_is_valid_json(self, model, math_task, creative_task):
        """Tool call response should be parseable JSON with 'choice' key."""
        import json

        response_format = ToolUseChoiceFormat()
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=response_format,
            template=BINARY_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)

        response_text = model.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        # Verify response structure
        parsed = json.loads(response_text)
        assert "choice" in parsed
        assert parsed["choice"].upper() in ("A", "B")


# =============================================================================
# Rating Tests
# =============================================================================


class TestRatingRegexFormat:
    """Test rating measurement with RegexRatingFormat."""

    def test_parses_rating_successfully(self, model, math_task):
        """Should parse numeric rating from model response."""
        measurer = TaskScoreMeasurer()
        builder = PreTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
            template=PRE_TASK_RATING_TEMPLATE,
        )

        prompt = builder.build(math_task)
        response_text = model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)


class TestRatingXMLFormat:
    """Test rating measurement with XMLRatingFormat."""

    def test_parses_rating_from_xml(self, model, creative_task):
        """Should parse rating from XML tags."""
        measurer = TaskScoreMeasurer()
        builder = PreTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=XMLRatingFormat(
                scale_min=measurer.scale_min,
                scale_max=measurer.scale_max,
            ),
            template=PRE_TASK_RATING_TEMPLATE,
        )

        prompt = builder.build(creative_task)
        response_text = model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)


# =============================================================================
# Post-Task Rating Tests
# =============================================================================


class TestPostTaskRatingRegexFormat:
    """Test post-task rating measurement with RegexRatingFormat."""

    def test_parses_post_task_rating_successfully(self, model, math_task):
        """Should parse rating after task completion."""
        measurer = TaskScoreMeasurer()
        builder = PostTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
            template=POST_TASK_RATING_TEMPLATE,
        )

        # Simulate a task completion
        completion = TaskCompletion(task=math_task, text="The answer is 4.")

        prompt = builder.build(math_task, completion)
        assert prompt.kind == PreferenceType.POST_TASK_STATED
        assert len(prompt.messages) == 3  # user, assistant, user

        response_text = model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)


class TestPostTaskRatingXMLFormat:
    """Test post-task rating measurement with XMLRatingFormat."""

    def test_parses_post_task_rating_from_xml(self, model, creative_task):
        """Should parse post-task rating from XML tags."""
        measurer = TaskScoreMeasurer()
        builder = PostTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=XMLRatingFormat(
                scale_min=measurer.scale_min,
                scale_max=measurer.scale_max,
            ),
            template=POST_TASK_RATING_TEMPLATE,
        )

        # Simulate a task completion
        completion = TaskCompletion(
            task=creative_task,
            text="Waves crash on shore\nSalt spray kisses the warm wind\nPeace beneath the blue",
        )

        prompt = builder.build(creative_task, completion)
        assert prompt.kind == PreferenceType.POST_TASK_STATED
        assert len(prompt.messages) == 3  # user, assistant, user

        response_text = model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)


class TestRatingToolUseFormat:
    """Test rating measurement with ToolUseRatingFormat."""

    def test_tool_call_returns_valid_rating(self, model, math_task):
        """Tool call should return valid JSON that parses to a rating."""
        response_format = ToolUseRatingFormat()
        builder = PreTaskRatingPromptBuilder(
            measurer=TaskScoreMeasurer(),
            response_format=response_format,
            template=PRE_TASK_RATING_TEMPLATE,
        )

        prompt = builder.build(math_task)

        # Call API with tools
        response_text = model.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        # Response should be valid JSON with rating
        result = prompt.measurer.parse(response_text, prompt)
        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)

    def test_tool_call_response_is_valid_json(self, model, creative_task):
        """Tool call response should be parseable JSON with 'rating' key."""
        import json

        response_format = ToolUseRatingFormat()
        builder = PreTaskRatingPromptBuilder(
            measurer=TaskScoreMeasurer(),
            response_format=response_format,
            template=PRE_TASK_RATING_TEMPLATE,
        )

        prompt = builder.build(creative_task)

        response_text = model.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        # Verify response structure
        parsed = json.loads(response_text)
        assert "rating" in parsed
        assert isinstance(parsed["rating"], (int, float))


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestMeasureDatasetPreferences:
    """Test the full measure_dataset_preferences pipeline."""

    def test_binary_measurement_pipeline(self, model, math_task, creative_task):
        """Should run binary measurements and return valid results."""
        binary_builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        config = DatasetMeasurementConfig(
            measurement_types=frozenset({"binary"}),
            pairing_strategy=PairingStrategy.ALL_PAIRS,
            num_samples=1,
            temperature=0.0,
        )

        result = measure_dataset_preferences(
            model=model,
            tasks=[math_task, creative_task],
            binary_builder=binary_builder,
            config=config,
        )

        assert len(result["binary_comparisons"]) == 1
        comparison = result["binary_comparisons"][0]
        assert len(comparison["samples"]) == 1
        sample = comparison["samples"][0]
        assert isinstance(sample["response"].result, BinaryPreferenceMeasurement)
        assert sample["response"].result.choice in ("a", "b")

    def test_rating_measurement_pipeline(self, model, math_task, creative_task):
        """Should run rating measurements and return valid results."""
        measurer = TaskScoreMeasurer()
        rating_builder = PreTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
            template=PRE_TASK_RATING_TEMPLATE,
        )

        config = DatasetMeasurementConfig(
            measurement_types=frozenset({"rating"}),
            num_samples=1,
            temperature=0.0,
        )

        result = measure_dataset_preferences(
            model=model,
            tasks=[math_task, creative_task],
            rating_builder=rating_builder,
            config=config,
        )

        assert len(result["task_ratings"]) == 2
        for task_result in result["task_ratings"]:
            assert len(task_result["samples"]) == 1
            sample = task_result["samples"][0]
            assert isinstance(sample["response"].result, TaskScore)
            assert isinstance(sample["response"].result.score, float)

    def test_measurement_with_recorder(self, model, math_task, creative_task):
        """Should record measurements with different formats to YAML file."""
        from pathlib import Path
        from src import MeasurementRecorder

        output_path = Path(__file__).parent / "measurement_results.yaml"

        # Clean up any existing file
        if output_path.exists():
            output_path.unlink()

        tasks = [math_task, creative_task]
        measurer = TaskScoreMeasurer()

        config = DatasetMeasurementConfig(
            measurement_types=frozenset({"binary", "rating"}),
            pairing_strategy=PairingStrategy.ALL_PAIRS,
            num_samples=1,
            temperature=0.0,
        )

        with MeasurementRecorder(output_path) as recorder:
            # 1. Regex formats
            measure_dataset_preferences(
                model=model,
                tasks=tasks,
                binary_builder=BinaryPromptBuilder(
                    measurer=BinaryPreferenceMeasurer(),
                    preference_type=PreferenceType.DISPOSITIONAL,
                    response_format=RegexChoiceFormat(),
                    template=BINARY_CHOICE_TEMPLATE,
                ),
                rating_builder=PreTaskRatingPromptBuilder(
                    measurer=measurer,
                    response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
                    template=PRE_TASK_RATING_TEMPLATE,
                ),
                config=config,
                recorder=recorder,
            )

            # 2. XML formats
            measure_dataset_preferences(
                model=model,
                tasks=tasks,
                binary_builder=BinaryPromptBuilder(
                    measurer=BinaryPreferenceMeasurer(),
                    preference_type=PreferenceType.DISPOSITIONAL,
                    response_format=XMLChoiceFormat(),
                    template=BINARY_CHOICE_TEMPLATE,
                ),
                rating_builder=PreTaskRatingPromptBuilder(
                    measurer=measurer,
                    response_format=XMLRatingFormat(measurer.scale_min, measurer.scale_max),
                    template=PRE_TASK_RATING_TEMPLATE,
                ),
                config=config,
                recorder=recorder,
            )

            # 3. Tool use formats
            measure_dataset_preferences(
                model=model,
                tasks=tasks,
                binary_builder=BinaryPromptBuilder(
                    measurer=BinaryPreferenceMeasurer(),
                    preference_type=PreferenceType.DISPOSITIONAL,
                    response_format=ToolUseChoiceFormat(),
                    template=BINARY_CHOICE_TEMPLATE,
                ),
                rating_builder=PreTaskRatingPromptBuilder(
                    measurer=measurer,
                    response_format=ToolUseRatingFormat(measurer.scale_min, measurer.scale_max),
                    template=PRE_TASK_RATING_TEMPLATE,
                ),
                config=config,
                recorder=recorder,
            )

        assert output_path.exists()
        content = output_path.read_text()

        # Verify all measurement types recorded
        assert "binary" in content
        assert "rating" in content

        # Verify all format types recorded (including any that had errors)
        assert "RegexChoiceFormat" in content
        assert "XMLChoiceFormat" in content
        assert "ToolUseChoiceFormat" in content
        assert "RegexRatingFormat" in content
        assert "XMLRatingFormat" in content
        assert "ToolUseRatingFormat" in content

        # Log any errors for visibility (errors are recorded, not raised)
        import yaml

        with open(output_path) as f:
            records = yaml.safe_load(f)

        errors = [r for r in records if "error" in r.get("result", {})]
        if errors:
            print(f"\n{len(errors)} measurements had parse errors (recorded, not raised):")
            for e in errors:
                print(f"  - {e['response_format']}: {e['result']['error']}")
