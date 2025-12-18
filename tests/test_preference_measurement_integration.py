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
    BinaryPreferenceMeasurer,
    TaskScoreMeasurer,
    RegexChoiceFormat,
    XMLChoiceFormat,
    RegexRatingFormat,
    XMLRatingFormat,
    BINARY_CHOICE_TEMPLATE,
    PRE_TASK_RATING_TEMPLATE,
    BinaryPreferenceMeasurement,
    TaskScore,
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
        """Should record measurements to YAML file."""
        from pathlib import Path
        from src import MeasurementRecorder

        output_path = Path(__file__).parent / "measurement_results.yaml"

        binary_builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.DISPOSITIONAL,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        measurer = TaskScoreMeasurer()
        rating_builder = PreTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
            template=PRE_TASK_RATING_TEMPLATE,
        )

        config = DatasetMeasurementConfig(
            measurement_types=frozenset({"binary", "rating"}),
            pairing_strategy=PairingStrategy.ALL_PAIRS,
            num_samples=1,
            temperature=0.0,
        )

        with MeasurementRecorder(output_path) as recorder:
            measure_dataset_preferences(
                model=model,
                tasks=[math_task, creative_task],
                binary_builder=binary_builder,
                rating_builder=rating_builder,
                config=config,
                recorder=recorder,
            )

        assert output_path.exists()
        content = output_path.read_text()
        assert "binary" in content
        assert "rating" in content
