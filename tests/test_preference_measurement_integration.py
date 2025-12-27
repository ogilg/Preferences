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
    CompletionChoiceFormat,
    RegexRatingFormat,
    XMLRatingFormat,
    ToolUseChoiceFormat,
    ToolUseRatingFormat,
    BINARY_CHOICE_TEMPLATE,
    BINARY_COMPLETION_TEMPLATE,
    PRE_TASK_RATING_TEMPLATE,
    POST_TASK_RATING_TEMPLATE,
    BinaryPreferenceMeasurement,
    TaskScore,
    TaskCompletion,
    PreferenceType,
    measure_binary_preferences,
    measure_ratings,
)
from src.preferences.config import DatasetMeasurementConfig, PairingStrategy


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


@pytest.fixture(scope="module")
def completion_model():
    """Model with higher token limit for task completion tests."""
    return HyperbolicModel(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_new_tokens=128,
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
            preference_type=PreferenceType.PRE_TASK_STATED,
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
            preference_type=PreferenceType.PRE_TASK_STATED,
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
            preference_type=PreferenceType.PRE_TASK_STATED,
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
            preference_type=PreferenceType.PRE_TASK_STATED,
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


class TestBinaryChoiceCompletionFormat:
    """Test binary preference measurement with CompletionChoiceFormat (revealed preference)."""

    def test_parses_choice_from_task_completion(self, completion_model, math_task, creative_task):
        """Model completes a task and we parse which one it chose."""
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_REVEALED,
            response_format=CompletionChoiceFormat(),
            template=BINARY_COMPLETION_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        assert prompt.kind == PreferenceType.PRE_TASK_REVEALED

        response_text = completion_model.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")
        # Response should contain actual task completion content
        assert len(response_text) > 10  # Not just "Task A:" but actual content

    def test_response_contains_task_indicator(self, completion_model, math_task, creative_task):
        """Response should start with Task A: or Task B: indicator."""
        builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_REVEALED,
            response_format=CompletionChoiceFormat(),
            template=BINARY_COMPLETION_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        response_text = completion_model.generate(prompt.messages, temperature=0.0)

        response_lower = response_text.lower()
        assert "task a" in response_lower or "task b" in response_lower


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


class TestPostTaskRatingToolUseFormat:
    """Test post-task rating measurement with ToolUseRatingFormat."""

    def test_tool_call_returns_valid_rating(self, model, math_task):
        """Tool call should return valid JSON rating after task completion."""
        response_format = ToolUseRatingFormat()
        builder = PostTaskRatingPromptBuilder(
            measurer=TaskScoreMeasurer(),
            response_format=response_format,
            template=POST_TASK_RATING_TEMPLATE,
        )

        completion = TaskCompletion(task=math_task, text="The answer is 4.")
        prompt = builder.build(math_task, completion)
        assert prompt.kind == PreferenceType.POST_TASK_STATED
        assert len(prompt.messages) == 3  # user, assistant, user

        response_text = model.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        result = prompt.measurer.parse(response_text, prompt)
        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)

    def test_tool_call_response_is_valid_json(self, model, creative_task):
        """Tool call response should be parseable JSON with 'rating' key."""
        import json

        response_format = ToolUseRatingFormat()
        builder = PostTaskRatingPromptBuilder(
            measurer=TaskScoreMeasurer(),
            response_format=response_format,
            template=POST_TASK_RATING_TEMPLATE,
        )

        completion = TaskCompletion(
            task=creative_task,
            text="Waves crash on shore\nSalt spray kisses the warm wind\nPeace beneath the blue",
        )
        prompt = builder.build(creative_task, completion)

        response_text = model.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        parsed = json.loads(response_text)
        assert "rating" in parsed
        assert isinstance(parsed["rating"], (int, float))


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


class TestMeasurePreferences:
    """Test the measure_binary_preferences and measure_ratings functions."""

    def test_binary_measurement_pipeline(self, model, math_task, creative_task):
        """Should run binary measurements and return valid results."""
        binary_builder = BinaryPromptBuilder(
            measurer=BinaryPreferenceMeasurer(),
            preference_type=PreferenceType.PRE_TASK_STATED,
            response_format=RegexChoiceFormat(),
            template=BINARY_CHOICE_TEMPLATE,
        )

        pairs = [(math_task, creative_task)]
        results = measure_binary_preferences(
            model=model,
            pairs=pairs,
            builder=binary_builder,
            temperature=0.0,
        )

        assert len(results) == 1
        assert isinstance(results[0], BinaryPreferenceMeasurement)
        assert results[0].choice in ("a", "b")

    def test_rating_measurement_pipeline(self, model, math_task, creative_task):
        """Should run rating measurements and return valid results."""
        measurer = TaskScoreMeasurer()
        rating_builder = PreTaskRatingPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
            template=PRE_TASK_RATING_TEMPLATE,
        )

        tasks = [math_task, creative_task]
        results = measure_ratings(
            model=model,
            tasks=tasks,
            builder=rating_builder,
            temperature=0.0,
        )

        assert len(results) == 2
        for score in results:
            assert isinstance(score, TaskScore)
            assert isinstance(score.score, float)

    def test_measurement_with_recorder(self, model, math_task, creative_task):
        """Should record measurements with different formats to YAML file."""
        from pathlib import Path
        from src import MeasurementRecorder
        from src.preferences.measurement_recorder import MeasurementRecord

        output_path = Path(__file__).parent / "measurement_results.yaml"

        # Clean up any existing file
        if output_path.exists():
            output_path.unlink()

        measurer = TaskScoreMeasurer()

        def record_measurement(recorder, builder, tasks_for_record, measurement_type):
            """Helper to run a measurement and record it."""
            prompt = builder.build(*tasks_for_record)
            tools = getattr(prompt.response_format, "tools", None)
            response_text = model.generate(prompt.messages, temperature=0.0, tools=tools)

            try:
                response = prompt.measurer.parse(response_text, prompt)
                if hasattr(response.result, "choice"):
                    result_dict = {"choice": response.result.choice}
                else:
                    result_dict = {"score": response.result.score}
            except Exception as e:
                result_dict = {"error": str(e)}

            prompt_text = "\n\n".join(
                f"[{m['role']}]\n{m['content']}" for m in prompt.messages
            )
            record = MeasurementRecord(
                model=model.model_name,
                measurement_type=measurement_type,
                tasks=[{"id": t.id, "prompt": t.prompt} for t in tasks_for_record],
                response_format=type(prompt.response_format).__name__,
                template=builder.template.name,
                temperature=0.0,
                sample_index=0,
                prompt=prompt_text,
                response=response_text,
                result=result_dict,
            )
            recorder.record(record)

        with MeasurementRecorder(output_path) as recorder:
            # Binary choice formats
            for fmt in [
                RegexChoiceFormat(),
                XMLChoiceFormat(),
                ToolUseChoiceFormat(),
            ]:
                builder = BinaryPromptBuilder(
                    measurer=BinaryPreferenceMeasurer(),
                    preference_type=PreferenceType.PRE_TASK_STATED,
                    response_format=fmt,
                    template=BINARY_CHOICE_TEMPLATE,
                )
                record_measurement(recorder, builder, (math_task, creative_task), PreferenceType.PRE_TASK_STATED.name)

            # Completion format (revealed preference)
            completion_model = HyperbolicModel(
                model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                max_new_tokens=128,
            )
            completion_builder = BinaryPromptBuilder(
                measurer=BinaryPreferenceMeasurer(),
                preference_type=PreferenceType.PRE_TASK_REVEALED,
                response_format=CompletionChoiceFormat(),
                template=BINARY_COMPLETION_TEMPLATE,
            )
            prompt = completion_builder.build(math_task, creative_task)
            response_text = completion_model.generate(prompt.messages, temperature=0.0)
            try:
                response = prompt.measurer.parse(response_text, prompt)
                result_dict = {"choice": response.result.choice}
            except Exception as e:
                result_dict = {"error": str(e)}
            prompt_text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in prompt.messages)
            recorder.record(MeasurementRecord(
                model=completion_model.model_name,
                measurement_type=PreferenceType.PRE_TASK_REVEALED.name,
                tasks=[{"id": math_task.id, "prompt": math_task.prompt}, {"id": creative_task.id, "prompt": creative_task.prompt}],
                response_format=type(completion_builder.response_format).__name__,
                template=completion_builder.template.name,
                temperature=0.0,
                sample_index=0,
                prompt=prompt_text,
                response=response_text,
                result=result_dict,
            ))

            # Rating formats (pre-task)
            for fmt in [
                RegexRatingFormat(measurer.scale_min, measurer.scale_max),
                XMLRatingFormat(scale_min=measurer.scale_min, scale_max=measurer.scale_max),
                ToolUseRatingFormat(scale_min=measurer.scale_min, scale_max=measurer.scale_max),
            ]:
                builder = PreTaskRatingPromptBuilder(
                    measurer=measurer,
                    response_format=fmt,
                    template=PRE_TASK_RATING_TEMPLATE,
                )
                record_measurement(recorder, builder, (math_task,), PreferenceType.PRE_TASK_STATED.name)

            # Post-task rating
            post_task_builder = PostTaskRatingPromptBuilder(
                measurer=measurer,
                response_format=RegexRatingFormat(measurer.scale_min, measurer.scale_max),
                template=POST_TASK_RATING_TEMPLATE,
            )
            completion = TaskCompletion(task=math_task, text="The answer is 4.")
            prompt = post_task_builder.build(math_task, completion)
            response_text = model.generate(prompt.messages, temperature=0.0)
            try:
                response = prompt.measurer.parse(response_text, prompt)
                result_dict = {"score": response.result.score}
            except Exception as e:
                result_dict = {"error": str(e)}
            prompt_text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in prompt.messages)
            recorder.record(MeasurementRecord(
                model=model.model_name,
                measurement_type=PreferenceType.POST_TASK_STATED.name,
                tasks=[{"id": math_task.id, "prompt": math_task.prompt}],
                response_format=type(post_task_builder.response_format).__name__,
                template=post_task_builder.template.name,
                temperature=0.0,
                sample_index=0,
                prompt=prompt_text,
                response=response_text,
                result=result_dict,
            ))

        assert output_path.exists()
        content = output_path.read_text()

        # Verify all preference types recorded
        assert PreferenceType.PRE_TASK_STATED.name in content
        assert PreferenceType.PRE_TASK_REVEALED.name in content
        assert PreferenceType.POST_TASK_STATED.name in content

        # Verify all format types recorded (including any that had errors)
        assert "RegexChoiceFormat" in content
        assert "XMLChoiceFormat" in content
        assert "ToolUseChoiceFormat" in content
        assert "CompletionChoiceFormat" in content
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
