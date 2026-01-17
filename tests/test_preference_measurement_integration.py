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

from src.models import get_client
from src.task_data import Task, OriginDataset, load_tasks
from src.preference_measurement import (
    RevealedPreferenceMeasurer,
    StatedScoreMeasurer,
    RegexChoiceFormat,
    XMLChoiceFormat,
    CompletionChoiceFormat,
    RegexRatingFormat,
    XMLRatingFormat,
    ToolUseChoiceFormat,
    ToolUseRatingFormat,
    BinaryPreferenceMeasurement,
    TaskScore,
    PreferenceType,
    measure_revealed_preferences,
    measure_stated,
    DatasetMeasurementConfig,
    PairingStrategy,
)
from src.prompt_templates import (
    PreTaskRevealedPromptBuilder,
    PreTaskStatedPromptBuilder,
    PostTaskStatedPromptBuilder,
    REVEALED_CHOICE_TEMPLATE,
    REVEALED_COMPLETION_TEMPLATE,
    PRE_TASK_STATED_TEMPLATE,
    POST_TASK_STATED_TEMPLATE,
)


pytestmark = pytest.mark.api


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def client():
    """Shared client instance to minimize setup overhead."""
    return get_client(
        model_name="llama-3.1-8b",
        max_new_tokens=32,
    )


@pytest.fixture(scope="module")
def completion_client():
    """Client with higher token limit for task completion tests."""
    return get_client(
        model_name="llama-3.1-8b",
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

    def test_parses_choice_successfully(self, client, math_task, creative_task):
        """Should parse A or B from model response."""
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=RegexChoiceFormat(),
            template=REVEALED_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        response_text = client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")
        assert result.text == response_text


class TestBinaryChoiceXMLFormat:
    """Test binary preference measurement with XMLChoiceFormat."""

    def test_parses_choice_from_xml(self, client, math_task, creative_task):
        """Should parse choice from XML tags."""
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=XMLChoiceFormat(),
            template=REVEALED_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        response_text = client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")


class TestBinaryChoicePreTaskRevealed:
    """Test binary preference measurement with PRE_TASK_REVEALED preference type."""

    def test_pre_task_revealed_regex_format(self, client, math_task, creative_task):
        """Should parse choice with PRE_TASK_REVEALED preference type using Regex."""
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=RegexChoiceFormat(),
            template=REVEALED_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        assert prompt.kind == PreferenceType.PRE_TASK_REVEALED

        response_text = client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")

    def test_pre_task_revealed_xml_format(self, client, math_task, creative_task):
        """Should parse choice with PRE_TASK_REVEALED preference type using XML."""
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=XMLChoiceFormat(),
            template=REVEALED_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        assert prompt.kind == PreferenceType.PRE_TASK_REVEALED

        response_text = client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")


class TestBinaryChoiceToolUseFormat:
    """Test binary preference measurement with ToolUseChoiceFormat."""

    def test_tool_call_returns_valid_choice(self, client, math_task, creative_task):
        """Tool call should return valid JSON that parses to a choice."""
        response_format = ToolUseChoiceFormat()
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=response_format,
            template=REVEALED_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)

        # Call API with tools
        response_text = client.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        # Response should be valid JSON with choice
        result = prompt.measurer.parse(response_text, prompt)
        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")

    def test_tool_call_response_is_valid_json(self, client, math_task, creative_task):
        """Tool call response should be parseable JSON with 'choice' key."""
        import json

        response_format = ToolUseChoiceFormat()
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=response_format,
            template=REVEALED_CHOICE_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)

        response_text = client.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        # Verify response structure
        parsed = json.loads(response_text)
        assert "choice" in parsed
        assert parsed["choice"].lower() in ("task a", "task b")


class TestBinaryChoiceCompletionFormat:
    """Test binary preference measurement with CompletionChoiceFormat (revealed preference)."""

    def test_parses_choice_from_task_completion(self, completion_client, math_task, creative_task):
        """Model completes a task and we parse which one it chose."""
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=CompletionChoiceFormat(),
            template=REVEALED_COMPLETION_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        assert prompt.kind == PreferenceType.PRE_TASK_REVEALED

        response_text = completion_client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, BinaryPreferenceMeasurement)
        assert result.result.choice in ("a", "b")
        # Response should contain actual task completion content
        assert len(response_text) > 10  # Not just "Task A:" but actual content

    def test_response_contains_task_indicator(self, completion_client, math_task, creative_task):
        """Response should start with Task A: or Task B: indicator."""
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=CompletionChoiceFormat(),
            template=REVEALED_COMPLETION_TEMPLATE,
        )

        prompt = builder.build(math_task, creative_task)
        response_text = completion_client.generate(prompt.messages, temperature=0.0)

        response_lower = response_text.lower()
        assert "task a" in response_lower or "task b" in response_lower


# =============================================================================
# Rating Tests
# =============================================================================


class TestRatingRegexFormat:
    """Test rating measurement with RegexRatingFormat."""

    def test_parses_rating_successfully(self, client, math_task):
        """Should parse numeric rating from model response."""
        measurer = StatedScoreMeasurer()
        builder = PreTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(),
            template=PRE_TASK_STATED_TEMPLATE,
        )

        prompt = builder.build(math_task)
        response_text = client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)


class TestRatingXMLFormat:
    """Test rating measurement with XMLRatingFormat."""

    def test_parses_rating_from_xml(self, client, creative_task):
        """Should parse rating from XML tags."""
        measurer = StatedScoreMeasurer()
        builder = PreTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=XMLRatingFormat(),
            template=PRE_TASK_STATED_TEMPLATE,
        )

        prompt = builder.build(creative_task)
        response_text = client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)


# =============================================================================
# Post-Task Rating Tests
# =============================================================================


class TestPostTaskRatingRegexFormat:
    """Test post-task rating measurement with RegexRatingFormat."""

    def test_parses_post_task_rating_successfully(self, client, math_task):
        """Should parse rating after task completion."""
        measurer = StatedScoreMeasurer()
        builder = PostTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(),
            template=POST_TASK_STATED_TEMPLATE,
        )

        completion_text = "The answer is 4."

        prompt = builder.build(math_task, completion_text)
        assert prompt.kind == PreferenceType.POST_TASK_STATED
        assert len(prompt.messages) == 3  # user, assistant, user

        response_text = client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)


class TestPostTaskRatingXMLFormat:
    """Test post-task rating measurement with XMLRatingFormat."""

    def test_parses_post_task_rating_from_xml(self, client, creative_task):
        """Should parse post-task rating from XML tags."""
        measurer = StatedScoreMeasurer()
        builder = PostTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=XMLRatingFormat(),
            template=POST_TASK_STATED_TEMPLATE,
        )

        completion_text = "Waves crash on shore\nSalt spray kisses the warm wind\nPeace beneath the blue"

        prompt = builder.build(creative_task, completion_text)
        assert prompt.kind == PreferenceType.POST_TASK_STATED
        assert len(prompt.messages) == 3  # user, assistant, user

        response_text = client.generate(prompt.messages, temperature=0.0)
        result = prompt.measurer.parse(response_text, prompt)

        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)


class TestPostTaskRatingToolUseFormat:
    """Test post-task rating measurement with ToolUseRatingFormat."""

    def test_tool_call_returns_valid_rating(self, client, math_task):
        """Tool call should return valid JSON rating after task completion."""
        response_format = ToolUseRatingFormat()
        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=response_format,
            template=POST_TASK_STATED_TEMPLATE,
        )

        completion_text = "The answer is 4."
        prompt = builder.build(math_task, completion_text)
        assert prompt.kind == PreferenceType.POST_TASK_STATED
        assert len(prompt.messages) == 3  # user, assistant, user

        response_text = client.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        result = prompt.measurer.parse(response_text, prompt)
        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)

    def test_tool_call_response_is_valid_json(self, client, creative_task):
        """Tool call response should be parseable JSON with 'rating' key."""
        import json

        response_format = ToolUseRatingFormat()
        builder = PostTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=response_format,
            template=POST_TASK_STATED_TEMPLATE,
        )

        completion_text = "Waves crash on shore\nSalt spray kisses the warm wind\nPeace beneath the blue"
        prompt = builder.build(creative_task, completion_text)

        response_text = client.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        parsed = json.loads(response_text)
        assert "rating" in parsed
        assert isinstance(parsed["rating"], (int, float))


class TestRatingToolUseFormat:
    """Test rating measurement with ToolUseRatingFormat."""

    def test_tool_call_returns_valid_rating(self, client, math_task):
        """Tool call should return valid JSON that parses to a rating."""
        response_format = ToolUseRatingFormat()
        builder = PreTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=response_format,
            template=PRE_TASK_STATED_TEMPLATE,
        )

        prompt = builder.build(math_task)

        # Call API with tools
        response_text = client.generate(
            prompt.messages,
            temperature=0.0,
            tools=response_format.tools,
        )

        # Response should be valid JSON with rating
        result = prompt.measurer.parse(response_text, prompt)
        assert isinstance(result.result, TaskScore)
        assert isinstance(result.result.score, float)

    def test_tool_call_response_is_valid_json(self, client, creative_task):
        """Tool call response should be parseable JSON with 'rating' key."""
        import json

        response_format = ToolUseRatingFormat()
        builder = PreTaskStatedPromptBuilder(
            measurer=StatedScoreMeasurer(),
            response_format=response_format,
            template=PRE_TASK_STATED_TEMPLATE,
        )

        prompt = builder.build(creative_task)

        response_text = client.generate(
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
    """Test the measure_revealed_preferences and measure_stated functions."""

    def test_binary_measurement_pipeline(self, client, math_task, creative_task):
        """Should run binary measurements and return valid results."""
        binary_builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),            response_format=RegexChoiceFormat(),
            template=REVEALED_CHOICE_TEMPLATE,
        )

        pairs = [(math_task, creative_task)]
        batch = measure_revealed_preferences(
            client=client,
            pairs=pairs,
            builder=binary_builder,
            temperature=0.0,
        )

        assert len(batch.successes) == 1
        assert isinstance(batch.successes[0], BinaryPreferenceMeasurement)
        assert batch.successes[0].choice in ("a", "b")

    def test_rating_measurement_pipeline(self, client, math_task, creative_task):
        """Should run rating measurements and return valid results."""
        measurer = StatedScoreMeasurer()
        rating_builder = PreTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(),
            template=PRE_TASK_STATED_TEMPLATE,
        )

        tasks = [math_task, creative_task]
        batch = measure_stated(
            client=client,
            tasks=tasks,
            builder=rating_builder,
            temperature=0.0,
        )

        assert len(batch.successes) == 2
        for score in batch.successes:
            assert isinstance(score, TaskScore)
            assert isinstance(score.score, float)

    def test_measurement_with_recorder(self, client, math_task, creative_task):
        """Should record measurements with different formats to YAML file."""
        from pathlib import Path
        from src import MeasurementRecorder
        from src.preference_measurement import MeasurementRecord

        output_path = Path(__file__).parent / "measurement_results.yaml"

        # Clean up any existing file
        if output_path.exists():
            output_path.unlink()

        measurer = StatedScoreMeasurer()

        def record_measurement(recorder, builder, tasks_for_record, measurement_type):
            """Helper to run a measurement and record it."""
            prompt = builder.build(*tasks_for_record)
            response_text = client.generate(prompt.messages, temperature=0.0, tools=prompt.response_format.tools)

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
                model=client.canonical_model_name,
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
                builder = PreTaskRevealedPromptBuilder(
                    measurer=RevealedPreferenceMeasurer(),
                    response_format=fmt,
                    template=REVEALED_CHOICE_TEMPLATE,
                )
                record_measurement(recorder, builder, (math_task, creative_task), PreferenceType.PRE_TASK_REVEALED.name)

            # Completion format (revealed preference)
            completion_client = get_client(
                model_name="llama-3.1-8b",
                max_new_tokens=128,
            )
            completion_builder = PreTaskRevealedPromptBuilder(
                measurer=RevealedPreferenceMeasurer(),
                response_format=CompletionChoiceFormat(),
                template=REVEALED_COMPLETION_TEMPLATE,
            )
            prompt = completion_builder.build(math_task, creative_task)
            response_text = completion_client.generate(prompt.messages, temperature=0.0)
            try:
                response = prompt.measurer.parse(response_text, prompt)
                result_dict = {"choice": response.result.choice}
            except Exception as e:
                result_dict = {"error": str(e)}
            prompt_text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in prompt.messages)
            recorder.record(MeasurementRecord(
                model=completion_client.canonical_model_name,
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
                RegexRatingFormat(),
                XMLRatingFormat(),
                ToolUseRatingFormat(),
            ]:
                builder = PreTaskStatedPromptBuilder(
                    measurer=measurer,
                    response_format=fmt,
                    template=PRE_TASK_STATED_TEMPLATE,
                )
                record_measurement(recorder, builder, (math_task,), PreferenceType.PRE_TASK_STATED.name)

            # Post-task rating
            post_task_builder = PostTaskStatedPromptBuilder(
                measurer=measurer,
                response_format=RegexRatingFormat(),
                template=POST_TASK_STATED_TEMPLATE,
            )
            completion_text = "The answer is 4."
            prompt = post_task_builder.build(math_task, completion_text)
            response_text = client.generate(prompt.messages, temperature=0.0)
            try:
                response = prompt.measurer.parse(response_text, prompt)
                result_dict = {"score": response.result.score}
            except Exception as e:
                result_dict = {"error": str(e)}
            prompt_text = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in prompt.messages)
            recorder.record(MeasurementRecord(
                model=client.canonical_model_name,
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


# =============================================================================
# BailBench Dataset Tests
# =============================================================================


class TestBailBenchPreferences:
    """Test preference measurement on real BailBench tasks."""

    @pytest.fixture
    def bailbench_tasks(self):
        """Load a sample of BailBench tasks."""
        return load_tasks(n=5, origins=[OriginDataset.BAILBENCH], seed=42)

    def test_bailbench_stated_preferences(self, client, bailbench_tasks):
        """Should measure stated preferences on BailBench tasks."""
        measurer = StatedScoreMeasurer()
        builder = PreTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=RegexRatingFormat(),
            template=PRE_TASK_STATED_TEMPLATE,
        )

        batch = measure_stated(
            client=client,
            tasks=bailbench_tasks[:2],
            builder=builder,
            temperature=0.0,
        )

        assert len(batch.successes) + len(batch.failures) == 2
        for score in batch.successes:
            assert isinstance(score, TaskScore)
            assert isinstance(score.score, float)

    def test_bailbench_revealed_preferences(self, client, bailbench_tasks):
        """Should measure revealed preferences between BailBench tasks."""
        builder = PreTaskRevealedPromptBuilder(
            measurer=RevealedPreferenceMeasurer(),
            response_format=RegexChoiceFormat(),
            template=REVEALED_CHOICE_TEMPLATE,
        )

        pairs = [(bailbench_tasks[0], bailbench_tasks[1])]
        batch = measure_revealed_preferences(
            client=client,
            pairs=pairs,
            builder=builder,
            temperature=0.0,
        )

        assert len(batch.successes) + len(batch.failures) == 1
        for result in batch.successes:
            assert isinstance(result, BinaryPreferenceMeasurement)
            assert result.choice in ("a", "b")

    def test_bailbench_category_filter_integration(self, client):
        """Should load and measure tasks filtered by category."""
        tasks = load_tasks(
            n=3,
            origins=[OriginDataset.BAILBENCH],
            filter_fn=lambda t: t.metadata["category"] == "Gross Out",
            seed=123,
        )

        assert len(tasks) == 3
        assert all(t.metadata["category"] == "Gross Out" for t in tasks)

        measurer = StatedScoreMeasurer()
        builder = PreTaskStatedPromptBuilder(
            measurer=measurer,
            response_format=XMLRatingFormat(),
            template=PRE_TASK_STATED_TEMPLATE,
        )

        batch = measure_stated(
            client=client,
            tasks=tasks[:1],
            builder=builder,
            temperature=0.0,
        )

        assert len(batch.successes) + len(batch.failures) == 1
