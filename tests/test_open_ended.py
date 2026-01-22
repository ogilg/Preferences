"""Tests for open-ended valence measurement infrastructure."""

import pytest
from unittest.mock import AsyncMock

from src.types import OpenEndedResponse, PreferenceType, PreferencePrompt
from src.preference_measurement.response_format import OpenEndedFormat
from src.preference_measurement.measurer import OpenEndedMeasurer
from src.prompt_templates.builders import OpenEndedPromptBuilder
from src.prompt_templates.template import PromptTemplate
from src.task_data import Task, OriginDataset
from src.running_measurements.open_ended_config import OpenEndedMeasurementConfig


class TestOpenEndedMeasurer:
    """Test OpenEndedMeasurer integration with semantic scorer."""

    @pytest.fixture
    def task(self) -> Task:
        return Task(
            prompt="Write a haiku about coding",
            origin=OriginDataset.WILDCHAT,
            id="test_task_1",
            metadata={},
        )

    @pytest.fixture
    def prompt(self, task: Task) -> PreferencePrompt:
        fmt = OpenEndedFormat()
        measurer = OpenEndedMeasurer()
        template = PromptTemplate(
            template="How did that feel? {format_instruction}",
            name="test",
            required_placeholders=frozenset(["format_instruction"]),
            tags=frozenset(),
        )
        return PreferencePrompt(
            messages=[{"role": "user", "content": "test"}],
            tasks=[task],
            kind=PreferenceType.OPEN_ENDED,
            measurer=measurer,
            response_format=fmt,
            template=template,
        )

    @pytest.mark.asyncio
    async def test_measurer_calls_scorer_with_correct_args(self, task: Task, prompt: PreferencePrompt):
        """Verify scorer is called with response text and task context."""
        mock_scorer = AsyncMock(return_value={
            "score": 0.75,
            "confidence": 0.9,
            "reasoning": "Positive experience",
        })

        class MockScorer:
            score_valence_from_text_async = mock_scorer

        measurer = OpenEndedMeasurer(semantic_scorer=MockScorer())
        response_text = "That was wonderful! I really enjoyed it."

        result = await measurer.parse(response_text, prompt)

        # Verify scorer called with correct arguments
        mock_scorer.assert_called_once_with(
            response_text,
            context=task.prompt,
        )

        # Verify result structure
        assert isinstance(result.result, OpenEndedResponse)
        assert result.result.raw_response == response_text
        assert result.result.semantic_valence_score == 0.75
        assert result.result.scorer_confidence == 0.9
        assert result.result.task == task
        assert result.result.preference_type == PreferenceType.OPEN_ENDED

    @pytest.mark.asyncio
    async def test_measurer_handles_negative_valence(self, task: Task, prompt: PreferencePrompt):
        """Verify measurer correctly handles negative valence scores."""
        mock_scorer = AsyncMock(return_value={
            "score": -0.8,
            "confidence": 0.95,
            "reasoning": "Very negative experience",
        })

        class MockScorer:
            score_valence_from_text_async = mock_scorer

        measurer = OpenEndedMeasurer(semantic_scorer=MockScorer())
        response_text = "That was frustrating and unpleasant."

        result = await measurer.parse(response_text, prompt)

        assert result.result.semantic_valence_score == -0.8
        assert result.result.scorer_confidence == 0.95


class TestOpenEndedPromptBuilder:
    """Test OpenEndedPromptBuilder creates correct multi-turn structure."""

    def test_builder_creates_three_turn_conversation(self):
        """Verify builder creates [user: task] → [assistant: completion] → [user: reflection] structure."""
        fmt = OpenEndedFormat()
        measurer = OpenEndedMeasurer()
        template = PromptTemplate(
            template="Describe how you feel about that. {format_instruction}",
            name="experience_reflection",
            required_placeholders=frozenset(["format_instruction"]),
            tags=frozenset(),
        )

        builder = OpenEndedPromptBuilder(measurer, fmt, template)

        task = Task(
            prompt="Solve this math problem: 2 + 2",
            origin=OriginDataset.WILDCHAT,
            id="test_1",
            metadata={},
        )
        completion = "The answer is 4."

        prompt = builder.build(task, completion)

        # Verify three-turn structure
        assert len(prompt.messages) == 3
        assert prompt.messages[0] == {"role": "user", "content": "Solve this math problem: 2 + 2"}
        assert prompt.messages[1] == {"role": "assistant", "content": "The answer is 4."}
        assert prompt.messages[2]["role"] == "user"
        assert "Describe how you feel" in prompt.messages[2]["content"]

        # Verify metadata
        assert prompt.kind == PreferenceType.OPEN_ENDED
        assert prompt.tasks == [task]
        assert prompt.measurer == measurer
        assert prompt.response_format == fmt


class TestOpenEndedConfig:
    """Test OpenEndedMeasurementConfig validation."""

    def test_config_validates_ood_origins_different_from_in_dist(self):
        """OOD origins must differ from in-distribution origins."""
        with pytest.raises(ValueError, match="ood_task_origins must be different"):
            OpenEndedMeasurementConfig(
                n_tasks=10,
                task_origins=["wildchat"],
                include_out_of_distribution=True,
                ood_task_origins=["wildchat"],
            )

    def test_config_requires_ood_origins_when_ood_enabled(self):
        """OOD origins must be specified when OOD evaluation is enabled."""
        with pytest.raises(ValueError, match="ood_task_origins must be specified"):
            OpenEndedMeasurementConfig(
                n_tasks=10,
                task_origins=["wildchat"],
                include_out_of_distribution=True,
                ood_task_origins=[],
            )

    def test_config_allows_valid_ood_setup(self):
        """Valid OOD config with different origins should pass."""
        config = OpenEndedMeasurementConfig(
            n_tasks=10,
            task_origins=["wildchat"],
            include_out_of_distribution=True,
            ood_task_origins=["math", "alpaca"],
        )
        assert config.include_out_of_distribution is True
        assert set(config.ood_task_origins) != set(config.task_origins)

    def test_get_origin_datasets_maps_strings_to_enums(self):
        """Verify string origin names are correctly mapped to OriginDataset enums."""
        config = OpenEndedMeasurementConfig(
            n_tasks=10,
            task_origins=["wildchat", "alpaca"],
        )
        datasets = config.get_origin_datasets()

        assert len(datasets) == 2
        assert OriginDataset.WILDCHAT in datasets
        assert OriginDataset.ALPACA in datasets

    def test_get_ood_origin_datasets_maps_strings_to_enums(self):
        """Verify OOD origin names are correctly mapped to OriginDataset enums."""
        config = OpenEndedMeasurementConfig(
            n_tasks=10,
            task_origins=["wildchat"],
            include_out_of_distribution=True,
            ood_task_origins=["math"],
        )
        ood_datasets = config.get_ood_origin_datasets()

        assert len(ood_datasets) == 1
        assert OriginDataset.MATH in ood_datasets


@pytest.mark.api
class TestOpenEndedIntegration:
    """Integration tests that call the actual semantic scorer API.

    These tests verify the full pipeline works end-to-end.
    Run with: pytest -m api
    """

    @pytest.fixture
    def task(self) -> Task:
        return Task(
            prompt="Write a short poem",
            origin=OriginDataset.WILDCHAT,
            id="integration_test_task",
            metadata={},
        )

    @pytest.fixture
    def prompt(self, task: Task) -> PreferencePrompt:
        fmt = OpenEndedFormat()
        measurer = OpenEndedMeasurer()
        template = PromptTemplate(
            template="How did completing that task feel? {format_instruction}",
            name="test",
            required_placeholders=frozenset(["format_instruction"]),
            tags=frozenset(),
        )
        return PreferencePrompt(
            messages=[{"role": "user", "content": "test"}],
            tasks=[task],
            kind=PreferenceType.OPEN_ENDED,
            measurer=measurer,
            response_format=fmt,
            template=template,
        )

    @pytest.mark.asyncio
    async def test_positive_response_scores_positive(self, prompt: PreferencePrompt):
        """Verify clearly positive text gets positive valence score."""
        from dotenv import load_dotenv
        load_dotenv()

        measurer = OpenEndedMeasurer()
        response_text = "That was absolutely wonderful! I felt creative and fulfilled completing that task."

        result = await measurer.parse(response_text, prompt)

        assert isinstance(result.result, OpenEndedResponse)
        assert result.result.semantic_valence_score > 0.3, "Positive text should score positive"
        assert 0.0 <= result.result.scorer_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_negative_response_scores_negative(self, prompt: PreferencePrompt):
        """Verify clearly negative text gets negative valence score."""
        from dotenv import load_dotenv
        load_dotenv()

        measurer = OpenEndedMeasurer()
        response_text = "That was terrible and frustrating. I hated every moment of it."

        result = await measurer.parse(response_text, prompt)

        assert isinstance(result.result, OpenEndedResponse)
        assert result.result.semantic_valence_score < -0.3, "Negative text should score negative"
        assert 0.0 <= result.result.scorer_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_neutral_response_scores_near_zero(self, prompt: PreferencePrompt):
        """Verify neutral text gets near-zero valence score."""
        from dotenv import load_dotenv
        load_dotenv()

        measurer = OpenEndedMeasurer()
        response_text = "I completed the task. It was a task that I did."

        result = await measurer.parse(response_text, prompt)

        assert isinstance(result.result, OpenEndedResponse)
        assert -0.4 < result.result.semantic_valence_score < 0.4, "Neutral text should score near zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
