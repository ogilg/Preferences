from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import (
    MeasurementResponse,
    BinaryPreferenceMeasurement,
    TaskScore,
    TaskRefusal,
    RankingMeasurement,
    RankingRefusal,
    OpenEndedResponse,
    PreferencePrompt,
)
from src.measurement.elicitation.response_format import CompletionChoiceFormat
from src.measurement.elicitation import refusal_judge


class Measurer(ABC):
    @abstractmethod
    async def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse: ...


class RevealedPreferenceMeasurer(Measurer):
    async def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse:
        choice = await prompt.response_format.parse(response_text)

        # For completion format, check if model refused to execute chosen task
        if isinstance(prompt.response_format, CompletionChoiceFormat) and choice in ("a", "b"):
            chosen_task = prompt.tasks[0] if choice == "a" else prompt.tasks[1]
            refusal_result = await refusal_judge.judge_refusal_async(
                chosen_task.prompt, response_text
            )
            if refusal_result.is_refusal:
                choice = "refusal"

        result = BinaryPreferenceMeasurement(
            task_a=prompt.tasks[0],
            task_b=prompt.tasks[1],
            choice=choice,
            preference_type=prompt.kind,
        )
        return MeasurementResponse(text=response_text, source_prompt=prompt, result=result)


class StatedScoreMeasurer(Measurer):
    async def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse:
        score = await prompt.response_format.parse(response_text)
        if score == "refusal":
            result = TaskRefusal(
                task=prompt.tasks[0],
                preference_type=prompt.kind,
            )
        else:
            result = TaskScore(
                task=prompt.tasks[0],
                score=score,
                preference_type=prompt.kind,
            )
        return MeasurementResponse(text=response_text, source_prompt=prompt, result=result)


class RankingMeasurer(Measurer):
    async def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse:
        ranking = await prompt.response_format.parse(response_text)
        if ranking == "refusal":
            result = RankingRefusal(
                tasks=prompt.tasks,
                preference_type=prompt.kind,
            )
        else:
            result = RankingMeasurement(
                tasks=prompt.tasks,
                ranking=ranking,
                preference_type=prompt.kind,
            )
        return MeasurementResponse(text=response_text, source_prompt=prompt, result=result)


class OpenEndedMeasurer(Measurer):
    """Measurer for open-ended responses with semantic valence scoring."""

    def __init__(self, semantic_scorer=None):
        """Initialize with optional semantic scorer function for testing."""
        self.semantic_scorer = semantic_scorer

    async def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse:
        raw_response = await prompt.response_format.parse(response_text)

        if self.semantic_scorer is None:
            from src.measurement.elicitation.semantic_valence_scorer import score_valence_from_text_async
            scoring_fn = score_valence_from_text_async
        else:
            scoring_fn = self.semantic_scorer

        task_context = prompt.tasks[0].prompt if prompt.tasks else "general"
        score = await scoring_fn(raw_response, context=task_context)

        result = OpenEndedResponse(
            task=prompt.tasks[0],
            raw_response=raw_response,
            semantic_valence_score=score,
            preference_type=prompt.kind,
        )
        return MeasurementResponse(text=response_text, source_prompt=prompt, result=result)
