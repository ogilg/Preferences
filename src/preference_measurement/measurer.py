from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import (
    MeasurementResponse,
    BinaryPreferenceMeasurement,
    TaskScore,
    TaskRefusal,
    RankingMeasurement,
    RankingRefusal,
    PreferencePrompt,
)


class Measurer(ABC):
    @abstractmethod
    async def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse: ...


class RevealedPreferenceMeasurer(Measurer):
    async def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse:
        choice = await prompt.response_format.parse(response_text)
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
