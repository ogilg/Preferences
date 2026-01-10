from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import (
    MeasurementResponse,
    BinaryPreferenceMeasurement,
    TaskScore,
    PreferencePrompt,
)


class Measurer(ABC):
    @abstractmethod
    def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse: ...


class BinaryPreferenceMeasurer(Measurer):
    def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse:
        choice = prompt.response_format.parse(response_text)
        result = BinaryPreferenceMeasurement(
            task_a=prompt.tasks[0],
            task_b=prompt.tasks[1],
            choice=choice,
            preference_type=prompt.kind,
        )
        return MeasurementResponse(text=response_text, source_prompt=prompt, result=result)


class TaskScoreMeasurer(Measurer):
    def parse(self, response_text: str, prompt: PreferencePrompt) -> MeasurementResponse:
        score = prompt.response_format.parse(response_text)
        result = TaskScore(
            task=prompt.tasks[0],
            score=score,
            preference_type=prompt.kind,
        )
        return MeasurementResponse(text=response_text, source_prompt=prompt, result=result)
