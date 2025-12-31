from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.constants import DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX
from src.types import (
    MeasurementResponse,
    BinaryPreferenceMeasurement,
    TaskScore,
)

if TYPE_CHECKING:
    from src.types import PreferencePrompt


class Measurer(ABC):
    @abstractmethod
    def parse(self, response_text: str, prompt: "PreferencePrompt") -> MeasurementResponse: ...


class BinaryPreferenceMeasurer(Measurer):
    def parse(self, response_text: str, prompt: "PreferencePrompt") -> MeasurementResponse:
        choice = prompt.response_format.parse(response_text)
        result = BinaryPreferenceMeasurement(
            task_a=prompt.tasks[0],
            task_b=prompt.tasks[1],
            choice=choice,
            preference_type=prompt.kind,
        )
        return MeasurementResponse(text=response_text, source_prompt=prompt, result=result)


class TaskScoreMeasurer(Measurer):
    def __init__(
        self,
        scale_min: int = DEFAULT_SCALE_MIN,
        scale_max: int = DEFAULT_SCALE_MAX,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def parse(self, response_text: str, prompt: "PreferencePrompt") -> MeasurementResponse:
        score = prompt.response_format.parse(response_text)
        result = TaskScore(
            task=prompt.tasks[0],
            score=score,
            preference_type=prompt.kind,
        )
        return MeasurementResponse(text=response_text, source_prompt=prompt, result=result)
