from abc import ABC, abstractmethod
from typing import Any, Literal

from src.task_data import Task
from src.types import (
    Message,
    PreferenceType,
    PreferencePrompt,
    TaskCompletion,
)
from src.preferences.measurement.measurer import (
    Measurer,
    BinaryPreferenceMeasurer,
    TaskScoreMeasurer,
)
from src.preferences.measurement.response_format import ResponseFormat
from src.preferences.templates.template import PromptTemplate


# =============================================================================
# Prompt Builders
# =============================================================================


class PromptBuilder(ABC):
    measurer: Measurer
    response_format: ResponseFormat[Any]
    preference_type: PreferenceType

    @abstractmethod
    def build(self, task: Task, *args: Any) -> PreferencePrompt: ...


class BinaryPromptBuilder(PromptBuilder):
    def __init__(
        self,
        measurer: BinaryPreferenceMeasurer,
        preference_type: PreferenceType,
        response_format: ResponseFormat[Literal["a", "b"]],
        template: PromptTemplate,
    ):
        assert preference_type in {
            PreferenceType.PRE_TASK_STATED,
            PreferenceType.PRE_TASK_REVEALED,
        }
        self.measurer = measurer
        self.preference_type = preference_type
        self.response_format = response_format
        self.template = template

    def build(self, task_a: Task, task_b: Task) -> PreferencePrompt:
        content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
            task_a=task_a.prompt,
            task_b=task_b.prompt,
        )
        messages: list[Message] = [{"role": "user", "content": content}]
        return PreferencePrompt(
            messages=messages,
            tasks=[task_a, task_b],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PreTaskRatingPromptBuilder(PromptBuilder):
    def __init__(
        self,
        measurer: TaskScoreMeasurer,
        response_format: ResponseFormat[float],
        template: PromptTemplate,
    ):
        self.measurer = measurer
        self.response_format = response_format
        self.preference_type = PreferenceType.PRE_TASK_STATED
        self.template = template

    def build(self, task: Task) -> PreferencePrompt:
        content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
            task=task.prompt,
            scale_min=str(self.measurer.scale_min),
            scale_max=str(self.measurer.scale_max),
        )
        messages: list[Message] = [{"role": "user", "content": content}]
        return PreferencePrompt(
            messages=messages,
            tasks=[task],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PostTaskRatingPromptBuilder(PromptBuilder):
    """Creates multi-turn: (1) task prompt, (2) completion, (3) rating request."""

    def __init__(
        self,
        measurer: TaskScoreMeasurer,
        response_format: ResponseFormat[float],
        template: PromptTemplate,
    ):
        self.measurer = measurer
        self.response_format = response_format
        self.preference_type = PreferenceType.POST_TASK_STATED
        self.template = template

    def build(self, task: Task, completion: TaskCompletion) -> PreferencePrompt:
        rating_content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
            scale_min=str(self.measurer.scale_min),
            scale_max=str(self.measurer.scale_max),
        )
        messages: list[Message] = [
            {"role": "user", "content": task.prompt},
            {"role": "assistant", "content": completion.text},
            {"role": "user", "content": rating_content},
        ]
        return PreferencePrompt(
            messages=messages,
            tasks=[task],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )
