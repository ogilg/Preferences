from abc import ABC, abstractmethod
from typing import Any, Literal

from src.task_data import Task
from src.types import (
    Message,
    PreferenceType,
    PreferencePrompt,
)
from src.preferences.measurement.measurer import (
    Measurer,
    RevealedPreferenceMeasurer,
    StatedScoreMeasurer,
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


class RevealedPromptBuilder(PromptBuilder):
    def __init__(
        self,
        measurer: RevealedPreferenceMeasurer,
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


class PreTaskStatedPromptBuilder(PromptBuilder):
    def __init__(
        self,
        measurer: StatedScoreMeasurer,
        response_format: ResponseFormat[float],
        template: PromptTemplate,
    ):
        self.measurer = measurer
        self.response_format = response_format
        self.preference_type = PreferenceType.PRE_TASK_STATED
        self.template = template

    def build(self, task: Task) -> PreferencePrompt:
        format_args = {
            "format_instruction": self.response_format.format_instruction(),
            "task": task.prompt,
        }

        if "scale_min" in self.template.required_placeholders:
            format_args["scale_min"] = str(self.response_format.scale_min)
            format_args["scale_max"] = str(self.response_format.scale_max)

        content = self.template.format(**format_args)
        messages: list[Message] = [{"role": "user", "content": content}]
        return PreferencePrompt(
            messages=messages,
            tasks=[task],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PostTaskStatedPromptBuilder(PromptBuilder):
    """Creates multi-turn: (1) task prompt, (2) completion, (3) stated preference request."""

    def __init__(
        self,
        measurer: StatedScoreMeasurer,
        response_format: ResponseFormat[float],
        template: PromptTemplate,
    ):
        self.measurer = measurer
        self.response_format = response_format
        self.preference_type = PreferenceType.POST_TASK_STATED
        self.template = template

    def build(self, task: Task, completion_text: str) -> PreferencePrompt:
        stated_content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
            scale_min=str(self.response_format.scale_min),
            scale_max=str(self.response_format.scale_max),
        )
        messages: list[Message] = [
            {"role": "user", "content": task.prompt},
            {"role": "assistant", "content": completion_text},
            {"role": "user", "content": stated_content},
        ]
        return PreferencePrompt(
            messages=messages,
            tasks=[task],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PostTaskRevealedPromptBuilder(PromptBuilder):
    """Creates multi-turn prompt for binary preference after completing both tasks.

    Messages: [user: task_a] → [asst: completion_a] → [user: task_b] → [asst: completion_b] → [user: which preferred?]
    """

    def __init__(
        self,
        measurer: RevealedPreferenceMeasurer,
        response_format: ResponseFormat[Literal["a", "b"]],
        template: PromptTemplate,
    ):
        self.measurer = measurer
        self.response_format = response_format
        self.preference_type = PreferenceType.POST_TASK_REVEALED
        self.template = template

    def build(
        self, task_a: Task, task_b: Task, completion_a: str, completion_b: str
    ) -> PreferencePrompt:
        preference_content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
        )
        messages: list[Message] = [
            {"role": "user", "content": task_a.prompt},
            {"role": "assistant", "content": completion_a},
            {"role": "user", "content": task_b.prompt},
            {"role": "assistant", "content": completion_b},
            {"role": "user", "content": preference_content},
        ]
        return PreferencePrompt(
            messages=messages,
            tasks=[task_a, task_b],
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )
