from abc import ABC, abstractmethod
from typing import Any, Literal

from ..task_data import Task
from ..types import (
    Message,
    PreferenceType,
    PreferencePrompt,
    TaskCompletion,
)
from .measurer import (
    Measurer,
    BinaryPreferenceMeasurer,
    TaskScoreMeasurer,
)
from .response_format import ResponseFormat
from .templates import PromptTemplate


# =============================================================================
# Prompt Builders
# =============================================================================


class PromptBuilder(ABC):
    """Abstract base class for building preference measurement prompts.

    Attributes:
        measurer: The measurer used to parse responses.
        response_format: The format used for response parsing.
        preference_type: The type of preference being measured.
    """

    measurer: Measurer
    response_format: ResponseFormat[Any]
    preference_type: PreferenceType

    @abstractmethod
    def build(self, task: Task, *args: Any) -> PreferencePrompt:
        """Build a prompt from a task and optional additional arguments."""
        ...


class BinaryPromptBuilder(PromptBuilder):
    """Builds prompts for binary choice between two tasks."""

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
        )


class PreTaskRatingPromptBuilder(PromptBuilder):
    """Builds prompts for pre-task rating."""

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
        )


class PostTaskRatingPromptBuilder(PromptBuilder):
    """Builds prompts for post-task rating.

    Creates a multi-turn conversation:
    1. User: original task prompt
    2. Assistant: task completion
    3. User: rating request
    """

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
        )


