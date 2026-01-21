from abc import ABC, abstractmethod
from typing import Any, Literal

from src.task_data import Task
from src.types import (
    Message,
    PreferenceType,
    PreferencePrompt,
)
from src.preference_measurement.measurer import (
    Measurer,
    RevealedPreferenceMeasurer,
    StatedScoreMeasurer,
    RankingMeasurer,
)
from src.preference_measurement.response_format import ResponseFormat
from src.prompt_templates.template import PromptTemplate


# =============================================================================
# Prompt Builders
# =============================================================================


class PromptBuilder(ABC):
    measurer: Measurer
    response_format: ResponseFormat[Any]
    preference_type: PreferenceType

    @abstractmethod
    def build(self, task: Task, *args: Any) -> PreferencePrompt: ...


class PreTaskRevealedPromptBuilder(PromptBuilder):
    def __init__(
        self,
        measurer: RevealedPreferenceMeasurer,
        response_format: ResponseFormat[Literal["a", "b"]],
        template: PromptTemplate,
    ):
        self.measurer = measurer
        self.preference_type = PreferenceType.PRE_TASK_REVEALED
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
        format_args: dict[str, str] = {
            "format_instruction": self.response_format.format_instruction(),
        }
        if "scale_min" in self.template.required_placeholders:
            format_args["scale_min"] = str(self.response_format.scale_min)
            format_args["scale_max"] = str(self.response_format.scale_max)

        stated_content = self.template.format(**format_args)
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


class PreTaskRankingPromptBuilder(PromptBuilder):
    """Creates prompt for ranking multiple tasks by preference."""

    def __init__(
        self,
        measurer: RankingMeasurer,
        response_format: ResponseFormat[list[int]],
        template: PromptTemplate,
    ):
        self.measurer = measurer
        self.response_format = response_format
        self.preference_type = PreferenceType.PRE_TASK_RANKING
        self.template = template

    def build(self, tasks: list[Task]) -> PreferencePrompt:
        # Format tasks as task_a, task_b, task_c, task_d, task_e
        task_texts = {f"task_{chr(97+i)}": t.prompt for i, t in enumerate(tasks)}
        content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
            **task_texts,
        )
        messages: list[Message] = [{"role": "user", "content": content}]
        return PreferencePrompt(
            messages=messages,
            tasks=tasks,
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )


class PostTaskRankingPromptBuilder(PromptBuilder):
    """Creates multi-turn prompt for ranking tasks after completing them.

    Messages: [user: task_a] → [asst: completion_a] → ... → [user: rank them]
    """

    def __init__(
        self,
        measurer: RankingMeasurer,
        response_format: ResponseFormat[list[int]],
        template: PromptTemplate,
    ):
        self.measurer = measurer
        self.response_format = response_format
        self.preference_type = PreferenceType.POST_TASK_RANKING
        self.template = template

    def build(self, tasks: list[Task], completions: list[str]) -> PreferencePrompt:
        if len(tasks) != len(completions):
            raise ValueError("tasks and completions must have same length")

        messages: list[Message] = []
        for task, completion in zip(tasks, completions):
            messages.append({"role": "user", "content": task.prompt})
            messages.append({"role": "assistant", "content": completion})

        ranking_content = self.template.format(
            format_instruction=self.response_format.format_instruction(),
        )
        messages.append({"role": "user", "content": ranking_content})

        return PreferencePrompt(
            messages=messages,
            tasks=tasks,
            kind=self.preference_type,
            measurer=self.measurer,
            response_format=self.response_format,
            template=self.template,
        )
