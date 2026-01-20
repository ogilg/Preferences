from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from .task_data import Task
    from .preferences.measurement.measurer import Measurer
    from .preferences.measurement.response_format import ResponseFormat
    from .preferences.templates import PromptTemplate


class Message(TypedDict):
    """A single message in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class PreferenceType(Enum):
    PRE_TASK_STATED = auto()
    PRE_TASK_REVEALED = auto()
    POST_TASK_STATED = auto()
    POST_TASK_REVEALED = auto()


@dataclass
class BinaryPreferenceMeasurement:
    task_a: "Task"
    task_b: "Task"
    choice: Literal["a", "b", "refusal"]
    preference_type: PreferenceType


@dataclass
class TaskScore:
    task: "Task"
    score: float
    preference_type: PreferenceType


@dataclass
class TaskRefusal:
    task: "Task"
    preference_type: PreferenceType


@dataclass
class PreferencePrompt:
    messages: list[Message]
    tasks: list["Task"]
    kind: PreferenceType
    measurer: "Measurer"
    response_format: "ResponseFormat[Any]"
    template: "PromptTemplate"


@dataclass
class MeasurementResponse:
    text: str
    source_prompt: PreferencePrompt
    result: BinaryPreferenceMeasurement | TaskScore | TaskRefusal


@dataclass
class MeasurementBatch[T: (BinaryPreferenceMeasurement, TaskScore, TaskRefusal)]:
    successes: list[T]
    failures: list[tuple[PreferencePrompt, str]]
