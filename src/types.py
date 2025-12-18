from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from typing import Any
    from .task_data import Task
    from .preferences.measurer import Measurer
    from .preferences.response_format import ResponseFormat


class Message(TypedDict):
    """A single message in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


class PreferenceType(Enum):
    PRE_TASK_STATED = auto()
    PRE_TASK_REVEALED = auto()  
    POST_TASK_STATED = auto()


@dataclass
class TaskCompletion:
    task: "Task"
    text: str


@dataclass
class BinaryPreferenceMeasurement:
    task_a: "Task"
    task_b: "Task"
    choice: Literal["a", "b"]
    preference_type: PreferenceType


@dataclass
class TaskScore:
    task: "Task"
    score: float
    preference_type: PreferenceType


@dataclass
class PreferencePrompt:
    messages: list[Message]
    tasks: list["Task"]
    kind: PreferenceType
    measurer: "Measurer"
    response_format: "ResponseFormat[Any]"


@dataclass
class MeasurementResponse:
    text: str
    source_prompt: PreferencePrompt
    result: BinaryPreferenceMeasurement | TaskScore
