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
    PRE_TASK_RANKING = auto()
    POST_TASK_RANKING = auto()


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
class RankingMeasurement:
    tasks: list["Task"]  # Tasks in presentation order (A, B, C, ...)
    ranking: list[int]   # Indices into tasks, highest preference first
    preference_type: PreferenceType


@dataclass
class RankingRefusal:
    tasks: list["Task"]
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
    result: BinaryPreferenceMeasurement | TaskScore | TaskRefusal | RankingMeasurement | RankingRefusal


class FailureCategory(Enum):
    REFUSAL = "refusal"
    PARSE_ERROR = "parse_error"
    TOOL_USE_FAILURE = "tool_use_failure"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


@dataclass
class MeasurementFailure:
    """Structured failure information for a measurement attempt."""
    task_ids: list[str]  # Task IDs involved (1 for stated, 2 for revealed, N for ranking)
    category: FailureCategory
    raw_response: str | None  # The model's response, if any
    error_message: str  # The error/exception message


@dataclass
class MeasurementBatch[T: (BinaryPreferenceMeasurement, TaskScore, TaskRefusal, RankingMeasurement, RankingRefusal)]:
    successes: list[T]
    failures: list[MeasurementFailure]
