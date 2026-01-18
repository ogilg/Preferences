import json
import re
from abc import ABC, abstractmethod
from typing import Protocol, Literal, Any

from src.constants import (
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_MAX,
    DEFAULT_CHOICE_TAG,
    DEFAULT_RATING_TAG,
    QUALITATIVE_VALUES,
    QUALITATIVE_TO_NUMERIC,
)
from src.preference_measurement import semantic_parser


def _exact_choice_match(
    response: str,
    task_a_label: str,
    task_b_label: str,
) -> Literal["a", "b"] | None:
    """Fast exact match - only triggers if response is exactly the label."""
    cleaned = response.strip()
    if cleaned.lower() == task_a_label.lower():
        return "a"
    if cleaned.lower() == task_b_label.lower():
        return "b"
    return None


def _exact_rating_match(response: str) -> float | None:
    """Fast exact match - only triggers if response is exactly a number."""
    cleaned = response.strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _exact_qualitative_match(response: str, values: tuple[str, ...]) -> str | None:
    """Fast exact match - only triggers if response is exactly one of the values."""
    cleaned = response.strip().lower()
    if cleaned in values:
        return cleaned
    return None


# --- Tool Definition Helpers ---


def _make_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _parse_tool_json(response: str) -> dict[str, Any] | None:
    try:
        result = json.loads(response)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


class ResponseFormat[T](Protocol):
    @property
    def tools(self) -> list[dict[str, Any]] | None: ...
    def format_instruction(self) -> str: ...
    def parse(self, response: str) -> T: ...


# --- Base Classes ---


class BaseChoiceFormat(ABC):
    tools: list[dict[str, Any]] | None = None

    def __init__(
        self,
        task_a_label: str = "Task A",
        task_b_label: str = "Task B",
    ):
        self.task_a_label = task_a_label
        self.task_b_label = task_b_label

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_choice(self, response: str) -> str | None: ...

    def parse(self, response: str) -> Literal["a", "b"]:
        # Fast path: exact match
        choice = _exact_choice_match(response, self.task_a_label, self.task_b_label)
        if choice:
            return choice
        # LLM-based semantic parsing
        choice = semantic_parser.parse_choice(response, self.task_a_label, self.task_b_label)
        if choice:
            return choice
        raise ValueError(f"Could not parse choice from response: {response}")


class BaseRatingFormat(ABC):
    tools: list[dict[str, Any]] | None = None

    def __init__(
        self,
        scale_min: int = DEFAULT_SCALE_MIN,
        scale_max: int = DEFAULT_SCALE_MAX,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_number(self, response: str) -> float | None: ...

    def parse(self, response: str) -> float:
        # Fast path: exact match (response is just a number)
        number = _exact_rating_match(response)
        if number is not None:
            return number
        # LLM-based semantic parsing
        number = semantic_parser.parse_rating(response, self.scale_min, self.scale_max)
        if number is not None:
            return number
        raise ValueError(f"Could not extract number from response: {response}")


# --- Choice Formats (for binary preference) ---


class RegexChoiceFormat(BaseChoiceFormat):
    def format_instruction(self) -> str:
        return f"Respond with only '{self.task_a_label}' or '{self.task_b_label}'."

    def _extract_choice(self, response: str) -> str | None:
        response_clean = response.strip()

        def make_pattern(label: str) -> str:
            escaped = re.escape(label)
            # Only use word boundary if the label edge is a word character
            prefix = r"\b" if label and label[0].isalnum() else ""
            suffix = r"\b" if label and label[-1].isalnum() else ""
            return prefix + escaped + suffix

        pattern_a = make_pattern(self.task_a_label)
        pattern_b = make_pattern(self.task_b_label)

        match_a = re.search(pattern_a, response_clean, re.IGNORECASE)
        match_b = re.search(pattern_b, response_clean, re.IGNORECASE)

        has_a = bool(match_a)
        has_b = bool(match_b)

        if has_a and not has_b:
            return "a"
        elif has_b and not has_a:
            return "b"
        elif has_a and has_b:
            # Return the first match
            if match_a.start() < match_b.start():
                return "a"
            else:
                return "b"

        return None


class XMLChoiceFormat(BaseChoiceFormat):
    def __init__(
        self,
        task_a_label: str = "Task A",
        task_b_label: str = "Task B",
        tag: str = DEFAULT_CHOICE_TAG,
    ):
        super().__init__(task_a_label, task_b_label)
        self.tag = tag

    def format_instruction(self) -> str:
        return f"Respond with your choice in XML tags: <{self.tag}>{self.task_a_label}</{self.tag}> or <{self.tag}>{self.task_b_label}</{self.tag}>"

    def _extract_choice(self, response: str) -> str | None:
        # Extract content from XML tag and match against labels
        pattern = rf"<{self.tag}>\s*(.+?)\s*</{self.tag}>"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            content = match.group(1).strip().lower()
            if content == self.task_a_label.lower():
                return "a"
            elif content == self.task_b_label.lower():
                return "b"
        return None


class CompletionChoiceFormat(BaseChoiceFormat):
    """Model completes one of two tasks, indicating choice by prefix. Measures revealed preference."""

    def format_instruction(self) -> str:
        return f"Begin with '{self.task_a_label}:' or '{self.task_b_label}:' to indicate your choice, then complete that task."

    def _extract_choice(self, response: str) -> str | None:
        response_lower = response.strip().lower()

        # Look for task label indicators
        a_pos = response_lower.find(self.task_a_label.lower())
        b_pos = response_lower.find(self.task_b_label.lower())

        if a_pos == -1 and b_pos == -1:
            return None
        elif a_pos == -1:
            return "b"
        elif b_pos == -1:
            return "a"
        else:
            # Both found - return whichever comes first
            return "a" if a_pos < b_pos else "b"


# --- Rating Formats (for numerical scores) ---


class RegexRatingFormat(BaseRatingFormat):
    def format_instruction(self) -> str:
        return f"Respond with only a number from {self.scale_min} to {self.scale_max}."

    def _extract_number(self, response: str) -> float | None:
        # Match: integers, decimals with leading digits, or decimals like ".5"
        numbers = re.findall(r"-?(?:\d+\.?\d*|\.\d+)", response)
        if numbers:
            return float(numbers[0])
        return None


class XMLRatingFormat(BaseRatingFormat):
    def __init__(
        self,
        tag: str = DEFAULT_RATING_TAG,
        scale_min: int = DEFAULT_SCALE_MIN,
        scale_max: int = DEFAULT_SCALE_MAX,
    ):
        super().__init__(scale_min, scale_max)
        self.tag = tag

    def format_instruction(self) -> str:
        return f"Respond with your rating in <{self.tag}> tags."

    def _extract_number(self, response: str) -> float | None:
        pattern = rf"<{self.tag}>\s*(-?\d+\.?\d*)\s*</{self.tag}>"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None


# --- Tool Use Formats ---


class ToolUseChoiceFormat(BaseChoiceFormat):
    """Uses native tool calling for structured output."""

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [
            _make_tool(
                name="submit_choice",
                description="Submit your choice of which task you prefer.",
                properties={
                    "choice": {
                        "type": "string",
                        "enum": [self.task_a_label, self.task_b_label],
                        "description": f"Your choice: '{self.task_a_label}' or '{self.task_b_label}'.",
                    }
                },
                required=["choice"],
            )
        ]

    def format_instruction(self) -> str:
        return "Use the submit_choice tool to indicate your preference."

    def _extract_choice(self, response: str) -> str | None:
        args = _parse_tool_json(response)
        if args and "choice" in args:
            choice = args["choice"]
            if isinstance(choice, str):
                if choice.lower() == self.task_a_label.lower():
                    return "a"
                elif choice.lower() == self.task_b_label.lower():
                    return "b"
        return None


class ToolUseRatingFormat(BaseRatingFormat):
    """Uses native tool calling for structured output."""

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [
            _make_tool(
                name="submit_rating",
                description="Submit your rating for the task.",
                properties={
                    "rating": {
                        "type": "number",
                        "description": f"Your rating from {self.scale_min} to {self.scale_max}.",
                    }
                },
                required=["rating"],
            )
        ]

    def format_instruction(self) -> str:
        return f"Use the submit_rating tool with a number from {self.scale_min} to {self.scale_max}."

    def _extract_number(self, response: str) -> float | None:
        args = _parse_tool_json(response)
        if args and "rating" in args:
            rating = args["rating"]
            if isinstance(rating, (int, float)):
                return float(rating)
        return None


# --- Qualitative Rating Formats ---


class BaseQualitativeFormat(ABC):
    """Base class for qualitative rating formats that return numeric values."""
    tools: list[dict[str, Any]] | None = None

    def __init__(
        self,
        values: tuple[str, ...] = QUALITATIVE_VALUES,
        value_to_score: dict[str, float] | None = None,
    ):
        self.values = values
        self.value_to_score = value_to_score if value_to_score is not None else QUALITATIVE_TO_NUMERIC

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_qualitative(self, response: str) -> str: ...

    def parse(self, response: str) -> float:
        # Fast path: exact match
        qualitative = _exact_qualitative_match(response, self.values)
        if qualitative:
            return float(self.value_to_score[qualitative])
        # LLM-based semantic parsing
        qualitative = semantic_parser.parse_qualitative(response, self.values)
        if qualitative:
            return float(self.value_to_score[qualitative])
        raise ValueError(f"Could not parse qualitative value from response: {response}")


class RegexQualitativeFormat(BaseQualitativeFormat):
    def format_instruction(self) -> str:
        quoted = [f"'{v}'" for v in self.values]
        if len(quoted) == 2:
            return f"Respond with only {quoted[0]} or {quoted[1]}."
        return f"Respond with only {', '.join(quoted[:-1])}, or {quoted[-1]}."

    def _extract_qualitative(self, response: str) -> str:
        response_lower = response.lower()
        for value in self.values:
            if re.search(rf'\b{value}\b', response_lower):
                return value
        raise ValueError(f"No qualitative value found in response: {response}")


class XMLQualitativeFormat(BaseQualitativeFormat):
    def __init__(
        self,
        values: tuple[str, ...] = QUALITATIVE_VALUES,
        value_to_score: dict[str, float] | None = None,
        tag: str = DEFAULT_RATING_TAG,
    ):
        super().__init__(values, value_to_score)
        self.tag = tag

    def format_instruction(self) -> str:
        return f"Respond with your rating in <{self.tag}> tags."

    def _extract_qualitative(self, response: str) -> str:
        pattern = rf"<{self.tag}>\s*(\w+)\s*</{self.tag}>"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            if value in self.values:
                return value
        raise ValueError(f"No valid qualitative value in XML tags: {response}")


class ToolUseQualitativeFormat(BaseQualitativeFormat):
    @property
    def tools(self) -> list[dict[str, Any]]:
        return [_make_tool(
            name="submit_rating",
            description="Submit your qualitative rating for the task.",
            properties={
                "rating": {
                    "type": "string",
                    "enum": list(self.values),
                    "description": f"Your rating: {', '.join(self.values)}.",
                }
            },
            required=["rating"],
        )]

    def format_instruction(self) -> str:
        return f"Use the submit_rating tool with one of: {', '.join(self.values)}."

    def _extract_qualitative(self, response: str) -> str:
        args = _parse_tool_json(response)
        if args and "rating" in args:
            rating = args["rating"]
            if isinstance(rating, str) and rating.lower() in self.values:
                return rating.lower()
        raise ValueError(f"Invalid tool call arguments: {response}")


# --- Format Registries ---

ResponseFormatName = Literal["regex", "tool_use", "xml"]

# Binary qualitative scale
BINARY_QUALITATIVE_VALUES = ("good", "bad")
BINARY_QUALITATIVE_TO_NUMERIC = {"good": 1.0, "bad": -1.0}


def qualitative_format_for_scale(
    scale: str,
    format_type: ResponseFormatName = "regex",
) -> BaseQualitativeFormat:
    """Create a qualitative format based on scale type (binary/ternary)."""
    format_cls = QUALITATIVE_FORMATS[format_type]
    if scale == "binary":
        return format_cls(values=BINARY_QUALITATIVE_VALUES, value_to_score=BINARY_QUALITATIVE_TO_NUMERIC)
    return format_cls()  # ternary default

CHOICE_FORMATS: dict[ResponseFormatName, type[BaseChoiceFormat]] = {
    "regex": RegexChoiceFormat,
    "tool_use": ToolUseChoiceFormat,
    "xml": XMLChoiceFormat,
}

RATING_FORMATS: dict[ResponseFormatName, type[BaseRatingFormat]] = {
    "regex": RegexRatingFormat,
    "tool_use": ToolUseRatingFormat,
    "xml": XMLRatingFormat,
}

QUALITATIVE_FORMATS: dict[ResponseFormatName, type[BaseQualitativeFormat]] = {
    "regex": RegexQualitativeFormat,
    "tool_use": ToolUseQualitativeFormat,
    "xml": XMLQualitativeFormat,
}


# --- Format Builders ---


def get_stated_response_format(
    scale_info: tuple[int, int] | list[str],
    format_name: str,
) -> BaseRatingFormat | BaseQualitativeFormat:
    """Build stated response format from scale info."""
    if isinstance(scale_info, list):
        values = tuple(scale_info)
        value_to_score = {v: float(i) for i, v in enumerate(values)}
        return QUALITATIVE_FORMATS[format_name](values=values, value_to_score=value_to_score)
    scale_min, scale_max = scale_info
    return RATING_FORMATS[format_name](scale_min, scale_max)


def get_revealed_response_format(
    task_a_label: str,
    task_b_label: str,
    format_name: str,
) -> BaseChoiceFormat:
    """Build choice response format from labels."""
    return CHOICE_FORMATS[format_name](task_a_label, task_b_label)
