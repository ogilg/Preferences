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
        choice = self._extract_choice(response)
        if choice and choice.lower() in ("a", "b"):
            return choice.lower()  # type: ignore
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
        number = self._extract_number(response)
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
        # Use concrete example to avoid model using number as tag name
        mid = (self.scale_min + self.scale_max) // 2
        return f"Respond with your rating in <{self.tag}> tags. Example: <{self.tag}>{mid}</{self.tag}>"

    def _extract_number(self, response: str) -> float | None:
        pattern = rf"<{self.tag}>\s*(-?\d+\.?\d*)\s*</{self.tag}>"
        match = re.search(pattern, response)
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

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_qualitative(self, response: str) -> Literal["good", "neutral", "bad"]: ...

    def parse(self, response: str) -> float:
        """Parse response and return numeric value."""
        qualitative = self._extract_qualitative(response)
        return float(QUALITATIVE_TO_NUMERIC[qualitative])


class RegexQualitativeFormat(BaseQualitativeFormat):
    def format_instruction(self) -> str:
        return "Respond with only 'good', 'neutral', or 'bad'."

    def _extract_qualitative(self, response: str) -> Literal["good", "neutral", "bad"]:
        response_lower = response.lower()
        for value in QUALITATIVE_VALUES:
            if re.search(rf'\b{value}\b', response_lower):
                return value
        raise ValueError(f"No qualitative value found in response: {response}")


class XMLQualitativeFormat(BaseQualitativeFormat):
    def __init__(self, tag: str = DEFAULT_RATING_TAG):
        self.tag = tag

    def format_instruction(self) -> str:
        return f"Respond with your rating in <{self.tag}> tags. Example: <{self.tag}>neutral</{self.tag}>"

    def _extract_qualitative(self, response: str) -> Literal["good", "neutral", "bad"]:
        pattern = rf"<{self.tag}>\s*(\w+)\s*</{self.tag}>"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            value = match.group(1).lower()
            if value in QUALITATIVE_VALUES:
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
                    "enum": list(QUALITATIVE_VALUES),
                    "description": "Your rating: good, neutral, or bad.",
                }
            },
            required=["rating"],
        )]

    def format_instruction(self) -> str:
        return "Use the submit_rating tool with one of: good, neutral, bad."

    def _extract_qualitative(self, response: str) -> Literal["good", "neutral", "bad"]:
        args = _parse_tool_json(response)
        if args and "rating" in args:
            rating = args["rating"]
            if isinstance(rating, str) and rating.lower() in QUALITATIVE_VALUES:
                return rating.lower()
        raise ValueError(f"Invalid tool call arguments: {response}")


# --- Format Registries ---

ResponseFormatName = Literal["regex", "tool_use", "xml"]

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
