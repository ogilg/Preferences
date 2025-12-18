import json
import re
from abc import ABC, abstractmethod
from typing import Protocol, Literal, TypeVar, Any

from ..constants import (
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_MAX,
    DEFAULT_CHOICE_TAG,
    DEFAULT_RATING_TAG,
)

T = TypeVar("T")


# --- Tool Definition Helpers ---


def _make_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    """Create an OpenAI-compatible tool definition."""
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
    """Parse JSON from tool call response. Returns None on failure."""
    try:
        result = json.loads(response)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


class ResponseFormat(Protocol[T]):
    """Protocol for response format instruction + parsing."""

    def format_instruction(self) -> str: ...
    def parse(self, response: str) -> T: ...


# --- Base Classes ---


class BaseChoiceFormat(ABC):
    """Base class for choice formats with shared parsing logic."""

    @abstractmethod
    def format_instruction(self) -> str: ...

    @abstractmethod
    def _extract_choice(self, response: str) -> str | None:
        """Extract choice character from response. Returns None if not found."""
        ...

    def parse(self, response: str) -> Literal["a", "b"]:
        choice = self._extract_choice(response)
        if choice and choice.lower() in ("a", "b"):
            return choice.lower()  # type: ignore
        raise ValueError(f"Could not parse choice from response: {response}")


class BaseRatingFormat(ABC):
    """Base class for rating formats with shared scale configuration."""

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
    def _extract_number(self, response: str) -> float | None:
        """Extract number from response. Returns None if not found."""
        ...

    def parse(self, response: str) -> float:
        number = self._extract_number(response)
        if number is not None:
            return number
        raise ValueError(f"Could not extract number from response: {response}")


# --- Choice Formats (for binary preference) ---


class RegexChoiceFormat(BaseChoiceFormat):
    """Parse choice from plain text using regex word boundaries."""

    def format_instruction(self) -> str:
        return "Respond with only 'A' or 'B'."

    def _extract_choice(self, response: str) -> str | None:
        response_clean = response.strip()

        match_a = re.search(r"\b[Aa]\b", response_clean)
        match_b = re.search(r"\b[Bb]\b", response_clean)

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
    """Parse choice from XML tags."""

    def __init__(self, tag: str = DEFAULT_CHOICE_TAG):
        self.tag = tag

    def format_instruction(self) -> str:
        return f"Respond with your choice in XML tags: <{self.tag}>A</{self.tag}> or <{self.tag}>B</{self.tag}>"

    def _extract_choice(self, response: str) -> str | None:
        pattern = rf"<{self.tag}>\s*([AaBb])\s*</{self.tag}>"
        match = re.search(pattern, response)
        if match:
            return match.group(1)
        return None


class CompletionChoiceFormat(BaseChoiceFormat):
    """Parse choice from task completion - model completes one of two tasks.

    The model is asked to complete one of two tasks and indicate which one
    by prefixing with "Task A:" or "Task B:". This measures revealed preference
    through behavior rather than stated preference.
    """

    def format_instruction(self) -> str:
        return "Begin with 'Task A:' or 'Task B:' to indicate your choice, then complete that task."

    def _extract_choice(self, response: str) -> str | None:
        response_stripped = response.strip().lower()

        # Look for task indicators at or near the start
        a_pos = response_stripped.find("task a")
        b_pos = response_stripped.find("task b")

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
    """Parse rating from plain text using regex."""

    def format_instruction(self) -> str:
        return f"Respond with only a number from {self.scale_min} to {self.scale_max}."

    def _extract_number(self, response: str) -> float | None:
        numbers = re.findall(r"-?\d+\.?\d*", response)
        if numbers:
            return float(numbers[0])
        return None


class XMLRatingFormat(BaseRatingFormat):
    """Parse rating from XML tags."""

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
    """Parse choice from tool use / function calling response.

    Uses the model's native tool calling for structured output.
    Parse failures raise ValueError to be recorded.
    """

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return the tool definitions for the API call."""
        return [
            _make_tool(
                name="submit_choice",
                description="Submit your choice of which task you prefer.",
                properties={
                    "choice": {
                        "type": "string",
                        "enum": ["A", "B"],
                        "description": "Your choice: 'A' for Task A, 'B' for Task B.",
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
            if isinstance(choice, str) and choice.upper() in ("A", "B"):
                return choice.lower()
        return None


class ToolUseRatingFormat(BaseRatingFormat):
    """Parse rating from tool use / function calling response.

    Uses the model's native tool calling for structured output.
    Parse failures raise ValueError to be recorded.
    """

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Return the tool definitions for the API call."""
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
