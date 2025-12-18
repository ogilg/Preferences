import re
from abc import ABC, abstractmethod
from typing import Protocol, Literal, TypeVar

from ..constants import (
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_MAX,
    DEFAULT_CHOICE_TAG,
    DEFAULT_RATING_TAG,
)

T = TypeVar("T")


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
        return f"Respond with your rating in XML tags: <{self.tag}>NUMBER</{self.tag}> where NUMBER is from {self.scale_min} to {self.scale_max}."

    def _extract_number(self, response: str) -> float | None:
        pattern = rf"<{self.tag}>\s*(-?\d+\.?\d*)\s*</{self.tag}>"
        match = re.search(pattern, response)
        if match:
            return float(match.group(1))
        return None
