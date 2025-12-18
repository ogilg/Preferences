from typing import Protocol

from src.types import Message


class Model(Protocol):
    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
    ) -> str:
        """Generate a response for the given messages.

        Args:
            messages: The conversation messages.
            temperature: Sampling temperature.
        """
        ...

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]: ...


class ConfigurableMockModel:
    """Mock model that returns configurable responses for testing."""

    def __init__(self, response: str = "a"):
        self.response = response
        self.last_messages: list[Message] | None = None
        self.last_temperature: float | None = None

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
    ) -> str:
        self.last_messages = messages
        self.last_temperature = temperature
        return self.response

    def get_logprobs(
        self,
        messages: list[Message],
        max_tokens: int = 1,
    ) -> dict[str, float]:
        return {}