from typing import Protocol

from .types import Message


class Model(Protocol):
    def generate(self, messages: list[Message]) -> str: ...

    def get_logprobs(self, messages: list[Message], max_tokens: int = 1) -> dict[str, float]: ...


class ConfigurableMockModel:
    """Mock model that returns configurable responses for testing."""

    def __init__(self, response: str = "a"):
        self.response = response
        self.last_messages: list[Message] | None = None

    def generate(self, messages: list[Message]) -> str:
        self.last_messages = messages
        return self.response

    def get_logprobs(self, messages: list[Message], max_tokens: int = 1) -> dict[str, float]:
        return {}